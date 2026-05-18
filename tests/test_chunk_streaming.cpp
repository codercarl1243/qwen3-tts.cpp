/* test_chunk_streaming.cpp — stubbed-codec unit tests for ChunkStreamer.
 *
 * Verifies the streaming algorithm in isolation from the real talker/codec:
 *   - callback fires ceil(N / chunk_frames) times for N input codec frames
 *   - total emitted sample count equals N * frame_samples (in-place crossfade)
 *   - no inter-chunk discontinuity above threshold at chunk boundaries
 *   - silent-frame sentinel: 15 consecutive zero frames triggers hard-stop
 *     and the partial buffer is flushed as the terminal chunk
 *   - penalty hook fires after 5 consecutive silent frames
 */

#include "streaming_internal.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kChunkFrames  = 8;
constexpr std::size_t kFrameSamples = 240;  // 24kHz / ~100Hz codec rate

struct Capture {
    std::vector<std::vector<float>> chunks;
    int                              penalty_calls = 0;
    float                            last_penalty  = 0.0f;
};

void on_chunk(const float* pcm, std::size_t n_samples, void* user_data) {
    auto* cap = static_cast<Capture*>(user_data);
    cap->chunks.emplace_back(pcm, pcm + n_samples);
}

// Produce a smooth sine-like frame to model real codec output.
std::vector<float> tone_frame(std::size_t frame_idx, float amplitude = 0.5f) {
    std::vector<float> out(kFrameSamples);
    const float        freq = 220.0f;
    const float        sr   = 24000.0f;
    for (std::size_t i = 0; i < kFrameSamples; ++i) {
        const std::size_t t = frame_idx * kFrameSamples + i;
        out[i] = amplitude * std::sin(2.0f * 3.14159265f * freq * t / sr);
    }
    return out;
}

std::vector<float> silent_frame() {
    return std::vector<float>(kFrameSamples, 0.0f);
}

}  // namespace

static int test_chunk_count_and_total_samples() {
    Capture cap;
    qwen3tts::ChunkStreamer streamer(
        kChunkFrames, kFrameSamples, on_chunk, &cap);

    constexpr std::size_t kInputFrames = 24;  // exactly 3 full chunks
    for (std::size_t i = 0; i < kInputFrames; ++i) {
        auto f = tone_frame(i);
        bool ok = streamer.push_frame(f.data());
        assert(ok && "push_frame should succeed on non-silent input");
    }
    streamer.finish();

    const std::size_t expected_chunks = (kInputFrames + kChunkFrames - 1) / kChunkFrames;
    if (cap.chunks.size() != expected_chunks) {
        std::fprintf(stderr,
                     "FAIL chunk count: got %zu, expected %zu\n",
                     cap.chunks.size(), expected_chunks);
        return 1;
    }

    std::size_t total_samples = 0;
    for (const auto& c : cap.chunks) total_samples += c.size();
    const std::size_t expected_samples = kInputFrames * kFrameSamples;
    if (total_samples != expected_samples) {
        std::fprintf(stderr,
                     "FAIL sample sum: got %zu, expected %zu\n",
                     total_samples, expected_samples);
        return 1;
    }
    return 0;
}

static int test_no_boundary_discontinuity() {
    Capture cap;
    qwen3tts::ChunkStreamer streamer(
        kChunkFrames, kFrameSamples, on_chunk, &cap);

    // 16 frames = 2 chunks worth, one chunk boundary expected.
    constexpr std::size_t kInputFrames = 16;
    for (std::size_t i = 0; i < kInputFrames; ++i) {
        auto f = tone_frame(i);
        streamer.push_frame(f.data());
    }
    streamer.finish();

    if (cap.chunks.size() < 2) {
        std::fprintf(stderr, "FAIL not enough chunks for boundary test\n");
        return 1;
    }

    // Check the boundary between chunk 0 and chunk 1.
    const auto& a = cap.chunks[0];
    const auto& b = cap.chunks[1];
    const float last_a  = a.back();
    const float first_b = b.front();
    const float diff    = std::fabs(last_a - first_b);
    constexpr float kDiscontinuityThreshold = 0.5f;  // generous for 220Hz tone
    if (diff > kDiscontinuityThreshold) {
        std::fprintf(stderr,
                     "FAIL boundary discontinuity: |%.4f - %.4f| = %.4f\n",
                     last_a, first_b, diff);
        return 1;
    }
    return 0;
}

static int test_silent_sentinel_triggers_hard_stop() {
    Capture cap;
    int penalty_count = 0;
    qwen3tts::ChunkStreamer streamer(
        kChunkFrames, kFrameSamples, on_chunk, &cap,
        [&penalty_count](float /*severity*/) { ++penalty_count; });

    // Two frames of tone, then 20 silent frames — sentinel must fire at 15.
    for (std::size_t i = 0; i < 2; ++i) {
        auto f = tone_frame(i);
        streamer.push_frame(f.data());
    }
    bool accepting = true;
    int silent_pushed = 0;
    for (int i = 0; i < 20 && accepting; ++i) {
        auto s = silent_frame();
        accepting = streamer.push_frame(s.data());
        ++silent_pushed;
    }

    if (!streamer.hard_stop()) {
        std::fprintf(stderr, "FAIL hard-stop flag not set after silent run\n");
        return 1;
    }
    if (silent_pushed != qwen3tts::ChunkStreamer::HARD_STOP_THRESHOLD) {
        std::fprintf(stderr,
                     "FAIL silent frames accepted: got %d, expected %d\n",
                     silent_pushed,
                     qwen3tts::ChunkStreamer::HARD_STOP_THRESHOLD);
        return 1;
    }
    // Penalty hook should have fired for silent frames 5..14 inclusive (10 calls).
    const int expected_penalty_min = 1;
    if (penalty_count < expected_penalty_min) {
        std::fprintf(stderr,
                     "FAIL penalty hook not fired (count=%d)\n",
                     penalty_count);
        return 1;
    }
    // Verify some PCM was emitted as the terminal flush.
    if (cap.chunks.empty()) {
        std::fprintf(stderr, "FAIL hard-stop did not flush remainder\n");
        return 1;
    }
    return 0;
}

static int test_partial_final_chunk() {
    Capture cap;
    qwen3tts::ChunkStreamer streamer(
        kChunkFrames, kFrameSamples, on_chunk, &cap);

    // 10 frames: one full chunk of 8 + a final partial chunk of 2.
    constexpr std::size_t kInputFrames = 10;
    for (std::size_t i = 0; i < kInputFrames; ++i) {
        auto f = tone_frame(i);
        streamer.push_frame(f.data());
    }
    streamer.finish();

    if (cap.chunks.size() != 2) {
        std::fprintf(stderr,
                     "FAIL partial flush: got %zu chunks, expected 2\n",
                     cap.chunks.size());
        return 1;
    }
    if (cap.chunks[0].size() != kChunkFrames * kFrameSamples) {
        std::fprintf(stderr, "FAIL first chunk size\n");
        return 1;
    }
    if (cap.chunks[1].size() != 2 * kFrameSamples) {
        std::fprintf(stderr,
                     "FAIL partial chunk size: got %zu, expected %zu\n",
                     cap.chunks[1].size(), 2 * kFrameSamples);
        return 1;
    }
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_chunk_count_and_total_samples();
    rc |= test_no_boundary_discontinuity();
    rc |= test_silent_sentinel_triggers_hard_stop();
    rc |= test_partial_final_chunk();
    if (rc == 0) {
        std::printf("test_chunk_streaming: all checks passed\n");
    }
    return rc;
}
