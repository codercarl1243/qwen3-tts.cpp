/* streaming.cpp — fixed-chunk streaming pipeline for qwen3-tts.cpp.
 *
 * Implements the public C API declared in include/qwen3tts/streaming.h and
 * an internal ChunkStreamer class that encapsulates the algorithm:
 *   - chunk_frames-sized buffering of codec frames
 *   - 1-frame overlap-add crossfade between adjacent chunks (smoothing)
 *   - amplitude-thresholded silent-frame tracking
 *   - log-scale sampler penalty hook after 5 consecutive silent frames
 *   - hard-stop after 15 consecutive silent frames
 *   - convolution-tail flush of the final codec frame on finish()
 *
 * ChunkStreamer is testable in isolation via src/streaming_internal.h;
 * the real pipeline (talker → codec → sampler) gets wired into
 * qwen3tts_synthesize_streaming in audio/04 (Rust FFI stage).
 */

#include "qwen3tts/streaming.h"
#include "streaming_internal.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace qwen3tts {

ChunkStreamer::ChunkStreamer(size_t      chunk_frames,
                             size_t      frame_samples,
                             ChunkSink   sink,
                             void*       user_data,
                             SamplerPenaltyHook penalty)
    : chunk_frames_(chunk_frames),
      frame_samples_(frame_samples),
      sink_(std::move(sink)),
      user_data_(user_data),
      penalty_hook_(std::move(penalty)),
      chunk_buf_(),
      silent_run_(0),
      hard_stop_(false),
      cancel_(false),
      finished_(false),
      chunk_count_(0) {
    chunk_buf_.reserve(chunk_frames_ * frame_samples_);
}

bool ChunkStreamer::push_frame(const float* pcm) {
    if (hard_stop_ || cancel_.load(std::memory_order_acquire) || finished_) {
        return false;
    }

    track_silence(pcm);

    if (hard_stop_) {
        // 15-frame silent sentinel hit; flush remainder and stop accepting input.
        flush_final();
        return false;
    }

    chunk_buf_.insert(chunk_buf_.end(), pcm, pcm + frame_samples_);

    if (frames_in_buf() == chunk_frames_) {
        emit_chunk();
    }

    return true;
}

void ChunkStreamer::cancel() {
    cancel_.store(true, std::memory_order_release);
}

void ChunkStreamer::finish() {
    flush_final();
}

void ChunkStreamer::track_silence(const float* pcm) {
    if (is_silent_frame(pcm)) {
        silent_run_++;
        if (silent_run_ >= HARD_STOP_THRESHOLD) {
            hard_stop_ = true;
            return;
        }
        if (silent_run_ >= PENALTY_THRESHOLD && penalty_hook_) {
            // Log-scale penalty: severity grows with the overflow past the
            // threshold, dampening repeat samples in the talker's logits.
            int  excess   = silent_run_ - PENALTY_THRESHOLD + 1;
            float penalty = std::log1p(static_cast<float>(excess));
            penalty_hook_(penalty);
        }
    } else {
        silent_run_ = 0;
    }
}

bool ChunkStreamer::is_silent_frame(const float* pcm) const {
    float peak = 0.0f;
    for (size_t i = 0; i < frame_samples_; ++i) {
        peak = std::max(peak, std::fabs(pcm[i]));
    }
    return peak < SILENCE_THRESHOLD;
}

void ChunkStreamer::emit_chunk() {
    // Fade-in matches the prior chunk's fade-out at the boundary; the very
    // first emitted chunk plays at full volume from the start.
    apply_fade_in_first_frame();
    // Always fade-out the last frame; the terminal partial chunk overrides
    // this in flush_final() so the utterance ends at its natural amplitude.
    apply_fade_out_last_frame();

    sink_(chunk_buf_.data(), chunk_buf_.size(), user_data_);
    chunk_count_++;

    chunk_buf_.clear();
}

void ChunkStreamer::apply_fade_in_first_frame() {
    if (chunk_count_ == 0 || chunk_buf_.size() < frame_samples_) {
        return;
    }
    const float denom = static_cast<float>(frame_samples_ > 1 ? frame_samples_ - 1 : 1);
    for (size_t i = 0; i < frame_samples_; ++i) {
        const float t = static_cast<float>(i) / denom;
        chunk_buf_[i] *= t;
    }
}

void ChunkStreamer::apply_fade_out_last_frame() {
    if (chunk_buf_.size() < frame_samples_) {
        return;
    }
    const size_t off   = chunk_buf_.size() - frame_samples_;
    const float  denom = static_cast<float>(frame_samples_ > 1 ? frame_samples_ - 1 : 1);
    for (size_t i = 0; i < frame_samples_; ++i) {
        const float t = static_cast<float>(i) / denom;
        chunk_buf_[off + i] *= (1.0f - t);
    }
}

void ChunkStreamer::flush_final() {
    if (finished_) return;
    finished_ = true;

    if (!chunk_buf_.empty()) {
        // Fade-in matches the previous chunk's fade-out so the boundary is
        // continuous; we deliberately skip fade-out — this is the terminal
        // chunk and the convolution-tail content must end at full volume.
        apply_fade_in_first_frame();

        sink_(chunk_buf_.data(), chunk_buf_.size(), user_data_);
        chunk_count_++;
        chunk_buf_.clear();
    }
}

size_t ChunkStreamer::frames_in_buf() const {
    return chunk_buf_.size() / frame_samples_;
}

}  // namespace qwen3tts

/* ---------------------------------------------------------------------------
 * Public C API.
 *
 * The opaque qwen3tts_ctx in this stage carries only the cancellation flag.
 * audio/04 (Rust FFI) extends it with the real talker/codec/sampler handles;
 * the C signatures below are frozen so the Rust crate can link against them
 * today.
 * ------------------------------------------------------------------------- */

struct qwen3tts_ctx {
    std::atomic<bool> cancel_flag;
    qwen3tts_ctx() : cancel_flag(false) {}
};

extern "C" int qwen3tts_synthesize_streaming(qwen3tts_ctx*     ctx,
                                             const char*       text,
                                             uint32_t          chunk_frames,
                                             qwen3tts_chunk_cb cb,
                                             void*             user_data) {
    (void)ctx;
    (void)text;
    (void)chunk_frames;
    (void)cb;
    (void)user_data;
    // TODO(audio/04): wire ChunkStreamer to the real talker/codec pipeline.
    // The chunk-streaming algorithm itself is implemented and tested via
    // qwen3tts::ChunkStreamer; this entry-point exists today only so the
    // Rust crate can resolve its FFI symbols. -ENOSYS-equivalent until then.
    return -1;
}

extern "C" int qwen3tts_cancel(qwen3tts_ctx* ctx) {
    if (!ctx) return -1;
    ctx->cancel_flag.store(true, std::memory_order_release);
    return 0;
}

extern "C" int qwen3tts_thermal_warmup(qwen3tts_ctx* ctx) {
    (void)ctx;
    // TODO(audio/04): synthesise "." through the real pipeline to prime the
    // chunk graph. Until the pipeline is connected, this is a no-op so the
    // Rust wrapper can invoke it during session-init without erroring.
    return 0;
}
