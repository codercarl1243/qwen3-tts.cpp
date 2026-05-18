/* streaming_internal.h — non-public C++ surface for the chunk-streaming
 * algorithm. Visible to tests via the project's src/ include path; not part
 * of the installed public header set.
 */
#ifndef QWEN3TTS_STREAMING_INTERNAL_H
#define QWEN3TTS_STREAMING_INTERNAL_H

#include <atomic>
#include <cstddef>
#include <functional>
#include <vector>

namespace qwen3tts {

class ChunkStreamer {
   public:
    using ChunkSink = std::function<void(const float* pcm,
                                         std::size_t   n_samples,
                                         void*         user_data)>;
    using SamplerPenaltyHook = std::function<void(float severity)>;

    static constexpr float SILENCE_THRESHOLD     = 1e-4f;
    static constexpr int   PENALTY_THRESHOLD     = 5;
    static constexpr int   HARD_STOP_THRESHOLD   = 15;

    ChunkStreamer(std::size_t        chunk_frames,
                  std::size_t        frame_samples,
                  ChunkSink          sink,
                  void*              user_data,
                  SamplerPenaltyHook penalty = nullptr);

    /* Push a single codec frame (frame_samples float32 samples).
     * Returns true while the streamer is willing to accept more frames,
     * false once hard-stop / cancel / finish has terminated the stream. */
    bool push_frame(const float* pcm);

    /* Signal end of input. Emits any partial chunk as the terminal chunk;
     * idempotent. */
    void finish();

    /* Cooperative cancellation; observed at the next push_frame() call. */
    void cancel();

    bool        hard_stop() const { return hard_stop_; }
    bool        finished() const  { return finished_;  }
    int         silent_run() const { return silent_run_; }
    int         chunk_count() const { return chunk_count_; }
    std::size_t chunk_frames() const { return chunk_frames_; }
    std::size_t frame_samples() const { return frame_samples_; }

   private:
    void        track_silence(const float* pcm);
    bool        is_silent_frame(const float* pcm) const;
    void        emit_chunk();
    void        apply_fade_in_first_frame();
    void        apply_fade_out_last_frame();
    void        flush_final();
    std::size_t frames_in_buf() const;

    std::size_t        chunk_frames_;
    std::size_t        frame_samples_;
    ChunkSink          sink_;
    void*              user_data_;
    SamplerPenaltyHook penalty_hook_;

    std::vector<float> chunk_buf_;

    int               silent_run_;
    bool              hard_stop_;
    std::atomic<bool> cancel_;
    bool              finished_;
    int               chunk_count_;
};

}  // namespace qwen3tts

#endif  // QWEN3TTS_STREAMING_INTERNAL_H
