/* qwen3tts/streaming.h — public C API for fixed-chunk streaming synthesis.
 *
 * Kaiwa fork addition (codercarl1243/qwen3-tts.cpp@kaiwa-streaming).
 * Exposes the streaming entry-points that Rust FFI (audio/04) binds to:
 *   - qwen3tts_synthesize_streaming(): emits 24kHz mono float32 PCM via callback
 *   - qwen3tts_cancel(): cooperative cancellation at the next chunk boundary
 *   - qwen3tts_thermal_warmup(): builds the chunk graph before user-visible work
 *
 * Algorithm details (overlap-add crossfade, silent-frame sampler penalty,
 * convolution-tail flush) live in src/pipeline/streaming.cpp.
 */
#ifndef QWEN3TTS_STREAMING_H
#define QWEN3TTS_STREAMING_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque streaming context. Holds the loaded Qwen3Tts engine plus the
 * cancellation flag; constructed via qwen3tts_context_new and released via
 * qwen3tts_context_free. Tests may construct it directly via the C++ helpers
 * in streaming.cpp. */
typedef struct qwen3tts_ctx qwen3tts_ctx;

/* Create a streaming context, loading models from model_dir.
 *
 * model_dir must be a directory bundle containing the GGUF weights and
 * tokenizer expected by Qwen3Tts::load_models.
 *
 * Returns NULL on failure (model directory missing or weights fail to load).
 * The returned pointer must be freed with qwen3tts_context_free. */
qwen3tts_ctx* qwen3tts_context_new(const char* model_dir);

/* Release a context created by qwen3tts_context_new. No-op on NULL. */
void qwen3tts_context_free(qwen3tts_ctx* ctx);

/* Chunk callback: invoked once per fixed-size PCM chunk.
 *   pcm        — 24kHz mono float32 samples, valid for the duration of the call
 *   n_samples  — number of float samples (== chunk_frames * frame_samples)
 *   user_data  — opaque pointer round-tripped from synthesize_streaming
 *
 * The callback must not retain pcm beyond the call — copy if needed.
 */
typedef void (*qwen3tts_chunk_cb)(const float* pcm,
                                  size_t       n_samples,
                                  void*        user_data);

/* Streaming entry-point.
 *
 * Runs the talker/predictor loop and emits PCM in groups of `chunk_frames`
 * codec frames via `cb`. Each chunk is decoded with left-context warm-up
 * (the preceding frames are re-decoded to give the causal conv stack its true
 * receptive field, then their samples are dropped) so chunk joins are clean
 * without an overlap-add crossfade. The trailing partial chunk is flushed when
 * the talker emits EOS or hits its frame cap.
 *
 * n_samples passed to `cb` is the decoded payload size for the chunk; it is
 * approximately chunk_frames * frame_samples but may vary slightly at edges,
 * so callers must use n_samples rather than assuming a fixed chunk length.
 *
 * Returns 0 on success; non-zero on error. On cancellation the call returns 0
 * after the held-back final chunk is flushed.
 */
int qwen3tts_synthesize_streaming(qwen3tts_ctx*       ctx,
                                  const char*         text,
                                  uint32_t            chunk_frames,
                                  qwen3tts_chunk_cb   cb,
                                  void*               user_data);

/* Request that the in-flight qwen3tts_synthesize_streaming call abort at the
 * next chunk boundary. Idempotent; safe to call from any thread. */
int qwen3tts_cancel(qwen3tts_ctx* ctx);

/* Synthesise a single "." discarding output, to build and cache the
 * chunk-frames graph. Called once at session init. Cost is backend-dependent
 * (~50ms Metal / ~100–150ms CUDA/Vulkan / ~200–500ms CPU). */
int qwen3tts_thermal_warmup(qwen3tts_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3TTS_STREAMING_H */
