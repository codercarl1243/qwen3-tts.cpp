# AGENTS.md — qwen3-tts.cpp fork

Fork-specific reference for the Qwen3-TTS C++ pipeline (talker → code predictor → vocoder). This holds what clangd can't give you: architecture, exact token layout, and divergence from the base. Navigation/build gotchas are in `CLAUDE.md`.

## Build

- CMake 3.14+, C++17. GGML is vendored under `./ggml`.
- GGML first: `cmake -S ggml -B ggml/build -DGGML_METAL=ON && cmake --build ggml/build -j4`
- Project: `cmake -S . -B build && cmake --build build -j4`

## C++ conventions

- C++17, no exceptions, no RTTI. `#pragma once` guards.
- Log via `fprintf(stderr, ...)`, not `std::cerr`.
- Methods return `bool`; error detail in the `error_msg_` member.
- GGML contexts own tensor memory — free with `ggml_free()`.
- `snake_case` functions/vars, `PascalCase` classes, `UPPER_CASE` macros. Public types in the `qwen3_tts` namespace.
- Includes are module-qualified: `#include "common/gguf_loader.h"`.

## GGML forward pass

Every pass: build graph → `ggml_backend_sched_alloc_graph` → set inputs (`ggml_backend_tensor_set`) → `ggml_backend_sched_graph_compute` → read outputs (`ggml_backend_tensor_get`) → `ggml_backend_sched_reset`.

Gotcha: `ggml_cast` to F32 before `ggml_mul_mat` when the weight is F16 (`ffn_down` in talker and code predictor).

Backend: `init_preferred_backend()` (`src/common/gguf_loader.cpp`) orders `IGPU → GPU → ACCEL → CPU`; add a CPU fallback backend to the scheduler when the runtime backend isn't CPU. The vocoder runs a private backend (`init_private_backend`) for talker‖vocoder concurrency.

## Architecture

Two sub-models in `src/transformer/tts_transformer.cpp` (~3200 lines; `generate`, `forward_prefill`, `forward_step`, `predict_codes_autoregressive`):

1. **Talker** — 28-layer Qwen2 (1024 hidden, 16 heads, 8 KV, 128 head_dim). In: prefill or step embedding `[1,1024]`. Out: hidden states + codec logits via `codec_head`.
2. **Code predictor** — 5-layer, same attention config, own KV cache (max 16). In: talker hidden + codebook-0 embedding. Out: 15 codebook predictions, autoregressive.

### Prefill embedding (10 positions, single-word input)

```
Pos 0:  text_projection(<|im_start|>)
Pos 1:  text_projection(assistant)
Pos 2:  text_projection(\n)
Pos 3:  tts_pad + codec_embd(think_id)
Pos 4:  tts_pad + codec_embd(think_bos_id)
Pos 5:  tts_pad + codec_embd(language_id)
Pos 6:  tts_pad + codec_embd(think_eos_id)
Pos 7:  tts_pad + speaker_embedding
Pos 8:  tts_bos + codec_embd(pad_id)
Pos 9+: text_projection(text_token[i]) + codec_embd(bos_id or pad_id)
```

Must mirror the Python pipeline exactly.

### Special token IDs

Authoritative in `src/transformer/tts_transformer.h`:

```
tts_bos=151672   tts_eos=151673   tts_pad=151671
codec_bos=2149   codec_eos=2150   codec_pad=2148
codec_think=2154 codec_think_bos=2156 codec_think_eos=2157
english_language_id=2050
```

## Divergence & limitations

- F16 weights diverge from Python float32 in autoregressive decode — speech codes differ, audio is perceptually equivalent.
- M-RoPE uses 1D positions (equivalent for single-batch).
- `--top-p` is parsed but unused in sampling.
- Code predictor is the bottleneck (~71% of generation: 15 sequential passes/frame); talker ~27%.

## Testing

Reference comparison: `bash scripts/run_all_tests.sh`, or `./build/test_transformer --ref-dir reference/`. Pass = prefill logits cosine > 0.99 (speech codes diverge under F16).
