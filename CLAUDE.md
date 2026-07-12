# qwen3-tts.cpp (fork: kaiwa-streaming)

Our diverged fork of the Qwen3-TTS C++ pipeline. Fork-specific architecture, token IDs, and prefill layout live in `AGENTS.md` — read it before touching the transformer.

Navigate with `LSP`/clangd against `build/compile_commands.json` (regenerate: `cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON`). Don't `Read` whole files — `src/transformer/tts_transformer.cpp` is ~3200 lines.

Editing `.cpp`/`.h` here does not trigger a cargo rebuild of the Rust crate; `touch CMakeLists.txt` to force one.
