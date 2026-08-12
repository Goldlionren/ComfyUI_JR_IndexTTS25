# ComfyUI_JR_IndexTTS25

Native, in-process IndexTTS-2.5 nodes validated for Python 3.13.11, PyTorch
2.11.0+cu130, Windows, and NVIDIA CUDA. The plugin never starts a second
Python process and its requirements intentionally do not install or replace
ComfyUI's Torch or Transformers stack.

## Model and source discovery

The loader checks, in order:

1. its explicit `model_path_override` / `source_path_override` inputs;
2. `INDEXTTS25_MODEL_DIR` / `INDEXTTS25_SOURCE` environment variables;
3. ComfyUI `models/IndexTTS-2.5` and `models/indextts/IndexTTS-2.5`;
4. this development workspace's `models/IndexTTS-2.5` and `source/index-tts`.

No model is downloaded automatically. Numba cache files are redirected to the
ComfyUI temporary directory so importing IndexTTS does not write to the real
ComfyUI Python `site-packages` directory.

## Nodes

- JR IndexTTS 2.5 Loader
- JR IndexTTS 2.5 Voice Preset
- JR IndexTTS 2.5 Emotion Control
- JR IndexTTS 2.5 Generate
- JR IndexTTS 2.5 Multi-Talk Generate
- JR IndexTTS 2.5 Runtime Diagnostics

Multi-Talk syntax is `[Speaker]: text`. Connect Voice Preset nodes whose
`speaker_name` values match the tags. Unknown speakers use the first connected
voice as a safe fallback.
