# Compatibility release status

Release: `0.5.1`

Stable base commit: `f99efba23d50cf40e0d1c74c1145d62df43f1646`

Development branch: `codex/python312-dependency-ranges`; release target: GitHub `main`

The dual-backend release completed user acceptance and is approved for the
main plugin branch.

The release extends the existing Windows/Linux and Python 3.12/3.13 work with
an Intel XPU runtime path. It does not replace the stable NVIDIA CUDA package
matrix and it does not install or replace ComfyUI's PyTorch packages.

## Preserved stable baseline

- Windows 11
- Python 3.13.11
- torch 2.11.0+cu130
- torchaudio 2.11.0+cu130
- Torch CUDA 13.0
- NVIDIA RTX 5090

Release validation performed on the stable machine:

- Python compilation: PASS
- Runtime matrix tests: PASS
- Existing plugin and runtime-matrix tests: PASS (39 tests)
- Ten-node registration in the real ComfyUI Python: PASS
- Strict Windows 3.13 runtime validation: PASS
- Release import and strict CUDA environment validation: PASS

The existing stable Windows production workflow remains the CUDA real-inference
baseline. No ComfyUI Python packages were modified during compatibility work.

## Ubuntu/Python 3.12 status

- Dependency markers and runtime matrix: implemented
- Static Python compatibility: PASS under the development checks
- Real Ubuntu import: awaiting external machine test
- Real Ubuntu model load: awaiting external machine test
- Real Ubuntu GPU WAV inference: awaiting external machine test

Follow `compatibility/ubuntu_py312/README.md`. Return the generated
`ubuntu_py312_report.json` and the complete ComfyUI startup log before adding
Ubuntu/Python 3.12 as a separately verified baseline.

## Ubuntu/Python 3.13/Intel XPU status

- Target runtime: Ubuntu 24.04, Python 3.13.x, torch/torchaudio 2.11.0+xpu
- Target hardware: Intel Arc A770 16GB and Intel Arc Pro B60
- Device-aware cache, seed, BF16 and diagnostics: implemented
- CPU reference-audio Mel/Kaldi preprocessing: implemented
- XPU runtime matrix and operator probe: implemented
- CUDA kernel, DeepSpeed, third-party flash-attn and torch.compile: disabled
- Built-in Qwen emotion model: not part of the first XPU acceptance gate
- Ubuntu XPU user acceptance, model download/load and real TTS workflow: PASS (user reported)
- Target hardware includes Intel Arc A770 16GB and Intel Arc Pro B60

The reusable evidence probe remains in `compatibility/xpu/runtime_probe.py` for
future release regression and per-GPU diagnostics.
