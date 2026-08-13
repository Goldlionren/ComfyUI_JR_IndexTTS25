# Compatibility candidate status

Candidate: `0.4.0rc1`

Stable base commit: `f99efba23d50cf40e0d1c74c1145d62df43f1646`

Development branch: `codex/python312-ubuntu-compat`

This directory is a test candidate, not a production release.

## Preserved stable baseline

- Windows 11
- Python 3.13.11
- torch 2.11.0+cu130
- torchaudio 2.11.0+cu130
- Torch CUDA 13.0
- NVIDIA RTX 5090

Candidate validation performed on the stable machine:

- Python compilation: PASS
- Runtime matrix tests: PASS
- Existing plugin unit tests: PASS (33 tests total with the matrix tests)
- Candidate import in the real ComfyUI Python: PASS
- Ten-node registration in the real ComfyUI Python: PASS
- Strict Windows 3.13 runtime validation: PASS
- New candidate cold-load real inference: TIMEOUT at 600 seconds before GPU work began

The timeout is not recorded as an inference PASS. An existing ComfyUI Python
process was already using the GPU during this cold-load attempt. The stable
production plugin was not overwritten, stopped, or modified.

## Ubuntu/Python 3.12 status

- Dependency markers and runtime matrix: implemented
- Static Python compatibility: PASS under the development checks
- Real Ubuntu import: awaiting external machine test
- Real Ubuntu model load: awaiting external machine test
- Real Ubuntu GPU WAV inference: awaiting external machine test

Follow `compatibility/ubuntu_py312/README.md`. Return the generated
`ubuntu_py312_report.json` and the complete ComfyUI startup log before this
candidate is promoted.
