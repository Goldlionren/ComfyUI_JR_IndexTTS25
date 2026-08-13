# Ubuntu + Python 3.12 compatibility candidate

This folder is for a controlled compatibility test. It does not install or replace
PyTorch, TorchAudio, TorchVision, Transformers, or NumPy automatically.

## Candidate matrix

- Ubuntu 22.04 or 24.04, x86-64
- NVIDIA GPU and a working NVIDIA driver
- Python 3.12.x used by ComfyUI
- `torch==2.11.0+cu130`
- `torchaudio==2.11.0+cu130`
- Torch CUDA runtime `13.0`

Windows + Python 3.13.11 remains the stable baseline. This candidate must not be
copied over the stable Windows production installation.

## 1. Put the candidate in ComfyUI

Copy the whole `ComfyUI_JR_IndexTTS25` directory to:

```text
ComfyUI/custom_nodes/ComfyUI_JR_IndexTTS25
```

Use a fresh Ubuntu ComfyUI installation or back up an existing copy first.

## 2. Check the core runtime before installing plugin dependencies

Run these commands with the same Python that starts ComfyUI:

```bash
cd /path/to/ComfyUI
./venv/bin/python -c "import platform, torch, torchaudio; print(platform.python_version()); print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Stop if the result is not Python 3.12.x, Torch/TorchAudio 2.11.0+cu130,
Torch CUDA 13.0, and `True`. Do not repair the environment by blindly upgrading
PyTorch inside an existing ComfyUI installation.

## 3. Install only the plugin side dependencies

```bash
./venv/bin/python -m pip install -r custom_nodes/ComfyUI_JR_IndexTTS25/requirements.txt
```

Linux uses `WeTextProcessing`; Windows keeps using `wetext`. Python 3.12 uses its
built-in `audioop`, while `audioop-lts` is installed only on Python 3.13 or newer.
The requirements file intentionally does not declare ComfyUI's Torch,
TorchAudio, TorchVision, Transformers, Tokenizers, NumPy, or Safetensors.

The `ffmpeg` executable must also be available:

```bash
ffmpeg -version
```

## 4. Start ComfyUI and inspect registration

Start ComfyUI normally. Confirm that the log registers all ten `JR IndexTTS 2.5`
nodes and has no missing-package traceback.

## 5. Run the evidence probe

Use a real model directory and a real speaker reference WAV:

```bash
cd /path/to/ComfyUI
./venv/bin/python custom_nodes/ComfyUI_JR_IndexTTS25/compatibility/ubuntu_py312/runtime_probe.py \
  --comfyui-root "$PWD" \
  --plugin-dir "$PWD/custom_nodes/ComfyUI_JR_IndexTTS25" \
  --model-dir "/path/to/IndexTTS-2.5" \
  --reference-audio "/path/to/reference.wav" \
  --output-dir "$PWD/output/jr_indextts25_ubuntu_probe" \
  --real-inference
```

The probe writes:

- `ubuntu_py312_report.json`
- `ubuntu_py312_output.wav` after successful real GPU inference

Send the JSON report and the complete ComfyUI startup log back for review. A
successful import alone is not sufficient evidence; the report must contain
`"real_inference": "PASS"`.
