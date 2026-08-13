# Ubuntu + Python 3.12 测试说明

这是 `0.4.0rc1` 兼容候选包，不是正式版。请在 Ubuntu 测试机上使用，不要覆盖目前能正常工作的 Windows + Python 3.13 生产插件。

## 目标环境

- Ubuntu 22.04 或 24.04，x86-64
- NVIDIA GPU 与正常工作的驱动
- ComfyUI 实际使用 Python 3.12.x
- `torch==2.11.0+cu130`
- `torchaudio==2.11.0+cu130`
- `torch.version.cuda == 13.0`

## 操作步骤

1. 把候选包内层的 `ComfyUI_JR_IndexTTS25` 完整放到：

   ```text
   ComfyUI/custom_nodes/ComfyUI_JR_IndexTTS25
   ```

2. 使用“实际启动 ComfyUI 的 Python”检查核心环境。以下假设虚拟环境叫 `venv`：

   ```bash
   cd /你的路径/ComfyUI
   ./venv/bin/python -c "import platform, torch, torchaudio; print(platform.python_version()); print(torch.__version__); print(torchaudio.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
   ```

   如果不是 Python 3.12.x、Torch/TorchAudio 2.11.0+cu130、Torch CUDA 13.0 和 `True`，先停止测试。不要在已有 ComfyUI 环境中盲目升级 Torch。

3. 只安装插件侧依赖：

   ```bash
   ./venv/bin/python -m pip install -r custom_nodes/ComfyUI_JR_IndexTTS25/requirements.txt
   ```

   Linux 会安装 `WeTextProcessing`；Python 3.12 不会安装只适用于 3.13+ 的 `audioop-lts`。该 requirements 文件不声明 Torch、TorchAudio、TorchVision、Transformers、Tokenizers、NumPy 或 Safetensors。

4. 确认系统能找到 FFmpeg：

   ```bash
   ffmpeg -version
   ```

5. 正常启动 ComfyUI，保存完整启动日志，并确认 10 个 `JR IndexTTS 2.5` 节点均已注册。

6. 停止 ComfyUI 后执行真实证据测试（路径换成你机器上的实际路径）：

   ```bash
   cd /你的路径/ComfyUI
   ./venv/bin/python custom_nodes/ComfyUI_JR_IndexTTS25/compatibility/ubuntu_py312/runtime_probe.py \
     --comfyui-root "$PWD" \
     --plugin-dir "$PWD/custom_nodes/ComfyUI_JR_IndexTTS25" \
     --model-dir "/模型路径/IndexTTS-2.5" \
     --reference-audio "/参考音频路径/reference.wav" \
     --output-dir "$PWD/output/jr_indextts25_ubuntu_probe" \
     --real-inference
   ```

## 请带回的结果

- `output/jr_indextts25_ubuntu_probe/ubuntu_py312_report.json`
- 完整的 ComfyUI 启动日志
- 若成功，生成的 `ubuntu_py312_output.wav`

只有报告中同时出现 `"status": "PASS"` 和 `"real_inference": "PASS"`，才算 Ubuntu 真实推理通过。单纯能导入插件不算完成。
