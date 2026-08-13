# Intel XPU 使用与验证说明

Intel Arc XPU 已合入插件主线，与 NVIDIA CUDA 共用插件代码。两种平台仍必须
使用各自独立的 ComfyUI Python 环境和对应 PyTorch wheel，不能互相覆盖。

## 已验证目标

- Ubuntu 24.04 x86-64
- Python 3.13.x
- `torch==2.11.0+xpu`
- `torchaudio==2.11.0+xpu`
- Intel Arc A770 16GB
- Intel Arc Pro B60

稳定基线只启用基础推理。CUDA kernel、DeepSpeed、第三方 flash-attn、
`torch.compile` 均保持关闭。内置 Qwen 情绪模型暂不作为 XPU 首轮通过条件；
Loader、Voice Preset、Generate、Multi Talk 和外部 llama.cpp 情绪分析是首要目标。

## 重要安全规则

不要在已经能工作的 NVIDIA ComfyUI 环境里把 CUDA 版 PyTorch 替换成 XPU 版。
请使用 Ubuntu XPU 机器上独立的 ComfyUI 环境。本插件的 `requirements.txt` 故意不声明
PyTorch、TorchAudio、TorchVision、Transformers 或 NumPy，也不会自动替换这些核心包。

## 1. 确认环境

使用启动 ComfyUI 的同一个 Python 执行：

```bash
cd /path/to/ComfyUI
./venv/bin/python -c "import platform, torch, torchaudio; print(platform.python_version()); print(torch.__version__); print(torchaudio.__version__); print(torch.xpu.is_available()); print(torch.xpu.get_device_name(0) if torch.xpu.is_available() else None)"
```

预期看到 Python 3.13.x、Torch/TorchAudio 2.11.0+xpu、`True` 和正确的显卡名称。
如果不符合，请先停止，不要在现有环境中盲目升级或降级核心包。

## 2. 安放插件

把完整的 `ComfyUI_JR_IndexTTS25` 文件夹复制到：

```text
ComfyUI/custom_nodes/ComfyUI_JR_IndexTTS25
```

如果目标已经存在，请先备份。然后只安装插件侧依赖：

```bash
./venv/bin/python -m pip install -r custom_nodes/ComfyUI_JR_IndexTTS25/requirements.txt
```

这一步不应安装或替换 Torch 系列包。系统还需要可执行的 `ffmpeg`。

## 3. ComfyUI 节点设置

在 `JR IndexTTS 2.5 Loader` 中设置：

- `device`: `xpu:0`
- `precision`: 初次使用 `fp32`
- `enable_qwen_emotion`: 初次使用关闭
- `strict_environment`: 开启
- 模型路径和源码路径按正常文档设置；仓库内置源码时可留空源码路径

FP32 真实推理通过后，再把 `precision` 改为 `bf16` 单独测试。不要先用 BF16
判断基础兼容性。

## 4. 运行证据探针

准备真实 IndexTTS-2.5 模型目录和一段真实参考 WAV，然后执行：

```bash
cd /path/to/ComfyUI
./venv/bin/python custom_nodes/ComfyUI_JR_IndexTTS25/compatibility/xpu/runtime_probe.py \
  --comfyui-root "$PWD" \
  --model-dir "/path/to/IndexTTS-2.5" \
  --reference-audio "/path/to/reference.wav" \
  --output-dir "$PWD/output/jr_indextts25_xpu_probe" \
  --device xpu:0 \
  --precision fp32 \
  --real-inference
```

FP32 通过后可再运行一次 `--precision bf16`，但请使用不同输出目录。

如果 A770 和 B60 同时装在一台机器里，先用环境检查命令确认两张卡的索引，
然后分别使用 `--device xpu:0` 和 `--device xpu:1`。探针报告会记录全部 XPU
设备以及本次实际选中的设备，不能根据显卡插槽位置猜测索引。

探针会生成：

- `xpu_report.json`
- 成功时的 `xpu_output.wav`

请分别从 A770 和 Arc Pro B60 返回 JSON、WAV 以及完整 ComfyUI 启动日志。
只有报告出现 `"real_inference": "PASS"` 才算真实推理通过；仅插件 import 成功不算。
