# ComfyUI_JR_IndexTTS25

IndexTTS-2.5 的 ComfyUI 原生节点。插件在 ComfyUI 进程内加载官方模型，输出标准 `AUDIO`，支持持久化声模、四种情绪控制、多人台词、逐句情绪标签、LLM 发音标注、长文本参数和生成进度。

当前首版：`v0.1.0`

> 这是一个针对特定新运行栈完成兼容开发和真实 GPU 推理验证的版本：Windows、NVIDIA CUDA、Python 3.13.11、PyTorch 2.11.0+cu130。模型和官方源码不会自动下载。插件也不会自动安装、升级或替换 ComfyUI 的 Torch、Transformers、NumPy 等核心包。

## 功能一览

- 单人零样本声音克隆与 `AUDIO` 输出
- 声模首次登记、稳定 ID、名称、音频数据和持久化管理
- 手动情绪向量、情绪参考音频、台词自动情绪分析、独立情绪描述
- llama.cpp / OpenAI-compatible API 情绪分析，不占用 IndexTTS 内置 Qwen 的显存
- 可选官方内置 QwenEmotion
- 中文、英文、日文的 LLM 发音标注
- 最多 10 路声模的多人台词与逐句情绪覆盖
- `do_sample`、temperature、top-p、top-k、beam 等高级采样参数
- 长文本分段参数与 ComfyUI 进度条
- 模型缓存查看、单模型卸载、全部卸载

全部节点位于 `JR/Audio/IndexTTS 2.5` 分类，共 9 个。

## 运行环境

严格模式验证以下环境：

| 项目 | 已验证值 |
|---|---|
| OS | Windows |
| Python | 3.13.11 |
| PyTorch | 2.11.0+cu130 |
| Torch CUDA | 13.0 |
| GPU | NVIDIA CUDA GPU |
| IndexTTS | IndexTTS-2.5，固定上游基线 `a371df7` 并应用本仓库补丁 |

`strict_environment=false` 只会关闭版本拦截，不代表其他版本已经兼容。不要为了安装本插件降低 Python、PyTorch 或 CUDA 版本。

## 安装

### 1. 安装节点

在 ComfyUI 的 `custom_nodes` 目录执行：

```powershell
git clone https://github.com/Goldlionren/ComfyUI_JR_IndexTTS25.git
```

最终目录应为：

```text
ComfyUI/
└─ custom_nodes/
   └─ ComfyUI_JR_IndexTTS25/
      ├─ __init__.py
      ├─ nodes.py
      ├─ backend/
      └─ patches/
```

如果目标目录已经存在，请先备份或用 Git 正常更新，不要把新版直接覆盖到混合目录中。

### 2. 准备官方 IndexTTS-2.5 源码

推荐把源码放到插件内的 `index_tts`：

```powershell
cd ComfyUI\custom_nodes\ComfyUI_JR_IndexTTS25
git clone https://github.com/index-tts/index-tts.git index_tts
cd index_tts
git checkout a371df7d0746a0ae7fdf075798b6b04e34a0132e
git am ..\patches\0001-Add-Python-3.13-and-Torch-2.11-compatibility.patch ..\patches\0002-Fix-sampling-control-and-Arabic-normalization.patch
```

补丁提供本项目所需的 Python 3.13 / Torch 2.11 / Transformers 兼容层、SoundFile WAV 写入、进度回调、采样参数透传和阿拉伯语归一化分支。不要对官方源码执行 `pip install .`，否则其依赖解析可能改动真实 ComfyUI 环境。

也可以把源码放在其他位置，然后填写 Loader 的 `source_path_override`，或设置环境变量 `INDEXTTS25_SOURCE`。

### 3. 准备模型

插件不下载模型。把完整 IndexTTS-2.5 权重放到以下任一位置：

```text
ComfyUI/models/IndexTTS-2.5/
ComfyUI/models/indextts/IndexTTS-2.5/
```

至少需要：

```text
IndexTTS-2.5/
├─ config.yaml
├─ gpt.pth
├─ s2mel.pth
├─ codec.pth
├─ multilingual_zh_ja_yue_char_del.tiktoken
├─ wav2vec2bert_stats.pt
├─ hf_cache/                  # 官方运行时所需的本地 Hugging Face 内容
└─ qwen0.6bemo4-merge/        # 仅 builtin_qwen 情绪后端需要
```

模型在别处时，填写 Loader 的 `model_path_override`，或设置 `INDEXTTS25_MODEL_DIR`。插件只读取模型，不移动大模型，也不改变目录结构。

### 4. 依赖边界

`requirements.txt` 只列 IndexTTS 运行时缺失的基础依赖，刻意不列出 Torch、TorchAudio、TorchVision、Transformers、Tokenizers、NumPy 和 Safetensors。安装前先确认 ComfyUI 当前环境确实符合上面的版本要求；不要使用 `pip install -U`。

可选依赖分开管理：

- `requirements-builtin-qwen.txt`：仅使用内置 QwenEmotion 时需要。
- `requirements-japanese.txt`：仅日文分词/发音需要；缺少 fugashi/MeCab 不影响中文和英文。

本仓库不会在节点执行时自动安装任何包。

## 最快上手

首次登记一个声音：

```text
Load Audio → JR IndexTTS 2.5 Voice Preset ┐
                                          ├→ JR IndexTTS 2.5 Generate → Preview/Save Audio
JR IndexTTS 2.5 Loader ───────────────────┘
```

以后直接使用已保存声模：

```text
JR IndexTTS 2.5 Load Voice Preset ────────┐
                                          ├→ JR IndexTTS 2.5 Generate → AUDIO
JR IndexTTS 2.5 Loader ───────────────────┘
```

需要情绪时，把 `JR IndexTTS 2.5 Emotion Control` 的输出接到 Generate 的 `emotion`。

## 节点完整说明

### 1. JR IndexTTS 2.5 Loader

真实加载并缓存 IndexTTS-2.5，输出 `JR_INDEXTTS25_MODEL`。

| 参数 | 含义 |
|---|---|
| `model_path_override` | 模型目录的显式绝对路径。留空时自动搜索。 |
| `source_path_override` | 已应用兼容补丁的官方源码根目录。根目录内必须有 `indextts/infer_v2_5.py`。 |
| `device` | 推理设备，默认 `cuda:0`。 |
| `precision` | `fp32` 或 `bf16`。BF16 主要作用于 GPT 部分，并要求 GPU 支持。 |
| `enable_qwen_emotion` | 是否随主模型加载官方内置 QwenEmotion。使用 llama.cpp 情绪后端时应关闭，以节省显存。 |
| `strict_environment` | 开启时强制检查 Python 3.13.11、Torch 2.11.0+cu130、CUDA 13.0 和 CUDA 可用性。 |

相同配置会命中进程内模型缓存。基线固定关闭 CUDA kernel JIT、DeepSpeed、`torch.compile` 和加速实验路径，优先保证 Windows 新栈稳定。

### 2. JR IndexTTS 2.5 Voice Preset

输入一段参考人声，保存到声模库，并输出 `JR_INDEXTTS25_VOICE`。

| 参数 | 含义 |
|---|---|
| `reference_audio` | ComfyUI `AUDIO`。建议使用清晰、单人、低噪声、无背景音乐的参考音频。 |
| `speaker_name` | 声模显示名称，也是多人台词里的角色匹配名。 |
| `overwrite_existing` | 同名但音频不同时，是否覆盖。默认关闭，防止误覆盖。 |

第一次执行会生成：稳定 `id`、`name`、PCM16 `prompt.wav`、SHA-256、采样率、时长和时间戳。相同名称且相同音频会复用原记录。

声模库路径按以下顺序确定：

1. 环境变量 `INDEXTTS25_PRESET_DIR`；
2. `ComfyUI/models/indextts/voice_presets`；
3. ComfyUI 外独立运行时的插件本地 `voice_presets`。

### 3. JR IndexTTS 2.5 Load Voice Preset

无需重新输入参考音频，直接从声模库输出与 Voice Preset 完全相同的 `JR_INDEXTTS25_VOICE`。

| 参数 | 含义 |
|---|---|
| `preset` | 已保存声模下拉框，格式为 `name :: id`。新增声模后刷新节点定义或重启/刷新 ComfyUI 可更新列表。 |
| `preset_id_or_name_override` | 可直接输入稳定 ID 或名称；非空时优先于下拉框，适合刚创建后立即引用。 |

### 4. JR IndexTTS 2.5 Voice Preset Manager

查看或管理持久化声模，输出格式化 JSON。

| 参数 | 含义 |
|---|---|
| `action=list` | 列出所有声模，包含 `id`、`name`、音频路径、采样率、时长、哈希等。 |
| `action=inspect` | 查看 `preset_id_or_name` 指定的一条记录。 |
| `action=rename` | 把指定记录改为 `new_name`，稳定 ID 不变。 |
| `action=delete` | 删除指定声模目录；必须同时设置 `confirm_delete=true`。 |
| `preset_id_or_name` | `inspect`、`rename`、`delete` 的目标，可填 ID 或名称。 |
| `new_name` | `rename` 的新名称。 |
| `confirm_delete` | 删除保护开关。其他 action 不使用。 |

删除会移除声模记录及其 `prompt.wav`，请自行确保不再需要该声音。

### 5. JR IndexTTS 2.5 Emotion Control

输出 `JR_INDEXTTS25_EMOTION`。四种 mode 的逻辑不同：

| mode | 用法 |
|---|---|
| `emotion_vector` | 手动设置八维向量。适合可重复、可精确调节的情绪。 |
| `reference_audio` | 把 `emotion_reference_audio` 的表演情绪迁移到目标声音。 |
| `auto_from_text` | 分析 Generate 的合成台词本身，自动产生八维向量。 |
| `emotion_text` | 分析独立的 `emotion_text` 描述，如“压低声音、克制、略带失望”。这里的 `emotion_text` 就是“独立情绪描述文本”。 |

八维顺序与官方一致：

```text
happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
开心，生气，悲伤，害怕，厌恶，忧郁，惊讶，平静
```

| 参数 | 含义 |
|---|---|
| 八个情绪滑块 | `emotion_vector` 模式下各维的原始强度，范围 0–1。 |
| `strength` | 情绪条件对最终声音的作用强度 `emo_alpha`，建议先用 0.5–0.7。 |
| `apply_official_bias` | 对手动向量应用官方偏置/归一化方式。 |
| `emotion_reference_audio` | 仅 `reference_audio` 模式必填。 |
| `emotion_text` | 仅 `emotion_text` 模式必填；不是要朗读的正文。 |
| `random_sampling` | 交给官方情绪路径启用随机采样。需要稳定复现时关闭。 |
| `text_emotion_backend` | `llama.cpp_openai_api` 或 `builtin_qwen`。 |
| `openai_api_url` | llama.cpp/OpenAI-compatible 服务地址，默认 `http://127.0.0.1:10000`。可填服务根、`/v1` 或完整 chat completions URL。 |
| `openai_api_key` | 本地 llama.cpp 通常留空；远程兼容服务按需填写。 |
| `openai_model` | 留空时自动读取 `/v1/models` 返回的第一个模型 ID。 |
| `llm_timeout_seconds` | LLM 请求超时。 |

推荐的本地 llama.cpp 配置：

```text
mode: auto_from_text 或 emotion_text
text_emotion_backend: llama.cpp_openai_api
openai_api_url: http://127.0.0.1:10000
openai_api_key: 留空
openai_model: 留空
strength: 0.5–0.7
```

llama.cpp 返回值会被限制并转换成官方八维向量，再送入 IndexTTS 原生推理。此路径不需要 Loader 开启 Qwen。`builtin_qwen` 才要求 `enable_qwen_emotion=true`、完整的 Qwen 模型目录和可选依赖。

### 6. JR IndexTTS 2.5 Pronunciation Enhance (LLM)

让 llama.cpp/OpenAI-compatible LLM 只给歧义词增加官方发音标注，输出增强后的字符串，再连接到 Generate 的 `text`。

| 参数 | 含义 |
|---|---|
| `text` | 原始待朗读文本。 |
| `language` | `ZH`、`EN` 或 `JA`。 |
| `openai_api_url/key/model/timeout` | 与 Emotion Control 相同。model 留空时自动发现。 |
| `instruction` | 给 LLM 的补充要求。默认只标真正需要指定读音的词，不改写原文。 |

官方标注语法：

```text
中文：银<行|HANG2>今天休息。
英文：A <minute|M IH1 . N AH0 T> detail.
日文：彼は<上手|じょうず>です。
```

中文自动增强会严格要求一个汉字对应一个带声调数字的拼音音节。节点会校验：去掉 `<原文|读音>` 标注后必须与原文完全一致；如果 LLM 擅自改写、删字或加字，节点会报错而不是静默采用。

也可以完全不经过该节点，直接在 Generate 的正文中手写上述官方标注。

### 7. JR IndexTTS 2.5 Generate

单声音生成，输入 Loader 的 `model`、任一 Voice 节点的 `voice`，输出标准 ComfyUI `AUDIO`。

| 参数 | 含义 |
|---|---|
| `text` | 要合成的正文，可包含官方发音标注。 |
| `language` | 官方当前菜单：`ZH`、`EN`、`JA`、`AR`、`ES`。 |
| `duration_factor` | 时长倍率，0.5–2.0。增大通常更慢、更长。 |
| `text_normalization` | 使用官方文本归一化。西班牙语/阿拉伯语若缺少 NeMo 会安全退回原文；中文/英文使用各自路径。 |
| `interval_silence_ms` | 长文本分段之间插入的静音。 |
| `seed` | Torch/CUDA 随机种子。相同环境和参数下用于复现。 |
| `unload_model_after` | 生成结束后从插件模型缓存卸载该模型并清理 CUDA 缓存。批量生成时建议关闭。 |
| `emotion` | 可选，连接 Emotion Control。未连接时使用自然/默认情绪。 |

高级参数见“采样与长文本参数”一节。

### 8. JR IndexTTS 2.5 Multi-Talk Generate

按标签选择多个声音，逐段生成后拼成一个 `AUDIO`。

| 参数 | 含义 |
|---|---|
| `dialogue` | 多人台词文本，语法见下一节。 |
| `voice_1` … `voice_10` | 最多连接 10 个 Voice Preset/Load Voice Preset。至少连接一路。 |
| `emotion` | 可选的共享情绪，也为逐句 `auto`/`emo_text` 提供 LLM 后端配置。 |
| `gap_ms` | 不同台词段之间插入的静音。 |
| 其他参数 | 与 Generate 相同。每段 seed 为 `seed + 段索引`。 |

角色名按 Voice 的 `speaker_name`/声模名进行不区分大小写匹配。未知角色会回退到 `voice_1`，所以正式工作流应检查拼写。

### 9. JR IndexTTS 2.5 Runtime Diagnostics

输出 JSON 运行报告。

| action | 含义 |
|---|---|
| `report` | 显示 Python、Torch、Torch CUDA、CUDA 可用性、GPU 和插件模型缓存。 |
| `unload_model` | 卸载输入的 `model`；必须连接 model。 |
| `unload_all` | 卸载本插件缓存的全部 IndexTTS 模型并清理 CUDA 缓存。 |

它不会卸载或修改其他 custom node 的模型，也不会修改任何 Python package。

## 多人台词语法

### 基本格式

冒号支持半角 `:` 和全角 `：`：

```text
[紫灵]: 你好，今天过得怎么样？
[旁白]：夜色慢慢沉了下来。
```

角色名必须对应某个已连接 Voice 的名称。每个新 `[角色]` 标签开始一个新片段，正文可以跨行，直到下一个角色标签。

### 逐句情绪格式

在角色名后用 `|` 增加选项：

```text
[紫灵|开心=0.55|平静=0.20|强度=0.8]: 你好，今天过得怎么样？
[旁白|sad=0.65]: 她沉默了一会儿。
[紫灵|auto|strength=0.6]: 我真的不知道应该怎么办。
[旁白|emo_text=低沉、克制、略带悲伤|strength=0.6]: 夜色慢慢沉了下来。
[旁白|natural]: 这句不使用共享情绪。
```

情绪名称可写英文或中文别名：

| 官方维度 | 中文别名 |
|---|---|
| `happy` | `开心`、`快乐` |
| `angry` | `生气`、`愤怒` |
| `sad` | `悲伤`、`伤心` |
| `afraid` | `害怕`、`恐惧` |
| `disgusted` | `厌恶`、`嫌弃` |
| `melancholic` | `忧郁`、`低沉` |
| `surprised` | `惊讶`、`吃惊` |
| `calm` | `平静`、`冷静` |

可用选项：

- `开心=0.6` / `happy=0.6`：指定向量值，范围 0–1；省略 `=值` 时默认 0.8。
- `strength=0.6` / `alpha=0.6` / `强度=0.6`：该句情绪作用强度。
- `auto` / `自动` / `自动情绪`：分析该句正文。
- `emo_text=描述` / `emotion_text=描述` / `情绪文本=描述`：使用独立表演描述。
- `random` / `随机`：开启；也可写 `random=false`。
- `bias` / `official_bias` / `官方偏置`：官方偏置开关，可写 `bias=false`。
- `natural` / `none` / `自然` / `无情绪`：该句不使用情绪；必须单独出现。

布尔值接受 `true/false`、`yes/no`、`on/off`、`1/0`、`是/否`、`开/关`。手动向量不能与 `auto` 或 `emo_text` 混用。

逐句标签优先于节点共享 `emotion`。逐句 `auto`/`emo_text` 会继承共享 Emotion Control 的 llama.cpp URL、key、model 和 timeout；如果没有连接共享 Emotion Control，则使用默认 `http://127.0.0.1:10000` 的 llama.cpp 后端。

## 采样与长文本参数

Generate 和 Multi-Talk Generate 都提供以下参数：

| 参数 | 作用 |
|---|---|
| `do_sample` | 开启时按概率采样，声音表现通常更有变化；关闭时走确定性/搜索式生成，`temperature`、`top_p`、`top_k` 的采样作用会减弱或不生效。 |
| `temperature` | 采样随机度。越高越发散，越低越保守；通常先用 0.8。 |
| `top_p` | nucleus sampling，只从累计概率范围内采样。1.0 近似不裁剪。 |
| `top_k` | 只保留概率最高的 K 个候选；`0` 表示禁用 top-k。 |
| `num_beams` | beam search 数。更高可能更稳定但更慢、更占显存；默认 3。 |
| `repetition_penalty` | 重复惩罚。过高可能损伤自然度；当前官方默认路径为 10.0。 |
| `length_penalty` | 搜索时的长度偏好，正值更偏长，负值更偏短。 |
| `max_mel_tokens` | 单次生成允许的最大声学 token 数，限制最长音频；范围 50–1815。 |
| `max_text_tokens_per_segment` | 长文本切段的最大文本 token 数；越小分段越多，越大单段负担越高。 |

本仓库第二个源码补丁修复了上游把 `do_sample` 固定成 `True` 的问题，因此 UI 中关闭它会真实传入 GPT 生成逻辑。

## 生成进度

两个生成节点都把 IndexTTS 原生阶段映射到 ComfyUI ProgressBar，包括文本处理、情绪分析、逐段生成和完成。Multi-Talk 会按片段数量汇总总进度。

进度代表当前处理阶段，不是严格的逐 token 百分比；某些模型步骤本身没有更细粒度回调，所以进度可能在一个阶段停留后跳到下一阶段。

## 路径搜索顺序

模型目录：

1. Loader `model_path_override`
2. `INDEXTTS25_MODEL_DIR`
3. `ComfyUI/models/IndexTTS-2.5`
4. `ComfyUI/models/indextts/IndexTTS-2.5`
5. 开发工作区相邻的 `models/IndexTTS-2.5`

源码目录：

1. Loader `source_path_override`
2. `INDEXTTS25_SOURCE`
3. `ComfyUI_JR_IndexTTS25/index_tts`
4. 开发工作区相邻的 `source/index-tts`

缓存目录可用 `INDEXTTS25_CACHE_DIR` 指定。默认放在 ComfyUI 临时目录下，Numba 缓存不会写入真实 ComfyUI Python 的 `site-packages`。

## 常见问题

### Loader 显示找不到模型或源码

检查填写的是目录而不是某个 `.pth` 文件。`model_path_override` 应指向同时包含 `config.yaml`、`gpt.pth`、`s2mel.pth` 和其他必需文件的 `IndexTTS-2.5` 根目录；`source_path_override` 应指向包含 `indextts` 文件夹的源码根目录。

### 新参数或新节点没有出现

完全停止并重启 ComfyUI。检查 `custom_nodes` 下是否同时残留旧目录、备份目录或重名副本；ComfyUI 可能把备份也当作节点加载。备份应放到 `custom_nodes` 外。

### llama.cpp 情绪分析连接失败

确认服务已启动，并能访问：

```text
http://127.0.0.1:10000/v1/models
http://127.0.0.1:10000/v1/chat/completions
```

model 留空时节点会先请求 `/v1/models`。如果服务要求鉴权，填写 key；如果返回多个模型，可显式填写模型 ID。

### 提示缺少 fugashi / MeCab

这是日文可选依赖限制。不要为了中文/英文测试安装它；只有确实使用日文时，再评估 `requirements-japanese.txt`。

### 西班牙语或阿拉伯语缺少 NeMo

Windows 环境中 NeMo/Pynini 并非默认依赖。插件的兼容源码会在 NeMo 不可用时保留原文继续合成；这表示高级文本归一化降级，不等于核心 TTS 不可用。

### TorchAudio 保存 WAV 报 TorchCodec/FFmpeg DLL 错误

本项目兼容补丁已把官方推理的 WAV 写入改为 SoundFile PCM16。请确认应用了 `patches` 中的两个补丁，不要通过升级/降级 TorchAudio 来绕过。

## 已知边界

- 首版严格验证目标是 Windows + Python 3.13.11 + Torch 2.11.0+cu130；其他平台/版本尚未声明兼容。
- `builtin_qwen` 会增加模型加载时间和 GPU 显存占用；默认推荐外部 llama.cpp。
- 日文需要额外分词依赖；中文、英文不依赖它。
- 西班牙语/阿拉伯语在缺少 NeMo 时会跳过高级文本归一化。
- 进度是阶段级进度，不是逐 token 进度。
- 模型权重、官方源码和声音数据不包含在本仓库中。

## 致谢与上游

- IndexTTS / IndexTTS-2.5：<https://github.com/index-tts/index-tts>
- ComfyUI：<https://github.com/comfyanonymous/ComfyUI>

本项目是独立的 ComfyUI 集成与兼容层。请同时遵守上游源码、模型权重和参考音频各自的许可及使用条款。

## v0.1.0

- 发布 9 个 IndexTTS-2.5 ComfyUI 节点。
- 完成 Python 3.13.11 / Torch 2.11.0+cu130 / CUDA 13 兼容层。
- 支持持久化声模库、多人逐句情绪、llama.cpp 情绪分析和发音增强。
- 支持高级生成参数、长文本参数、进度条和模型缓存管理。
