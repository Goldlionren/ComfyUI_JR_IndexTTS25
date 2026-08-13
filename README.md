# ComfyUI_JR_IndexTTS25

IndexTTS-2.5 的 ComfyUI 原生节点。插件直接在 ComfyUI 进程内加载模型并输出标准 `AUDIO`，支持声音克隆、持久化声模、四种情绪控制、多人台词、小说转角色脚本、逐句情绪、LLM 发音标注、长文本和生成进度。

当前版本：`v0.3.0`

> 已验证目标环境：Windows、NVIDIA CUDA、Python 3.13.11、PyTorch 2.11.0+cu130、Torch CUDA 13.0。兼容后的 IndexTTS 运行源码已经内置，普通用户不需要再次下载源码或应用补丁。

## 目录

- [功能与节点](#功能与节点)
- [安装前确认](#安装前确认)
- [安装插件](#安装插件)
- [第一次运行：自动下载模型](#第一次运行自动下载模型)
- [第一个声音克隆工作流](#第一个声音克隆工作流)
- [以后直接加载声模](#以后直接加载声模)
- [十个节点完整说明](#十个节点完整说明)
- [小说转多人语音工作流](#小说转多人语音工作流)
- [多人台词语法](#多人台词语法)
- [采样与长文本参数](#采样与长文本参数)
- [路径与文件位置](#路径与文件位置)
- [更新插件](#更新插件)
- [常见问题](#常见问题)
- [许可证与上游](#许可证与上游)

## 功能与节点

全部节点位于 `JR/Audio/IndexTTS 2.5`，共 10 个：

| 节点 | 用途 | 输出 |
|---|---|---|
| JR IndexTTS 2.5 Loader | 下载、加载和缓存模型 | `JR_INDEXTTS25_MODEL` |
| JR IndexTTS 2.5 Voice Preset | 从参考音频登记声模 | `JR_INDEXTTS25_VOICE` |
| JR IndexTTS 2.5 Load Voice Preset | 无需参考音频，加载已保存声模 | `JR_INDEXTTS25_VOICE` |
| JR IndexTTS 2.5 Voice Preset Manager | 列出、查看、改名或删除声模 | `STRING` JSON |
| JR IndexTTS 2.5 Emotion Control | 手动、参考音频或文本情绪 | `JR_INDEXTTS25_EMOTION` |
| JR IndexTTS 2.5 Pronunciation Enhance (LLM) | 用 LLM 添加官方发音标注 | `STRING` |
| JR IndexTTS 2.5 Novel to Dialogue (LLM) | 把小说转换为 Multi-Talk 角色脚本 | `STRING` × 3 |
| JR IndexTTS 2.5 Generate | 单声音合成 | `AUDIO` |
| JR IndexTTS 2.5 Multi-Talk Generate | 最多 10 个声音的多人合成 | `AUDIO` |
| JR IndexTTS 2.5 Runtime Diagnostics | 查看环境、缓存或卸载模型 | `STRING` JSON |

主要能力：

- 中文、英文、日文、阿拉伯文和西班牙文合成
- 零样本声音克隆和持久化声模库
- 手动八维情绪、情绪参考音频、台词自动分析、独立情绪描述
- llama.cpp / OpenAI-compatible API 情绪分析
- 可选的官方内置 QwenEmotion
- 中文、英文、日文 LLM 发音增强
- 小说旁白/引语/说话人识别与原文保护
- 多人台词与逐句情绪标签
- `do_sample`、temperature、top-p、top-k、beam 等官方生成参数
- 长文本切段、段间静音和 ComfyUI ProgressBar
- 模型缓存查看、单模型卸载和全部卸载

## 安装前确认

### 1. 运行环境

严格模式检查：

| 项目 | 需要的版本 |
|---|---|
| 系统 | Windows |
| Python | 3.13.11 |
| PyTorch | 2.11.0+cu130 |
| TorchAudio | 2.11.0+cu130 |
| Torch CUDA | 13.0 |
| GPU | NVIDIA CUDA GPU |

`strict_environment=false` 只关闭版本拦截，不代表其他版本已经兼容。遇到版本错误时，不要直接降级或升级 Torch、Transformers、NumPy。

### 2. 磁盘和网络

首次下载包括 IndexTTS-2.5 主模型和运行所需辅助模型，会占用数 GB 磁盘。下载期间需要能访问 Hugging Face 或 ModelScope；中断后已有文件会保留，可以再次运行继续补全。

### 3. 参考音频

建议准备一段：

- 只有一个人说话；
- 人声清楚、噪声低；
- 没有背景音乐、混响或其他人的声音；
- 开头和结尾没有很长的空白。

参考音频只决定声音特征。需要控制表演情绪时，另外连接 Emotion Control。

## 安装插件

### 方法一：Git 克隆

先完全停止 ComfyUI，然后进入 `ComfyUI/custom_nodes`：

```powershell
git clone https://github.com/Goldlionren/ComfyUI_JR_IndexTTS25.git
```

最终结构应为：

```text
ComfyUI/
└─ custom_nodes/
   └─ ComfyUI_JR_IndexTTS25/
      ├─ __init__.py
      ├─ nodes.py
      ├─ backend/
      ├─ index_tts/          # 已内置、已兼容的官方运行源码
      ├─ patches/            # 兼容修改的审计副本
      ├─ requirements.txt
      └─ README.md
```

不需要再克隆 `index-tts/index-tts`，也不需要运行 `git am`。

### 安装基础依赖

必须使用“实际启动 ComfyUI 的那个 Python”。不要使用系统里另一个 Python，也不要使用 `pip install -U`。

ComfyUI-aki-v3 示例，在整合包根目录执行：

```powershell
.\python\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_JR_IndexTTS25\requirements.txt
```

官方 Windows Portable 的 Python 路径通常类似：

```powershell
.\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_JR_IndexTTS25\requirements.txt
```

标准虚拟环境通常类似：

```powershell
.\venv\Scripts\python.exe -m pip install -r .\custom_nodes\ComfyUI_JR_IndexTTS25\requirements.txt
```

上面三条命令只选择与你的目录结构匹配的一条。根 `requirements.txt` 不声明 Torch、TorchAudio、TorchVision、Transformers、Tokenizers、NumPy 或 Safetensors，避免主动替换 ComfyUI 核心栈。

可选依赖（下面以 ComfyUI-aki-v3 为例；其他安装只替换 Python 路径）：

```powershell
# 只有使用内置 QwenEmotion / ModelScope 时才需要
.\python\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_JR_IndexTTS25\requirements-builtin-qwen.txt

# 只有日文 JA 分词/发音时才需要
.\python\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_JR_IndexTTS25\requirements-japanese.txt
```

中文和英文测试不需要安装 fugashi/MeCab。

安装完成后重新启动 ComfyUI，右键节点菜单搜索 `JR IndexTTS`。

## 第一次运行：自动下载模型

新用户不需要填写任何源码路径。添加 `JR IndexTTS 2.5 Loader`，推荐首次设置：

```text
model_path_override: 留空
download_model: true
source_path_override: 留空
device: cuda:0
precision: fp32
enable_qwen_emotion: false
strict_environment: true
```

第一次执行工作流时：

1. Loader 自动使用插件内置的 `index_tts`；
2. 主模型下载到 `ComfyUI/models/IndexTTS-2.5`；
3. 模型初始化时补齐 `hf_cache` 中的辅助模型；
4. 完成后模型保留在磁盘，并缓存在当前 ComfyUI 进程中；
5. 以后可以关闭 `download_model`，已有完整模型会直接复用。

如果想把模型下载到其他磁盘：

```text
model_path_override: D:\AI-Models\IndexTTS-2.5
download_model: true
```

`model_path_override` 必须是模型根目录，不是 `gpt.pth` 文件本身。

核心模型目录至少包含：

```text
IndexTTS-2.5/
├─ config.yaml
├─ gpt.pth
├─ s2mel.pth
├─ codec.pth
├─ multilingual_zh_ja_yue_char_del.tiktoken
├─ wav2vec2bert_stats.pt
├─ hf_cache/
└─ qwen0.6bemo4-merge/     # 只在 builtin_qwen 时加载
```

下载模型表示使用者自行确认并接受上游模型许可证。插件不会在 Loader 执行时安装 Python package。

## 第一个声音克隆工作流

### 工作流连接

```text
Load Audio ─→ JR IndexTTS 2.5 Voice Preset ─┐
                                             ├→ JR IndexTTS 2.5 Generate ─→ Preview Audio / Save Audio
JR IndexTTS 2.5 Loader ──────────────────────┘
```

### 操作步骤

1. 用 ComfyUI 的 `Load Audio` 载入参考人声；
2. 连接到 `Voice Preset.reference_audio`；
3. `speaker_name` 填一个容易识别的名字，例如 `紫灵`；
4. Loader 按上一节设置；
5. 把 Loader 的 `model` 和 Voice Preset 的 `voice` 连接到 Generate；
6. Generate 的 `text` 输入短句，例如 `你好，这是我的第一次语音测试。`；
7. `language=ZH`，其他生成参数先保持默认；
8. 把 Generate 的 `audio` 连接到 Preview Audio 或 Save Audio；
9. Queue 工作流。

第一次执行 Voice Preset 会把参考人声登记到声模库。它同时输出可立即生成的 `voice`，所以不需要再接一次 Load Voice Preset。

## 以后直接加载声模

声模保存后，不再需要 Load Audio：

```text
JR IndexTTS 2.5 Load Voice Preset ───────────┐
                                             ├→ JR IndexTTS 2.5 Generate ─→ AUDIO
JR IndexTTS 2.5 Loader ──────────────────────┘
```

在 `preset` 下拉框选择 `name :: vp_xxx`。如果刚创建的声模还没出现在下拉框，刷新节点定义、刷新浏览器或重启 ComfyUI；也可以把 ID/名称直接填入 `preset_id_or_name_override`。

默认声模库：

```text
ComfyUI/models/indextts/voice_presets/
```

每个声模保存稳定 ID、显示名称、`prompt.wav`、采样率、时长和 SHA-256。移动或备份声模库时，应完整复制整个声模目录。

## 小说转多人语音工作流

先启动 llama.cpp 的 OpenAI-compatible API。默认地址是 `http://127.0.0.1:10000`，API key 和 model 通常都可以留空。model 留空时插件会自动读取 llama.cpp 当前加载的模型。

连接方式：

```text
小说原文 → Novel to Dialogue.dialogue ───────────────┐
                                                      ├→ Multi-Talk Generate → AUDIO
Loader.model ─────────────────────────────────────────┤
Load Voice Preset（旁白、紫灵等）────────────────────┘
```

示例原文：

```text
天色渐渐变暗了，紫灵走在山谷中略显凄凉。她仰头看天，自言自语道“时间差不多了，我要赶紧加快速度赶过去了。”
```

`speaker_only` 输出：

```text
[旁白]: 天色渐渐变暗了，紫灵走在山谷中略显凄凉。她仰头看天，自言自语道
[紫灵]: 时间差不多了，我要赶紧加快速度赶过去了。
```

`llm_emotion_tags` 输出还会带现有 Multi-Talk 能直接解析的八维情绪标签。`strict_text_preservation=true` 时，LLM 只要删字、加字、改写、重复或打乱正文，节点就会拒绝结果并报错。

识别出的角色名必须与连接到 Multi-Talk 的 Voice Preset 名称一致。建议在 `known_speakers` 预先填写重要角色，例如：

```text
旁白, 紫灵, 林志远
```

## 十个节点完整说明

### 1. JR IndexTTS 2.5 Loader

下载、加载和缓存模型，输出 `JR_INDEXTTS25_MODEL`。

| 参数 | 初学者设置 | 说明 |
|---|---|---|
| `model_path_override` | 留空 | 留空时搜索已有模型；下载时默认 `ComfyUI/models/IndexTTS-2.5`。填写时必须指向模型根目录。 |
| `download_model` | 首次开启 | 目标不完整时从官方仓库下载并补全；默认关闭。完整模型不会重复下载。 |
| `source_path_override` | 留空 | 自动使用插件内置 `index_tts`。仅开发者测试其他源码时填写。 |
| `device` | `cuda:0` | 第一块 NVIDIA GPU。多卡可尝试 `cuda:1`。 |
| `precision` | 先用 `fp32` | `bf16` 主要作用于 GPT 部分，不代表整个模型都转换为 BF16。 |
| `enable_qwen_emotion` | `false` | 只有 Emotion Control 使用 `builtin_qwen` 时开启；会增加显存占用。 |
| `strict_environment` | `true` | 检查目标 Python、Torch、CUDA 和 GPU。关闭不代表其他版本已兼容。 |

相同设置会复用进程内模型缓存。Windows 基线主动关闭 CUDA kernel JIT、DeepSpeed、第三方 flash-attn accel 和 `torch.compile`，优先保证稳定。

### 2. JR IndexTTS 2.5 Voice Preset

输入参考音频、保存声模，并立即输出 `JR_INDEXTTS25_VOICE`。

| 参数 | 说明 |
|---|---|
| `reference_audio` | ComfyUI `AUDIO`。建议清晰、单人、低噪声、无背景音乐。 |
| `speaker_name` | 声模名称；也是 Multi-Talk 的角色匹配名。空值会使用 `Narrator`。 |
| `overwrite_existing` | 同名但音频不同时是否替换。默认关闭以防误覆盖；覆盖后稳定 ID 保持不变。 |

同名且音频内容相同时复用已有记录。同名但音频不同时会报错，除非开启 `overwrite_existing`。

### 3. JR IndexTTS 2.5 Load Voice Preset

从持久化声模库加载声音，无需重新输入参考音频。

| 参数 | 说明 |
|---|---|
| `preset` | 下拉框，格式为 `name :: id`。 |
| `preset_id_or_name_override` | 可填稳定 ID 或名称；非空时优先于下拉框。 |

输出类型与 Voice Preset 完全相同，可以连接 Generate 或 Multi-Talk。

### 4. JR IndexTTS 2.5 Voice Preset Manager

管理声模库，输出格式化 JSON。建议连接任意 Show/Preview Text 节点查看。

| `action` | 需要填写 | 作用 |
|---|---|---|
| `list` | 无 | 列出所有声模及 `id`、`name`、音频路径、采样率、时长和哈希。 |
| `inspect` | `preset_id_or_name` | 查看一个声模。 |
| `rename` | `preset_id_or_name`、`new_name` | 修改名称，稳定 ID 不变。 |
| `delete` | `preset_id_or_name`、`confirm_delete=true` | 删除记录和对应 `prompt.wav`。 |

删除不可由节点撤销。重要声模请先备份声模目录。

### 5. JR IndexTTS 2.5 Emotion Control

输出 `JR_INDEXTTS25_EMOTION`，连接 Generate/Multi-Talk 的可选 `emotion` 输入。不连接时使用模型自然情绪。

四种 mode：

| mode | 情绪来自哪里 | 需要的额外输入 |
|---|---|---|
| `emotion_vector` | 手动八维滑块 | 调整八个情绪值 |
| `reference_audio` | 另一段音频的表演情绪 | 连接 `emotion_reference_audio` |
| `auto_from_text` | Generate 实际要朗读的正文 | 选择文本情绪后端 |
| `emotion_text` | 独立表演描述 | 在 `emotion_text` 写描述并选择后端 |

`emotion_text` 是情绪说明，不是朗读正文，例如：

```text
压低声音，克制，略带失望，但不要哭腔。
```

官方八维顺序：

```text
happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
开心，生气，悲伤，害怕，厌恶，忧郁，惊讶，平静
```

| 参数 | 说明 |
|---|---|
| 八个情绪滑块 | 只在 `emotion_vector` 中使用，范围 0–1。总量超过 0.8 时会按比例归一化。 |
| `strength` | 情绪对最终声音的作用强度。文本自动情绪建议先从 0.5–0.7 开始。 |
| `apply_official_bias` | 对手动向量使用官方各维偏置，默认开启。 |
| `emotion_reference_audio` | 只在 `reference_audio` mode 中需要。 |
| `emotion_text` | 只在 `emotion_text` mode 中需要。 |
| `random_sampling` | 官方情绪条件的随机采样；它与 Generate 的 `do_sample` 不是同一个开关。 |
| `text_emotion_backend` | `llama.cpp_openai_api` 或 `builtin_qwen`。 |
| `openai_api_url` | OpenAI-compatible 服务地址，可填根地址、`/v1` 或完整 chat completions URL。 |
| `openai_api_key` | 本地 llama.cpp 通常留空。 |
| `openai_model` | 留空时先请求 `/v1/models` 并使用返回的第一个模型 ID。 |
| `llm_timeout_seconds` | LLM 请求超时，默认 120 秒。 |

#### 推荐：外部 llama.cpp 自动情绪

先启动支持 OpenAI API 的 llama.cpp server，然后设置：

```text
Loader.enable_qwen_emotion: false
Emotion Control.mode: auto_from_text 或 emotion_text
Emotion Control.text_emotion_backend: llama.cpp_openai_api
Emotion Control.openai_api_url: http://127.0.0.1:10000
Emotion Control.openai_api_key: 留空
Emotion Control.openai_model: 留空
Emotion Control.strength: 0.5–0.7
```

外部 LLM 返回会被限制并转换为官方八维向量，再送入原生推理。这条路径不占用内置 Qwen 的额外显存。

#### 可选：内置 QwenEmotion

设置：

```text
Loader.enable_qwen_emotion: true
Emotion Control.text_emotion_backend: builtin_qwen
```

这条路径要求模型目录中存在完整 `qwen0.6bemo4-merge`，并可能需要 `requirements-builtin-qwen.txt`。它会增加模型加载时间和 GPU 显存占用。

### 6. JR IndexTTS 2.5 Pronunciation Enhance (LLM)

调用 llama.cpp/OpenAI-compatible LLM，只给歧义词添加官方发音标注，输出 `enhanced_text`。把它连接到 Generate 的 `text`。

| 参数 | 说明 |
|---|---|
| `text` | 原始正文。 |
| `language` | `ZH`、`EN` 或 `JA`。 |
| `openai_api_url/key/model/timeout` | 与 Emotion Control 相同；model 留空时自动发现。 |
| `instruction` | 给 LLM 的额外要求。默认只标必要词，不改写原文。 |

官方标注示例：

```text
中文：银<行|HANG2>今天休息，他沿路<行|XING2>走。
英文：A <minute|M IH1 . N AH0 T> detail.
日文：彼は<上手|じょうず>です。
```

节点会校验：去掉标注后的正文必须与原文完全一致。LLM 如果改写、删字或加字，节点会报错，不会静默使用错误结果。也可以不使用该节点，直接在 Generate 正文中手写官方标注。

### 7. JR IndexTTS 2.5 Novel to Dialogue (LLM)

调用 llama.cpp/OpenAI-compatible LLM，把小说中的旁白、引语和说话人转换为 Multi-Talk 标签。主输出 `dialogue` 可以直接连接 Multi-Talk Generate 的 `dialogue`。

| 参数 | 初学者设置 | 说明 |
|---|---|---|
| `novel_text` | 粘贴小说原文 | 支持连接其他 `STRING` 节点。不能为空。 |
| `narrator_name` | `旁白` | 所有叙述内容使用的角色名；需要准备同名 Voice。 |
| `known_speakers` | 填主要角色 | 用逗号、分号或换行分隔；名称应与 Voice Preset 一致。允许 LLM 发现新角色。 |
| `emotion_mode` | `llm_emotion_tags` | 见下面三种模式。 |
| `emotion_strength` | `0.7` | 只缩放 LLM 生成的情绪向量，0 为关闭，1 为原值。最终总量仍限制在 0.8。 |
| `chunk_size_chars` | `2000` | 长篇小说分块大小。优先在引号外的句末切分，并向下一块传递最近角色上下文。 |
| `strict_text_preservation` | `true` | 检查正文没有被删改、重复或乱序。正式生成必须保持开启。 |
| `openai_api_url` | `http://127.0.0.1:10000` | llama.cpp/OpenAI-compatible 服务地址。 |
| `openai_api_key` | 留空 | 本地 llama.cpp 通常不需要。 |
| `openai_model` | 留空 | 自动读取 `/v1/models` 返回的第一个模型。 |
| `llm_temperature` | `0.1` | 保持低值，减少改写和不稳定分段。 |
| `llm_max_tokens` | `4096` | 单个分块允许的最大输出 token。响应截断时增大或降低分块大小。 |
| `llm_timeout_seconds` | `300` | 每个分块的请求超时。 |
| `instruction` | 通常留空 | 可写角色别名或特殊引号规则，不要要求润色、翻译或改写正文。 |

三种情绪模式：

| mode | 输出形式 | 特点 |
|---|---|---|
| `llm_emotion_tags` | `[紫灵|calm=0.35]: ...` | 推荐。同一次小说分析直接生成官方八维情绪标签，不需要逐句再请求 LLM。 |
| `speaker_only` | `[紫灵]: ...` | 只区分旁白和角色，不添加逐句情绪。 |
| `auto_emotion` | `[紫灵|auto]: ...` | Multi-Talk 生成每一段时再调用情绪分析，调用次数更多。 |

三个输出：

| 输出 | 用途 |
|---|---|
| `dialogue` | 直接连接 Multi-Talk Generate。 |
| `speaker_list` | 每行一个识别出的角色，用来检查是否准备了对应声模。 |
| `conversion_report` | JSON 报告，包含分块数、角色、模式、严格校验状态和警告。 |

节点只删除成对引号的分隔符，不应改写引号内外的正文。特别长、角色关系复杂或代词很多的章节，建议按章节运行并检查 `speaker_list` 和转换结果后再生成音频。

### 8. JR IndexTTS 2.5 Generate

单声音生成。必须连接 Loader 的 `model` 和任一 Voice 节点的 `voice`，输出标准 ComfyUI `AUDIO`（22,050 Hz 单声道）。

| 参数 | 说明 |
|---|---|
| `text` | 要朗读的正文，可接 Pronunciation Enhance。不能为空。 |
| `language` | `ZH`、`EN`、`JA`、`AR`、`ES`。应与正文主要语言匹配。 |
| `duration_factor` | 0.5–2.0；增大通常让结果更长、更慢。先用 1.0。 |
| `text_normalization` | 官方文本归一化，通常开启。 |
| `interval_silence_ms` | 长文本内部切段之间的静音，默认 200 ms。 |
| `seed` | Torch/CUDA 种子。相同环境与参数有助于复现，但 GPU 不保证跨机器逐样本完全一致。 |
| `unload_model_after` | 生成后卸载本插件模型。连续生成时保持关闭；开启后下次使用需让 Loader 重新执行。 |
| `emotion` | 可选，连接 Emotion Control；未连接时使用自然情绪。 |

高级生成参数见“采样与长文本参数”。

### 9. JR IndexTTS 2.5 Multi-Talk Generate

按角色标签选择多个声音，逐段生成并拼成一个 `AUDIO`。

| 参数 | 说明 |
|---|---|
| `model` | 连接 Loader。 |
| `dialogue` | 多人台词，语法见下一节。 |
| `language` | 当前整段共用一个语言选项。 |
| `voice_1` … `voice_10` | 最多 10 个 Voice；至少连接一路。 |
| `emotion` | 可选共享情绪；也为逐句 `auto` / `emo_text` 提供 LLM URL、key、model 和 timeout。 |
| `gap_ms` | 不同台词段之间的静音。 |
| `duration_factor`、`text_normalization` | 与 Generate 相同。 |
| `seed` | 第一段使用 seed，后续段依次使用 `seed + 段索引`。 |
| `unload_model_after` | 整段完成后卸载模型；连续生成时保持关闭。 |

角色名按 Voice 的声模名称进行不区分大小写匹配。未知角色会回退到第一个已连接 Voice，因此正式生成前应检查角色拼写。

### 10. JR IndexTTS 2.5 Runtime Diagnostics

输出 JSON。建议连接 Show/Preview Text 节点查看。

| action | 说明 |
|---|---|
| `report` | 查看 Python、Torch、Torch CUDA、CUDA、GPU 和本插件模型缓存。 |
| `unload_model` | 卸载连接的单个模型；必须连接 `model`。 |
| `unload_all` | 卸载本插件缓存的全部 IndexTTS 模型并清理 CUDA 缓存。 |

它不会卸载其他 custom node 的模型，也不会安装、升级或删除 Python package。

## 多人台词语法

### 基本格式

半角和全角冒号都支持：

```text
[紫灵]: 你好，今天过得怎么样？
[旁白]：夜色慢慢沉了下来。
```

角色名应与已连接 Voice 的 `speaker_name`/声模名一致。每个新 `[角色]` 开始一个片段；正文可以跨行，直到下一个角色标签。

### 逐句情绪

在角色名后用 `|` 添加选项：

```text
[紫灵|开心=0.55|平静=0.20|强度=0.8]: 你好，今天过得怎么样？
[旁白|sad=0.65]: 她沉默了一会儿。
[紫灵|auto|strength=0.6]: 我真的不知道应该怎么办。
[旁白|emo_text=低沉、克制、略带悲伤|strength=0.6]: 夜色慢慢沉了下来。
[旁白|natural]: 这句不使用共享情绪。
```

情绪别名：

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

- `开心=0.6` / `happy=0.6`：指定 0–1 的向量值；省略数值时默认 0.8。
- `strength=0.6` / `alpha=0.6` / `强度=0.6`：该句情绪作用强度。
- `auto` / `自动` / `自动情绪`：分析该句正文。
- `emo_text=描述` / `emotion_text=描述` / `情绪文本=描述`：分析独立表演描述。
- `random` / `随机`：开启情绪随机采样，也可写 `random=false`。
- `bias` / `official_bias` / `官方偏置`：官方偏置，可写 `bias=false`。
- `natural` / `none` / `自然` / `无情绪`：本句不使用情绪，必须单独出现。

布尔值接受 `true/false`、`yes/no`、`on/off`、`1/0`、`是/否`、`开/关`。手动向量不能与 `auto` 或 `emo_text` 混用。

逐句标签优先于共享 `emotion`。逐句 `auto`/`emo_text` 会继承共享 Emotion Control 的 LLM 配置；没有连接共享 Emotion Control 时，默认调用 `http://127.0.0.1:10000`。

## 采样与长文本参数

Generate 和 Multi-Talk 都提供：

| 参数 | 默认 | 作用 |
|---|---:|---|
| `do_sample` | true | 开启概率采样，表现通常更有变化；关闭时更偏确定性/搜索式生成。 |
| `temperature` | 0.8 | 采样随机度。越高越发散，越低越保守。 |
| `top_p` | 0.8 | nucleus sampling 的累计概率范围。1.0 近似不裁剪。 |
| `top_k` | 30 | 只保留最高概率的 K 个候选；0 表示关闭 top-k。 |
| `num_beams` | 3 | beam 数；提高可能更稳定，但更慢、更占显存。 |
| `repetition_penalty` | 10.0 | 重复惩罚。过高可能影响自然度。 |
| `length_penalty` | 0.0 | 搜索时的长度偏好；正值偏长，负值偏短。 |
| `max_mel_tokens` | 1500 | 单段最大声学 token，范围 50–1815。 |
| `max_text_tokens_per_segment` | 120 | 长文本每段最大文本 token，范围 20–600。 |

初学者建议全部保持默认。只想提高可复现性时，可以先尝试：

```text
do_sample: false
num_beams: 3
seed: 固定值
```

`do_sample=false` 时，temperature、top-p、top-k 的采样作用会减弱或不生效。本项目内置源码已修复上游将 `do_sample` 固定为 true 的问题，所以 UI 开关会真实传入 GPT。

长文本出现断句不自然时，优先调整标点和 `max_text_tokens_per_segment`；不要一开始同时修改所有采样参数。

## 生成进度

Generate 和 Multi-Talk 会把文本处理、情绪分析、逐段生成和完成阶段映射到 ComfyUI ProgressBar。Multi-Talk 按片段数量汇总进度。

这是阶段级进度，不是严格逐 token 百分比。某些模型阶段没有更细回调，因此进度可能停留一段时间后跳到下一阶段。

## 路径与文件位置

### 模型搜索顺序

1. Loader `model_path_override`
2. `INDEXTTS25_MODEL_DIR`
3. `ComfyUI/models/IndexTTS-2.5`
4. `ComfyUI/models/indextts/IndexTTS-2.5`
5. 开发工作区相邻的 `models/IndexTTS-2.5`

当 `download_model=true`：

- `model_path_override` 非空：以该路径为下载目标；
- `model_path_override` 留空：固定下载到 `ComfyUI/models/IndexTTS-2.5`；
- 已找到完整模型：直接复用。

### 源码搜索顺序

1. Loader `source_path_override`
2. `INDEXTTS25_SOURCE`
3. `ComfyUI_JR_IndexTTS25/index_tts`（普通用户默认）
4. 开发工作区相邻的 `source/index-tts`

正常安装永远可以把 `source_path_override` 留空。若报源码缺失，通常说明 Git 下载不完整；确认存在：

```text
ComfyUI_JR_IndexTTS25/index_tts/indextts/infer_v2_5.py
```

### 声模库

1. `INDEXTTS25_PRESET_DIR`
2. `ComfyUI/models/indextts/voice_presets`
3. 非 ComfyUI 独立运行时的插件本地 `voice_presets`

### 临时缓存

`INDEXTTS25_CACHE_DIR` 可以指定缓存目录。默认使用 ComfyUI temp 下的 `jr_indextts25`，Numba 缓存不会写入 ComfyUI Python 的 `site-packages`。

## 更新插件

完全停止 ComfyUI，在插件目录执行：

```powershell
git pull
```

然后重新启动 ComfyUI。不要把备份目录放在 `custom_nodes` 内，否则 ComfyUI 可能同时加载新旧两份节点。

更新插件不会自动删除模型或声模；仍建议在重要更新前备份：

```text
ComfyUI/models/IndexTTS-2.5
ComfyUI/models/indextts/voice_presets
```

## 常见问题

### 找不到 JR IndexTTS 节点

确认插件目录不是多嵌套一层：

```text
正确：custom_nodes/ComfyUI_JR_IndexTTS25/__init__.py
错误：custom_nodes/ComfyUI_JR_IndexTTS25/ComfyUI_JR_IndexTTS25/__init__.py
```

查看 ComfyUI 启动窗口中的 custom node import 错误。缺包时必须使用启动 ComfyUI 的 Python 安装 `requirements.txt`。

### Loader 报环境版本不兼容

先用 Runtime Diagnostics 查看 Python、Torch 和 Torch CUDA。目标是 Python 3.13.11、Torch 2.11.0+cu130、CUDA 13.0。不要把 `strict_environment=false` 当作通用修复。

### Loader 报找不到源码

普通用户不填写 `source_path_override`。确认 Git 克隆完整，并存在 `index_tts/indextts/infer_v2_5.py`。不要把路径填成 `indextts` 子目录或某个 `.py` 文件。

### Loader 报找不到模型

第一次使用请开启 `download_model`。如果手动填写路径，它应指向同时包含 `config.yaml`、`gpt.pth`、`s2mel.pth` 等文件的模型根目录。

### 模型下载中断

已有文件会保留。检查磁盘空间和网络后，再次开启 `download_model` 并运行。不要手动删除已经完成的大文件，除非确认文件损坏。

### llama.cpp 情绪或发音增强连接失败

确认服务已启动，并能访问：

```text
http://127.0.0.1:10000/v1/models
http://127.0.0.1:10000/v1/chat/completions
```

API URL 可填写 `http://127.0.0.1:10000`。model 留空时节点先请求 `/v1/models`；服务需要鉴权时才填写 key。

### builtin_qwen 报错

同时确认：

- Loader `enable_qwen_emotion=true`；
- Emotion Control `text_emotion_backend=builtin_qwen`；
- 模型目录存在完整 `qwen0.6bemo4-merge`；
- 已评估并安装 `requirements-builtin-qwen.txt`；
- GPU 显存足够。

使用外部 llama.cpp 时，Loader 的 Qwen 开关应关闭。

### 新声模没有出现在下拉框

刷新节点定义、浏览器页面或重启 ComfyUI。也可以在 Load Voice Preset 的 `preset_id_or_name_override` 直接填写声模名称或稳定 ID。

### 同名声模保存失败

同名、同音频会复用；同名、不同音频默认拒绝覆盖。确认确实要替换后才开启 `overwrite_existing`。

### Multi-Talk 角色用了错误声音

检查 `[角色]` 是否与 Voice 的声模名称一致。未知角色会回退到第一个连接的 Voice。建议每个角色使用唯一名称。

### 提示缺少 fugashi / MeCab

这是日文可选依赖。中文和英文不需要安装；确实使用 JA 时再安装 `requirements-japanese.txt`。

### 西班牙语或阿拉伯语缺少 NeMo

Windows 环境中 NeMo/Pynini 不是默认依赖。内置兼容源码会在 NeMo 不可用时保留原文继续合成，表示高级文本归一化降级，不影响核心 TTS 调用。

### TorchAudio 保存 WAV 报 TorchCodec/FFmpeg DLL

插件内置源码已经使用 SoundFile PCM16 写入，不应通过升级/降级 TorchAudio 解决。确认没有用 `source_path_override` 指向未打补丁的官方源码。

### CUDA 显存不足

依次尝试：

1. 关闭 Loader `enable_qwen_emotion`，改用外部 llama.cpp；
2. 缩短文本或降低 `max_text_tokens_per_segment`；
3. 关闭其他占用显存的工作流/模型；
4. 生成完成后使用 Runtime Diagnostics `unload_model` / `unload_all`；
5. GPU 支持时再测试 `bf16`。

### 卸载后提示 model handle has been unloaded

`unload_model_after` 或 Diagnostics 卸载会让当前 handle 失效。让 Loader 重新执行后再生成；连续调试时保持 `unload_model_after=false`。

## 已知边界

- 已声明兼容目标是 Windows + Python 3.13.11 + Torch 2.11.0+cu130；其他组合尚未承诺。
- `builtin_qwen` 增加模型加载时间和显存占用，默认推荐外部 llama.cpp。
- 日文需要可选分词依赖；中文和英文不依赖它。
- 西班牙语和阿拉伯语缺少 NeMo 时跳过高级文本归一化。
- Multi-Talk 当前整段共用一个 `language`。
- 进度是阶段级，不是逐 token。
- 模型权重和声模数据不包含在 GitHub 仓库中。

## 许可证与上游

插件内置了经过兼容修改的 IndexTTS 运行源码，并保留：

- `index_tts/LICENSE`
- `index_tts/LICENSE_ZH.txt`
- `index_tts/DISCLAIMER`
- `index_tts/VENDORED_SOURCE.md`

使用、复制和再分发受上游许可证约束；中英文冲突时以中文版本为准。

Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

上游：

- IndexTTS / IndexTTS-2.5：<https://github.com/index-tts/index-tts>
- ComfyUI：<https://github.com/comfyanonymous/ComfyUI>

内置源码固定信息：

- 上游基线：`a371df7d0746a0ae7fdf075798b6b04e34a0132e`
- 最终兼容 revision：`30fecfa188455a560aeea6f6dc60bc2f7c19bb14`

## 更新记录

### v0.3.0

- 新增 Novel to Dialogue (LLM)，把小说旁白和角色引语转换为 Multi-Talk 可直接使用的标签脚本。
- 支持 `speaker_only`、`auto_emotion` 和一次性 `llm_emotion_tags` 三种转换模式。
- 新增长文本分块、已知角色与上下文传递、角色列表和 JSON 转换报告。
- 严格原文保护会拒绝 LLM 的删字、加字、改写、重复或乱序结果。
- 节点总数增加到 10 个；现有 Loader、Voice、Emotion、Generate 和 Multi-Talk 接口保持不变。

### v0.2.0

- Loader 增加官方模型下载开关和默认 ComfyUI 模型目录。
- 内置固定 revision、已验证且已应用兼容修改的 IndexTTS 运行源码。
- 新用户无需第二次克隆源码或手动应用补丁，`source_path_override` 默认留空。
- 提供 9 个节点、持久化声模、多人逐句情绪、llama.cpp 情绪/发音增强、高级采样参数、长文本和进度条。

### v0.1.0

- 完成 Python 3.13.11 / Torch 2.11.0+cu130 / CUDA 13 兼容基线。
- 发布第一版 IndexTTS-2.5 ComfyUI 节点。
