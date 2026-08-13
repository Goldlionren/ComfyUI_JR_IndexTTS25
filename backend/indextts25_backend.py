from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
import torch

from .openai_compatible import (
    DEFAULT_OPENAI_API_URL,
    DEFAULT_OPENAI_MODEL,
    analyze_emotion_text,
)


MODEL_FILES = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
)
MODEL_REPO_ID = "IndexTeam/IndexTTS-2.5"
MODEL_CACHE: dict[tuple[str, str, bool, bool, str], "IndexTTS25Handle"] = {}
MODEL_CACHE_LOCK = threading.RLock()
MODEL_DOWNLOAD_LOCK = threading.RLock()
EMOTION_NAMES = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
EMOTION_NAME_ALIASES = {
    "开心": "happy",
    "快乐": "happy",
    "生气": "angry",
    "愤怒": "angry",
    "悲伤": "sad",
    "伤心": "sad",
    "害怕": "afraid",
    "恐惧": "afraid",
    "厌恶": "disgusted",
    "嫌弃": "disgusted",
    "忧郁": "melancholic",
    "低沉": "melancholic",
    "惊讶": "surprised",
    "吃惊": "surprised",
    "平静": "calm",
    "冷静": "calm",
}
DIALOGUE_TAG = re.compile(r"\[([^\]]+)\]\s*[:：]\s*")


@dataclass(frozen=True)
class VoicePreset:
    name: str
    audio: dict[str, Any]
    preset_id: str | None = None


@dataclass(frozen=True)
class EmotionControl:
    vector: tuple[float, ...] | None
    audio: dict[str, Any] | None = None
    alpha: float = 1.0
    text: str | None = None
    use_text: bool = False
    random_sampling: bool = False
    text_backend: str = "llama.cpp_openai_api"
    openai_api_url: str = DEFAULT_OPENAI_API_URL
    openai_api_key: str = ""
    openai_model: str = DEFAULT_OPENAI_MODEL
    llm_timeout_seconds: int = 120


@dataclass(frozen=True)
class DialogueSegment:
    speaker: str
    text: str
    emotion: EmotionControl | None = None


@dataclass
class IndexTTS25Handle:
    runtime: Any
    model_dir: Path
    source_dir: Path
    device: str
    use_bf16: bool
    use_qwen_emo: bool
    cache_key: tuple[str, str, bool, bool, str]
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comfy_temp_dir() -> Path | None:
    try:
        import folder_paths

        return Path(folder_paths.get_temp_directory()).resolve()
    except Exception:
        return None


def _cache_root() -> Path:
    configured = os.environ.get("INDEXTTS25_CACHE_DIR", "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        comfy_temp = _comfy_temp_dir()
        root = comfy_temp / "jr_indextts25" if comfy_temp else _plugin_root() / ".cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configure_numba_cache() -> Path:
    cache_dir = _cache_root() / "numba"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))
    return cache_dir


def _candidate_source_dirs(override: str = "") -> list[Path]:
    candidates: list[Path] = []
    if override.strip():
        candidates.append(Path(override).expanduser())
    env_path = os.environ.get("INDEXTTS25_SOURCE", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            _plugin_root() / "index_tts",
            _plugin_root().parent / "source" / "index-tts",
        ]
    )
    return [path.resolve() for path in candidates]


def resolve_source_dir(override: str = "") -> Path:
    for candidate in _candidate_source_dirs(override):
        if (candidate / "indextts" / "infer_v2_5.py").is_file():
            return candidate
    searched = "\n  - ".join(str(path) for path in _candidate_source_dirs(override))
    raise FileNotFoundError(
        "IndexTTS-2.5 source was not found. Set INDEXTTS25_SOURCE or use source_path_override. "
        f"Searched:\n  - {searched}"
    )


def _candidate_model_dirs(override: str = "") -> list[Path]:
    candidates: list[Path] = []
    if override.strip():
        candidates.append(Path(override).expanduser())
    env_path = os.environ.get("INDEXTTS25_MODEL_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    try:
        import folder_paths

        models_root = Path(folder_paths.models_dir)
        candidates.extend(
            [models_root / "IndexTTS-2.5", models_root / "indextts" / "IndexTTS-2.5"]
        )
    except Exception:
        pass
    candidates.append(_plugin_root().parent / "models" / "IndexTTS-2.5")
    return [path.resolve() for path in candidates]


def resolve_model_dir(override: str = "") -> Path:
    for candidate in _candidate_model_dirs(override):
        if all((candidate / name).is_file() for name in MODEL_FILES):
            return candidate
    searched = "\n  - ".join(str(path) for path in _candidate_model_dirs(override))
    raise FileNotFoundError(
        "A complete IndexTTS-2.5 model directory was not found. Set INDEXTTS25_MODEL_DIR "
        f"or use model_path_override. Searched:\n  - {searched}"
    )


def _default_model_download_dir() -> Path:
    try:
        import folder_paths

        return (Path(folder_paths.models_dir) / "IndexTTS-2.5").resolve()
    except Exception:
        return (_plugin_root().parent / "models" / "IndexTTS-2.5").resolve()


def model_download_dir(override: str = "") -> Path:
    value = (override or "").strip()
    return Path(value).expanduser().resolve() if value else _default_model_download_dir()


def missing_model_files(model_dir: Path) -> list[str]:
    return [name for name in MODEL_FILES if not (model_dir / name).is_file()]


def _download_model_snapshot(source_dir: Path, target_dir: Path) -> None:
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    downloader = importlib.import_module("indextts.utils.model_download")
    downloader.snapshot_download(MODEL_REPO_ID, local_dir=str(target_dir))


def ensure_model_available(
    model_path_override: str = "",
    source_dir: Path | None = None,
    *,
    download_model: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
) -> Path:
    override = (model_path_override or "").strip()
    if override:
        explicit_target = model_download_dir(override)
        if not missing_model_files(explicit_target):
            return explicit_target
    else:
        # Preserve discovery of an already-complete environment/configured model.
        try:
            return resolve_model_dir("")
        except FileNotFoundError:
            pass

    target_dir = model_download_dir(model_path_override)
    missing = missing_model_files(target_dir)
    if not download_model:
        raise FileNotFoundError(
            f"IndexTTS-2.5 model is incomplete at {target_dir} (missing: {', '.join(missing)}). "
            "Enable download_model in Loader to download it, or set model_path_override "
            "to an existing complete model directory."
        )
    if source_dir is None:
        raise ValueError("source_dir is required to download the official IndexTTS-2.5 model")

    with MODEL_DOWNLOAD_LOCK:
        # A parallel Loader may have completed the same target while this one waited.
        missing = missing_model_files(target_dir)
        if not missing:
            return target_dir
        if target_dir.exists() and not target_dir.is_dir():
            raise NotADirectoryError(f"Model download target is not a directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(0.01, f"downloading {MODEL_REPO_ID} to {target_dir}")
        try:
            _download_model_snapshot(source_dir, target_dir)
        except Exception as error:
            raise RuntimeError(
                f"Failed to download {MODEL_REPO_ID} to {target_dir}: "
                f"{type(error).__name__}: {error}"
            ) from error

        missing = missing_model_files(target_dir)
        if missing:
            raise RuntimeError(
                f"Model download finished but {target_dir} is still incomplete "
                f"(missing: {', '.join(missing)}). Existing files were preserved."
            )
        if progress_callback is not None:
            progress_callback(0.08, "model download complete")
        return target_dir


def assert_runtime_compatible(strict: bool = True) -> dict[str, Any]:
    info = runtime_diagnostics()
    errors: list[str] = []
    if sys.version_info[:3] != (3, 13, 11):
        errors.append(f"Python 3.13.11 required, found {sys.version.split()[0]}")
    if not torch.__version__.startswith("2.11.0+cu130"):
        errors.append(f"torch 2.11.0+cu130 required, found {torch.__version__}")
    if torch.version.cuda != "13.0":
        errors.append(f"Torch CUDA 13.0 required, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
    if strict and errors:
        raise RuntimeError("Incompatible ComfyUI runtime:\n- " + "\n- ".join(errors))
    info["compatibility_errors"] = errors
    return info


def _import_runtime(source_dir: Path):
    _configure_numba_cache()
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    module = importlib.import_module("indextts.infer_v2_5")
    return module.IndexTTS2


def load_model(
    model_path_override: str = "",
    source_path_override: str = "",
    device: str = "cuda:0",
    precision: str = "fp32",
    enable_qwen_emotion: bool = False,
    strict_environment: bool = True,
    download_model: bool = False,
    progress_callback: Callable[[float, str], None] | None = None,
) -> IndexTTS25Handle:
    assert_runtime_compatible(strict=strict_environment)
    source_dir = resolve_source_dir(source_path_override)
    model_dir = ensure_model_available(
        model_path_override,
        source_dir,
        download_model=download_model,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(0.1, "loading IndexTTS-2.5 model")
    use_bf16 = precision.lower() == "bf16"
    if precision.lower() not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but the selected CUDA runtime does not support it")
    key = (str(model_dir), device, use_bf16, bool(enable_qwen_emotion), str(source_dir))
    with MODEL_CACHE_LOCK:
        cached = MODEL_CACHE.get(key)
        if cached is not None:
            if progress_callback is not None:
                progress_callback(1.0, "model loaded from cache")
            return cached
        runtime_class = _import_runtime(source_dir)
        runtime = runtime_class(
            cfg_path=str(model_dir / "config.yaml"),
            model_dir=str(model_dir),
            use_bf16=use_bf16,
            device=device,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            use_qwen_emo=bool(enable_qwen_emotion),
        )
        handle = IndexTTS25Handle(
            runtime=runtime,
            model_dir=model_dir,
            source_dir=source_dir,
            device=device,
            use_bf16=use_bf16,
            use_qwen_emo=bool(enable_qwen_emotion),
            cache_key=key,
        )
        MODEL_CACHE[key] = handle
        if progress_callback is not None:
            progress_callback(1.0, "model loaded")
        return handle


def clear_model_cache(handle: IndexTTS25Handle | None = None) -> int:
    with MODEL_CACHE_LOCK:
        if handle is None:
            removed = list(MODEL_CACHE.values())
            MODEL_CACHE.clear()
        else:
            removed_handle = MODEL_CACHE.pop(handle.cache_key, None)
            removed = [removed_handle] if removed_handle is not None else []
    removed_count = len(removed)
    for item in removed:
        item.runtime = None
    removed.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    return removed_count


def audio_to_mono_numpy(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    if not isinstance(audio, dict) or "waveform" not in audio:
        raise ValueError("ComfyUI AUDIO must contain a waveform")
    waveform = audio["waveform"]
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().to(device="cpu", dtype=torch.float32).numpy()
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
        array = array.mean(axis=0) if array.shape[0] > 1 else array[0]
    elif array.ndim == 2:
        if array.shape[0] <= 8:
            array = array.mean(axis=0)
        elif array.shape[1] <= 8:
            array = array.mean(axis=1)
        else:
            raise ValueError(f"Ambiguous AUDIO waveform shape: {array.shape}")
    elif array.ndim != 1:
        raise ValueError(f"Unsupported AUDIO waveform shape: {array.shape}")
    sample_rate = int(audio.get("sample_rate", audio.get("sampler_rate", 0)))
    if sample_rate <= 0:
        raise ValueError("ComfyUI AUDIO sample_rate is missing or invalid")
    if not np.isfinite(array).all() or not array.size:
        raise ValueError("Reference AUDIO is empty or contains NaN/Inf")
    return np.clip(array, -1.0, 1.0), sample_rate


def _materialize_audio(audio: dict[str, Any], kind: str) -> Path:
    waveform, sample_rate = audio_to_mono_numpy(audio)
    digest = hashlib.sha256()
    digest.update(str(sample_rate).encode("ascii"))
    digest.update(waveform.tobytes())
    directory = _cache_root() / "audio_prompts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{digest.hexdigest()[:24]}.wav"
    if not path.is_file():
        sf.write(path, waveform, sample_rate, subtype="PCM_16", format="WAV")
    return path


def normalize_emotion_vector(vector: tuple[float, ...], apply_bias: bool = True) -> tuple[float, ...]:
    if len(vector) != 8:
        raise ValueError(f"IndexTTS-2.5 emotion vector must contain 8 values, got {len(vector)}")
    values = [max(0.0, min(1.0, float(value))) for value in vector]
    if apply_bias:
        bias = (0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625)
        values = [value * factor for value, factor in zip(values, bias)]
    total = sum(values)
    if total > 0.8:
        values = [value * 0.8 / total for value in values]
    return tuple(values)


def _numpy_to_audio(waveform: np.ndarray, sample_rate: int) -> dict[str, Any]:
    array = np.asarray(waveform)
    if array.dtype == np.int16:
        array = array.astype(np.float32) / 32768.0
    else:
        array = array.astype(np.float32)
    array = np.squeeze(array)
    if array.ndim != 1:
        raise ValueError(f"Unexpected generated waveform shape: {array.shape}")
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("IndexTTS-2.5 returned an empty or non-finite waveform")
    return {"waveform": torch.from_numpy(array.copy()).reshape(1, 1, -1), "sample_rate": int(sample_rate)}


def generate_audio(
    handle: IndexTTS25Handle,
    voice: VoicePreset,
    text: str,
    language: str,
    emotion: EmotionControl | None = None,
    duration_factor: float = 1.0,
    text_normalization: bool = True,
    interval_silence_ms: int = 200,
    seed: int = 0,
    do_sample: bool = True,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 30,
    num_beams: int = 3,
    repetition_penalty: float = 10.0,
    length_penalty: float = 0.0,
    max_mel_tokens: int = 1500,
    max_text_tokens_per_segment: int = 120,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict[str, Any]:
    if handle.runtime is None:
        raise RuntimeError("This IndexTTS-2.5 model handle has been unloaded; run Loader again")
    if not text.strip():
        raise ValueError("Text is empty")
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ValueError("duration_factor must be between 0.5 and 2.0")
    if not 0.1 <= float(temperature) <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0")
    if not 0.0 <= float(top_p) <= 1.0:
        raise ValueError("top_p must be between 0 and 1")
    if not 0 <= int(top_k) <= 100:
        raise ValueError("top_k must be between 0 and 100")
    if not 1 <= int(num_beams) <= 10:
        raise ValueError("num_beams must be between 1 and 10")
    if not 0.1 <= float(repetition_penalty) <= 20.0:
        raise ValueError("repetition_penalty must be between 0.1 and 20")
    if not -2.0 <= float(length_penalty) <= 2.0:
        raise ValueError("length_penalty must be between -2 and 2")
    if not 50 <= int(max_mel_tokens) <= 1815:
        raise ValueError("max_mel_tokens must be between 50 and 1815")
    if not 20 <= int(max_text_tokens_per_segment) <= 600:
        raise ValueError("max_text_tokens_per_segment must be between 20 and 600")
    speaker_path = _materialize_audio(voice.audio, "speaker")
    emotion_path = _materialize_audio(emotion.audio, "emotion") if emotion and emotion.audio else None
    emotion_vector = list(emotion.vector) if emotion and emotion.vector is not None else None
    emotion_alpha = float(emotion.alpha) if emotion else 1.0
    use_emotion_text = bool(emotion and emotion.use_text)
    emotion_text = emotion.text if use_emotion_text else None
    random_emotion = bool(emotion and emotion.random_sampling)
    if use_emotion_text and emotion and emotion.text_backend == "llama.cpp_openai_api":
        if progress_callback is not None:
            progress_callback(0.02, "analyzing emotion with llama.cpp...")
        emotion_vector = list(
            analyze_emotion_text(
                emotion_text if emotion_text is not None else text,
                api_url=emotion.openai_api_url,
                api_key=emotion.openai_api_key,
                model=emotion.openai_model,
                timeout_seconds=emotion.llm_timeout_seconds,
            )
        )
        use_emotion_text = False
        emotion_text = None
    elif use_emotion_text and not handle.use_qwen_emo:
        raise RuntimeError(
            "Built-in text emotion requires Loader enable_qwen_emotion=True. "
            "Enable it or select llama.cpp_openai_api in Emotion Control."
        )
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)
    with handle.lock, torch.inference_mode():
        previous_progress = getattr(handle.runtime, "gr_progress", None)
        if progress_callback is not None:
            handle.runtime.gr_progress = lambda value, desc=None: progress_callback(
                max(0.0, min(1.0, float(value))), str(desc or "")
            )
        try:
            result = handle.runtime.infer(
                spk_audio_prompt=str(speaker_path),
                text=text,
                output_path=None,
                lang=language,
                emo_audio_prompt=str(emotion_path) if emotion_path else None,
                emo_alpha=emotion_alpha,
                emo_vector=emotion_vector,
                use_emo_text=use_emotion_text,
                emo_text=emotion_text,
                use_random=random_emotion,
                interval_silence=int(interval_silence_ms),
                duration_factor=float(duration_factor),
                text_normalization=bool(text_normalization),
                verbose=False,
                max_text_tokens_per_segment=int(max_text_tokens_per_segment),
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=None if int(top_k) == 0 else int(top_k),
                num_beams=int(num_beams),
                repetition_penalty=float(repetition_penalty),
                length_penalty=float(length_penalty),
                max_mel_tokens=int(max_mel_tokens),
            )
        finally:
            if progress_callback is not None:
                handle.runtime.gr_progress = previous_progress
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected IndexTTS-2.5 result: {type(result).__name__}")
    sample_rate, waveform = result
    if progress_callback is not None:
        progress_callback(1.0, "complete")
    return _numpy_to_audio(waveform, int(sample_rate))


def _parse_bool(value: str, option: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on", "是", "开"}:
        return True
    if normalized in {"0", "false", "no", "off", "否", "关"}:
        return False
    raise ValueError(f"Dialogue emotion option {option!r} expects true/false, got {value!r}")


def _dialogue_emotion(options: list[str]) -> EmotionControl | None:
    if not options:
        return None
    vector_values = {name: 0.0 for name in EMOTION_NAMES}
    has_vector = False
    use_text = False
    emotion_text: str | None = None
    alpha = 1.0
    random_sampling = False
    apply_bias = True

    for raw_option in options:
        option = raw_option.strip()
        if not option:
            continue
        if "=" in option:
            raw_key, raw_value = option.split("=", 1)
            key = raw_key.strip().casefold()
            value = raw_value.strip()
        else:
            key = option.casefold()
            value = ""

        key = EMOTION_NAME_ALIASES.get(key, key)
        if key in EMOTION_NAMES:
            amount = 0.8 if value == "" else float(value)
            if not 0.0 <= amount <= 1.0:
                raise ValueError(f"Dialogue emotion {key!r} must be between 0 and 1")
            vector_values[key] = amount
            has_vector = True
        elif key in {"strength", "alpha", "强度"}:
            alpha = float(value)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError("Dialogue emotion strength must be between 0 and 1")
        elif key in {"auto", "自动", "自动情绪"}:
            use_text = True
            emotion_text = None
        elif key in {"emo_text", "emotion_text", "情绪文本", "情感文本"}:
            if not value:
                raise ValueError("Dialogue emo_text cannot be empty")
            use_text = True
            emotion_text = value
        elif key in {"random", "随机"}:
            random_sampling = True if value == "" else _parse_bool(value, option)
        elif key in {"bias", "official_bias", "官方偏置"}:
            apply_bias = True if value == "" else _parse_bool(value, option)
        elif key in {"natural", "none", "自然", "无情绪"}:
            if len(options) != 1:
                raise ValueError(f"Dialogue option {option!r} cannot be combined with other emotion options")
            # Keep an explicit empty control so Multi-Talk does not fall back to
            # a shared Emotion Control for this segment.
            return EmotionControl(vector=None)
        else:
            raise ValueError(f"Unknown dialogue emotion option: {option!r}")

    if has_vector and use_text:
        raise ValueError("Dialogue emotion vector options cannot be combined with auto/emo_text")
    if use_text:
        return EmotionControl(
            vector=None,
            alpha=alpha,
            text=emotion_text,
            use_text=True,
            random_sampling=random_sampling,
        )
    if has_vector:
        vector = normalize_emotion_vector(
            tuple(vector_values[name] for name in EMOTION_NAMES),
            apply_bias=apply_bias,
        )
        return EmotionControl(vector=vector, alpha=alpha, random_sampling=random_sampling)
    raise ValueError("Dialogue emotion tag did not specify an emotion")


def parse_dialogue_segments(text: str) -> list[DialogueSegment]:
    raw = (text or "").strip()
    if not raw:
        return []
    matches = list(DIALOGUE_TAG.finditer(raw))
    if not matches:
        return [DialogueSegment(speaker="Narrator", text=raw)]
    segments: list[DialogueSegment] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        if not content:
            continue
        header = [part.strip() for part in match.group(1).split("|")]
        speaker = header[0]
        if not speaker:
            raise ValueError("Dialogue speaker name cannot be empty")
        segments.append(
            DialogueSegment(
                speaker=speaker,
                text=content,
                emotion=_dialogue_emotion(header[1:]),
            )
        )
    return segments


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    return [(segment.speaker, segment.text) for segment in parse_dialogue_segments(text)]


def cache_diagnostics() -> list[dict[str, Any]]:
    with MODEL_CACHE_LOCK:
        return [
            {
                "model_dir": str(handle.model_dir),
                "source_dir": str(handle.source_dir),
                "device": handle.device,
                "bf16": handle.use_bf16,
                "qwen_emotion": handle.use_qwen_emo,
                "loaded": handle.runtime is not None,
            }
            for handle in MODEL_CACHE.values()
        ]


def runtime_diagnostics() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR"),
        "models": cache_diagnostics(),
    }


def diagnostics_json() -> str:
    return json.dumps(runtime_diagnostics(), ensure_ascii=False, indent=2)
