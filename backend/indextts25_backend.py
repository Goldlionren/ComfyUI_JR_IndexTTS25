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
from typing import Any

import numpy as np
import soundfile as sf
import torch


MODEL_FILES = (
    "config.yaml",
    "gpt.pth",
    "s2mel.pth",
    "codec.pth",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "wav2vec2bert_stats.pt",
)
MODEL_CACHE: dict[tuple[str, str, bool, bool, str], "IndexTTS25Handle"] = {}
MODEL_CACHE_LOCK = threading.RLock()
DIALOGUE_TAG = re.compile(r"\[([^\]]+)\]\s*[:：]\s*")


@dataclass(frozen=True)
class VoicePreset:
    name: str
    audio: dict[str, Any]


@dataclass(frozen=True)
class EmotionControl:
    vector: tuple[float, ...] | None
    audio: dict[str, Any] | None = None
    alpha: float = 1.0


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
) -> IndexTTS25Handle:
    assert_runtime_compatible(strict=strict_environment)
    model_dir = resolve_model_dir(model_path_override)
    source_dir = resolve_source_dir(source_path_override)
    use_bf16 = precision.lower() == "bf16"
    if precision.lower() not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but the selected CUDA runtime does not support it")
    key = (str(model_dir), device, use_bf16, bool(enable_qwen_emotion), str(source_dir))
    with MODEL_CACHE_LOCK:
        cached = MODEL_CACHE.get(key)
        if cached is not None:
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
) -> dict[str, Any]:
    if handle.runtime is None:
        raise RuntimeError("This IndexTTS-2.5 model handle has been unloaded; run Loader again")
    if not text.strip():
        raise ValueError("Text is empty")
    if not 0.5 <= float(duration_factor) <= 2.0:
        raise ValueError("duration_factor must be between 0.5 and 2.0")
    speaker_path = _materialize_audio(voice.audio, "speaker")
    emotion_path = _materialize_audio(emotion.audio, "emotion") if emotion and emotion.audio else None
    emotion_vector = list(emotion.vector) if emotion and emotion.vector is not None else None
    emotion_alpha = float(emotion.alpha) if emotion else 1.0
    torch.manual_seed(int(seed) & 0x7FFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) & 0x7FFFFFFF)
    with handle.lock, torch.inference_mode():
        result = handle.runtime.infer(
            spk_audio_prompt=str(speaker_path),
            text=text,
            output_path=None,
            lang=language,
            emo_audio_prompt=str(emotion_path) if emotion_path else None,
            emo_alpha=emotion_alpha,
            emo_vector=emotion_vector,
            use_emo_text=False,
            interval_silence=int(interval_silence_ms),
            duration_factor=float(duration_factor),
            text_normalization=bool(text_normalization),
            verbose=False,
        )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(f"Unexpected IndexTTS-2.5 result: {type(result).__name__}")
    sample_rate, waveform = result
    return _numpy_to_audio(waveform, int(sample_rate))


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    raw = (text or "").strip()
    if not raw:
        return []
    matches = list(DIALOGUE_TAG.finditer(raw))
    if not matches:
        return [("Narrator", raw)]
    segments: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()
        if content:
            segments.append((match.group(1).strip(), content))
    return segments


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
