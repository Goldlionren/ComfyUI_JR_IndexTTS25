from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import torch

from .backend.indextts25_backend import (
    EmotionControl,
    VoicePreset,
    assert_runtime_compatible,
    clear_model_cache,
    generate_audio,
    load_model,
    normalize_emotion_vector,
    parse_dialogue_segments,
    runtime_diagnostics,
)
from .backend.openai_compatible import (
    DEFAULT_OPENAI_API_URL,
    DEFAULT_OPENAI_MODEL,
    NOVEL_EMOTION_MODES,
    convert_novel_to_dialogue,
    enhance_pronunciation_text,
)
from .backend.voice_presets import (
    delete_voice_preset,
    list_voice_presets,
    load_voice_preset_audio,
    rename_voice_preset,
    resolve_voice_preset,
    save_voice_preset,
    voice_preset_choices,
    voice_preset_library_dir,
    voice_preset_library_fingerprint,
)


CATEGORY = "JR/Audio/IndexTTS 2.5"
MODEL_TYPE = "JR_INDEXTTS25_MODEL"
VOICE_TYPE = "JR_INDEXTTS25_VOICE"
EMOTION_TYPE = "JR_INDEXTTS25_EMOTION"
LANGUAGES = ["ZH", "EN", "JA", "AR", "ES"]


def _new_progress_bar():
    try:
        import comfy.utils

        return comfy.utils.ProgressBar(1000)
    except (ImportError, AttributeError):
        return None


def _progress_callback(progress_bar, start: float = 0.0, span: float = 1.0):
    if progress_bar is None:
        return None

    def report(value: float, _description: str = ""):
        absolute = max(0.0, min(1.0, start + span * float(value)))
        progress_bar.update_absolute(round(absolute * 1000), 1000)

    return report


def _advanced_generation_inputs():
    return {
        "do_sample": ("BOOLEAN", {"default": True}),
        "temperature": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.05}),
        "top_p": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
        "top_k": ("INT", {"default": 30, "min": 0, "max": 100, "step": 1}),
        "num_beams": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
        "repetition_penalty": (
            "FLOAT",
            {"default": 10.0, "min": 0.1, "max": 20.0, "step": 0.1},
        ),
        "length_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
        "max_mel_tokens": ("INT", {"default": 1500, "min": 50, "max": 1815, "step": 10}),
        "max_text_tokens_per_segment": (
            "INT",
            {"default": 120, "min": 20, "max": 600, "step": 2},
        ),
    }


def _inherit_text_emotion_config(segment_emotion, shared_emotion):
    if segment_emotion is None or not segment_emotion.use_text or shared_emotion is None:
        return segment_emotion
    return replace(
        segment_emotion,
        text_backend=shared_emotion.text_backend,
        openai_api_url=shared_emotion.openai_api_url,
        openai_api_key=shared_emotion.openai_api_key,
        openai_model=shared_emotion.openai_model,
        llm_timeout_seconds=shared_emotion.llm_timeout_seconds,
    )


class JR_IndexTTS25_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path_override": ("STRING", {"default": ""}),
                "download_model": ("BOOLEAN", {"default": False}),
                "source_path_override": ("STRING", {"default": ""}),
                "device": ("STRING", {"default": "cuda:0"}),
                "precision": (["fp32", "bf16"], {"default": "fp32"}),
                "enable_qwen_emotion": ("BOOLEAN", {"default": False}),
                "strict_environment": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (MODEL_TYPE,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        model_path_override,
        download_model,
        source_path_override,
        device,
        precision,
        enable_qwen_emotion,
        strict_environment,
    ):
        progress_bar = _new_progress_bar()
        return (
            load_model(
                model_path_override=model_path_override,
                download_model=download_model,
                source_path_override=source_path_override,
                device=device,
                precision=precision,
                enable_qwen_emotion=enable_qwen_emotion,
                strict_environment=strict_environment,
                progress_callback=_progress_callback(progress_bar),
            ),
        )


class JR_IndexTTS25_VoicePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "speaker_name": ("STRING", {"default": "Narrator"}),
            },
            "optional": {
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def create(self, reference_audio, speaker_name, overwrite_existing=False):
        name = (speaker_name or "Narrator").strip() or "Narrator"
        from .backend.indextts25_backend import audio_to_mono_numpy

        waveform, sample_rate = audio_to_mono_numpy(reference_audio)
        record = save_voice_preset(
            name,
            waveform,
            sample_rate,
            overwrite=bool(overwrite_existing),
        )
        _, stored_audio = load_voice_preset_audio(record.id)
        return (VoicePreset(name=record.name, audio=stored_audio, preset_id=record.id),)


class JR_IndexTTS25_LoadVoicePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (voice_preset_choices(),),
                "preset_id_or_name_override": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return voice_preset_library_fingerprint()

    def load(self, preset, preset_id_or_name_override):
        reference = (preset_id_or_name_override or "").strip() or preset
        record, audio = load_voice_preset_audio(reference)
        return (VoicePreset(name=record.name, audio=audio, preset_id=record.id),)


class JR_IndexTTS25_VoicePresetManager:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "action": (["list", "inspect", "rename", "delete"], {"default": "list"}),
                "preset_id_or_name": ("STRING", {"default": ""}),
                "new_name": ("STRING", {"default": ""}),
                "confirm_delete": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("presets_json",)
    FUNCTION = "manage"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def manage(self, action, preset_id_or_name, new_name, confirm_delete):
        reference = (preset_id_or_name or "").strip()
        result: dict[str, Any] = {
            "action": action,
            "library_dir": str(voice_preset_library_dir(create=False)),
        }
        if action == "inspect":
            if not reference:
                raise ValueError("inspect requires preset_id_or_name")
            result["preset"] = resolve_voice_preset(reference).to_dict()
        elif action == "rename":
            if not reference or not (new_name or "").strip():
                raise ValueError("rename requires preset_id_or_name and new_name")
            result["preset"] = rename_voice_preset(reference, new_name).to_dict()
        elif action == "delete":
            if not reference:
                raise ValueError("delete requires preset_id_or_name")
            if not confirm_delete:
                raise ValueError("delete requires confirm_delete=True")
            result["deleted"] = delete_voice_preset(reference).to_dict()
        result["presets"] = [record.to_dict() for record in list_voice_presets()]
        result["count"] = len(result["presets"])
        return (json.dumps(result, ensure_ascii=False, indent=2),)


class JR_IndexTTS25_EmotionControl:
    @classmethod
    def INPUT_TYPES(cls):
        sliders = {
            name: ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            for name in ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
        }
        return {
            "required": {
                "mode": (
                    ["emotion_vector", "reference_audio", "auto_from_text", "emotion_text"],
                    {"default": "emotion_vector"},
                ),
                **sliders,
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_official_bias": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "emotion_reference_audio": ("AUDIO",),
                "emotion_text": ("STRING", {"multiline": True, "default": ""}),
                "random_sampling": ("BOOLEAN", {"default": False}),
                "text_emotion_backend": (
                    ["llama.cpp_openai_api", "builtin_qwen"],
                    {"default": "llama.cpp_openai_api"},
                ),
                "openai_api_url": ("STRING", {"default": DEFAULT_OPENAI_API_URL}),
                "openai_api_key": ("STRING", {"default": ""}),
                "openai_model": ("STRING", {"default": DEFAULT_OPENAI_MODEL}),
                "llm_timeout_seconds": ("INT", {"default": 120, "min": 1, "max": 1800, "step": 1}),
            },
        }

    RETURN_TYPES = (EMOTION_TYPE,)
    RETURN_NAMES = ("emotion",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(
        self,
        mode,
        happy,
        angry,
        sad,
        afraid,
        disgusted,
        melancholic,
        surprised,
        calm,
        strength,
        apply_official_bias,
        emotion_reference_audio=None,
        emotion_text="",
        random_sampling=False,
        text_emotion_backend="llama.cpp_openai_api",
        openai_api_url=DEFAULT_OPENAI_API_URL,
        openai_api_key="",
        openai_model=DEFAULT_OPENAI_MODEL,
        llm_timeout_seconds=120,
    ):
        if mode == "reference_audio":
            if emotion_reference_audio is None:
                raise ValueError("reference_audio mode requires emotion_reference_audio")
            return (
                EmotionControl(
                    vector=None,
                    audio=emotion_reference_audio,
                    alpha=float(strength),
                    text_backend=text_emotion_backend,
                    openai_api_url=openai_api_url,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                    llm_timeout_seconds=int(llm_timeout_seconds),
                ),
            )
        if mode == "auto_from_text":
            return (
                EmotionControl(
                    vector=None,
                    alpha=float(strength),
                    text=None,
                    use_text=True,
                    random_sampling=bool(random_sampling),
                    text_backend=text_emotion_backend,
                    openai_api_url=openai_api_url,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                    llm_timeout_seconds=int(llm_timeout_seconds),
                ),
            )
        if mode == "emotion_text":
            description = (emotion_text or "").strip()
            if not description:
                raise ValueError("emotion_text mode requires emotion_text")
            return (
                EmotionControl(
                    vector=None,
                    alpha=float(strength),
                    text=description,
                    use_text=True,
                    random_sampling=bool(random_sampling),
                    text_backend=text_emotion_backend,
                    openai_api_url=openai_api_url,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                    llm_timeout_seconds=int(llm_timeout_seconds),
                ),
            )
        vector = normalize_emotion_vector(
            (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm),
            apply_bias=bool(apply_official_bias),
        )
        return (
            EmotionControl(
                vector=vector,
                audio=None,
                alpha=float(strength),
                random_sampling=bool(random_sampling),
                text_backend=text_emotion_backend,
                openai_api_url=openai_api_url,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
                llm_timeout_seconds=int(llm_timeout_seconds),
            ),
        )


class JR_IndexTTS25_PronunciationEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "language": (["ZH", "EN", "JA"], {"default": "ZH"}),
                "openai_api_url": ("STRING", {"default": DEFAULT_OPENAI_API_URL}),
                "openai_api_key": ("STRING", {"default": ""}),
                "openai_model": ("STRING", {"default": DEFAULT_OPENAI_MODEL}),
                "llm_timeout_seconds": ("INT", {"default": 120, "min": 1, "max": 1800, "step": 1}),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "只标注确实存在歧义、且需要指定读音的词，不要改写原文。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_text",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def enhance(self, text, language, openai_api_url, openai_api_key, openai_model, llm_timeout_seconds, instruction):
        progress_bar = _new_progress_bar()
        callback = _progress_callback(progress_bar)
        if callback is not None:
            callback(0.0, "requesting pronunciation enhancement")
        result = enhance_pronunciation_text(
            text=text,
            language=language,
            api_url=openai_api_url,
            api_key=openai_api_key,
            model=openai_model,
            timeout_seconds=int(llm_timeout_seconds),
            instruction=instruction,
        )
        if callback is not None:
            callback(1.0, "complete")
        return (result,)


class JR_IndexTTS25_NovelToDialogue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "novel_text": ("STRING", {"multiline": True, "default": ""}),
                "narrator_name": ("STRING", {"default": "旁白"}),
                "known_speakers": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：紫灵, 林志远（名称应与 Voice Preset 一致）",
                    },
                ),
                "emotion_mode": (list(NOVEL_EMOTION_MODES), {"default": "llm_emotion_tags"}),
                "emotion_strength": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "chunk_size_chars": (
                    "INT",
                    {"default": 2000, "min": 200, "max": 10000, "step": 100},
                ),
                "strict_text_preservation": ("BOOLEAN", {"default": True}),
                "openai_api_url": ("STRING", {"default": DEFAULT_OPENAI_API_URL}),
                "openai_api_key": ("STRING", {"default": ""}),
                "openai_model": ("STRING", {"default": DEFAULT_OPENAI_MODEL}),
                "llm_temperature": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "llm_max_tokens": (
                    "INT",
                    {"default": 4096, "min": 256, "max": 32768, "step": 256},
                ),
                "llm_timeout_seconds": (
                    "INT",
                    {"default": 300, "min": 1, "max": 1800, "step": 1},
                ),
                "instruction": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "可选：补充角色别名或小说特殊写法，不要要求改写正文。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("dialogue", "speaker_list", "conversion_report")
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def convert(
        self,
        novel_text,
        narrator_name,
        known_speakers,
        emotion_mode,
        emotion_strength,
        chunk_size_chars,
        strict_text_preservation,
        openai_api_url,
        openai_api_key,
        openai_model,
        llm_temperature,
        llm_max_tokens,
        llm_timeout_seconds,
        instruction,
    ):
        progress_bar = _new_progress_bar()
        result = convert_novel_to_dialogue(
            novel_text,
            narrator_name=narrator_name,
            known_speakers=known_speakers,
            emotion_mode=emotion_mode,
            emotion_strength=float(emotion_strength),
            api_url=openai_api_url,
            api_key=openai_api_key,
            model=openai_model,
            temperature=float(llm_temperature),
            max_tokens=int(llm_max_tokens),
            timeout_seconds=int(llm_timeout_seconds),
            chunk_size_chars=int(chunk_size_chars),
            strict_text_preservation=bool(strict_text_preservation),
            instruction=instruction,
            progress_callback=_progress_callback(progress_bar),
        )
        report = {
            "status": "PASS" if not result.warnings else "PASS_WITH_WARNINGS",
            "chunk_count": result.chunk_count,
            "speaker_count": len(result.speakers),
            "speakers": list(result.speakers),
            "emotion_mode": emotion_mode,
            "strict_text_preservation": bool(strict_text_preservation),
            "warnings": list(result.warnings),
        }
        return (
            result.dialogue,
            "\n".join(result.speakers),
            json.dumps(report, ensure_ascii=False, indent=2),
        )


class JR_IndexTTS25_Generate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "voice": (VOICE_TYPE,),
                "text": ("STRING", {"multiline": True, "default": "你好，这是 IndexTTS 二点五。"}),
                "language": (LANGUAGES, {"default": "ZH"}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "text_normalization": ("BOOLEAN", {"default": True}),
                "interval_silence_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "unload_model_after": ("BOOLEAN", {"default": False}),
                **_advanced_generation_inputs(),
            },
            "optional": {"emotion": (EMOTION_TYPE,)},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        model,
        voice,
        text,
        language,
        duration_factor,
        text_normalization,
        interval_silence_ms,
        seed,
        unload_model_after,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1500,
        max_text_tokens_per_segment=120,
        emotion=None,
    ):
        progress_bar = _new_progress_bar()
        try:
            audio = generate_audio(
                model,
                voice,
                text,
                language,
                emotion=emotion,
                duration_factor=duration_factor,
                text_normalization=text_normalization,
                interval_silence_ms=interval_silence_ms,
                seed=seed,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                max_mel_tokens=max_mel_tokens,
                max_text_tokens_per_segment=max_text_tokens_per_segment,
                progress_callback=_progress_callback(progress_bar),
            )
            return (audio,)
        finally:
            if unload_model_after:
                clear_model_cache(model)


class JR_IndexTTS25_MultiTalkGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {"emotion": (EMOTION_TYPE,)}
        for index in range(1, 11):
            optional[f"voice_{index}"] = (VOICE_TYPE,)
        return {
            "required": {
                "model": (MODEL_TYPE,),
                "dialogue": ("STRING", {"multiline": True, "default": "[Narrator]:欢迎使用多角色语音。\n[Alice]:Hello from IndexTTS."}),
                "language": (LANGUAGES, {"default": "ZH"}),
                "gap_ms": ("INT", {"default": 200, "min": 0, "max": 5000, "step": 10}),
                "duration_factor": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.05}),
                "text_normalization": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "unload_model_after": ("BOOLEAN", {"default": False}),
                **_advanced_generation_inputs(),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        model,
        dialogue,
        language,
        gap_ms,
        duration_factor,
        text_normalization,
        seed,
        unload_model_after,
        do_sample=True,
        temperature=0.8,
        top_p=0.8,
        top_k=30,
        num_beams=3,
        repetition_penalty=10.0,
        length_penalty=0.0,
        max_mel_tokens=1500,
        max_text_tokens_per_segment=120,
        emotion=None,
        **kwargs,
    ):
        voices = [kwargs.get(f"voice_{index}") for index in range(1, 11)]
        voices = [voice for voice in voices if voice is not None]
        if not voices:
            raise ValueError("Multi-Talk requires at least one voice input")
        voice_map = {voice.name.casefold(): voice for voice in voices}
        default_voice = voices[0]
        segments = parse_dialogue_segments(dialogue)
        if not segments:
            raise ValueError("No dialogue segments found; use [Speaker]: text")
        progress_bar = _new_progress_bar()
        waveforms = []
        sample_rate = None
        try:
            for index, segment in enumerate(segments):
                voice = voice_map.get(segment.speaker.casefold(), default_voice)
                segment_emotion = _inherit_text_emotion_config(segment.emotion, emotion)
                audio = generate_audio(
                    model,
                    voice,
                    segment.text,
                    language,
                    emotion=segment_emotion if segment_emotion is not None else emotion,
                    duration_factor=duration_factor,
                    text_normalization=text_normalization,
                    interval_silence_ms=0,
                    seed=int(seed) + index,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    max_mel_tokens=max_mel_tokens,
                    max_text_tokens_per_segment=max_text_tokens_per_segment,
                    progress_callback=_progress_callback(
                        progress_bar,
                        start=index / len(segments),
                        span=1.0 / len(segments),
                    ),
                )
                current_rate = int(audio["sample_rate"])
                if sample_rate is None:
                    sample_rate = current_rate
                elif current_rate != sample_rate:
                    raise RuntimeError(f"Generated sample-rate mismatch: {current_rate} != {sample_rate}")
                waveforms.append(audio["waveform"])
            gap_frames = int((sample_rate or 22050) * max(0, int(gap_ms)) / 1000)
            gap = torch.zeros((1, 1, gap_frames), dtype=torch.float32)
            parts = []
            for index, waveform in enumerate(waveforms):
                parts.append(waveform)
                if index + 1 < len(waveforms) and gap_frames:
                    parts.append(gap)
            return ({"waveform": torch.cat(parts, dim=-1), "sample_rate": int(sample_rate or 22050)},)
        finally:
            if unload_model_after:
                clear_model_cache(model)


class JR_IndexTTS25_RuntimeDiagnostics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"action": (["report", "unload_model", "unload_all"], {"default": "report"})},
            "optional": {"model": (MODEL_TYPE,)},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("diagnostics",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, action, model=None):
        cleared = None
        if action == "unload_model":
            if model is None:
                raise ValueError("unload_model requires a model input")
            cleared = clear_model_cache(model)
        elif action == "unload_all":
            cleared = clear_model_cache()
        info = runtime_diagnostics()
        info["action"] = action
        if cleared is not None:
            info["cleared_models"] = cleared
        try:
            info["validated_environment"] = assert_runtime_compatible(strict=False)
        except Exception as error:
            info["diagnostic_error"] = f"{type(error).__name__}: {error}"
        return (json.dumps(info, ensure_ascii=False, indent=2),)


NODE_CLASS_MAPPINGS = {
    "JR_IndexTTS25_Loader": JR_IndexTTS25_Loader,
    "JR_IndexTTS25_VoicePreset": JR_IndexTTS25_VoicePreset,
    "JR_IndexTTS25_LoadVoicePreset": JR_IndexTTS25_LoadVoicePreset,
    "JR_IndexTTS25_VoicePresetManager": JR_IndexTTS25_VoicePresetManager,
    "JR_IndexTTS25_EmotionControl": JR_IndexTTS25_EmotionControl,
    "JR_IndexTTS25_PronunciationEnhance": JR_IndexTTS25_PronunciationEnhance,
    "JR_IndexTTS25_NovelToDialogue": JR_IndexTTS25_NovelToDialogue,
    "JR_IndexTTS25_Generate": JR_IndexTTS25_Generate,
    "JR_IndexTTS25_MultiTalkGenerate": JR_IndexTTS25_MultiTalkGenerate,
    "JR_IndexTTS25_RuntimeDiagnostics": JR_IndexTTS25_RuntimeDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_IndexTTS25_Loader": "JR IndexTTS 2.5 Loader",
    "JR_IndexTTS25_VoicePreset": "JR IndexTTS 2.5 Voice Preset",
    "JR_IndexTTS25_LoadVoicePreset": "JR IndexTTS 2.5 Load Voice Preset",
    "JR_IndexTTS25_VoicePresetManager": "JR IndexTTS 2.5 Voice Preset Manager",
    "JR_IndexTTS25_EmotionControl": "JR IndexTTS 2.5 Emotion Control",
    "JR_IndexTTS25_PronunciationEnhance": "JR IndexTTS 2.5 Pronunciation Enhance (LLM)",
    "JR_IndexTTS25_NovelToDialogue": "JR IndexTTS 2.5 Novel to Dialogue (LLM)",
    "JR_IndexTTS25_Generate": "JR IndexTTS 2.5 Generate",
    "JR_IndexTTS25_MultiTalkGenerate": "JR IndexTTS 2.5 Multi-Talk Generate",
    "JR_IndexTTS25_RuntimeDiagnostics": "JR IndexTTS 2.5 Runtime Diagnostics",
}
