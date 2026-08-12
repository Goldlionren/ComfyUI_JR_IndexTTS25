from __future__ import annotations

import json
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
    parse_dialogue,
    runtime_diagnostics,
)


CATEGORY = "JR/Audio/IndexTTS 2.5"
MODEL_TYPE = "JR_INDEXTTS25_MODEL"
VOICE_TYPE = "JR_INDEXTTS25_VOICE"
EMOTION_TYPE = "JR_INDEXTTS25_EMOTION"
LANGUAGES = ["ZH", "EN", "JA", "ES", "DE", "FR", "KO", "RU"]


class JR_IndexTTS25_Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path_override": ("STRING", {"default": ""}),
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

    def load(self, model_path_override, source_path_override, device, precision, enable_qwen_emotion, strict_environment):
        return (
            load_model(
                model_path_override=model_path_override,
                source_path_override=source_path_override,
                device=device,
                precision=precision,
                enable_qwen_emotion=enable_qwen_emotion,
                strict_environment=strict_environment,
            ),
        )


class JR_IndexTTS25_VoicePreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_audio": ("AUDIO",),
                "speaker_name": ("STRING", {"default": "Narrator"}),
            }
        }

    RETURN_TYPES = (VOICE_TYPE,)
    RETURN_NAMES = ("voice",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, reference_audio, speaker_name):
        name = (speaker_name or "Narrator").strip() or "Narrator"
        return (VoicePreset(name=name, audio=reference_audio),)


class JR_IndexTTS25_EmotionControl:
    @classmethod
    def INPUT_TYPES(cls):
        sliders = {
            name: ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            for name in ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")
        }
        return {
            "required": {
                "mode": (["emotion_vector", "reference_audio"], {"default": "emotion_vector"}),
                **sliders,
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "apply_official_bias": ("BOOLEAN", {"default": True}),
            },
            "optional": {"emotion_reference_audio": ("AUDIO",)},
        }

    RETURN_TYPES = (EMOTION_TYPE,)
    RETURN_NAMES = ("emotion",)
    FUNCTION = "create"
    CATEGORY = CATEGORY

    def create(self, mode, happy, angry, sad, afraid, disgusted, melancholic, surprised, calm, strength, apply_official_bias, emotion_reference_audio=None):
        if mode == "reference_audio":
            if emotion_reference_audio is None:
                raise ValueError("reference_audio mode requires emotion_reference_audio")
            return (EmotionControl(vector=None, audio=emotion_reference_audio, alpha=float(strength)),)
        vector = normalize_emotion_vector(
            (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm),
            apply_bias=bool(apply_official_bias),
        )
        return (EmotionControl(vector=vector, audio=None, alpha=float(strength)),)


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
            },
            "optional": {"emotion": (EMOTION_TYPE,)},
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model, voice, text, language, duration_factor, text_normalization, interval_silence_ms, seed, unload_model_after, emotion=None):
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
            },
            "optional": optional,
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(self, model, dialogue, language, gap_ms, duration_factor, text_normalization, seed, unload_model_after, emotion=None, **kwargs):
        voices = [kwargs.get(f"voice_{index}") for index in range(1, 11)]
        voices = [voice for voice in voices if voice is not None]
        if not voices:
            raise ValueError("Multi-Talk requires at least one voice input")
        voice_map = {voice.name.casefold(): voice for voice in voices}
        default_voice = voices[0]
        segments = parse_dialogue(dialogue)
        if not segments:
            raise ValueError("No dialogue segments found; use [Speaker]: text")
        waveforms = []
        sample_rate = None
        try:
            for index, (speaker, text) in enumerate(segments):
                voice = voice_map.get(speaker.casefold(), default_voice)
                audio = generate_audio(
                    model,
                    voice,
                    text,
                    language,
                    emotion=emotion,
                    duration_factor=duration_factor,
                    text_normalization=text_normalization,
                    interval_silence_ms=0,
                    seed=int(seed) + index,
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
    "JR_IndexTTS25_EmotionControl": JR_IndexTTS25_EmotionControl,
    "JR_IndexTTS25_Generate": JR_IndexTTS25_Generate,
    "JR_IndexTTS25_MultiTalkGenerate": JR_IndexTTS25_MultiTalkGenerate,
    "JR_IndexTTS25_RuntimeDiagnostics": JR_IndexTTS25_RuntimeDiagnostics,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JR_IndexTTS25_Loader": "JR IndexTTS 2.5 Loader",
    "JR_IndexTTS25_VoicePreset": "JR IndexTTS 2.5 Voice Preset",
    "JR_IndexTTS25_EmotionControl": "JR IndexTTS 2.5 Emotion Control",
    "JR_IndexTTS25_Generate": "JR IndexTTS 2.5 Generate",
    "JR_IndexTTS25_MultiTalkGenerate": "JR IndexTTS 2.5 Multi-Talk Generate",
    "JR_IndexTTS25_RuntimeDiagnostics": "JR IndexTTS 2.5 Runtime Diagnostics",
}
