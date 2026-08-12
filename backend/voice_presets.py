from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


PRESET_SCHEMA_VERSION = 1
PRESET_DIR_ENV = "INDEXTTS25_PRESET_DIR"
PRESET_LOCK = threading.RLock()
CHOICE_SEPARATOR = " :: "
OVERRIDE_CHOICE = "<select by ID or name override>"


@dataclass(frozen=True)
class VoicePresetRecord:
    id: str
    name: str
    audio_file: str
    sample_rate: int
    samples: int
    duration_seconds: float
    audio_sha256: str
    created_at: str
    updated_at: str
    directory: Path

    @property
    def audio_path(self) -> Path:
        return self.directory / self.audio_file

    @property
    def choice(self) -> str:
        return f"{self.name}{CHOICE_SEPARATOR}{self.id}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("directory", None)
        payload["audio_path"] = str(self.audio_path)
        return payload


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def voice_preset_library_dir(create: bool = False) -> Path:
    configured = os.environ.get(PRESET_DIR_ENV, "").strip()
    if configured:
        root = Path(configured).expanduser().resolve()
    else:
        try:
            import folder_paths

            root = Path(folder_paths.models_dir).resolve() / "indextts" / "voice_presets"
        except Exception:
            root = _plugin_root() / "voice_presets"
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def safe_preset_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', "_", (name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._")
    if not cleaned:
        raise ValueError("Voice preset name cannot be empty")
    return cleaned


def _record_from_directory(directory: Path) -> VoicePresetRecord | None:
    metadata_path = directory / "preset.json"
    if not metadata_path.is_file():
        return None
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != PRESET_SCHEMA_VERSION:
            return None
        audio_file = str(payload["audio_file"])
        if Path(audio_file).name != audio_file:
            return None
        record = VoicePresetRecord(
            id=str(payload["id"]),
            name=str(payload["name"]),
            audio_file=audio_file,
            sample_rate=int(payload["sample_rate"]),
            samples=int(payload["samples"]),
            duration_seconds=float(payload["duration_seconds"]),
            audio_sha256=str(payload["audio_sha256"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            directory=directory.resolve(),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None
    if not record.id.startswith("vp_") or not record.name.strip() or not record.audio_path.is_file():
        return None
    return record


def list_voice_presets() -> list[VoicePresetRecord]:
    root = voice_preset_library_dir(create=False)
    if not root.is_dir():
        return []
    with PRESET_LOCK:
        records = [record for path in root.iterdir() if path.is_dir() if (record := _record_from_directory(path))]
    return sorted(records, key=lambda record: (record.name.casefold(), record.id))


def voice_preset_choices() -> list[str]:
    records = list_voice_presets()
    return [OVERRIDE_CHOICE, *(record.choice for record in records)]


def voice_preset_library_fingerprint() -> str:
    digest = hashlib.sha256()
    for record in list_voice_presets():
        digest.update(record.id.encode("utf-8"))
        digest.update(record.name.encode("utf-8"))
        digest.update(record.updated_at.encode("utf-8"))
    return digest.hexdigest()


def resolve_voice_preset(reference: str) -> VoicePresetRecord:
    value = (reference or "").strip()
    if CHOICE_SEPARATOR in value:
        value = value.rsplit(CHOICE_SEPARATOR, 1)[1].strip()
    for record in list_voice_presets():
        if value == record.id or value.casefold() == record.name.casefold():
            return record
    raise KeyError(f"Voice preset not found: {reference!r}")


def _pcm16_and_hash(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, str]:
    if int(sample_rate) <= 0:
        raise ValueError("Voice preset sample rate must be positive")
    array = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("Voice preset audio is empty or contains NaN/Inf")
    pcm16 = np.rint(np.clip(array, -1.0, 1.0) * 32767.0).astype("<i2")
    digest = hashlib.sha256()
    digest.update(str(int(sample_rate)).encode("ascii"))
    digest.update(pcm16.tobytes())
    return pcm16, digest.hexdigest()


def save_voice_preset(
    name: str,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    overwrite: bool = False,
) -> VoicePresetRecord:
    display_name = (name or "").strip()
    directory_name = safe_preset_name(display_name)
    pcm16, audio_hash = _pcm16_and_hash(waveform, sample_rate)
    now = datetime.now(timezone.utc).isoformat()
    root = voice_preset_library_dir(create=True)

    with PRESET_LOCK:
        existing = next(
            (record for record in list_voice_presets() if record.name.casefold() == display_name.casefold()),
            None,
        )
        if existing is not None and existing.audio_sha256 == audio_hash:
            load_voice_preset_audio(existing.id)
            return existing
        if existing is not None and not overwrite:
            raise FileExistsError(
                f"Voice preset {display_name!r} already exists with different audio. "
                "Enable overwrite_existing to replace it."
            )

        if existing is None:
            directory = root / directory_name
            if directory.exists():
                raise FileExistsError(f"Voice preset directory already exists: {directory}")
            directory.mkdir(parents=False)
            preset_id = f"vp_{uuid.uuid4().hex}"
            created_at = now
        else:
            directory = existing.directory
            preset_id = existing.id
            created_at = existing.created_at

        audio_file = "prompt.wav"
        audio_path = directory / audio_file
        audio_temp = directory / ".prompt.wav.tmp"
        metadata_path = directory / "preset.json"
        metadata_temp = directory / ".preset.json.tmp"
        payload = {
            "schema_version": PRESET_SCHEMA_VERSION,
            "id": preset_id,
            "name": display_name,
            "audio_file": audio_file,
            "sample_rate": int(sample_rate),
            "samples": int(pcm16.size),
            "duration_seconds": float(pcm16.size / int(sample_rate)),
            "audio_sha256": audio_hash,
            "created_at": created_at,
            "updated_at": now,
        }
        try:
            sf.write(audio_temp, pcm16, int(sample_rate), subtype="PCM_16", format="WAV")
            metadata_temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(audio_temp, audio_path)
            os.replace(metadata_temp, metadata_path)
        finally:
            audio_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)

        record = _record_from_directory(directory)
        if record is None:
            raise RuntimeError(f"Voice preset was written but could not be validated: {directory}")
        return record


def load_voice_preset_audio(reference: str) -> tuple[VoicePresetRecord, dict[str, Any]]:
    record = resolve_voice_preset(reference)
    waveform, sample_rate = sf.read(record.audio_path, dtype="int16", always_2d=True)
    if waveform.shape[1] != 1:
        raise RuntimeError(f"Stored voice preset must be mono: {record.audio_path}")
    pcm16 = np.asarray(waveform[:, 0], dtype="<i2")
    digest = hashlib.sha256()
    digest.update(str(int(sample_rate)).encode("ascii"))
    digest.update(pcm16.tobytes())
    if (
        int(sample_rate) != record.sample_rate
        or int(pcm16.size) != record.samples
        or digest.hexdigest() != record.audio_sha256
    ):
        raise RuntimeError(f"Stored voice preset audio failed integrity validation: {record.audio_path}")
    mono = pcm16.astype(np.float32) / 32768.0
    audio = {
        "waveform": torch.from_numpy(mono.copy()).reshape(1, 1, -1),
        "sample_rate": int(sample_rate),
    }
    return record, audio


def rename_voice_preset(reference: str, new_name: str) -> VoicePresetRecord:
    display_name = (new_name or "").strip()
    new_directory_name = safe_preset_name(display_name)
    with PRESET_LOCK:
        record = resolve_voice_preset(reference)
        for other in list_voice_presets():
            if other.id != record.id and other.name.casefold() == display_name.casefold():
                raise FileExistsError(f"Voice preset name already exists: {display_name!r}")
        new_directory = voice_preset_library_dir(create=True) / new_directory_name
        if new_directory != record.directory:
            if new_directory.exists():
                raise FileExistsError(f"Voice preset directory already exists: {new_directory}")
            record.directory.rename(new_directory)
        metadata_path = new_directory / "preset.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["name"] = display_name
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        metadata_temp = new_directory / ".preset.json.tmp"
        try:
            metadata_temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(metadata_temp, metadata_path)
        finally:
            metadata_temp.unlink(missing_ok=True)
        renamed = _record_from_directory(new_directory)
        if renamed is None:
            raise RuntimeError(f"Renamed voice preset could not be validated: {new_directory}")
        return renamed


def delete_voice_preset(reference: str) -> VoicePresetRecord:
    with PRESET_LOCK:
        record = resolve_voice_preset(reference)
        shutil.rmtree(record.directory)
        return record
