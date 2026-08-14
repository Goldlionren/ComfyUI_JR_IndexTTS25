from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from packaging.requirements import Requirement

from ComfyUI_JR_IndexTTS25.backend import indextts25_backend as backend


def test_required_runtime_dependencies_are_declared():
    plugin_root = Path(__file__).resolve().parents[2]
    requirement_lines = (plugin_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    assert "munch>=4.0.0,<5" in requirement_lines
    assert "matplotlib>=3.10.8,<4" in requirement_lines
    requirements = [
        Requirement(line)
        for line in requirement_lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(
        specifier.operator == "=="
        for requirement in requirements
        for specifier in requirement.specifier
    )


def test_audioop_lts_is_only_selected_when_stdlib_audioop_is_removed():
    plugin_root = Path(__file__).resolve().parents[2]
    requirement_lines = (plugin_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    audioop_requirement = Requirement(
        next(line for line in requirement_lines if line.startswith("audioop-lts"))
    )
    assert audioop_requirement.marker is not None
    assert audioop_requirement.marker.evaluate({"python_version": "3.12"}) is False
    assert audioop_requirement.marker.evaluate({"python_version": "3.13"}) is True


def test_explicit_xpu_device_routes_to_xpu_without_hardware():
    assert backend._device_backend("xpu:0") == "xpu"
    assert backend._device_index("xpu:1") == 1
    assert backend._device_index("xpu") == 0


def test_xpu_cache_seed_and_bf16_use_xpu_api(monkeypatch):
    calls: list[tuple[str, int | None]] = []
    fake_xpu = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append(("empty_cache", None)),
        manual_seed_all=lambda seed: calls.append(("manual_seed_all", seed)),
        is_bf16_supported=lambda: True,
    )
    monkeypatch.setattr(backend.torch, "xpu", fake_xpu)

    backend._empty_accelerator_cache("xpu:0")
    backend._seed_accelerator("xpu:0", 250)

    assert backend._bf16_supported("xpu:0") is True
    assert calls == [("empty_cache", None), ("manual_seed_all", 250)]
