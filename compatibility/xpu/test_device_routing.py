from __future__ import annotations

from types import SimpleNamespace

from ComfyUI_JR_IndexTTS25.backend import indextts25_backend as backend


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
