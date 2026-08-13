from __future__ import annotations

from ComfyUI_JR_IndexTTS25.backend.indextts25_backend import runtime_compatibility_errors


def errors(**overrides):
    values = {
        "python_version": (3, 13, 11),
        "operating_system": "Windows",
        "machine": "AMD64",
        "torch_version": "2.11.0+cu130",
        "torchaudio_version": "2.11.0+cu130",
        "accelerator_backend": "cuda",
        "accelerator_runtime": "13.0",
        "accelerator_available": True,
    }
    values.update(overrides)
    return runtime_compatibility_errors(**values)


def test_existing_windows_python313_baseline_remains_supported():
    assert errors() == []


def test_windows_python312_candidate_is_supported():
    assert errors(python_version=(3, 12, 12)) == []


def test_ubuntu_python312_candidate_is_supported():
    assert errors(
        python_version=(3, 12, 12), operating_system="Linux", machine="x86_64"
    ) == []


def test_ubuntu_python313_xpu_candidate_is_supported():
    assert errors(
        operating_system="Linux",
        machine="x86_64",
        torch_version="2.11.0+xpu",
        torchaudio_version="2.11.0+xpu",
        accelerator_backend="xpu",
        accelerator_runtime=None,
    ) == []


def test_wrong_core_stack_is_rejected():
    result = errors(
        python_version=(3, 11, 9),
        torch_version="2.10.0+cu128",
        torchaudio_version="2.10.0+cu128",
        accelerator_runtime="12.8",
        accelerator_available=False,
    )
    assert len(result) == 5


def test_xpu_wheel_mismatch_is_rejected():
    result = errors(
        operating_system="Linux",
        machine="x86_64",
        torch_version="2.11.0+cu130",
        torchaudio_version="2.11.0+cu130",
        accelerator_backend="xpu",
        accelerator_runtime=None,
    )
    assert len(result) == 2


def test_windows_xpu_is_not_claimed_by_the_ubuntu_candidate():
    result = errors(
        torch_version="2.11.0+xpu",
        torchaudio_version="2.11.0+xpu",
        accelerator_backend="xpu",
        accelerator_runtime=None,
    )
    assert result == ["Intel XPU candidate requires Linux, found Windows"]
