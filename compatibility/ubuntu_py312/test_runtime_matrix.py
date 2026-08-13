from __future__ import annotations

from ComfyUI_JR_IndexTTS25.backend.indextts25_backend import runtime_compatibility_errors


def errors(**overrides):
    values = {
        "python_version": (3, 13, 11),
        "operating_system": "Windows",
        "machine": "AMD64",
        "torch_version": "2.11.0+cu130",
        "torchaudio_version": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "cuda_available": True,
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


def test_wrong_core_stack_is_rejected():
    result = errors(
        python_version=(3, 11, 9),
        torch_version="2.10.0+cu128",
        torchaudio_version="2.10.0+cu128",
        torch_cuda="12.8",
        cuda_available=False,
    )
    assert len(result) == 5
