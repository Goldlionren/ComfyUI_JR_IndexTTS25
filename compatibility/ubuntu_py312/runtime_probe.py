from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Ubuntu/Python 3.12 IndexTTS-2.5 evidence probe")
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        help="Candidate plugin directory (defaults to ComfyUI/custom_nodes/ComfyUI_JR_IndexTTS25)",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--real-inference", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ubuntu_py312_report.json"
    report: dict[str, object] = {
        "status": "FAIL",
        "candidate": "Ubuntu x86-64 + Python 3.12.x + torch/torchaudio 2.11.0+cu130",
        "python": platform.python_version(),
        "executable": sys.executable,
        "operating_system": platform.system(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ffmpeg": shutil.which("ffmpeg"),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "torchaudio",
                "torchvision",
                "transformers",
                "tokenizers",
                "numpy",
                "numba",
                "llvmlite",
                "librosa",
                "munch",
                "soundfile",
                "WeTextProcessing",
                "wetext",
                "audioop-lts",
            )
        },
    }

    try:
        comfyui_root = args.comfyui_root.expanduser().resolve()
        custom_nodes = comfyui_root / "custom_nodes"
        plugin_dir = (
            args.plugin_dir.expanduser().resolve()
            if args.plugin_dir is not None
            else custom_nodes / "ComfyUI_JR_IndexTTS25"
        )
        if not (plugin_dir / "__init__.py").is_file():
            raise FileNotFoundError(f"Candidate plugin not found: {plugin_dir}")
        sys.path.insert(0, str(comfyui_root))
        sys.path.insert(0, str(plugin_dir.parent))

        import numpy as np
        import soundfile as sf
        import torch
        import ComfyUI_JR_IndexTTS25 as plugin
        from ComfyUI_JR_IndexTTS25.backend.indextts25_backend import (
            VoicePreset,
            assert_runtime_compatible,
            clear_model_cache,
            generate_audio,
            load_model,
        )

        report["torch_cuda"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        report["registered_node_count"] = len(plugin.NODE_CLASS_MAPPINGS)
        report["registered_nodes"] = sorted(plugin.NODE_CLASS_MAPPINGS)
        report["runtime_validation"] = assert_runtime_compatible(strict=True)
        if len(plugin.NODE_CLASS_MAPPINGS) != 10:
            raise RuntimeError(f"Expected 10 registered nodes, found {len(plugin.NODE_CLASS_MAPPINGS)}")
        report["plugin_import"] = "PASS"

        if args.real_inference:
            if args.model_dir is None or args.reference_audio is None:
                raise ValueError("--real-inference requires --model-dir and --reference-audio")
            model_dir = args.model_dir.expanduser().resolve()
            reference_path = args.reference_audio.expanduser().resolve()
            if not reference_path.is_file():
                raise FileNotFoundError(f"Reference audio not found: {reference_path}")
            waveform, sample_rate = sf.read(reference_path, dtype="float32", always_2d=True)
            audio = {
                "waveform": torch.from_numpy(waveform.T.copy()).unsqueeze(0),
                "sample_rate": int(sample_rate),
            }
            voice = VoicePreset(name="UbuntuProbe", audio=audio)
            started = time.perf_counter()
            handle = load_model(
                model_path_override=str(model_dir),
                device="cuda:0",
                precision="fp32",
                enable_qwen_emotion=False,
                strict_environment=True,
            )
            report["model_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            generated = generate_audio(
                handle,
                voice,
                "你好，这是 Ubuntu 和 Python 三点十二的真实语音测试。",
                "ZH",
                seed=312,
                do_sample=True,
            )
            report["inference_seconds"] = time.perf_counter() - started
            generated_waveform = generated["waveform"].detach().cpu().float().numpy()
            generated_rate = int(generated["sample_rate"])
            if generated_waveform.shape[:2] != (1, 1):
                raise RuntimeError(f"Unexpected AUDIO shape: {generated_waveform.shape}")
            if not generated_waveform.size or not np.isfinite(generated_waveform).all():
                raise RuntimeError("Generated audio is empty or non-finite")
            if not np.any(generated_waveform != 0):
                raise RuntimeError("Generated audio is silent")
            output_wav = output_dir / "ubuntu_py312_output.wav"
            sf.write(output_wav, generated_waveform[0, 0], generated_rate, subtype="PCM_16")
            report["real_inference"] = "PASS"
            report["output_wav"] = str(output_wav)
            report["output_sample_rate"] = generated_rate
            report["output_samples"] = int(generated_waveform.shape[-1])
            clear_model_cache(handle)
        else:
            report["real_inference"] = "SKIPPED"

        report["status"] = "PASS"
    except Exception as error:
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"REPORT={report_path}")

    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
