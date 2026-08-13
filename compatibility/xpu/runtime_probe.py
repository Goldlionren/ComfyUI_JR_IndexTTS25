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


def xpu_operator_preflight(torch, device: str) -> dict[str, object]:
    results: dict[str, object] = {}
    x = torch.randn(1, 8, 64, device=device)
    conv = torch.nn.Conv1d(8, 16, 3, padding=1).to(device)
    results["conv1d_shape"] = list(conv(x).shape)

    deconv = torch.nn.ConvTranspose1d(8, 4, 4, stride=2, padding=1).to(device)
    results["conv_transpose1d_shape"] = list(deconv(x).shape)

    q = torch.randn(1, 2, 32, 32, device=device)
    attention = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
    results["sdpa_shape"] = list(attention.shape)

    results["bf16_supported"] = bool(torch.xpu.is_bf16_supported())
    if results["bf16_supported"]:
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16):
            bf16_result = conv(x)
        results["bf16_autocast_dtype"] = str(bf16_result.dtype)
    else:
        results["bf16_autocast_dtype"] = "SKIPPED"

    torch.xpu.synchronize()
    results["status"] = "PASS"
    return results


def device_index(device: str) -> int:
    try:
        return max(0, int(device.rsplit(":", 1)[1])) if ":" in device else 0
    except ValueError:
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Intel XPU IndexTTS-2.5 evidence probe")
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        help="Candidate plugin directory (defaults to ComfyUI/custom_nodes/ComfyUI_JR_IndexTTS25)",
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--real-inference", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "xpu_report.json"
    report: dict[str, object] = {
        "status": "FAIL",
        "candidate": "Ubuntu 24.04 + Python 3.13 + torch/torchaudio 2.11.0+xpu",
        "device": args.device,
        "precision": args.precision,
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
                "soundfile",
                "WeTextProcessing",
                "wetext",
                "audioop-lts",
            )
        },
    }

    try:
        comfyui_root = args.comfyui_root.expanduser().resolve()
        plugin_dir = (
            args.plugin_dir.expanduser().resolve()
            if args.plugin_dir is not None
            else comfyui_root / "custom_nodes" / "ComfyUI_JR_IndexTTS25"
        )
        if not (plugin_dir / "__init__.py").is_file():
            raise FileNotFoundError(f"Candidate plugin not found: {plugin_dir}")
        sys.path.insert(0, str(comfyui_root))
        sys.path.insert(0, str(plugin_dir.parent))

        import numpy as np
        import soundfile as sf
        import torch

        if not hasattr(torch, "xpu") or not torch.xpu.is_available():
            raise RuntimeError("torch.xpu is not available")

        selected_device_index = device_index(args.device)
        xpu_device_count = int(torch.xpu.device_count())
        report["torch_xpu_version"] = getattr(torch.version, "xpu", None)
        report["xpu_available"] = True
        report["xpu_device_count"] = xpu_device_count
        if selected_device_index >= xpu_device_count:
            raise RuntimeError(
                f"Requested {args.device}, but only {xpu_device_count} XPU device(s) exist"
            )
        report["xpu_devices"] = [
            {"index": index, "name": torch.xpu.get_device_name(index)}
            for index in range(xpu_device_count)
        ]
        report["selected_device_index"] = selected_device_index
        report["gpu"] = torch.xpu.get_device_name(selected_device_index)
        report["gpu_capability"] = str(
            torch.xpu.get_device_capability(selected_device_index)
        )
        try:
            free_memory, total_memory = torch.xpu.mem_get_info(selected_device_index)
            report["xpu_memory"] = {
                "free": int(free_memory),
                "total": int(total_memory),
            }
        except Exception as error:
            report["xpu_memory_error"] = f"{type(error).__name__}: {error}"
        report["operator_preflight"] = xpu_operator_preflight(torch, args.device)

        import ComfyUI_JR_IndexTTS25 as plugin
        from ComfyUI_JR_IndexTTS25.backend.indextts25_backend import (
            VoicePreset,
            assert_runtime_compatible,
            clear_model_cache,
            generate_audio,
            load_model,
        )

        report["registered_node_count"] = len(plugin.NODE_CLASS_MAPPINGS)
        report["registered_nodes"] = sorted(plugin.NODE_CLASS_MAPPINGS)
        report["runtime_validation"] = assert_runtime_compatible(
            strict=True,
            requested_device=args.device,
        )
        if len(plugin.NODE_CLASS_MAPPINGS) != 10:
            raise RuntimeError(
                f"Expected 10 registered nodes, found {len(plugin.NODE_CLASS_MAPPINGS)}"
            )
        report["plugin_import"] = "PASS"

        if args.real_inference:
            if args.model_dir is None or args.reference_audio is None:
                raise ValueError("--real-inference requires --model-dir and --reference-audio")
            model_dir = args.model_dir.expanduser().resolve()
            reference_path = args.reference_audio.expanduser().resolve()
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            if not reference_path.is_file():
                raise FileNotFoundError(f"Reference audio not found: {reference_path}")

            waveform, sample_rate = sf.read(reference_path, dtype="float32", always_2d=True)
            audio = {
                "waveform": torch.from_numpy(waveform.T.copy()).unsqueeze(0),
                "sample_rate": int(sample_rate),
            }
            voice = VoicePreset(name="XPUProbe", audio=audio)
            started = time.perf_counter()
            handle = load_model(
                model_path_override=str(model_dir),
                device=args.device,
                precision=args.precision,
                enable_qwen_emotion=False,
                strict_environment=True,
            )
            report["model_load_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            generated = generate_audio(
                handle,
                voice,
                "你好，这是 IndexTTS 二点五的英特尔显卡真实语音测试。",
                "ZH",
                seed=250,
                do_sample=True,
            )
            torch.xpu.synchronize()
            report["inference_seconds"] = time.perf_counter() - started
            generated_waveform = generated["waveform"].detach().cpu().float().numpy()
            generated_rate = int(generated["sample_rate"])
            if generated_waveform.shape[:2] != (1, 1):
                raise RuntimeError(f"Unexpected AUDIO shape: {generated_waveform.shape}")
            if not generated_waveform.size or not np.isfinite(generated_waveform).all():
                raise RuntimeError("Generated audio is empty or non-finite")
            if not np.any(generated_waveform != 0):
                raise RuntimeError("Generated audio is silent")
            output_wav = output_dir / "xpu_output.wav"
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
