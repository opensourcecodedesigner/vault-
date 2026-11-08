"""Import and actual NF4 CUDA forward/backward check; never downloads a model."""
import argparse
import importlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys

from .artifacts import now, write_json


def diagnose():
    import torch
    import psutil
    result = {"created_at": now(), "python": sys.version, "platform": platform.platform(),
              "packages": {}, "cuda": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
              "ram_bytes": psutil.virtual_memory().total,
              "available_ram_bytes": psutil.virtual_memory().available,
              "disk_free_bytes": shutil.disk_usage(".").free, "checks": {}}
    for package in ["torch", "torchvision", "torchaudio", "transformers", "accelerate", "huggingface_hub",
                    "safetensors", "peft", "bitsandbytes", "psutil", "auditkit"]:
        try:
            importlib.import_module(package)
            result["packages"][package] = importlib.metadata.version(package.replace("_", "-"))
        except Exception as exc:
            result["checks"][package] = {"passed": False, "error_type": type(exc).__name__}
    if result["cuda"]:
        result["gpu"] = torch.cuda.get_device_name(0)
        result["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        result["compute_capability"] = list(torch.cuda.get_device_capability())
        try:
            import bitsandbytes as bnb
            torch.manual_seed(42)
            layer = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.bfloat16,
                                      quant_type="nf4", compress_statistics=True).to("cuda")
            data = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            output = layer(data)
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            result["checks"]["nf4_forward_backward"] = {"passed": bool(torch.isfinite(data.grad).all()),
                                                           "loss": loss.item()}
        except Exception as exc:
            result["checks"]["nf4_forward_backward"] = {"passed": False, "error_type": type(exc).__name__}
    else:
        result["checks"]["cuda"] = {"passed": False}
    pip = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    result["pip_check"] = {"exit_code": pip.returncode, "output": pip.stdout + pip.stderr}
    result["passed"] = pip.returncode == 0 and all(x["passed"] for x in result["checks"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    result = diagnose()
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
