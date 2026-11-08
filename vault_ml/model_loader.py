"""Pinned official model; no remote code and no silent fallback to a different model."""
import os
import time
from pathlib import Path

from .config import ModelConfig


class ModelLoadError(RuntimeError):
    pass


def input_device(model):
    """Accelerate's execution hook overrides a CPU/disk-stored embedding weight."""
    import torch
    embedding = model.get_input_embeddings()
    hook = getattr(embedding, "_hf_hook", None)
    execution = getattr(hook, "execution_device", None)
    if execution is not None:
        return torch.device(execution)
    device = embedding.weight.device
    if device.type == "meta":
        raise ModelLoadError("Embedding is on meta without an execution hook.")
    return device


def classify_load_error(exc):
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError
    current, seen = exc, set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, GatedRepoError) or (isinstance(current, HfHubHTTPError)
                and getattr(current.response, "status_code", None) in {401, 403}):
            return ("Tiny Aya access denied. Accept the model terms at "
                    "https://huggingface.co/CohereLabs/tiny-aya-global, then run "
                    r".\.venv\Scripts\hf.exe auth login locally. Never put tokens in artifacts.")
        if isinstance(current, LocalEntryNotFoundError):
            return "Pinned model files are not cached. Authenticate and download them with network access first."
        current = current.__cause__ or current.__context__
    # Do not log exception URLs, headers, credentials, or user data.
    return f"Model loading failed ({type(exc).__name__}). Check access, dependencies, disk space and memory."


def load_model(config: ModelConfig, *, training=False):
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from .resources import snapshot
    start = time.perf_counter()
    before = snapshot()
    if config.placement != "cpu" and not torch.cuda.is_available():
        raise ModelLoadError("CUDA requested but unavailable. Run diagnostics; CPU mode requires no quantization.")
    if training and config.placement != "cuda":
        raise ModelLoadError("Training requires explicit single-GPU placement; inference dispatch is not training offload.")
    common = dict(revision=config.revision, local_files_only=config.local_files_only, trust_remote_code=False,
                  cache_dir=str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface" / "hub"))
    try:
        architecture = AutoConfig.from_pretrained(config.model_id, **common)
        if architecture.model_type != "cohere2":
            raise ModelLoadError("Pinned model architecture is not cohere2; inspect before continuing.")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, **common)
        if not tokenizer.chat_template:
            raise ModelLoadError("Official tokenizer has no chat template; refusing to guess one.")
        kwargs = dict(dtype=torch.bfloat16 if config.placement != "cpu" else torch.float32,
                      attn_implementation="sdpa", **common)
        if config.quantization == "nf4":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        if config.placement == "cuda":
            free, _ = torch.cuda.mem_get_info()
            if config.quantization == "none" and free < 9 * 1024**3:
                raise ModelLoadError("Insufficient VRAM for unquantized single-GPU loading; use nf4 or auto offload.")
            kwargs["device_map"] = {"": 0}
        elif config.placement == "auto":
            if config.quantization != "none":
                raise ModelLoadError("Use unquantized auto for CPU/GPU inference offload; NF4 uses explicit GPU placement.")
            free, _ = torch.cuda.mem_get_info()
            kwargs.update(device_map="auto", max_memory={0: max(int(free * 0.75), 1024**3), "cpu": "12GiB"},
                          offload_folder=str(Path(".cache/offload").resolve()))
        else:
            kwargs["device_map"] = {"": "cpu"}
        model = AutoModelForCausalLM.from_pretrained(config.model_id, **kwargs)
        if config.adapter_id:
            from peft import PeftModel
            import hashlib
            import json
            manifest_path = Path(config.adapter_id) / "training-result.json"
            if not manifest_path.is_file():
                raise ModelLoadError("Adapter lacks VAULT training manifest; audit external adapters before admission.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("status") != "complete" or manifest["model"]["revision"] != config.revision
                    or manifest["model"]["model_id"] != config.model_id):
                raise ModelLoadError("Adapter base revision does not match requested model.")
            adapter_weights = Path(config.adapter_id) / "adapter_model.safetensors"
            if hashlib.sha256(adapter_weights.read_bytes()).hexdigest() != manifest["adapter_sha256"]:
                raise ModelLoadError("Adapter weights do not match the recorded training hash.")
            model = PeftModel.from_pretrained(model, config.adapter_id, is_trainable=training)
        model.eval()
        return model, tokenizer, {"load_seconds": time.perf_counter() - start,
                                  "before": before, "after": snapshot(),
                                  "architecture": architecture.model_type,
                                  "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {}).items()}}
    except ModelLoadError:
        raise
    except Exception as exc:
        raise ModelLoadError(classify_load_error(exc)) from None
