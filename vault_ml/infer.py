import argparse
import json
import sys
from pathlib import Path

from .artifacts import fingerprint, now, write_json
from .config import MODEL_REVISION, ModelConfig
from .contracts import InferenceRequest, make_response, messages_for
from .model_loader import ModelLoadError, input_device, load_model
from .resources import ResourceMonitor


class InferenceEngine:
    """One engine per process. Backend must serialize access and resolve adapters from an allowlist."""
    def __init__(self, config=ModelConfig()):
        self.config = config
        self.model, self.tokenizer, self.load_metadata = load_model(config)

    def generate_messages(self, messages):
        import torch
        from transformers import GenerationConfig, set_seed
        set_seed(self.config.seed)
        # Repeatable greedy decoding; cross-driver bitwise reproducibility is not promised.
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                     return_tensors="pt", return_dict=True)
        count = encoded["input_ids"].shape[-1]
        if count > self.config.max_input_tokens:
            raise ValueError("Prompt exceeds input budget; evidence is never silently truncated.")
        encoded = encoded.to(input_device(self.model))
        generation = GenerationConfig(do_sample=False, num_beams=1, use_cache=True,
                                      max_new_tokens=self.config.max_new_tokens,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None
                                                    else self.tokenizer.eos_token_id))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        with ResourceMonitor() as resources, torch.inference_mode():
            output = self.model.generate(**encoded, generation_config=generation)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        new_tokens = output[0, count:]
        answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        metadata = {"model_revision": self.config.revision, "model_config": self.config.to_dict(),
                    "generation_config": generation.to_dict(), "seed": self.config.seed,
                    "input_tokens": count, "output_tokens": len(new_tokens),
                    "output_hit_token_limit": len(new_tokens) == self.config.max_new_tokens,
                    "chat_template_sha256": fingerprint(self.tokenizer.chat_template),
                    "resources": resources.result, "load": self.load_metadata}
        return answer, metadata

    def infer(self, request: InferenceRequest):
        if request.model_id != self.config.model_id or request.adapter_id != self.config.adapter_id:
            raise ValueError("Request checkpoint differs from the loaded engine.")
        answer, metadata = self.generate_messages(messages_for(request))
        return make_response(request, answer, metadata)


def model_arguments(parser):
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--quantization", choices=["nf4", "none"], default="nf4")
    parser.add_argument("--placement", choices=["cuda", "auto", "cpu"], default="cuda")
    parser.add_argument("--adapter")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=192)


def config_from_args(args):
    return ModelConfig(revision=args.revision, quantization=args.quantization, placement=args.placement,
                       adapter_id=args.adapter, local_files_only=args.local_files_only,
                       max_new_tokens=args.max_new_tokens)


def main():
    parser = argparse.ArgumentParser(description="Local Tiny Aya inference; no network server.")
    model_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path, help="Only persist synthetic or explicitly permitted inputs.")
    args = parser.parse_args()
    try:
        config = config_from_args(args)
        request = (InferenceRequest.from_dict(json.loads(args.request.read_text(encoding="utf-8-sig")))
                   if args.request else InferenceRequest("cli", args.prompt, adapter_id=config.adapter_id))
        response = InferenceEngine(config).infer(request)
        if args.output:
            write_json(args.output, {"created_at": now(), "request": json.loads(json.dumps(request, default=lambda x: x.__dict__)),
                                     "response": response})
        print(json.dumps(response, ensure_ascii=False, indent=2))
        return 0
    except (ModelLoadError, ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({"error": f"Inference failed ({type(exc).__name__}); no response produced."}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
