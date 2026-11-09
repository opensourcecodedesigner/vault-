"""Bounded single-GPU QLoRA with explicit budgets and real adapter verification."""
import argparse
import gc
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .artifacts import fingerprint, now, run_directory, write_json
from .config import ModelConfig, TrainingConfig
from .contracts import InferenceRequest, messages_for
from .model_loader import input_device, load_model
from .resources import ResourceMonitor
from .training_data import selected_examples


def packed_sequences(tokenizer, examples, length, count):
    """Fixed processed-token budget; assistant-only loss. Never pad/reuse held-out data."""
    tokens, labels = [], []
    for example in examples:
        messages = messages_for(InferenceRequest.from_dict(example["request"]))
        prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": example["answer"]}],
                                              tokenize=True, add_generation_prompt=False, return_dict=False)
        if full[:len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError("Chat template does not produce an aligned assistant prefix; audit token masking.")
        tokens.extend(full)
        labels.extend([-100] * len(prefix) + full[len(prefix):])
    if not tokens or not any(x != -100 for x in labels):
        raise ValueError("Training dataset has no assistant targets.")
    # Stream repeats only training data; skip blocks with no supervised tokens.
    result, offset = [], 0
    while len(result) < count:
        index = [(offset + j) % len(tokens) for j in range(length)]
        ids, target = [tokens[j] for j in index], [labels[j] for j in index]
        offset += length
        if any(x != -100 for x in target[1:]):
            result.append({"input_ids": ids, "labels": target})
    return result


def train(model_config, training_config, root):
    import torch
    from peft import (LoraConfig, PeftModel, TaskType, get_peft_model, get_peft_model_state_dict,
                      prepare_model_for_kbit_training)
    from transformers import set_seed
    if model_config.adapter_id:
        raise ValueError("Fresh experiments must start from the untouched pinned model.")
    if model_config.quantization != "nf4" or model_config.placement != "cuda":
        raise ValueError("This runner requires single-GPU NF4 QLoRA; no inference offload for training.")
    directory = run_directory(root)
    result = {"created_at": now(), "model": model_config.to_dict(), "training": asdict(training_config),
              "status": "started", "loss_trajectory": [], "optimizer": {"name": "torch.optim.AdamW", "betas": [0.9, 0.999],
              "eps": 1e-8, "weight_decay": 0.0, "learning_rate": training_config.learning_rate}}
    write_json(directory / "manifest.json", result)
    start = time.perf_counter()
    try:
        set_seed(training_config.seed)
        torch.cuda.reset_peak_memory_stats()
        with ResourceMonitor() as resources:
            model, tokenizer, loading = load_model(model_config, training=True)
            result["load"] = loading
            # PEFT's standard preparation freezes base weights and enables checkpointing.
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                     gradient_checkpointing_kwargs={"use_reentrant": False})
            # Large vocabulary embeddings/output heads need BF16 to fit the laptop.
            # Keep normalization parameters FP32; no optimizer state for frozen weights.
            for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                if module is not None and not isinstance(module.weight, __import__("bitsandbytes").nn.Params4bit):
                    module.to(dtype=torch.bfloat16)
            model.config.use_cache = False
            lora = LoraConfig(r=training_config.rank, lora_alpha=training_config.alpha,
                              target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none",
                              task_type=TaskType.CAUSAL_LM)
            model = get_peft_model(model, lora)
            result["adapter_config"] = model.peft_config["default"].to_dict()
            # Enum fields serialize via their string inheritance in PEFT.
            result["adapter_config"]["target_modules"] = sorted(lora.target_modules)
            initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            trainable = [p for p in model.parameters() if p.requires_grad]
            result["trainable_parameters"] = sum(p.numel() for p in trainable)
            examples = selected_examples(training_config.variant)
            write_json(directory / "dataset.json", {"examples": examples, "hash": fingerprint(examples),
                                                      "heldout_used": False})
            sequences = packed_sequences(tokenizer, examples, training_config.sequence_length,
                                         training_config.steps * training_config.gradient_accumulation)
            result["processed_token_budget"] = sum(len(x["input_ids"]) for x in sequences)
            result["supervised_tokens"] = sum(sum(y != -100 for y in x["labels"][1:]) for x in sequences)
            result["dataset_hash"] = fingerprint(examples)
            result["packed_sequences_hash"] = fingerprint(sequences)
            optimizer = torch.optim.AdamW(trainable, lr=training_config.learning_rate, weight_decay=0.0)
            model.train()
            device = input_device(model)
            for step in range(training_config.steps):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for micro in range(training_config.gradient_accumulation):
                    row = sequences[step * training_config.gradient_accumulation + micro]
                    batch = {k: torch.tensor([v], dtype=torch.long, device=device) for k, v in row.items()}
                    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss.")
                    step_loss += loss.item() / training_config.gradient_accumulation
                    (loss / training_config.gradient_accumulation).backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Nonfinite gradient norm.")
                optimizer.step()
                torch.cuda.synchronize()
                result["loss_trajectory"].append({"step": step+1, "loss": step_loss, "gradient_norm": gradient_norm.item()})
                print(json.dumps(result["loss_trajectory"][-1]), flush=True)
            state = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}
            changed = [k for k in initial if not torch.equal(initial[k], state[k])]
            if not changed or not all(torch.isfinite(v).all() for v in state.values()):
                raise RuntimeError("Adapter parameters are unchanged or nonfinite.")
            adapter_dir = directory / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
            tokenizer.save_pretrained(adapter_dir)
            del optimizer, trainable
            base = model.unload()
            del model
            gc.collect()
            torch.cuda.empty_cache()
            reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
            reloaded_state = get_peft_model_state_dict(reloaded)
            equal = set(state) == set(reloaded_state) and all(torch.equal(v, reloaded_state[k].cpu()) for k, v in state.items())
            if not equal:
                raise RuntimeError("Reloaded adapter differs from trained parameters.")
            result["adapter_verification"] = {"changed_tensor_count": len(changed), "total_tensor_count": len(state),
                                              "finite_parameters": True, "reload_exact_match": equal,
                                              "method": "Real saved/reloaded PEFT tensors; generation change not required or inferred."}
            result["adapter_id"] = str(adapter_dir.resolve())
            result["adapter_sha256"] = hashlib.sha256((adapter_dir / "adapter_model.safetensors").read_bytes()).hexdigest()
        result.update(status="complete", runtime_seconds=time.perf_counter()-start, resources=resources.result)
        write_json(adapter_dir / "training-result.json", result)
        write_json(directory / "result.json", result)
        return directory
    except Exception as exc:
        result.update(status="failed", runtime_seconds=time.perf_counter()-start,
                      error_type=type(exc).__name__, error=str(exc) if isinstance(exc, ValueError) else
                      "Training failed; inspect the local command log for compatibility/memory diagnosis.")
        if "resources" in locals() and hasattr(resources, "result"):
            result["resources"] = resources.result
        write_json(directory / "failure.json", result)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON with model and training fields (also usable on Linux GPU).")
    parser.add_argument("--variant", choices=["smoke", "M1", "M2", "M2-control", "M2_control", "M2-safety", "M2_safety"], default="smoke")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--output-root", default="experiments/training")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config else {}
        model_kwargs = dict(config.get("model", {}))
        if args.local_files_only:
            model_kwargs["local_files_only"] = True
        model_config = ModelConfig(**model_kwargs)
        training_config = TrainingConfig(**config.get("training", {"variant": args.variant, "steps": args.steps,
                                                                    "sequence_length": args.sequence_length}))
        print(train(model_config, training_config, args.output_root))
        return 0
    except Exception as exc:
        print(f"Training failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
