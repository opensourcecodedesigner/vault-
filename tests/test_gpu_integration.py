"""Explicit opt-in. Random miniature Cohere2 is a compatibility test, not Tiny Aya inference."""
import os
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("VAULT_RUN_GPU_TESTS") == "1", "set VAULT_RUN_GPU_TESTS=1 to run GPU compatibility tests")
class GPUCompatibilityTests(unittest.TestCase):
    def test_random_cohere2_peft_backward_save_reload(self):
        import torch
        from transformers import Cohere2Config, Cohere2ForCausalLM
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
        self.assertTrue(torch.cuda.is_available())
        torch.manual_seed(42)
        config = Cohere2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=16, max_position_embeddings=128, sliding_window=64,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2)
        base = Cohere2ForCausalLM(config).to("cuda")
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
                                               task_type="CAUSAL_LM", lora_dropout=0.0))
        before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        ids = torch.randint(3, 128, (1, 32), device="cuda")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        loss = model(input_ids=ids, labels=ids).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        trained = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        self.assertTrue(any(not torch.equal(v, before[k]) for k, v in trained.items()))
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            reloaded = PeftModel.from_pretrained(model.unload(), temp)
            self.assertTrue(all(torch.equal(v, get_peft_model_state_dict(reloaded)[k]) for k, v in trained.items()))

    def test_random_cohere2_nf4_qlora_backward_reload(self):
        import torch
        from transformers import BitsAndBytesConfig, Cohere2Config, Cohere2ForCausalLM
        from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, prepare_model_for_kbit_training
        torch.manual_seed(42)
        config = Cohere2Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                               num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                               head_dim=16, max_position_embeddings=128, sliding_window=64,
                               pad_token_id=0, bos_token_id=1, eos_token_id=2)
        with tempfile.TemporaryDirectory() as temp:
            from pathlib import Path
            base_path = Path(temp) / "random-test-base"
            Cohere2ForCausalLM(config).save_pretrained(base_path)
            model = Cohere2ForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map={"": 0},
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
            model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})
            for module in (model.get_input_embeddings(), model.get_output_embeddings()):
                if module is not None:
                    module.to(dtype=torch.bfloat16)
            model.config.use_cache = False
            model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
            before = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
            ids = torch.randint(3, 128, (1, 32), device="cuda")
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
            model.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=ids).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            trained = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
            self.assertTrue(any(not torch.equal(v, before[k]) for k, v in trained.items()))
            adapter = Path(temp) / "random-test-adapter"
            model.save_pretrained(adapter)
            reloaded = PeftModel.from_pretrained(model.unload(), adapter)
            self.assertTrue(all(torch.equal(v, get_peft_model_state_dict(reloaded)[k]) for k, v in trained.items()))


if __name__ == "__main__":
    unittest.main()
