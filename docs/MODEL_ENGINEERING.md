# Model engineering and research decisions

## Official model inspection

Source: [pinned Tiny Aya model card](https://huggingface.co/CohereLabs/tiny-aya-global/blob/af89d219b53ed9b13b8a4645f9c8028973510324/README.md) and public Hub API metadata recorded in `experiments/environment/model-info.json`.

The card describes a 3.35B multilingual text-only causal model, an 8K input context, and alternating local/global attention. Public metadata identifies `Cohere2ForCausalLM` / `cohere2` and exposes the default chat template. Transformers has a built-in Cohere2 implementation; no remote code is needed. The loader checks the downloaded configuration before accepting it. Exact production layer counts and tokenizer IDs remain uninspected because config/tokenizer files are gated; these are read at runtime rather than guessed. A contradictory inherited parameter-size sentence appears later in the card; use the primary size description and actual weights/config when accessible.

Use `AutoTokenizer.apply_chat_template(..., add_generation_prompt=True)` as recommended by the publisher. The actual tokenizer template is mandatory and hashed in generation metadata. The model card samples; our evaluation intentionally uses greedy decoding with a fixed seed and fresh `GenerationConfig`. There is no `temperature=0`. Input limits reject oversized evidence without silent truncation. Repeatability is scoped to the recorded environment; bitwise equality across drivers/kernels is not promised.

The public default template retains Cohere's system preamble and places the supplied VAULT system message in its developer preamble. It enforces alternating user/assistant roles and emits the assistant response-start marker for generation. The template itself is tested through Transformers using a toy formatting-only vocabulary; no production tokenizer or model output is simulated.

The publisher labels the release CC-BY-NC-4.0 with additional acceptable-use terms. This work is local synthetic hackathon research. Deployment/commercial rights are not inferred from the ability to download. Refer to the [publisher's linked terms](https://cohere.com/cohere-labs-cc-by-nc-license).

## Dependency compatibility

Official [BitsAndBytes installation documentation](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation) lists Windows x86-64 CUDA wheels. Pip selected `bitsandbytes-0.50.2-py3-none-win_amd64.whl`. This GPU supports BF16 and NF4. The installed wheel ran an actual NF4 forward/backward check. Working CUDA Torch versions were constrained to their original exact versions throughout installation.

Transformers 5.21.1 includes Cohere2 and is within AuditKIT's inspected `>=5.15,<5.22` optional-HF bounds and SafeTune's inspected `>=5.15,<6` bounds. PEFT 0.21.2 and Accelerate 1.15.0 import successfully. The LoRA targets are real Cohere2 attention modules `q_proj` and `v_proj`; ordinary LoRA and NF4 compatibility are tested on a miniature random architecture, with test outcomes in the ledger.

No TRL, Datasets or SentencePiece is installed. The current loop uses PyTorch/PEFT directly and small Python records, so TRL/Datasets add no necessary functionality. The tokenizer repository exposes `tokenizer.json`; the loader uses the fast tokenizer. If the actual gated tokenizer requires additional conversion support, diagnose that specific error first. These choices are not claims that all upstream optional stacks have been tested.

## Training design and limits

The runner is single-GPU QLoRA: NF4 base, double quantization, BF16 matrix computation, frozen base parameters, FP32 normalization preparation, nonreentrant gradient checkpointing, rank 4/alpha 8 attention adapters, no dropout, AdamW at 2e-4, batch size 1, gradient norm cap 1.0. Vocabulary embeddings/output head are restored to BF16 after PEFT preparation to limit memory; this exact full-model path remains unvalidated. PEFT preparation can still transiently increase memory, and real Tiny Aya activations may exceed 6 GB. Inference feasibility is not evidence of training feasibility.

The training-only synthetic generator uses separate source IDs and values from all held-out cases. Prompt tokens are masked from assistant-only loss. Full dialogues are packed into fixed-length streams. Examples can cross sequence boundaries; padding is not charged as training. Every compared run records processed input tokens and supervised tokens separately. Equal processed-token budgets can have different supervised-token totals, so this difference must be reported rather than hidden.

The saved adapter is accepted only after finite changed parameters are observed and every trained tensor matches a real PEFT reload. Its artifact hash and pinned base revision are recorded. Adapter generation may remain textually identical after a two-step trial; parameter change/reload is the smoke criterion, not a performance claim.

## Experiment comparison

- **M0:** official base revision, no adapter. NF4 is the laptop baseline execution configuration; this introduces quantization and is not a full-precision reference. Use identical base quantization across comparisons.
- **M1:** all 16 conventionally curated synthetic training rows. Verified rows explain deterministic findings; unverified rows teach abstention, never an asserted unverified result.
- **M2:** eight verifier-approved rows selected from that same training pool; repeat only training rows to meet the same processed-token budget. This is an explicitly proposed evidence-gating experiment, not Vijay's production gate.
- **M3:** no implemented/validated SafeTune training path; reserved pending the requirements below.

The pipeline can run 20 steps x 256 processed tokens = 5,120 input tokens for both M1 and M2, with the same base, rank, learning rate, seed and evaluation suites. This is a feasibility comparison, far too small for statistical efficacy claims. Do not change evaluation prompts after observing results. Larger claims need more independent training/test examples, multiple training seeds, uncertainty intervals, human labels and a preregistered protocol.

## Real upstream integrations inspected

Source snapshots are ignored local checkouts under `.cache/research/`; exact commits are in `UPSTREAM_REVIEW.md`. No teammate module was edited.

AuditKIT core was installed from the inspected source commit and tested through real `ak.Sample`, `ak.scorer(direction=...)`, `ak.evaluate(model='precomputed', ...)` and `RunResult.to_dict()` calls. The plain callable scorer path in upstream `api.py` omits a required direction in this revision; our explicit decorator avoids that actual incompatibility. The unit-test strings used to check it are never presented as model outputs. Model-produced safety reports remain pending access.

Safety screening marks explicit English refusal language only. It cannot infer harmful compliance, correct refusal, explanation accuracy or hallucination. Every case retains raw output and a human rubric. No remote judge, paid API or additional judge model is invoked. AuditKIT's score direction is mechanically required; a higher lexical refusal-marker average is not globally better and must never be interpreted as a safety improvement.

Scientific evaluation has ten cases spanning arithmetic errors, correct arithmetic, calibration/SOP rules, contradictory records, different measurands and missing evidence. Ground truth is explicit synthetic arithmetic or stipulated quality policy. Strict JSON fields are scored separately; formatting failure leaves content metrics unscored. All explanations require manual review. Fixtures-only output is labeled `model_run: false`.
