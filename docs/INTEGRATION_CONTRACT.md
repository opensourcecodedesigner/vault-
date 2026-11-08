# Proposed inference and training handoff contract

**Status: proposed for Vijay/Pawan review.** No production source identity, evidence authority or authorization agreement is assumed. This module is an in-process library, not an authenticated server. It does not independently verify claims, issue verifier findings or mutate source records.

## Pawan: inference

```python
from vault_ml.config import ModelConfig
from vault_ml.contracts import InferenceRequest
from vault_ml.infer import InferenceEngine

# Keep one instance in a dedicated worker. Serialize generate calls on this GPU.
engine = InferenceEngine(ModelConfig())
payload = {
    "request_id": "req-001",
    "model_id": "CohereLabs/tiny-aya-global",
    "adapter_id": None,
    "question": "Why was experiment EXP-101 flagged?",
    "evidence": [{
        "source_id": "EXP-101",
        "claim": "The reported total concentration is 3 mg/L.",
        "verified_value": "30 mg/L",
        "verification_status": "flagged"
    }]
}
response = engine.infer(InferenceRequest.from_dict(payload))
```

Model-dependent construction/execution above awaits access. Pure contract validation/serialization has been tested. For a trained local adapter, build a worker with `ModelConfig(adapter_id=<approved directory>)` and send the same identifier in requests. Only successfully trained adapters with matching base revision, manifest and weight hash are admitted. Pawan must resolve adapter IDs through a server-side allowlist; never map an untrusted HTTP field directly to a filesystem path. External adapters need their own admission review before creating a VAULT receipt.

Response keys: `request_id`, `model_id`, `adapter_id`, `answer`, `cited_source_ids`, `generation_metadata`, `warnings`. Metadata records base revision, model/generation configs, template version/hash, seed, token counts, truncation flag, loading placement/time and generation resource measurements. This metadata does not contain the incoming evidence text. No request or model output is persisted unless the caller explicitly saves it; the CLI prints the response to its invoking terminal.

Evidence objects require `source_id`, `claim`, `verification_status`; optional `verified_value`, `source_hash`. Status vocabulary: `verified`, `flagged`, `unverified`, `insufficient`, `conflicting`. IDs use up to 128 ASCII alphanumeric/`_.:-` characters. Hashes, when provided, are lowercase SHA256 strings. Duplicate source IDs are rejected; represent distinct conflicting records with distinct immutable version IDs. Unknown request/evidence fields fail validation. Requests allow at most 32 sources; claims and requests have character limits, followed by a token-budget check. Oversized prompts fail, preserving every source record intact.

The template asks for `[SOURCE:id]` citations. The response citation list contains only IDs supplied in the request. Unknown structured citations are removed from the answer with a warning. This is an identity allowlist check, not proof that the model's prose is supported by the cited record. Free-text assertions, disguised identifiers or prompt injection cannot be certified by a string filter. All conclusions remain model-generated explanations requiring review. Evaluation retains the unfiltered generated text separately.

`ValueError`: request/placement/token-budget mismatch. `ModelLoadError`: sanitized access, cache, architecture or model-load failure. CLI failure exit code: 1; argument parsing: 2. Catch errors at the worker boundary and map to your internal API; no HTTP route/status schema is invented here. Do not expose inference publicly without authorization, evidence access controls, serialized GPU access, bounded queues and suitable request limits.

## Vijay: evidence and examples

Supply immutable versioned source IDs and canonical source hashes, plus explicit verifier findings. Distinguish the raw source value, computed value, units, measurand, assumptions, verifier/version, source effective dates and unresolved contradictions. `verified_value` is the verifier-supplied result; the LM never replaces the verifier. Hash syntax validation alone does not authenticate a source. Pawan must authorize the caller and source access before passing evidence to this library.

The minimal request shape above can carry display-ready findings, but production team agreement is still needed on canonical bytes, source revisions, trusted verifier receipts, authority conflicts, signature/authorization rules and revocation. Until agreed, the response always warns that caller evidence authority is unauthenticated.

For training, propose a separate reviewed synthetic/permitted JSONL export with an ID, request/evidence, assistant target, provenance receipt, verifier-approved decision and split. The current generator's `verifier_approved` boolean is a local synthetic fixture flag; it is never sufficient for admission of arbitrary external data. External dataset ingestion is deliberately not enabled. CuratorKIT chat exports can be converted after contract review while preserving manifests/checksums. Do not send private experiments, secrets or held-out data into a training export. Unknown/alternative methods should retain their assumptions and conditional findings; disagreement with a conventional method is not itself a rejection rule.
