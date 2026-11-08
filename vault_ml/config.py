from dataclasses import asdict, dataclass

MODEL_ID = "CohereLabs/tiny-aya-global"
MODEL_REVISION = "af89d219b53ed9b13b8a4645f9c8028973510324"
PROMPT_VERSION = "vault-evidence-v1"


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    revision: str = MODEL_REVISION
    quantization: str = "nf4"
    placement: str = "cuda"
    max_input_tokens: int = 2048
    max_new_tokens: int = 192
    seed: int = 42
    adapter_id: str | None = None
    local_files_only: bool = False

    def __post_init__(self):
        if self.model_id != MODEL_ID:
            raise ValueError("This workstream only supports the official Tiny Aya Global model.")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision):
            raise ValueError("Pin revision to a full Hub commit SHA.")
        if self.quantization not in {"nf4", "none"} or self.placement not in {"cuda", "auto", "cpu"}:
            raise ValueError("Unsupported quantization or placement.")
        if self.quantization == "nf4" and self.placement == "cpu":
            raise ValueError("CPU mode requires --quantization none.")
        if not 1 <= self.max_input_tokens <= 8192 or not 1 <= self.max_new_tokens <= 2048:
            raise ValueError("Invalid token limit.")
        if self.max_input_tokens + self.max_new_tokens > 8192:
            raise ValueError("Input plus output budget must be at most 8192 tokens.")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 2
    sequence_length: int = 256
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 0.0002
    rank: int = 4
    alpha: int = 8
    seed: int = 42
    variant: str = "smoke"

    def __post_init__(self):
        if not 1 <= self.steps <= 10000 or not 32 <= self.sequence_length <= 2048:
            raise ValueError("Invalid training steps or sequence length.")
        if self.batch_size != 1 or not 1 <= self.gradient_accumulation <= 64:
            raise ValueError("Use batch size 1 and positive gradient accumulation.")
        if not 0 < self.learning_rate <= 0.01 or not 1 <= self.rank <= 64 or self.alpha < 1:
            raise ValueError("Invalid optimizer or LoRA settings.")
        if self.variant not in {"smoke", "M1", "M2"}:
            raise ValueError("M3 requires a separately validated SafeTune implementation.")
