"""Downstream model variants reported in the paper."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class ModelType(Enum):
    LR = "lr"
    ABMIL = "abmil"


class ClassifierType(Enum):
    LINEAR = "linear"
    MLP = "mlp"


@dataclass
class ModelVariant:
    name: str
    model_type: ModelType
    embedding_sources: List[str]
    attention_type: Optional[str] = None
    use_context_stratification: bool = True
    num_heads: int = 1
    classifier_type: ClassifierType = ClassifierType.LINEAR
    mlp_hidden_dim: Optional[int] = None
    use_aem: bool = False
    use_pdl: bool = False
    pdl_proj_mode: Optional[str] = None

    def resolve_pdl_proj_mode(self, default_mode: str = "identity") -> str:
        return self.pdl_proj_mode or default_mode


MODEL_VARIANTS = {
    "lr_ext_embed": ModelVariant(
        name="LR sequence embedding",
        model_type=ModelType.LR,
        embedding_sources=["ext_embed"],
        use_context_stratification=False,
    ),
    "lr_hc_cell": ModelVariant(
        name="LR protein + context",
        model_type=ModelType.LR,
        embedding_sources=["hc", "cell"],
    ),
    "lr_hc_cell_ext_embed": ModelVariant(
        name="LR protein + context + sequence",
        model_type=ModelType.LR,
        embedding_sources=["hc", "cell", "ext_embed"],
    ),
    "abmil_hc_cell_gated_8": ModelVariant(
        name="ABMIL protein + context",
        model_type=ModelType.ABMIL,
        embedding_sources=["hc", "cell"],
        attention_type="gated",
        num_heads=8,
    ),
    "abmil_hc_cell_ext_embed_gated_8": ModelVariant(
        name="ABMIL protein + context + sequence",
        model_type=ModelType.ABMIL,
        embedding_sources=["hc", "cell", "ext_embed"],
        attention_type="gated",
        num_heads=8,
    ),
    "abmil_hc_cell_gated_8_pdl": ModelVariant(
        name="ABMIL+PDL protein + context",
        model_type=ModelType.ABMIL,
        embedding_sources=["hc", "cell"],
        attention_type="gated",
        num_heads=8,
        use_pdl=True,
    ),
    "abmil_hc_cell_ext_embed_gated_8_pdl": ModelVariant(
        name="ABMIL+PDL protein + context + sequence",
        model_type=ModelType.ABMIL,
        embedding_sources=["hc", "cell", "ext_embed"],
        attention_type="gated",
        num_heads=8,
        use_pdl=True,
    ),
}


def parse_model_argument(model_arg: str) -> List[str]:
    model_arg = model_arg.strip().lower()
    if model_arg == "all":
        return list(MODEL_VARIANTS)

    keys = [key.strip() for key in model_arg.split(",")]
    unknown = [key for key in keys if key not in MODEL_VARIANTS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. "
            f"Available: {', '.join(MODEL_VARIANTS)}"
        )
    return keys
