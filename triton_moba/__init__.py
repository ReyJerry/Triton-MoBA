from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from .wrapper import moba_layer
from .naive_moba import moba_attn_varlen_naive
from .efficient_moba import moba_attn_varlen_efficient
from .triton_moba import moba_attn_varlen_triton
from .config import MoBAConfig


def register_moba(cfg: MoBAConfig):
    ALL_ATTENTION_FUNCTIONS["naive_moba"] = partial(moba_layer, moba_attn_varlen_naive, cfg)
    ALL_ATTENTION_FUNCTIONS["efficient_moba"] = partial(moba_layer, moba_attn_varlen_efficient, cfg)
    ALL_ATTENTION_FUNCTIONS["triton_moba"] = partial(moba_layer, moba_attn_varlen_triton, cfg)
