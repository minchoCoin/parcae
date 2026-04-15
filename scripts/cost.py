from dataclasses import dataclass

@dataclass
class FLOPBreakdown:
    total_params: int = 0
    embed_params: int = 0

    core_block_params: int = 0
    non_core_params: int = 0

    total_attn_flops: int = 0
    core_attn_fwd_per_step: int = 0
    non_core_attn_flops: int = 0

    is_recurrent: bool = False
    mean_recurrence: int = 1
    mean_backprop_depth: int = 1
    block_size: int = 0

    effective_forward_steps: float = 0.0
    effective_backprop_depth: float = 0.0
    has_curriculum: bool = False

    def __post_init__(self):
        if self.effective_forward_steps == 0.0:
            self.effective_forward_steps = float(max(self.mean_recurrence - self.mean_backprop_depth, 0))
        if self.effective_backprop_depth == 0.0:
            self.effective_backprop_depth = float(self.mean_backprop_depth)

    def flops_per_token(self, use_curriculum_adjusted: bool = True) -> int:
        if not self.is_recurrent:
            return 6 * self.total_params + self.total_attn_flops

        if use_curriculum_adjusted and self.has_curriculum:
            fwd_steps = self.effective_forward_steps + self.effective_backprop_depth
            bwd_steps = self.effective_backprop_depth
        else:
            fwd_steps = float(self.mean_recurrence)
            bwd_steps = float(self.mean_backprop_depth)

        non_core_flops = 6 * self.non_core_params + self.non_core_attn_flops

        core_fwd = 2 * self.core_block_params * fwd_steps
        core_bwd = 4 * self.core_block_params * bwd_steps
        core_attn_fwd = self.core_attn_fwd_per_step * fwd_steps
        core_attn_bwd = 2 * self.core_attn_fwd_per_step * bwd_steps
        core_flops = core_fwd + core_bwd + core_attn_fwd + core_attn_bwd

        return int(non_core_flops + core_flops)


def _attention_flops_fwd_per_layer(n_embd: int, num_heads: int, seq_len: int) -> int:
    head_dim = n_embd // num_heads
    return 4 * num_heads * head_dim * seq_len


def estimate_flops_gpt(model, config) -> FLOPBreakdown:
    breakdown = FLOPBreakdown()
    breakdown.is_recurrent = False
    breakdown.block_size = config.block_size

    all_params = sum(p.numel() for p in model.parameters())
    breakdown.embed_params = model.transformer.wte.weight.numel()
    breakdown.total_params = all_params - breakdown.embed_params

    n_layers = len(model.transformer.h)
    attn_fwd_per_layer = _attention_flops_fwd_per_layer(config.n_embd, config.num_attention_heads, config.block_size)
    breakdown.total_attn_flops = n_layers * attn_fwd_per_layer * 3

    return breakdown


def estimate_flops_recurrent(model, config) -> FLOPBreakdown:
    breakdown = FLOPBreakdown()
    breakdown.is_recurrent = True
    breakdown.mean_recurrence = config.mean_recurrence
    breakdown.mean_backprop_depth = config.mean_backprop_depth
    breakdown.block_size = config.block_size

    all_params = sum(p.numel() for p in model.parameters())
    breakdown.embed_params = model.transformer.wte.weight.numel()
    breakdown.total_params = all_params - breakdown.embed_params

    breakdown.core_block_params = sum(p.numel() for p in model.transformer.core_block.parameters())
    if hasattr(model, 'recurrence_norm') and model.recurrence_norm is not None:
        breakdown.core_block_params += sum(p.numel() for p in model.recurrence_norm.parameters())
    breakdown.non_core_params = breakdown.total_params - breakdown.core_block_params

    if config.recurrent_embedding_dimension != config.n_embd:
        core_config = config.recurrent_block_config
        core_n_embd = core_config.n_embd
        core_n_heads = core_config.num_attention_heads
    else:
        core_n_embd = config.n_embd
        core_n_heads = config.num_attention_heads

    n_core = len(model.transformer.core_block)
    breakdown.core_attn_fwd_per_step = n_core * _attention_flops_fwd_per_layer(
        core_n_embd, core_n_heads, config.block_size
    )

    n_prelude = len(model.transformer.prelude)
    n_coda = len(model.transformer.coda)
    attn_fwd_per_layer = _attention_flops_fwd_per_layer(config.n_embd, config.num_attention_heads, config.block_size)
    breakdown.non_core_attn_flops = (n_prelude + n_coda) * attn_fwd_per_layer * 3

    return breakdown
