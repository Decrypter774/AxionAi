"""
model.py — GPT-Style Transformer Language Model
~100M–300M parameters, CUDA-optimized, with:
  - Rotary Positional Embeddings (RoPE)
  - Grouped-Query Attention (GQA) optional
  - Pre-LayerNorm (more stable training)
  - SwiGLU activation (LLaMA-style FFN)
  - Gradient Checkpointing support
  - Flash Attention via scaled_dot_product_attention (PyTorch 2.0+)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 8000          # set to your tokenizer's vocab size

    # Architecture
    n_layers: int = 12              # transformer blocks
    n_heads: int = 12               # attention heads
    n_kv_heads: int = 4             # key/value heads (GQA; set = n_heads for MHA)
    d_model: int = 768              # embedding dimension
    d_ff: int = 2048                # feed-forward hidden dim (SwiGLU uses d_ff * 2/3 internally)
    max_seq_len: int = 1024         # context window

    # Regularization
    dropout: float = 0.1
    attn_dropout: float = 0.0

    # Training helpers
    use_gradient_checkpointing: bool = True
    tie_embeddings: bool = True     # tie input/output embedding weights

    # RoPE
    rope_theta: float = 10000.0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    # Preset configs ---------------------------------------------------

    @classmethod
    def small(cls) -> "ModelConfig":
        """~50M params — fast experimentation."""
        return cls(n_layers=6, n_heads=8, n_kv_heads=4, d_model=512, d_ff=1536)

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~130M params — good balance."""
        return cls(n_layers=12, n_heads=12, n_kv_heads=4, d_model=768, d_ff=2048)

    @classmethod
    def large(cls) -> "ModelConfig":
        """~350M params — needs 16GB VRAM."""
        return cls(n_layers=24, n_heads=16, n_kv_heads=8, d_model=1024, d_ff=4096)


# ─────────────────────────────────────────────
# Rotary Positional Embedding (RoPE)
# ─────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """
    RoPE: encodes position by rotating query/key vectors.
    No learned parameters — purely deterministic.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        # Precompute cos/sin tables [max_seq_len, head_dim/2]
        freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, freqs)              # [T, head_dim/2]
        freqs_cos = freqs.cos()                    # [T, head_dim/2]
        freqs_sin = freqs.sin()
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate the second half of head_dim."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q/k shape: [B, n_heads, T, head_dim]
        T = q.size(2)
        cos = self.freqs_cos[:T].unsqueeze(0).unsqueeze(0)  # [1,1,T,head_dim/2]
        sin = self.freqs_sin[:T].unsqueeze(0).unsqueeze(0)
        # Interleave to full head_dim
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ─────────────────────────────────────────────
# Attention
# ─────────────────────────────────────────────

class GroupedQueryAttention(nn.Module):
    """
    Multi-Head or Grouped-Query Attention with:
      - RoPE positional encoding
      - Flash Attention (PyTorch 2.0 scaled_dot_product_attention)
      - KV-cache support for inference
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads  # GQA repeat factor

        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads to match Q heads for GQA."""
        if self.n_rep == 1:
            return x
        B, n_kv, T, hd = x.shape
        return x.unsqueeze(2).expand(B, n_kv, self.n_rep, T, hd).reshape(B, n_kv * self.n_rep, T, hd)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k)

        # KV Cache (inference)
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v)

        # GQA: expand KV to match Q heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Flash Attention (uses memory-efficient CUDA kernels when available)
        # is_causal=True applies causal mask automatically
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=(kv_cache is None),   # causal during training; not needed with cache
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out), new_kv_cache


# ─────────────────────────────────────────────
# Feed-Forward (SwiGLU)
# ─────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward (used in PaLM, LLaMA).
    FFN(x) = SiLU(xW1) ⊗ (xW2) → linear
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU needs 2/3 of d_ff to match standard FFN param count
        hidden = int(d_ff * 2 / 3)
        hidden = (hidden + 63) // 64 * 64  # round to multiple of 64 for efficiency
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(self.drop(F.silu(self.w1(x)) * self.w2(x)))


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm transformer block:
      x → LayerNorm → Attention → residual
        → LayerNorm → FFN → residual
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = SwiGLUFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, Optional[tuple]]:
        attn_out, new_kv = self.attn(self.norm1(x), attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x, new_kv


# ─────────────────────────────────────────────
# Full GPT Model
# ─────────────────────────────────────────────

class GPT(nn.Module):
    """
    Full autoregressive transformer language model.

    Forward pass returns logits [B, T, vocab_size].
    During inference, pass use_cache=True for fast generation.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = nn.RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying: embedding and lm_head share weights
        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        # Init weights
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layers) (GPT-2 style)
        for name, p in self.named_parameters():
            if "out_proj" in name or "w3" in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        kv_caches: Optional[list] = None,
    ) -> dict:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} > max {self.cfg.max_seq_len}"

        x = self.emb_drop(self.token_emb(input_ids))

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if (kv_caches is not None) else None
            if self.cfg.use_gradient_checkpointing and self.training:
                # Gradient checkpointing: recompute activations on backward pass
                # Saves ~40% VRAM at cost of ~30% slower backward
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        out, _ = module(*inputs)
                        return out
                    return custom_forward
                x = checkpoint(create_custom_forward(block), x, use_reentrant=False)
                new_kv_caches.append(None)
            else:
                x, new_kv = block(x, kv_cache=cache)
                new_kv_caches.append(new_kv)

        x = self.norm_out(x)
        logits = self.lm_head(x)   # [B, T, vocab_size]

        loss = None
        if targets is not None:
            # Shift: predict next token
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, self.cfg.vocab_size),
                targets[:, 1:].reshape(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "loss": loss,
            "kv_caches": new_kv_caches if use_cache else None,
        }

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = filter(lambda p: p.requires_grad, self.parameters()) if trainable_only else self.parameters()
        return sum(p.numel() for p in params)


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    cfg = ModelConfig.medium()
    model = GPT(cfg)
    print(f"Parameters: {model.num_parameters()/1e6:.1f}M")

    dummy = torch.randint(0, cfg.vocab_size, (2, 128))
    out = model(dummy, targets=dummy)
    print(f"Loss: {out['loss'].item():.4f}")
    print(f"Logits shape: {out['logits'].shape}")
