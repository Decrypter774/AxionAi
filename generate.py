"""
generate.py — Text Generation / Inference
Sampling strategies:
  - Greedy (temperature=0)
  - Temperature sampling
  - Top-K sampling
  - Top-P (nucleus) sampling
  - Combined Top-K + Top-P + Temperature (recommended)

Uses KV-cache for fast autoregressive generation.

Usage:
  python generate.py --prompt "Once upon a time"
  python generate.py --prompt "The meaning of life is" --temperature 0.8 --top_p 0.9
  python generate.py --checkpoint checkpoints/checkpoint_best.pt --prompt "Hello"
  python generate.py --interactive   # chat loop
"""

import argparse
import time
from typing import List, Optional

import torch
import torch.nn.functional as F

from model import GPT, ModelConfig
from tokenizer import BPETokenizer


# ─────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────

def top_k_filter(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Zero out all logits except the top-k."""
    if top_k == 0:
        return logits
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    threshold = values[:, -1].unsqueeze(-1)
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out logits outside the nucleus (top-p) probability mass."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Shift right so the token that pushes us over threshold is included
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
    sorted_logits[sorted_indices_to_remove] = float("-inf")

    # Restore original order
    logits_filtered = torch.scatter(logits, -1, sorted_indices, sorted_logits)
    return logits_filtered


def repetition_penalty_filter(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float = 1.3,
    window: int = 64,
) -> torch.Tensor:
    """Penalize tokens that appeared recently to prevent loops."""
    if penalty == 1.0 or not generated_ids:
        return logits
    recent = set(generated_ids[-window:])
    for token_id in recent:
        if token_id < logits.size(-1):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    generated_ids: List[int] = [],
    repetition_penalty: float = 1.3,
) -> torch.Tensor:
    """
    Sample next token from logits [B, vocab_size].

    Args:
        temperature: < 1 = focused, > 1 = random, 0 = greedy
        top_k: keep only top-k tokens
        top_p: keep tokens with cumulative prob >= top_p
        repetition_penalty: > 1.0 discourages repeating recent tokens
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = repetition_penalty_filter(logits, generated_ids, repetition_penalty)
    logits = logits / temperature
    logits = top_k_filter(logits, top_k)
    logits = top_p_filter(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda",
    stream: bool = True,
    stop_at_eos: bool = True,
) -> str:
    """
    Autoregressive text generation with KV-cache.

    Args:
        model: trained GPT model
        tokenizer: matching BPETokenizer
        prompt: input text string
        max_new_tokens: how many new tokens to generate
        temperature/top_k/top_p: sampling parameters
        stream: print tokens as generated
        stop_at_eos: halt at <|eos|> token

    Returns:
        generated text (excluding the prompt)
    """
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    if stream:
        print(f"\n[Generating] Prompt: {prompt!r}\n{'─'*60}")
        print(prompt, end="", flush=True)

    t0 = time.time()
    generated_ids: List[int] = []
    kv_caches = None

    # Prefill: process the whole prompt at once
    out = model(input_tensor, use_cache=True)
    kv_caches = out["kv_caches"]
    next_logits = out["logits"][:, -1, :]  # [1, vocab]

    for _ in range(max_new_tokens):
        next_id = sample_token(next_logits, temperature, top_k, top_p, generated_ids, repetition_penalty=1.3)
        token_id = next_id[0, 0].item()
        generated_ids.append(token_id)

        if stop_at_eos and token_id == tokenizer.eos_id:
            break

        # Stream: decode full sequence to get correct word spacing
        if stream:
            full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            prev_text = tokenizer.decode(generated_ids[:-1], skip_special_tokens=True) if len(generated_ids) > 1 else ""
            new_part = full_text[len(prev_text):]
            print(new_part, end="", flush=True)

        # Decode single token, feed back in
        next_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        out = model(next_tensor, use_cache=True, kv_caches=kv_caches)
        kv_caches = out["kv_caches"]
        next_logits = out["logits"][:, -1, :]

    elapsed = time.time() - t0
    if stream:
        tok_per_sec = len(generated_ids) / max(elapsed, 1e-6)
        print(f"\n{'─'*60}")
        print(f"[Generated {len(generated_ids)} tokens in {elapsed:.2f}s  ({tok_per_sec:.1f} tok/s)]")

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Load model from checkpoint
# ─────────────────────────────────────────────

def load_model_from_checkpoint(checkpoint_path: str, device: str = "cuda") -> tuple[GPT, BPETokenizer]:
    """Load model + tokenizer from a checkpoint file."""
    print(f"[Generate] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild config from saved dict
    cfg_dict = ckpt.get("model_config", {})
    cfg = ModelConfig(**cfg_dict)

    # Load tokenizer
    tok_path = ckpt.get("tokenizer_path", "checkpoints/tokenizer.json")
    tokenizer = BPETokenizer.load(tok_path)

    # Build and load model
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    print(f"[Generate] Model loaded: {model.num_parameters()/1e6:.1f}M params on {device}")
    return model, tokenizer


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate text with a trained GPT model")
    p.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_best.pt")
    p.add_argument("--prompt", type=str, default="The future of artificial intelligence is")
    p.add_argument("--max_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--interactive", action="store_true", help="Enter interactive chat mode")
    p.add_argument("--no_stream", action="store_true", help="Disable streaming output")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model, tokenizer = load_model_from_checkpoint(args.checkpoint, device=args.device)

    if args.interactive:
        print("\n[Interactive Mode] Type your prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("You: ").strip()
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt:
                    continue
                generate(
                    model, tokenizer, prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device,
                    stream=not args.no_stream,
                )
                print()
            except KeyboardInterrupt:
                break
    else:
        generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            stream=not args.no_stream,
        )