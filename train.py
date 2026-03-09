"""
train.py — Training Loop
Features:
  - Mixed-precision training (torch.cuda.amp) — cuts VRAM ~50%
  - Gradient checkpointing — cuts VRAM ~40% (model.py handles this)
  - Cosine LR schedule with warmup
  - Gradient clipping
  - Checkpoint saving/loading
  - Weights & Biases logging (optional)
  - Periodic validation + perplexity reporting

Usage:
  python train.py                        # uses defaults (sample corpus)
  python train.py --data my_text.txt     # your own dataset
  python train.py --resume checkpoint_best.pt
  python train.py --config large         # use large model preset
"""

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from dataset import build_dataset, create_dataloaders, SAMPLE_TEXT
from model import GPT, ModelConfig
from tokenizer import BPETokenizer


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """LR: linear warmup then cosine decay to min_lr_ratio * base_lr."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler,
    step: int,
    loss: float,
    cfg: ModelConfig,
    tokenizer_path: str,
    save_path: str,
):
    """Save full training state."""
    state = {
        "step": step,
        "loss": loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "model_config": cfg.__dict__,
        "tokenizer_path": tokenizer_path,
    }
    torch.save(state, save_path)
    print(f"[Train] Checkpoint saved → {save_path}  (step={step}, loss={loss:.4f})")


def load_checkpoint(
    path: str,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    scheduler=None,
) -> int:
    """Load checkpoint. Returns the step number."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    step = ckpt.get("step", 0)
    print(f"[Train] Resumed from {path}  (step={step}, loss={ckpt.get('loss', '?'):.4f})")
    return step


@torch.no_grad()
def evaluate(model: GPT, val_loader, device: str, max_batches: int = 50) -> dict:
    """Compute validation loss and perplexity."""
    model.eval()
    total_loss, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        tgts = batch["targets"].to(device)
        with autocast(dtype=torch.bfloat16):
            out = model(ids, targets=tgts)
        total_loss += out["loss"].item()
        n += 1
    avg_loss = total_loss / max(n, 1)
    return {"val_loss": avg_loss, "perplexity": math.exp(min(avg_loss, 20))}


# ─────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────

def train(args):
    # ── Device ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}")
    if device == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on Ampere+
    torch.backends.cudnn.allow_tf32 = True

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────
    tok_path = os.path.join(args.output_dir, "tokenizer.json")
    if os.path.exists(tok_path):
        print("[Train] Loading existing tokenizer...")
        tokenizer = BPETokenizer.load(tok_path)
    else:
        print("[Train] Training tokenizer from scratch...")
        if args.data and os.path.isfile(args.data):
            from dataset import load_text_file
            corpus_text = load_text_file(args.data)
            corpus = [line for line in corpus_text.split("\n") if line.strip()]
        else:
            corpus = [line for line in SAMPLE_TEXT.split("\n") if line.strip()]
        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(corpus)
        tokenizer.save(tok_path)

    # ── Dataset ───────────────────────────────────────────────────────
    cache_path = os.path.join(args.output_dir, "tokens_cache.npy")
    if args.data and os.path.isfile(args.data):
        from dataset import load_text_file
        corpus_text = load_text_file(args.data)
        source = [line for line in corpus_text.split("\n") if line.strip()]
    else:
        print("[Train] No --data provided; using built-in sample corpus.")
        source = [line for line in SAMPLE_TEXT.split("\n") if line.strip()]

    train_ds, val_ds = build_dataset(
        tokenizer=tokenizer,
        source=source,
        seq_len=args.seq_len,
        cache_path=cache_path,
    )
    train_loader, val_loader = create_dataloaders(
        train_ds, val_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────
    preset_map = {"small": ModelConfig.small, "medium": ModelConfig.medium, "large": ModelConfig.large}
    cfg = preset_map.get(args.config, ModelConfig.medium)()
    cfg.vocab_size = tokenizer.vocab_size_actual
    cfg.max_seq_len = args.seq_len
    cfg.use_gradient_checkpointing = args.grad_checkpoint

    model = GPT(cfg).to(device)
    print(f"[Train] Model: {model.num_parameters()/1e6:.1f}M parameters")

    # Compile for speed on PyTorch 2.0+ (optional, can disable if issues)
    if args.compile and hasattr(torch, "compile"):
        print("[Train] Compiling model with torch.compile()...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Separate weight decay params (don't decay bias, norms, embeddings)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            if any(nd in name for nd in ["bias", "norm", "emb"]):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ── Scaler & Scheduler ────────────────────────────────────────────
    scaler = GradScaler(enabled=(device == "cuda"))
    total_steps = args.epochs * len(train_loader)
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, optimizer, scaler, scheduler)

    # ── WandB (optional) ──────────────────────────────────────────────
    use_wandb = False
    if args.wandb:
        try:
            import wandb
            wandb.init(project="mini-llm", config=vars(args))
            use_wandb = True
        except ImportError:
            print("[Train] wandb not installed, skipping logging.")

    # ── Training Loop ─────────────────────────────────────────────────
    print(f"\n[Train] Starting training: {args.epochs} epochs, {total_steps:,} steps")
    print(f"[Train] Batch size: {args.batch_size}  |  Seq len: {args.seq_len}")
    print(f"[Train] LR: {args.lr}  |  Warmup: {warmup_steps} steps\n")

    step = start_step
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ids = batch["input_ids"].to(device, non_blocking=True)
            tgts = batch["targets"].to(device, non_blocking=True)

            # Forward + loss (mixed precision)
            with autocast(dtype=torch.bfloat16):
                out = model(ids, targets=tgts)
                loss = out["loss"]

            # Backward
            scaler.scale(loss).backward()

            # Gradient accumulation support
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            step += 1
            epoch_loss += loss.item()

            # Logging
            if step % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                tokens_per_sec = args.log_interval * args.batch_size * args.seq_len / elapsed
                print(
                    f"Epoch {epoch+1} | Step {step:6d} | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {lr:.2e} | "
                    f"{tokens_per_sec/1000:.1f}k tok/s"
                )
                if use_wandb:
                    wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})
                t0 = time.time()

            # Validation
            if step % args.eval_interval == 0:
                metrics = evaluate(model, val_loader, device)
                print(
                    f"  ▶ Val Loss: {metrics['val_loss']:.4f}  "
                    f"Perplexity: {metrics['perplexity']:.2f}"
                )
                if use_wandb:
                    wandb.log({**{f"val/{k}": v for k, v in metrics.items()}, "step": step})

                # Save best checkpoint
                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    save_checkpoint(
                        model, optimizer, scaler, scheduler, step,
                        metrics["val_loss"], cfg, tok_path,
                        os.path.join(args.output_dir, "checkpoint_best.pt"),
                    )
                model.train()

            # Periodic checkpoint
            if step % args.save_interval == 0:
                save_checkpoint(
                    model, optimizer, scaler, scheduler, step,
                    loss.item(), cfg, tok_path,
                    os.path.join(args.output_dir, f"checkpoint_step{step}.pt"),
                )

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n[Epoch {epoch+1}] Avg Loss: {avg_loss:.4f}\n")

    # Final save
    save_checkpoint(
        model, optimizer, scaler, scheduler, step,
        avg_loss, cfg, tok_path,
        os.path.join(args.output_dir, "checkpoint_final.pt"),
    )
    print(f"\n[Train] Training complete! Best val loss: {best_val_loss:.4f}")
    if use_wandb:
        wandb.finish()


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train a GPT-style LLM")
    p.add_argument("--data", type=str, default=None, help="Path to .txt training file")
    p.add_argument("--output_dir", type=str, default="./checkpoints")
    p.add_argument("--config", type=str, default="medium", choices=["small", "medium", "large"])
    p.add_argument("--vocab_size", type=int, default=8000)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--compile", action="store_true", help="Use torch.compile()")
    p.add_argument("--grad_checkpoint", action="store_true", default=True)
    p.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
