import argparse
import json
import os
import time

import torch
from torch.nn import functional as F

from data import CharData, load_text
from models import build_model

ROOT = os.path.join(os.path.dirname(__file__), "..")

# Same settings for every model so the comparison is fair.
# Only the learning rate is different per model (see README for why).
BATCH_SIZE = 64
BLOCK_SIZE = 128   # max characters of context any model gets during training/eval
STEPS = 5000
EVAL_EVERY = 250
EVAL_BATCHES = 50
LR = {"bigram": 1e-2, "lstm": 2e-3, "transformer": 1e-3}
SEED = 1337


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device):
    # GPU calls are async, so wait for them before reading the clock
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def loss_fn(logits, targets, reduction="mean"):
    B, T, V = logits.shape
    return F.cross_entropy(logits.view(B * T, V), targets.view(B * T), reduction=reduction)


@torch.no_grad()
def estimate_loss(model, data, device):
    # uses the same fixed batches every call (and for every model) so the
    # loss curves are directly comparable
    model.eval()
    out = {}
    for split in ["train", "val"]:
        g = torch.Generator().manual_seed(0)
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = data.get_batch(split, BATCH_SIZE, BLOCK_SIZE, device, generator=g)
            losses.append(loss_fn(model(x), y).item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


@torch.no_grad()
def full_eval(model, data, split, device, bs=256):
    """Loss over the whole split, cut into non-overlapping 128-char chunks.

    Also returns the average loss at each position inside a chunk. Position 0
    has 1 char of context, position 127 has 128, so this shows how much each
    model actually gets out of a longer context.
    """
    model.eval()
    xs, ys = data.chunks(split, BLOCK_SIZE)
    total = torch.zeros(BLOCK_SIZE)
    for i in range(0, len(xs), bs):
        x, y = xs[i:i + bs].to(device), ys[i:i + bs].to(device)
        l = loss_fn(model(x), y, reduction="none").view(x.shape)
        total += l.sum(0).cpu()
    model.train()
    per_pos = total / len(xs)
    return per_pos.mean().item(), per_pos.tolist()


def count_bigram_val_loss(data):
    # sanity check for the trained bigram: build the same model by just
    # counting character pairs (add-one smoothing) and score it on val
    V = data.vocab_size
    counts = torch.zeros(V, V)
    counts.index_put_((data.train[:-1], data.train[1:]),
                      torch.ones(len(data.train) - 1), accumulate=True)
    probs = (counts + 1) / (counts + 1).sum(1, keepdim=True)
    return -probs[data.val[:-1], data.val[1:]].log().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["bigram", "lstm", "transformer"])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()
    dev = args.device

    torch.manual_seed(SEED)
    data = CharData(load_text())
    model = build_model(args.model, data.vocab_size, BLOCK_SIZE).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.model}: {n_params:,} parameters, training on {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR[args.model])
    history = []
    train_time = 0.0  # only counts the training steps, not the evals
    t0 = time.time()

    for step in range(args.steps + 1):
        if step % EVAL_EVERY == 0 or step == args.steps:
            sync(dev)
            train_time += time.time() - t0
            losses = estimate_loss(model, data, dev)
            history.append({"step": step, "train": losses["train"],
                            "val": losses["val"], "time": train_time})
            print(f"step {step:5d} | train {losses['train']:.4f} | "
                  f"val {losses['val']:.4f} | {train_time:6.1f}s")
            t0 = time.time()
        if step == args.steps:
            break

        x, y = data.get_batch("train", BATCH_SIZE, BLOCK_SIZE, dev)
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print("final eval on the full train and val splits...")
    train_loss, _ = full_eval(model, data, "train", dev)
    val_loss, val_by_pos = full_eval(model, data, "val", dev)
    print(f"train {train_loss:.4f} | val {val_loss:.4f}")

    # generate with a fixed seed so the samples are reproducible
    model.eval()
    torch.manual_seed(42)
    samples = {}
    for prompt in ["\n", "ROMEO:\n"]:
        idx = torch.tensor([data.encode(prompt)], device=dev)
        samples[prompt] = data.decode(model.generate(idx, 500)[0].tolist())

    extra = {}
    if args.model == "bigram":
        extra["count_bigram_val_loss"] = count_bigram_val_loss(data)
        print(f"count-based bigram val loss: {extra['count_bigram_val_loss']:.4f}")

    results = {
        "model": args.model,
        "params": n_params,
        "device": dev,
        "steps": args.steps,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "lr": LR[args.model],
        "train_time_sec": train_time,
        "ms_per_step": 1000 * train_time / args.steps,
        "final_train_loss": train_loss,
        "final_val_loss": val_loss,
        "val_loss_by_position": val_by_pos,
        "history": history,
        "samples": samples,
        **extra,
    }

    os.makedirs(os.path.join(ROOT, "results", "samples"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "checkpoints"), exist_ok=True)
    with open(os.path.join(ROOT, "results", f"{args.model}.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(ROOT, "results", "samples", f"{args.model}.txt"), "w") as f:
        for prompt, text in samples.items():
            f.write(f"===== prompt: {prompt!r} (500 new chars, temperature 1.0) =====\n")
            f.write(text + "\n\n")
    torch.save(model.state_dict(), os.path.join(ROOT, "checkpoints", f"{args.model}.pt"))
    print("saved results/ and checkpoints/")


if __name__ == "__main__":
    main()
