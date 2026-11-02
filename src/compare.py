"""Reads results/*.json from train.py and makes the comparison table + plots."""
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

from data import CharData, load_text
from models import build_model

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
MODELS = ["bigram", "lstm", "transformer"]
NAMES = {"bigram": "Bigram", "lstm": "LSTM", "transformer": "Transformer"}
COLORS = {"bigram": "#2a78d6", "lstm": "#eb6834", "transformer": "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 12, "axes.titlecolor": INK,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def load():
    out = {}
    for m in MODELS:
        with open(os.path.join(RES, f"{m}.json")) as f:
            out[m] = json.load(f)
    return out


def results_table(res):
    lines = [
        "| Model | Params | Train loss | Val loss | Val bits/char | Training time (5000 steps) | ms/step |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODELS:
        r = res[m]
        lines.append(
            f"| {NAMES[m]} | {r['params']:,} | {r['final_train_loss']:.3f} | "
            f"{r['final_val_loss']:.3f} | {r['final_val_loss'] / math.log(2):.2f} | "
            f"{r['train_time_sec']:.0f} s | {r['ms_per_step']:.1f} |"
        )
    table = "\n".join(lines)
    with open(os.path.join(RES, "results_table.md"), "w") as f:
        f.write(table + "\n")
    print(table)


def plot_loss_curves(res):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in MODELS:
        h = res[m]["history"]
        steps = [p["step"] for p in h]
        ax.plot(steps, [p["val"] for p in h], color=COLORS[m], lw=2, label=f"{NAMES[m]} (val)")
        ax.plot(steps, [p["train"] for p in h], color=COLORS[m], lw=1.2, ls="--",
                alpha=0.8, label=f"{NAMES[m]} (train)")
        nudge = {"bigram": 0, "lstm": 7, "transformer": -7}[m]
        ax.annotate(NAMES[m], (steps[-1], h[-1]["val"]), xytext=(6, nudge),
                    textcoords="offset points", va="center", color=INK2)
    ax.set_xlim(right=5600)
    ax.set_ylim(1.0, 3.2)  # cut off the step-0 spike (~4.2) so the end of training is readable
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss (nats/char)")
    ax.set_title("Train (dashed) and validation (solid) loss", loc="left")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "loss_curves.png"), dpi=150)
    plt.close(fig)


CONTEXT_BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]


def binned_context_loss(per_pos):
    # each single position is only ~870 characters of val text, so it's noisy;
    # averaging into doubling bins makes the trend readable
    return [sum(per_pos[a - 1:b]) / (b - a + 1) for a, b in CONTEXT_BINS]


def plot_context(res):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = list(range(len(CONTEXT_BINS)))
    for m in MODELS:
        ys = binned_context_loss(res[m]["val_loss_by_position"])
        ax.plot(xs, ys, color=COLORS[m], lw=2, marker="o", ms=5, label=NAMES[m])
        nudge = {"bigram": 0, "lstm": 7, "transformer": -7}[m]
        ax.annotate(NAMES[m], (xs[-1], ys[-1]), xytext=(8, nudge),
                    textcoords="offset points", va="center", color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(a) if a == b else f"{a}-{b}" for a, b in CONTEXT_BINS])
    ax.set_xlim(-0.3, len(xs) - 0.2)
    ax.set_xlabel("characters of context available")
    ax.set_ylabel("validation loss (nats/char)")
    ax.set_title("Validation loss vs. how much context the model had", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.68))
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "loss_vs_context.png"), dpi=150)
    plt.close(fig)


def load_model(name, data, device="cpu"):
    model = build_model(name, data.vocab_size, 128)
    model.load_state_dict(torch.load(os.path.join(ROOT, "checkpoints", f"{name}.pt"),
                                     map_location="cpu"))
    return model.to(device).eval()


# (layer, head, what it seems to do) - picked by printing each head's argmax
# on the snippet below, see README
HEADS = [
    (0, 2, "looks 1 char back"),
    (0, 0, "looks 2 chars back"),
    (1, 2, "looks back to the start of the line"),
    (2, 1, "often looks at the first letter of the word"),
]


def plot_attention(data, snippet):
    model = load_model("transformer", data)
    for b in model.blocks:
        b.attn.save_attention = True
    with torch.no_grad():
        model(torch.tensor([data.encode(snippet)]))

    labels = [{" ": "␣", "\n": "↵"}.get(c, c) for c in snippet]
    blues = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 12.5))
    for ax, (layer, head, desc) in zip(axes.flat, HEADS):
        att = model.blocks[layer].attn.last_att[0, head]
        ax.imshow(att, cmap=blues, vmin=0, vmax=1)
        ax.set_title(f"layer {layer + 1}, head {head + 1}: {desc}", color=INK, loc="left")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7, family="monospace")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7, family="monospace")
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_xlabel("attends to (earlier character)")
        ax.set_ylabel("current character")
    fig.suptitle(f"Attention weights on {snippet!r} (darker = more weight)",
                 color=INK, x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(RES, "attention_heads.png"), dpi=150)
    plt.close(fig)


def sample_stats(data, n_seqs=20, n_chars=500):
    """Rough, automatic measure of how realistic the generated text is:
    what fraction of generated words are real words from the training text,
    and what fraction of generated speaker names are real speaker names."""
    train_text = data.decode(data.train.tolist())
    vocab = set(re.findall(r"[a-z]+(?:'[a-z]+)*", train_text.lower()))
    speakers = set(re.findall(r"^([A-Za-z][A-Za-z ]*):$", train_text, re.M))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    stats = {}
    for m in MODELS:
        model = load_model(m, data, device)
        torch.manual_seed(0)
        start = torch.tensor([data.encode("\n")] * n_seqs, device=device)
        out = model.generate(start, n_chars)
        text = "\n".join(data.decode(row.tolist()) for row in out)
        words = re.findall(r"[a-z]+(?:'[a-z]+)*", text.lower())
        names = re.findall(r"^([A-Za-z][A-Za-z ]*):$", text, re.M)
        stats[m] = {
            "chars": n_seqs * n_chars,
            "words": len(words),
            "real_word_pct": 100 * sum(w in vocab for w in words) / len(words),
            "speaker_lines": len(names),
            "real_speaker_pct": 100 * sum(n in speakers for n in names) / max(len(names), 1),
        }
    with open(os.path.join(RES, "sample_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    for m in MODELS:
        st = stats[m]
        print(f"{m}: {st['real_word_pct']:.1f}% of {st['words']} words are real, "
              f"{st['real_speaker_pct']:.1f}% of {st['speaker_lines']} speaker names are real")


if __name__ == "__main__":
    res = load()
    results_table(res)
    for m in MODELS:
        binned = ", ".join(f"{v:.3f}" for v in binned_context_loss(res[m]["val_loss_by_position"]))
        print(f"{m} val loss by context bin: {binned}")
    print(f"count-based bigram val loss: {res['bigram']['count_bigram_val_loss']:.4f}")
    plot_loss_curves(res)
    plot_context(res)
    data = CharData(load_text())
    plot_attention(data, "KING RICHARD II:\nNow, my lord, what says the king?")
    sample_stats(data)
    print("plots saved to results/")
