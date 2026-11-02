import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CharLM(nn.Module):
    """Shared generate() for the models. Subclasses set self.context_len to
    how many previous characters they are allowed to look at."""

    context_len = None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            ctx = idx if self.context_len is None else idx[:, -self.context_len:]
            logits = self(ctx)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------------------
# Model 1: bigram
# ---------------------------------------------------------------------------

class BigramLM(CharLM):
    # The whole model is one V x V table. Row i holds the logits for whatever
    # character comes after character i. Nothing else goes into the prediction.
    context_len = 1

    def __init__(self, vocab_size):
        super().__init__()
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        return self.table(idx)  # (B, T, V)


# ---------------------------------------------------------------------------
# Model 2: LSTM (no attention)
# ---------------------------------------------------------------------------

class LSTMLM(CharLM):
    def __init__(self, vocab_size, n_embd=128, hidden=256, n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, hidden, num_layers=n_layers,
                            batch_first=True, dropout=dropout)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, idx, state=None, return_state=False):
        x = self.embed(idx)              # (B, T, n_embd)
        out, state = self.lstm(x, state)  # (B, T, hidden), everything so far squeezed into h, c
        logits = self.head(self.drop(out))
        if return_state:
            return logits, state
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # run the prompt through once, then feed one char at a time and keep
        # carrying the hidden state forward (no need to re-read the context)
        logits, state = self(idx, return_state=True)
        for _ in range(max_new_tokens):
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            logits, state = self(nxt, state, return_state=True)
        return idx


# ---------------------------------------------------------------------------
# Model 3: small GPT-style transformer
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # lower-triangular mask so position t can only look at positions <= t
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size))
                             .view(1, 1, block_size, block_size))
        self.save_attention = False  # flipped on in compare.py for the heatmap
        self.last_att = None

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # every position scores every earlier position, softmax turns the
        # scores into weights, and the output is a weighted sum of the values
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        if self.save_attention:
            self.last_att = att.detach().cpu()
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    # pre-norm block: attention (talk to other positions) then MLP (think per position)
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(CharLM):
    def __init__(self, vocab_size, block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.context_len = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)  # what the character is
        self.pos_emb = nn.Embedding(block_size, n_embd)  # where it is (learned positions)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout)
                                      for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)  # output layer -> logits over the vocab
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.blocks(x)
        return self.head(self.ln_f(x))


def build_model(name, vocab_size, block_size):
    if name == "bigram":
        return BigramLM(vocab_size)
    if name == "lstm":
        return LSTMLM(vocab_size)
    if name == "transformer":
        return TransformerLM(vocab_size, block_size=block_size)
    raise ValueError(f"unknown model {name}")
