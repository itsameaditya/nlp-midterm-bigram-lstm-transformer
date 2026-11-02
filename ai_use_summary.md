# AI use summary

**Tool:** Claude Code (Anthropic's Claude, Opus 5.5), running in my terminal on my laptop.
**When:** November 2, 2026 (the day the project was due).

## What I asked for

I gave Claude screenshots of the full assignment page (goal, required models, comparison
questions, deliverables, README questions, grading rubric) and asked it to do the project
end to end: write the code, train the three models, make the results table, and write the
README in a plain, readable style rather than the usual over-formatted AI style. I also
asked it to put the repo on my GitHub and tell me what to submit on Canvas.

## What Claude did

- Set up a Python virtual environment and downloaded Tiny Shakespeare.
- Wrote all of the code in `src/` (data loading, the three models, the training loop, the
  comparison script and plots).
- Chose the hyperparameters: batch size, context length, number of steps, learning rates,
  and the LSTM/transformer sizes (it sized them to have roughly the same number of
  parameters so the comparison would be fair).
- Ran the training on my Mac (Apple M5 GPU through PyTorch's MPS backend) and produced
  everything in `results/`.
- Drafted the README and this file, then made the git commits and pushed the repo to
  GitHub. The commits are marked as co-authored by Claude.

So to be clear: most of the code and the first draft of the writing came from the AI, not
from me typing it.

## Keeping it honest

- Every number in the README comes from `results/*.json`, which the scripts write. Nothing
  was typed in or adjusted by hand. Running the scripts again with the same seed should
  give the same numbers, apart from small differences in timing and GPU arithmetic.
- The generated text samples are the raw model output with a fixed seed. Nothing was
  cherry-picked or edited.
- There's a built-in sanity check for the bigram model: `train.py` also builds a bigram by
  just counting character pairs, and its validation loss should match the trained one
  (it does: 2.4817 trained vs 2.4819 counted).
- The README says where the comparison has limits, like the learning rates not being
  tuned with a proper sweep and every model being trained only once (one seed).
- The README explains each design decision so they can be checked rather than taken on
  trust.
