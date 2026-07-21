# Chapter 6 — figure ↔ package consistency findings

Goal: verify the shipped `recsys` package works correctly with the book's chapter-6
listings (sequential recommendation: SASRec, BERT4Rec, gBCE), and surface
inconsistencies. Every item below is **verified by executing the installed package**
(`.venv/bin/python`, editable `recsys`) against the listings, not by inspection alone.

The chapter states its code "is taken directly from
`recsys/fourstage_recsys/retrieval/sequential.py`". That file ships
`_SequentialBase`, `SASRec`, `BERT4Rec`, `gbce_loss`, `evaluate`, `truncate_and_pad`,
`SASRecTrainDataset`, `BERT4RecTrainDataset`, `SequentialEvalDataset`.

## Bottom line

- The **shipped models work**. Real `SASRec` and `BERT4Rec` (from the package) build on
  a tiny toy tensor and run `forward()`/`recommend()`/`gbce_loss()` cleanly on CPU: no
  NaN, correct output shapes, padding token 0 excluded from recommendations. The
  hand-rolled per-layer `masked_fill` loop does prevent the all-padding-row NaN it was
  written to avoid (verified with fully-padded rows).
- The **printed core listings (6.1, 6.2, 6.5-mask, 6.6) match the package** almost
  verbatim (only indentation / docstrings / a missing `no_grad` differ).
- One **real signature drift** (Listing 6.3's `gbce_loss`) and one **real syntax bug**
  (Listing 6.7) are the substantive findings. Listings 6.4, 6.8, 6.9, 6.10 are
  *illustrative* — they are printed in the book but were never added to the package.

## A. Book listing ↔ package drift (printed code doesn't match the library)

| # | Where | Issue | Reality in package | Proposed patch |
|---|-------|-------|--------------------|----------------|
| A1 | L6.3 `gbce_loss` | Printed def is `gbce_loss(pos_scores, neg_scores, num_items, t=0.75)` — **no `pos_mask`** — and averages `(pos_loss+neg_loss).mean()/(n_neg+1)` over *all* positions, including left-padding. | Package is `gbce_loss(pos_scores, neg_scores, pos_mask, num_items, t=0.75)`; it multiplies per-position loss by `pos_mask.float()` and averages over target positions only. The package docstring explicitly calls the "average over all positions" approach a bug (`softplus(-0)=log(2)≈0.693`, not 0, so padding adds an irreducible constant). | Reprint Listing 6.3 with the `pos_mask` parameter and the masked average, matching `sequential.py:302`. |
| A2 | L6.3 ↔ L6.5 (self-contradiction) | The book contradicts **itself**: Listing 6.3 defines a 3-data-arg `gbce_loss`, but Listing 6.5's training loop calls `gbce_loss(pos_scores, neg_scores, cloze_mask, num_items)` — 4 positional args (the *package* signature). Verified: feeding the 6.5 call into the printed 6.3 def raises `RuntimeError: Subtraction ... with a bool tensor is not supported` (the bool `cloze_mask` lands in the `num_items` slot). | Same as A1 — the 6.5 call is correct; the 6.3 **definition** is the wrong one. | Fixing A1 resolves A2. |
| A3 | L6.6 `recommend` | Printed body omits the `with torch.no_grad():` guard. | Package `BERT4Rec.recommend` wraps the hidden/scoring block in `torch.no_grad()`. | Add the `with torch.no_grad():` guard to the listing (functional either way; without it inference allocates an autograd graph). |
| A4 | L6.4 `evaluate_ablation` | Printed as a package-style eval function, and calls `compute_ndcg_at_k(...)`. | Neither `evaluate_ablation` nor `compute_ndcg_at_k` exists **anywhere** in `recsys/`. The shipped evaluator is `sequential.evaluate(model, eval_dataset, k=10, ...)`, which computes NDCG@k/HR@k inline. Also L6.4 scores against `model.item_emb.weight.T` (full table, incl. special tokens) while the shipped `recommend`/`evaluate` slice `[:num_items+1]`. | Either mark the listing as illustrative-only, or ship a `compute_ndcg_at_k` helper and an `evaluate_ablation` wrapper so the printed code runs. |
| A5 | L6.7 `SASRecWithEvents`, L6.8 `pinnerformer_loss`, L6.9 `RelativeAttention`, L6.10 `OneTransTokenizer` | Printed as concrete classes/functions. | **None** are in the package (`grep` across `recsys/` finds zero hits). They are section-6.6/6.7 illustrations of extensions/other architectures. | No package change needed — but the prose ("extends the SASRec implementation from Listing 6.1") reads as if shipped; consider labelling these as sketches. |

## B. Real syntax / correctness bugs in printed listings

| # | Where | Issue | Evidence | Proposed patch |
|---|-------|-------|----------|----------------|
| B1 | L6.7 | **Stray `:` — does not parse.** The `super().__init__(...)` call ends `num_layers, num_heads, dropout):` (ch06.md line 504). | `compile()` raises `SyntaxError: invalid syntax` at file line 20; harness xfails this listing (`compiles=false`). | Delete the trailing `:` after `dropout)`. |
| B2 | L6.7 (latent, after B1 fix) | `forward` calls `self.transformer(x, mask=..., src_key_padding_mask=...)` **directly** — the exact one-shot call the base class deliberately avoids. | The base class comment (`sequential.py:110-120`) and chapter §6.3.2 explain that a single `self.transformer(...)` call NaNs on all-padding rows under causal masking; that's why `_encode` loops layers by hand with `masked_fill`. | Route `SASRecWithEvents.forward` through the base `_encode` (add the event term, then `return self._encode(...)`) instead of calling `self.transformer` directly. |
| B3 | L6.5 (numbering) | **Duplicate listing number.** Two distinct listings — "BERT4Rec training mask" and "BERT4Rec training loop" — are both printed as **Listing 6.5** (ch06.md lines 387 and 418). | Both `.py` files carry `number: "6.5"`; the compile-test ids collide (`ch06-L6.5`). | Renumber the training loop to 6.6 and shift 6.6→6.7 … (or letter them 6.5a/6.5b). |
| B4 | prose | **Wrong cross-references to the gBCE listing.** §6.4 opens "The loss in listing 6.2 is not the one used in the original SASRec paper" and §6.6 says "the gBCE loss from listing 6.2"; the gBCE loss is **Listing 6.3** (6.2 is the SASRec class). | ch06.md lines 350, 526, 528. | Change these "listing 6.2" references to "listing 6.3". |
| B5 | L6.10 (title) | Title reads "**OnTrans** Tokenizer" but the class is `OneTransTokenizer` (and the architecture is "OneTrans"). | ch06.md line 624 / listing body. | Fix title typo → "OneTrans Tokenizer". |

## C. Package internal consistency (does the shipped code actually work?)

All **pass** — the `sequential.py` module is self-consistent and runnable. Verified on a
toy catalog (`num_items=20`, `max_len=10`, `hidden_dim=16`, 2 layers/heads, CPU):

| check | result |
|-------|--------|
| `SASRec(...).forward(seqs)` | `(2,10,16)`, no NaN (incl. fully-padded rows) |
| `SASRec(...).recommend(seqs, k=5)` | `(2,5)` indices, padding token 0 never returned |
| `BERT4Rec(...).mask_sequence(seqs)` | ≥1 masked position per row guaranteed; `mask_token == num_items+1` |
| `BERT4Rec(...).forward(masked)` | `(2,10,16)`, no NaN |
| `BERT4Rec(...).recommend(seqs, k=5)` | `(2,5)` indices, appends `[MASK]`, trims to `max_len` |
| `gbce_loss(pos, neg, pos_mask, num_items)` | scalar (`≈0.76`), finite |
| import surface | `SASRec, BERT4Rec, gbce_loss, evaluate, SASRecTrainDataset, BERT4RecTrainDataset, SequentialEvalDataset, truncate_and_pad` all import cleanly |

No package-internal bug was found in this chapter's module. (Contrast ch02, where the
wired pipeline had contract bugs — the ch06 sequential module is clean.)

## D. Import resolution — figures & notebooks

- **Figures**: every ch06 listing that references the package is a *verbatim copy of* the
  package code, not an `import` of it, so there is no figure-side import drift (unlike
  ch02 L2.15). The only figure that can't even parse is L6.7 (B1).
- **Notebooks** (`notebooks/chapter-06/`):
  - `chapter06_sequential.ipynb` — imports `from recsys.fourstage_recsys.retrieval.sequential import (SASRec, BERT4Rec, gbce_loss, evaluate, SASRecTrainDataset, BERT4RecTrainDataset, SequentialEvalDataset, truncate_and_pad)`. **All resolve.** Also imports `mlflow` (present in the venv).
  - `sasrec.ipynb` — imports `recsys.utils.colab` (setup_colab_environment/get_data_path/check_gpu) and `recsys.data.loaders` (load_movielens, load_movielens_links, load_tmdb_movie_descriptions). **All resolve.** This notebook defines its own `SASRec`/`gbce_loss` inline rather than importing the package.
  - `sasrec_chapter6 (1).ipynb` — space-named **duplicate** of `sasrec.ipynb` (also defines `SASRec`/`gbce_loss` inline). Recommend deleting the "(1)" copy to avoid divergence.
  - `gsasrec_calibration.ipynb` — defines `SASRec`/`gbce_loss` inline (calibration study); no package/`recsys` import.

## E. Non-parsing / non-executable classification

| listing | compiles | classification |
|---------|----------|----------------|
| L6.7 | **false** | **real-book-syntax-bug** — stray `:` in `super().__init__(...):` (B1). |
| L6.1, L6.2, L6.9, L6.10 | true | executable=false: define `nn.Module` subclasses; `nn`/base class absent from the shared harness namespace → `NameError` at class creation (needs-package). |
| L6.5 (training loop) | true | executable=false: top-level training fragment → `NameError('batch')` (needs-training). |
| L6.3, L6.4, L6.5 (mask), L6.6, L6.8 | true | executable=true / pass: self-contained function/method `def`s (6.3 imports its own `F`); the `def` execs even though calling several of them would need torch/model in scope. |

## How this was checked

```bash
cd /Users/jharris/Development/apress/code/modern-recommender-systems

# 1. exec every listing in the shared harness namespace (cumulative), record outcome
# 2. build real SASRec/BERT4Rec on a toy tensor; run forward/recommend/gbce_loss
# 3. inspect.signature() diff: printed listing vs sequential.py
# 4. reproduce the L6.3-vs-L6.5 self-contradiction (RuntimeError on bool subtraction)
.venv/bin/python scratchpad/verify.py

# harness stays green (5 executable listings pass, 6.7 xfails on compile)
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch06]" -q
.venv/bin/pytest "tests/figures/test_figures.py::test_listing_compiles" -q -k ch06
```

_These are reported, not fixed — they belong to the author's book text / package. No
`recsys` source or file outside `tests/figures/ch06/` was modified._
