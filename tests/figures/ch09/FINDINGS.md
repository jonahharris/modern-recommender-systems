# Chapter 9 — figure ↔ package consistency findings

Goal: verify the shipped `recsys` package works correctly with the chapter-9 listings
(printed as "Listing 7.x"; the notebooks label the same code "Listing 9.x"), and
surface inconsistencies. Chapter 9 is *Generative Recommendation & Fine-Tuning*: almost
all code is **inline** using `transformers`/`torch`; the package ships only two helper
modules under `recsys/models/recsys_speaking_LLM/`. Everything below is **verified by
executing the installed package** (`.venv/bin/python`, editable `recsys`) against the
listings and notebooks, not by inspection alone.

26 listings extracted. **9 executable** in the harness (all pass), **14 non-executable**
(need torch/a fine-tuned model, or are pseudocode/output), **3 non-parsing**
(2 pseudocode snippets + 1 output block).

## A. Package modules that don't import / don't work

Both helper modules are **orphans** — nothing in the package or the two chapter-9
notebooks imports them (the notebooks define the formatters and losses inline). They
ship broken and untested.

| # | Module | Error (verified) | Cause | Proposed patch (not applied) |
|---|--------|------------------|-------|------------------------------|
| A1 | `recsys/models/recsys_speaking_LLM/multi_token_prediction.py:58` | `SyntaxError: invalid syntax` at line 58 (`import` and `py_compile` both fail) | A stray markdown ```` ``` ```` code-fence was left in the `.py`. Two further latent bugs sit above it: (1) the module never imports its dependencies — `create_training_labels` uses `np.exp` (L21) and `compute_loss` uses `torch.sum` / `F.log_softmax` (L39-40), all undefined; (2) lines 44-57 are a top-level demo block (`trainer = MultiTokenPredictor(model, ...)`, `for batch in training_data: ... get_future_events(...)`) that runs on import and references undefined `model` / `training_data` / `get_future_events`. | Delete the ```` ``` ```` on line 58; add `import numpy as np`, `import torch`, `import torch.nn.functional as F` at the top; and either delete lines 44-57 or guard them under `if __name__ == "__main__":`. Verified: with imports added, the demo removed, and the fence gone, the class imports and `create_training_labels` runs correctly on toy data. |
| A2 | `recsys/models/recsys_speaking_LLM/sampled_softmax_loss.py` | Imports OK, but `SampledSoftmaxLoss().forward(...)` raises `AttributeError: 'SampledSoftmaxLoss' object has no attribute 'embedding'` (verified on toy tensors) | This module is an **out-of-sync, buggy copy** of printed Listing 7.26. `__init__(self, num_samples=10000)` never accepts or stores an `embedding`, yet `forward` reads `self.embedding.weight` (L33). `forward` also returns `F.cross_entropy(logits, target_positions)` where `target_positions` is **never defined** (L37) — a `NameError` waiting behind the `AttributeError`. | Sync the module to Listing 7.26: `def __init__(self, embedding, num_samples=10000): ... self.embedding = embedding` and add `target_positions = torch.arange(len(target_ids))` before the `cross_entropy` return. The `CompressedDecodingHead` class in the same file is fine (verified: forward returns the expected `[batch, seq, vocab]` shape). |

## B. Book listing ↔ package drift

| # | Where | Issue | Reality in package / proposed patch |
|---|-------|-------|-------------------------------------|
| B1 | Listing 7.26 (`SampledSoftmaxLoss`) vs `recsys/.../sampled_softmax_loss.py` | The **printed listing is correct** and runs (`__init__(self, embedding, num_samples=...)`, `self.embedding = embedding`, and `target_positions = torch.arange(len(target_ids))`); verified loss ≈ 9.97 on toy data. The **package copy is behind and broken** (A2). | The drift is package-side, not book-side. Patch the module to match Listing 7.26 (see A2). The chapter-9 **notebook** (`02-gen_recsys_alignment.ipynb`) already contains the correct inline `SampledSoftmaxLoss` (labeled "Listing 9.26"), verbatim-equal to the printed listing — so book and notebook agree; only the standalone package file diverges. |
| B2 | `multi_token_prediction.py` | The `MultiTokenPredictor` helper corresponds to the **Multi-Token Prediction** discussion in §7.9.1 but is **not printed as any listing** in the chapter, and (A1) it does not import. | No book listing to reconcile against; treat as an internal package defect. Fix per A1, or remove the module since nothing references it. |

No book listing in ch09 imports from `recsys.*` for its generative logic. The two
notebooks import only `recsys.utils.colab` (`setup_colab_environment`, `get_data_path`,
`check_gpu`), which imports cleanly — so there is **no book→package import drift** of the
kind seen in ch02 L2.15. The divergence is entirely inside the two orphan helper modules.

## C. Non-parsing listings (`compiles=false`) — classification

| # | Listing | Classification | Why it doesn't parse |
|---|---------|----------------|----------------------|
| C1 | 7.1 Discriminative scoring | **pseudocode** | Bare top-level `return top_k_items(item_scores)` — `return` outside a function is a `SyntaxError`. catalog/model/user/top_k_items are illustrative. |
| C2 | 7.2 Generative recommendation | **pseudocode** | Bare top-level `return recommendations` — same `SyntaxError`. user_history/llm illustrative. |
| C3 | 7.13 Example formatted sequences | **output-not-code** | Printed sample output (`User 1: 'AGE_25 LOC_US L1_5 ...'`), not Python. |

All three are **real book intent, not defects** — 7.1/7.2 are labeled illustrative
pseudocode in the prose, and 7.13 is example output. They are compile-checked only
(xfailed by the harness) and never executed.

## D. Executable listings — verified in the harness

Nine listings run in the shared namespace (`np`, `pd`, `csr_matrix`,
`cosine_similarity`, `ratings`, `movies`) and all pass:

- **Standalone**: 7.4 (`get_item_tokens`), 7.5 (`BaseFormatter`), 7.6 (`find_new_tokens`),
  7.10 (`get_time_token`), 7.11 (`rich_data`), 7.17 (`SemanticDecoder`),
  7.23 (`icl_prompt`), 7.24 (`icl_rerank`).
- **needs-prior** (inherit `BaseFormatter` from 7.5, which runs earlier in the cumulative
  namespace): 7.12 (`ContextualFormatter`), 7.14 (`FeedbackFormatter`),
  7.25 (`AnchoredFormatter`).

These execute as *definitions*; the harness confirms the formatter/decoder class and
function bodies parse and bind without error.

### Latent book bugs in otherwise-passing listings (surface only when the code is *called*)

| # | Listing | Latent issue |
|---|---------|--------------|
| D1 | 7.5 `BaseFormatter` | `__init__` calls `Counter()` but the listing never does `from collections import Counter`; instantiating `BaseFormatter(item_df)` raises `NameError`. The chapter-9 notebook imports `Counter` at the top, so it only bites a reader who copies the listing in isolation. Proposed: add `from collections import Counter` to the printed listing. |
| D2 | 7.12 `ContextualFormatter` | `format()` uses `random.random()` but the listing never imports `random` (again supplied by the notebook, not the listing). Proposed: add `import random`. |

The remaining 14 listings are non-executable in the harness: they need a downloaded
DistilGPT2 / a fine-tuned model / the HuggingFace `Trainer` / raw torch
(7.7, 7.8, 7.9, 7.15, 7.16, 7.18, 7.19, 7.20, 7.21, 7.26), or are pseudocode
(7.3, 7.22) — all compile-checked and, where torch-only (7.15/7.21), verified to import
cleanly in the venv.

## How this was checked

```bash
# import resolution for the two package helper modules
.venv/bin/python -c "import recsys.models.recsys_speaking_LLM.multi_token_prediction"   # SyntaxError @ line 58
.venv/bin/python -c "import recsys.models.recsys_speaking_LLM.sampled_softmax_loss"      # imports OK
.venv/bin/python -m py_compile recsys/models/recsys_speaking_LLM/multi_token_prediction.py  # pins the line

# signature + toy-data diff: printed Listing 7.26 vs the package class
#   book SampledSoftmaxLoss(embedding, num_samples).forward(hidden, targets, V) -> loss ~= 9.97
#   pkg  SampledSoftmaxLoss(num_samples).forward(...)               -> AttributeError: no 'embedding'
#   CompressedDecodingHead(16,8,100)(x[2,3,16])                     -> [2,3,100]  (OK)

# proposed multi_token_prediction fix rebuilt and run on toy events -> labels/weights correct

# notebook import resolution
.venv/bin/python -c "from recsys.utils.colab import setup_colab_environment, get_data_path, check_gpu"  # OK
#   grep: neither notebook imports the two speaking_LLM modules (they define formatters/losses inline);
#   notebook 02's inline SampledSoftmaxLoss ("Listing 9.26") == printed Listing 7.26, verbatim.

# harness (observed == expected)
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch09]" -q      # 1 passed
.venv/bin/pytest tests/figures/test_figures.py::test_listing_compiles -q -k ch09       # 23 passed, 3 xfailed
```

_Findings are reported with proposed patches; no `recsys` package source was modified._
