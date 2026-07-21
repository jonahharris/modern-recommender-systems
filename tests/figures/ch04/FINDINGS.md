# Chapter 4 — figure ↔ package consistency findings

Goal: verify the shipped `recsys` package works correctly with the book's chapter-4
listings (offline evaluation), and surface inconsistencies. Ch04 is about temporal
splitting, ALS recommendations via `implicit`, relevance sets, and the accuracy /
rank-aware metrics (precision, recall, NDCG, MRR, MAP). Every item below is **verified
by executing code** in the project venv (`.venv/bin/python`, `implicit 0.7.3`,
`scikit-learn 1.7.2`), not by inspection alone.

Headline: the package module for this chapter — `recsys/evaluation/metrics.py` —
**is correct and byte-identical to the notebook's metric cell**. The problems are all
**book-internal**: the printed Listing 4.8 is a *different*, syntactically broken and
numerically wrong reimplementation that matches neither the notebook nor the package,
plus duplicate/missing listing numbers and undefined-variable drift across 4.3–4.7.

## A. Package module status — `recsys/evaluation/metrics.py`

| # | Check | Result |
|---|-------|--------|
| A1 | `ndcg_at_k`, `mrr_at_k`, `average_precision_at_k` import cleanly | OK |
| A2 | Numerically correct? | **Yes.** `ndcg_at_k` matches a hand-computed DCG/IDCG and `sklearn.metrics.ndcg_score` to 1e-15 (0.65092…). `mrr_at_k` and `average_precision_at_k` match the standard definitions. |
| A3 | Matches the chapter-04 notebook? | **Identical.** `recsys/evaluation/metrics.py` (minus its `import numpy as np` header) is a byte-for-byte copy of `notebooks/chapter-04/evaluation.ipynb` cell 34 (`def ndcg_at_k / mrr_at_k / average_precision_at_k`). |
| A4 | Is the module actually used? | **No.** Nothing in `recsys/` imports it, `recsys/evaluation/__init__.py` is empty (no re-export), and neither the notebook nor the chapter imports `recsys.evaluation.metrics` — the notebook redefines the same functions inline. The module is a correct-but-orphaned copy. |

There is **no package↔notebook drift** for this chapter. No package source needs a patch
for correctness; the only optional improvement is exposure (see D1).

## B. Book listing ↔ package/notebook divergence (the printed Listing 4.8)

The prose (section 4.6.3) says *"Listing 4.8 shows code to calculate the different
metrics."* But the printed Listing 4.8 is **not** the metric code the notebook/package
use (the per-user `ndcg_at_k`/`mrr_at_k`/`average_precision_at_k`). It is a separate
sklearn-based loop that appears **nowhere** in the notebook, and it is both broken and
numerically wrong. Reproductions below.

| # | Where | Issue (verified) | Proposed patch (do not apply) |
|---|-------|------------------|-------------------------------|
| B1 | L4.8 md:536 | **Syntax error — does not parse.** `if rel == 1:` has no indented body; `relevant_found += 1` sits at the *same* indent → `IndentationError`. | Indent `relevant_found += 1` one level under the `if`. |
| B2 | L4.8 md:538–542 | **AP loop dedented out of the `for`.** `ap += relevant_found / (i + 1)`, `ap /= …`, and the `else` branch are dedented to the `for k` level, so (once B1 is fixed) `ap` is incremented **once, for the last position only**, not at each hit. | Re-indent `ap += relevant_found/(i+1)` inside the `for i, rel` loop and under `if rel == 1`; put the `ap /= …` / `append` / `else` at the correct `if np.sum(...) > 0` level. |
| B3 | L4.8 md:539 | **AP denominator diverges from the package.** L4.8 does `ap /= np.sum(top_k_rel)` (divide by number of *hits*). The package/notebook `average_precision_at_k` divides by `min(len(relevant_items), k)`. When not all relevant items land in top-k these disagree: for recs `[a,b,c,d,e]`, relevant `{a,c,x}` (3 relevant, 2 found), package AP = **0.556**, L4.8 AP = **0.833** — L4.8 overstates AP. | Divide by `min(len(relevant_items), k)` to match the package and the standard AP@k definition. |
| B4 | L4.8 md:515 | **NDCG uses `ndcg_score` incorrectly** — `ndcg_score([ideal_rel], [top_k_rel])` passes the *ideal* (descending-sorted) relevance as `y_true` and the actual relevance as `y_score`. That is not NDCG of the ranking. For `top_k_rel=[0,1,0,1,0]` it returns **0.769** vs the correct **0.651**. | Compute NDCG of the ranking, e.g. `ndcg_score([top_k_rel], [rank_scores])` with descending per-position scores, or just call the shipped `recsys.evaluation.metrics.ndcg_at_k(rec_items, rel_items, k)`. |
| B5 | L4.8 md:543–553 | Annotation legend is duplicated/misaligned: letters `#B #C #D` appear twice and the text↔code mapping is scrambled (e.g. `#C Precision@K` and `#D NDCG@K` are swapped relative to the code order). | Renumber the annotations to match code order. |

## C. Book-internal code issues in the runnable listings (4.1–4.7)

Verified by executing each extracted figure in the harness namespace
(`np, pd, csr_matrix, cosine_similarity, ratings, movies`).

| # | Listing | Issue | Proposed patch |
|---|---------|-------|----------------|
| C1 | 4.3 / 4.4 | **Duplicate listing number, missing 4.4.** Two headings are labeled *Listing 4.3* ("Sampling data", md:196; "Generating recommendations from the ALS model", md:221). Prose at md:219 calls the second one "Listing 4.4". No listing is labeled 4.4. | Renumber "Generating recommendations from the ALS model" to **Listing 4.4**. |
| C2 | 4.7 | **Undefined variable `ground_truth`.** Last line calls `calculate_precision_at_k(pop_recs, ground_truth, k=10)`, but the relevance frame was built as `relevance_set` in Listing 4.5; `ground_truth` is never defined. Would `NameError` even with real data. | Change `ground_truth` → `relevance_set` (the notebook uses a consistent `relevance_set`). |
| C3 | 4.3-ALS, 4.5, 4.6, 4.7 | **Column-name / variable drift vs the actual data.** The printed listings use `test`, `train`, `user_id`, `item_id`, `test['user_id']`, `user_item_matrix[user_id]`. The MovieLens frames use `userId`/`movieId`; the split listing (4.1) produces `test_ratings`/`train_val_ratings`, not `test`/`train`. The notebook reconciles this with `user_to_idx`/`movie_to_idx` index maps and `test_sampled`/`train_sampled` — none of which appear in the printed listings, so the printed listings can't run as shown. | Either show the index-mapping setup the notebook uses, or rename the printed variables to the frames the split listing actually produces. |
| C4 | 4.6 | Annotation `#E Calculate precision \= hits / k` mislabels the code, which computes `num_hits / num_recs` (recs-per-user, ≤ k). For a user with < k recommendations this is not hits/k. | Fix the annotation to `hits / (number of recommendations for the user)`, or divide by `k` to match the annotation. |
| C5 | 4.1 | Section 4.3.1 describes a **train / validation / test** split (weeks 1–7 / 8 / 9–10), but Listing 4.1 only codes a two-way `train_val` vs `test` split — no validation split is ever produced. | Add the second split (train vs val) or soften the prose to match the two-way code. |

## D. Non-parsing / non-code listing classification (`compiles=false`)

| Listing | Classification | Detail |
|---------|----------------|--------|
| 4.2 "Overview of splitting the data" | **output-not-code** | Console output (counts, date ranges). `757,052` etc. parse as leading-zero int literals → syntax error. Compile-checked only. |
| 4.8 "Calculate metrics for different K values" | **real-book-syntax-bug** | Genuine Python that fails to parse (`IndentationError`, B1) and, once parsed, is logically wrong (B2–B4). Compile-checked only. |

## E. Import resolution (figures + notebooks)

- `recsys.evaluation.metrics` — imports OK (A1).
- `notebooks/chapter-04/evaluation.ipynb` package imports resolve:
  `from recsys.utils.colab import setup_colab_environment, get_data_path, check_gpu`
  and `from recsys.data import loaders` both import cleanly.
- `implicit.als.AlternatingLeastSquares` (used by Listing 4.3-ALS and the notebook) imports;
  the ch03 dependency `recsys.models.als_model` exposes `ALSModel`. No import drift found for ch04.

### Optional package improvement (not a bug)

| # | Suggestion |
|---|------------|
| D1 | `recsys/evaluation/__init__.py` is empty and nothing imports `metrics.py`. If the intent is for readers to `from recsys.evaluation import ndcg_at_k, mrr_at_k, average_precision_at_k`, re-export them in `__init__.py`. The functions are correct; they're just not surfaced. |

## Manifest summary (8 listings)

| category | count | executable | expected |
|----------|:-----:|:----------:|----------|
| needs-fixture | 2 (4.1, 4.3-sampling) | true | pass |
| needs-training | 4 (4.3-ALS, 4.5, 4.6, 4.7) | false | skip |
| output | 1 (4.2) | false | skip |
| needs-package | 1 (4.8) | false | skip |

- Executable & passing: **2**  · expected-fail: **0**  · non-parsing (`compiles=false`): **2** (4.2, 4.8).
- The 4 `needs-training` listings define clean functions but their module-level tails
  reference eval-pipeline symbols (`test`, `train`, `als_recs`, a trained `als_model`,
  `user_item_matrix`) that no earlier listing or the fixture provides → executable=false,
  compile-checked only. None is a *book bug on the fixture*; they simply need the ch03
  ALS model and real split (except the genuine bugs C2/C3 noted above).

## How this was checked

```bash
# metric equivalence: package == notebook cell 34, byte-for-byte (minus import header)
.venv/bin/python  # diff recsys/evaluation/metrics.py vs evaluation.ipynb cell 34 -> NO DIFF

# numerical correctness + book-vs-package divergence
.venv/bin/python  # ndcg_at_k == manual DCG/IDCG == sklearn.ndcg_score (0.65092...)
                  # AP denominator: package min(#rel,k)=0.556 vs L4.8 #hits=0.833
                  # L4.8 ndcg_score([ideal],[actual])=0.769 vs correct 0.651

# executability of every figure in the shared harness namespace
.venv/bin/python  # 4.1 pass, 4.3-sampling pass; 4.3-ALS/4.5/4.6/4.7 NameError; 4.8 IndentationError

# harness stays green
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch04]" -q   # 1 passed
.venv/bin/pytest "tests/figures/test_figures.py::test_listing_compiles" -k ch04 -q  # 6 passed, 2 xfailed
```

_All findings are reported, not fixed. Section-B/C items belong to the author's chapter
text and listings; the package module (`recsys/evaluation/metrics.py`) is correct and was
not modified._
