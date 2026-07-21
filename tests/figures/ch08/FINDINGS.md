# Chapter 8 — figure ↔ package consistency findings

Goal: verify the shipped `recsys.semantic_ids.*` package works correctly with the book's
chapter-8 listings (RQ-VAE / Semantic IDs), and surface inconsistencies. All items below
are **verified by executing the installed package** (`.venv`, editable `recsys`, torch +
sklearn + sentence-transformers + mlflow) against the listings — not by inspection alone.

The chapter is unusual: **16 of 17 listings are torch/package method excerpts** taken
verbatim from `recsys/semantic_ids/`. The chapter explicitly says the listings are trimmed
("we omit these from the listing for clarity"), so most book↔package differences are
*intentional subsets*. The findings separate those documented omissions from **undocumented
drift** (mostly default-argument values) and from **real package bugs**.

## A. Book listing ↔ package drift (printed code doesn't match the library)

Legend: **subset** = book is an intentional/documented trim of the shipped method;
**DRIFT** = an undocumented divergence a reader would actually hit.

| # | Listing | Package symbol | Issue | Proposed patch |
|---|---------|----------------|-------|----------------|
| A1 | L8.5 | `RQVAE.__init__` | **DRIFT** default `codebook_sizes=[16, 32, 128]` in the book vs **`[8, 64, 512]`** in `rqvae.py`. A reader calling `RQVAE(dim, embed)` with no third arg gets different levels/sizes than the book claims. | Change the package default to `[16, 32, 128]` to match the book (and the pipeline's own `[8, 32, 128]`), or update the listing to print `[8, 64, 512]`. |
| A2 | L8.7 | `SemanticIDPipeline.__init__` | **DRIFT** in all three defaults: book `internal_dim=512` / `usage_loss_weight=2.0` / `codebook_sizes=[16, 32, 128]` vs package **`internal_dim=64`** / **`usage_loss_weight=10.0`** / **`codebook_sizes=[8, 32, 128]`**. The chapter prose specifically justifies `internal_dim=512` ("projects … down to this 512-dimensional space"), so the shipped `64` contradicts the text. | Set package defaults to `codebook_sizes=[16, 32, 128], internal_dim=512, usage_loss_weight=2.0`, or correct the listing + prose. |
| A3 | L8.10 | `SemanticIDPipeline.train` | **DRIFT** default `epochs=500` in the book vs **`epochs=100`** in the package. (L8.8 also calls `train(..., epochs=500)`, overriding it, so that call is fine.) | Align the package default to `epochs=500`, or change the listing signature to `epochs=100`. |
| A4 | L8.1 | `VectorQuantizerEMA.__init__` | **subset (undocumented)** — book omits the package's `register_buffer('_code_usage_count', torch.zeros(num_embeddings))` (used by `reset_dead_codes`). Not called out. | Add the line to the listing, or note dead-code buffers are omitted. |
| A5 | L8.3 | `VectorQuantizerEMA.forward` | **subset** — book drops `_code_usage_count` tracking + the random DEBUG print, and renames local `usage_loss_val` → `usage_loss`. Logic identical. The rename would break `test_semantic_ids.py::test_usage_loss_formula_is_fixed`, which greps the *package* source for the literal `usage_loss_val = 1.0 - entropy_ratio` (package passes; book text wouldn't). | Harmless; leave as documented-style trim. |
| A6 | L8.5 | `RQVAE.forward` | **subset (documented)** — book `forward(self, x)` omits variance-loss + residual-loss terms and the `return_usage_stats`/`variance_weight` params; chapter explicitly says these are "omitted … for clarity." | None needed. |
| A7 | L8.10 | `SemanticIDPipeline.train` | **subset** — book drops the package's `diversity_weight` param + diversity loss, mlflow logging, perplexity/collapse checks. With `diversity_weight=0` (package default) the package loss reduces exactly to the book's `recon + vq + 0.5*cos`. | None needed. |
| A8 | L8.11 | `SemanticIDPipeline.inference` | **subset / behavioral note** — book sorts by `['semantic_id','title']` and groups by the raw tuple column; package builds a temp `_semantic_id_str` string column, sorts/groups on that, then drops it. Equivalent `leaf_id`/`final_id` for tuple keys. | None needed. |
| A9 | L8.2/8.9/8.15/8.16 | `init_codebook` / `initialize_data_with_embeddings` / `evaluate_reconstruction` / `evaluate_clustering` | **subset** — each book listing matches its package function's core logic exactly; only diagnostic `print(...)` (and, for L8.16, an `n_unique < 2` guard) are trimmed. Signatures match. | None needed. |

No **import drift** in ch08: the chapter-8 notebook imports
`recsys.semantic_ids.{semantic_ids_pipeline,evaluations,utils,training}` and
`recsys.data.loaders` / `recsys.utils.colab` — **all resolve** and expose the names used.

## B. Package internal inconsistencies (the shipped code has bugs)

| # | Where | Issue | Evidence |
|---|-------|-------|----------|
| B1 | `evaluations.py` `test_semantic_coherence` | **Early-return bug.** The `# Summary` block and `return results` are indented **inside** the `for` loop, so the function returns after processing **only the first** of six `test_pairs`. Downstream, `evaluate_semantic_ids` computes `coherence_pass_rate` over a single result → semantic-coherence scoring is effectively broken. Not visible in L8.17 (prints only the `test_pairs` list). | Ran it on a df with all nine titles → **`len(results) == 1`** (expected 6); printed "Passed: 1/1". |
| B2 | `training.py` `__main__` | **Demo raises immediately.** `simple_data` has 5 `item_id`/`genres`/`format` but only **4 `title`s** (and 5 `description`s) → `pd.DataFrame(simple_data)` raises `ValueError: All arrays must be of the same length`. | Reproduced. |
| B3 | `training.py` `__main__` | **Unpack mismatch.** `run_pipeline_debug` returns 3 values (`df_enriched, pipeline, data_tensor`) but `__main__` does `df_enriched, pipeline = run_pipeline_debug(...)` → `ValueError: too many values to unpack`. (Only reached if B2 fixed; both in `if __name__ == "__main__"`, so the importable API is unaffected.) | Confirmed return arity = 3 via `inspect.getsource`. |
| B4 | `tests/test_semantic_ids.py` `test_no_pycache_in_semantic_ids` | **Test raises `NameError`.** References `HAS_PYTEST`, never defined/imported; branch taken whenever `recsys/semantic_ids/__pycache__` exists (it does after any import). `pytest.warn` is also not a real API. | `.venv/bin/pytest tests/test_semantic_ids.py` → **6 passed, 1 failed** (`NameError: name 'HAS_PYTEST' is not defined`). |

**Proposed patches**

- **B1** — dedent the `# Summary` block and `return results` by one level so they sit in the
  function body, after the `for` loop (currently loop-body indent → move to function-body
  indent). Highest-value fix in the chapter.
- **B2** — add a fifth title (e.g. `"Legends of the Sea"` is described but has no title slot)
  so all arrays are length 5.
- **B3** — `df_enriched, pipeline, data_tensor = run_pipeline_debug(...)`.
- **B4** — define `HAS_PYTEST = True` at module top, or replace the branch with
  `warnings.warn(msg)` (drop the `HAS_PYTEST`/`pytest.warn` guard entirely).

## C. The shipped semantic-IDs code otherwise works

Verified on tiny toy tensors (CPU):

- `VectorQuantizerEMA(num_embeddings=8, embedding_dim=16, usage_loss_weight=2.0)` in
  `.train()` on `torch.randn(20, 16)` → `(quantized[20,16], loss, idx[20])` with **loss ≥ 0**
  (the positive usage-loss fix from L8.3 holds). `init_codebook` runs K-means and copies
  centroids without error.
- `RQVAE(input_dim=32, embed_dim=16, codebook_sizes=[4, 8])` → forward returns
  `reconstructed[12,32]`, `total_loss`, `codes[12,2]`; both book-style `forward(x)` and
  package `forward(x, variance_weight=1.0)` work; output shapes correct.
- `tests/test_semantic_ids.py` — the **6 substantive tests pass** (`TestVectorQuantizerEMA`
  ×3, `TestRQVAE` ×3: loss-positivity, output shapes, variance regularization). Only the
  cache-detection test (B4) fails, on its own `NameError`, not on model behavior.
- `SemanticIDPipeline`, `train`, `inference`, `initialize_data_with_embeddings`,
  `prepare_data`, and all six `evaluations.py` functions import and introspect cleanly.
  End-to-end pipeline runs weren't executed (SentenceTransformer download + ~25 min
  training), but every unit is exercised individually.

## D. Non-parsing listings (compiles=false) — classification

| Listing | Why it doesn't parse | Classification |
|---------|----------------------|----------------|
| L8.4 `_update_codebooks` | Printed at class-body indentation (2-space-indented `def`), so the standalone `.py` starts with `unexpected indent`. Body matches the package method **exactly**. | **method-excerpt / not-standalone** (verbatim indent artifact) — *not* a reader-facing syntax bug. |
| L8.17 `test_pairs` | Only the `test_pairs` list from `test_semantic_coherence`, printed with a leading space → `unexpected indent`. | **fragment / not-standalone** (verbatim indent artifact). Values match the package; enclosing function has bug B1. |

Neither is a genuine Python syntax error a reader would type — both are extraction artifacts
of printing an indented method fragment on its own. Recorded with `compiles=false` so the
compile harness xfails them.

## E. Executability in the harness

The shared `chapter_namespace` provides only `np, pd, csr_matrix, cosine_similarity, ratings,
movies` — **no torch/nn/F/KMeans/SentenceTransformer**. Therefore:

- **executable=true (2):** L8.6 `prepare_data` (pure pandas def) and L8.16 `evaluate_clustering`
  (imports `sklearn.metrics`, defines a torch-free function). Both **verified to exec without
  raising → expected=pass**.
- **executable=false (15):** everything else — `needs-package` for the torch method/class
  excerpts (L8.1–8.5, 8.7, 8.9–8.11, 8.15, 8.17), `needs-real-data`/`output` for inspection
  snippets referencing an undefined enriched `df` (L8.8, 8.12–8.14). Several torch *method*
  defs (8.2/8.3/8.9/8.10/8.11/8.15) and the pipeline *class* (8.7) technically bind when
  exec'd (a bare `def`/`class` body doesn't need torch until called), but they're package code
  that can't be exercised in the harness, so classified `needs-package` and left non-executable.

`.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch08]"` → **green**;
the two executable listings pass as expected, the rest are skipped and covered by the compile
test.

## How this was checked

```bash
.venv/bin/pytest tests/test_semantic_ids.py -q            # 6 passed, 1 failed (B4)
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch08]" -q  # green
# plus introspection/execution probes recorded above:
#  - inspect.signature on every VQ/RQVAE/Pipeline/eval symbol (default-arg drift A1–A3)
#  - toy VectorQuantizerEMA / RQVAE forward on random tensors (Section C)
#  - test_semantic_coherence on a 9-title df -> len(results)==1 (B1)
#  - pd.DataFrame(simple_data) / return-arity of run_pipeline_debug (B2, B3)
#  - exec of each figure in a mimicked chapter_namespace (Section E)
#  - importlib resolution of every notebooks/chapter-08 import (all OK)
```

_These are reported, not fixed — they belong to the author's code. The highest-value fix is
**B1** (semantic-coherence early return); the default-value drifts **A1–A3** are the most
likely to confuse a reader following the prose._
