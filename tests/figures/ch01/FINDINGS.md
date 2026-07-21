# Chapter 1 — figure ↔ package consistency findings

Goal: verify the chapter-1 listings actually run and surface inconsistencies. Chapter 1
is **self-contained inline Python** (a comic-book catalog: uniform / content-based /
collaborative baselines) and imports **no `recsys` modules**, so there is no
package-drift to report. The audit therefore focuses on (a) whether each inline listing
runs, (b) internal book bugs, and (c) whether the chapter-1 notebook's imports resolve.
Every item below is **verified by executing the code**, not by inspection alone.

## A. Package-module import failures

None — no listing and no notebook cell in this chapter imports `recsys`. Not applicable.

## B. Book listing ↔ package drift

None — the chapter uses only `numpy`/`pandas`; nothing references the `recsys` package.
Not applicable.

## C. Book-internal code issues

All seven listings are pure `def`/dict-literal blocks, so **executing** each listing
succeeds (function bodies don't run). The bugs below are therefore **latent** — they fire
only when the reader *calls* the function, exactly as the book intends them to be used.
Each was reproduced by calling the function in the harness namespace (`np`, `pd` seeded).

| # | Where | Issue | Evidence (verified) | Proposed patch |
|---|-------|-------|---------------------|----------------|
| C1 | L1.2 `uniform_recommender` | body returns `[catalog[key] ...]` but the catalog dict (L1.1) is named `catalogue`; `catalog` is never defined | `uniform_recommender()` -> `NameError: name 'catalog' is not defined` | change `return [catalog[key] for key in random_keys]` -> `return [catalogue[key] for key in random_keys]` (this is what the notebook does) |
| C2 | L1.6 `get_recommendations` | same `catalog`/`catalogue` mismatch on its final line | `get_recommendations(mat, 0, sim_users)` -> `NameError: name 'catalog' is not defined` | change `return [catalog[id] for id in rec_ids]` -> `return [catalogue[id] for id in rec_ids]` |
| C3 | L1.7 `find_similar_items` | `np.dot(matrix.T[item_id], matrix.T)` — the second operand should be `matrix`. As written it dots a `(n_users,)` vector with an `(n_items, n_users)` matrix, so inner dims disagree whenever `n_users != n_items` | `find_similar_items(mat, 9)` on a 5x10 matrix -> `ValueError: shapes (5,) and (10,5) not aligned` | change `item_sim = np.dot(matrix.T[item_id], matrix.T)` -> `item_sim = np.dot(matrix.T[item_id], matrix)` (this is what the notebook's cell 28 does) |
| C4 | L1.2 / L1.6 / L1.7 | all use `np.*` but Chapter 1's printed listings **never `import numpy as np`**; a reader copying the listings verbatim hits `NameError: name 'np' is not defined` before the bugs above | listings run in the harness only because the fixture seeds `np`; a clean session fails | add `import numpy as np` to the first listing that uses it (the notebook imports it in its setup cell) |
| C5 | L1.5 — stated output is wrong | prose (after L1.5) and the notebook's `# Output:` comment claim `content_based_recommender(1, k=3)` -> `['The Smurfs', 'Thorgal', 'Spirou and Fantasio']`; the code is correct but its deterministic output differs | verified actual = `['The Smurfs', 'Spirou and Fantasio', 'Corto Maltese']`. Jaccard(Asterix, Thorgal)=1/9~0.111 ranks **below** Spirou/Corto/Little Frog (all 1/8=0.125), so Thorgal cannot be in the top 3 | correct the printed output to `['The Smurfs', 'Spirou and Fantasio', 'Corto Maltese']` (and fix the follow-on prose "all three share themes like humor, fantasy, and adventure" — Corto Maltese shares only "historical" with Asterix). No code change needed |

Note: L1.4 Jaccard is **correct** — verified `sim(1,2)=1/9~0.111` (share only "humor") and
`sim(1,3)=2/8=0.25` (share "humor"+"magic"), both matching the book's worked example.

## D. Non-parsing-listing classification

None. All 7 listings have `compiles=true` and parse cleanly — no `real-book-syntax-bug`,
`output-not-code`, or `pseudocode` blocks were extracted for this chapter.

## E. Notebook import resolution

`notebooks/chapter-01/Modern_Recommender_systems_chapter_1.ipynb` imports resolve cleanly
in the venv: `numpy, pandas, collections, matplotlib, seaborn, sentence_transformers,
plotly, plotly.express` all import OK. The notebook is a **corrected** copy of the
listings (it uses `catalogue` in L1.2/L1.6 and `np.dot(matrix.T[item_id], matrix)` in the
item-CF cell), so the printed-book bugs C1–C4 do not reproduce there.

Minor, out of scope: the notebook's last code cell defines `PopularityRetrieval(FileBasedRetrieval)`
and annotates `context: RecommendationContext`, but neither `FileBasedRetrieval` nor
`RecommendationContext` is imported/defined in the notebook — running that cell would
`NameError`. It appears to be an orphan cell pasted from a later chapter, unrelated to the
Chapter 1 narrative.

## How this was checked

```bash
# execute every ch01 listing cumulatively in the harness namespace, then CALL the
# functions to surface the latent bugs
.venv/bin/python  # cumulative exec of tests/figures/ch01/listing_*.py + probe calls
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch01]" -q   # green
.venv/bin/pytest "tests/figures/test_figures.py::test_listing_compiles" -k ch01 -q  # 7 passed
```

All 7 listings are marked `executable=true, expected=pass` because the harness executes
only the `def`/literal top level (which succeeds); the C1–C5 bugs are latent and are
documented here rather than encoded as `expected=fail`. _These are reported, not fixed —
they belong to the author's book code._
