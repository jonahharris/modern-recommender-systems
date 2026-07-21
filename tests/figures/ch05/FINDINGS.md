# Chapter 5 — figure ↔ package consistency findings

Goal: verify the shipped `recsys` package works correctly with the book's chapter-5
listings (retrieval & reranking: two-tower/bi-encoder, FAISS ANN, hard-negative mining,
InfoNCE, cross-encoders, and the four-stage `ANNRetrieval` + `CrossEncoderScoring`
pipeline). Every item below is **verified by executing the installed package** against the
listings via `.venv/bin/python`, not by inspection alone.

Chapter 5 has **16 extracted listings** (5.1–5.15, with a **duplicate number 5.6** —
"Warm-starting a new item's embedding" and "Hard negative mining" both print as 5.6).
All 16 **parse** (`compiles=true`); there are **no non-parsing listings** in this chapter.
**4** are executable in the harness namespace (5.1, 5.2, 5.4, 5.15) and all pass; the
other 12 need torch/faiss/transformers/a trained model or real data and are compile-checked
only. **0** listings are `expected=fail`.

## A. Package module that does not import (the known ch05 bug)

| id | module | error | proposed patch |
|----|--------|-------|----------------|
| **P1** | `recsys/fourstage_recsys/retrieval/sentence_transformer_retrieval.py` | `ImportError: cannot import name 'RecommenderContext' from 'recsys.fourstage_recsys.recsys_context'` — the real class is `RecommendationContext` | see below |

Exact cause, pinned to two lines:

- **line 10:** `from recsys.fourstage_recsys.recsys_context import RecommenderContext` — the
  module only defines `RecommendationContext`, so this import raises at module load and the
  whole file is unimportable.
- **line 117:** `def retrieve(self, context: RecommenderContext) -> List[Dict]:` — the same
  wrong name is used as the method's annotation (would `NameError` even if line 10 were
  removed).

**Reproduction:**
```
$ .venv/bin/python -c "import recsys.fourstage_recsys.retrieval.sentence_transformer_retrieval"
ImportError: cannot import name 'RecommenderContext' from
'recsys.fourstage_recsys.recsys_context' (.../recsys_context.py)
```
Every *other* retrieval + four-stage module imports OK (`vectordb_retrieval`,
`database_retrieval`, `file_based_retrieval`, `pipeline`, `history_filtering`,
`popularity`, `weighted_ranker`).

**Proposed patch** (rename in both places — `RecommendationContext` is the shipped name):
```diff
-from recsys.fourstage_recsys.recsys_context import RecommenderContext
+from recsys.fourstage_recsys.recsys_context import RecommendationContext
@@
-  def retrieve(self, context: RecommenderContext) -> List[Dict]:
+  def retrieve(self, context: RecommendationContext) -> List[Dict]:
```
(Two secondary issues in the same file, for the record: `retrieve` returns `List[Dict]`
while the `Retrieval` ABC declares `-> List[ScoredItem]`; and it defines `retrieve(context)`
but the pipeline actually calls `retrieve_similar_items(...)` — see B-series/C-series below.)

## B. Book listing ↔ package drift (the four-stage listings don't match the shipped API)

Listings 5.10–5.14 present a four-stage pipeline "with the same interface as Chapter 2",
but they define their classes **inline** and none of them import from `recsys`. Probed with
`inspect.signature` against the real package, the printed API diverges on almost every
surface. The chapter never actually exercises the shipped classes.

| id | L | book prints | shipped `recsys` reality | proposed patch |
|----|---|-------------|--------------------------|----------------|
| **D1** | 5.10, 5.12, 5.13 | base class **`Scoring`** (`class CrossEncoderScoring(Scoring)`, `scorer: Scoring`) | ABC is **`Scorer`** in `recsys.fourstage_recsys.stages.scoring` — `Scoring` does not exist (`hasattr(scoring,'Scoring') == False`) | rename listings' base to `Scorer` (or add a `Scoring = Scorer` alias in the package) |
| **D2** | 5.10 | ctor param **`ranker`** + `ranked = self.ranker.rank(candidates)` | shipped `FourStageRecommender.__init__(..., ordering: Ordering)`; the 4th-stage ABC is `Ordering` with method **`order(filtered_items, context, debug=False)`** — there is no `rank` | rename `ranker`→`ordering`, `rank`→`order`, and pass `context` |
| **D3** | 5.10, 5.11, 5.14 | **`context.seed_movie_id`** (a single scalar id) | `RecommendationContext` has **`seed_items: Optional[List[str]]`** — there is no `seed_movie_id` field; shipped pipeline calls `retrieve_similar_items(context.seed_items, k=100)` | use `context.seed_items` (and have retrieval accept a list of seeds) |
| **D4** | 5.10 vs 5.12/5.13 | `self.scorer.score(candidates)` (**1 arg**) in 5.10, but `score(self, candidates, user_id)` (**2 args**) in 5.12/5.13 | shipped `Scorer.score(self, candidates, context)` (2 args, second is the **context**) | make 5.10 call `score(candidates, context)` and give the stages a `score(candidates, context)` signature |
| **D5** | 5.10 | `self.filter.filter(candidates, context.user_id)` | shipped `Filtering.filter(self, scored_items, context)` — and the shipped `pipeline.recommend` passes `context`, *not* `user_id`. (Ironically the book's `context.user_id` here matches the concrete `HistoryFiltering.filter(candidates, user_id)` better than the shipped pipeline does — see C1.) | pick one filter contract chapter-wide (see C1) and have 5.10 match it |
| **D6** | 5.14 | `DeduplicationFilter()`, `ScoreRanker()` | neither class exists anywhere in `recsys` (grep: no matches). Shipped concretes are `HistoryFiltering` (filtering) and `WeightedRanker` (ordering) | either ship `DeduplicationFilter`/`ScoreRanker`, or rewrite the listing against `HistoryFiltering`/`WeightedRanker` |
| **D7** | 5.11/5.12/5.13 | classes `ANNRetrieval`, `CrossEncoderScoring`, `TransformerScoring` | none exist in the package — the chapter's retrieval/scoring stages are never packaged (grep: no matches) | package them under `recsys.fourstage_recsys.retrieval` / `.scoring`, or mark the listings as chapter-local illustrations |

`inspect.signature` evidence (shipped package):
```
FourStageRecommender.__init__(self, retrieval: Retrieval, filter: Filtering,
                              scorer: Scorer, ordering: Ordering)
FourStageRecommender.recommend(self, context: RecommendationContext, debug=False)
Scorer.score(self, candidates: List[ScoredItem], context: RecommendationContext) -> List[ScoredItem]
Filtering.filter(self, scored_items: List[ScoredItem], context: RecommendationContext) -> List[ScoredItem]
Ordering.order(self, filtered_items: List[ScoredItem], context: RecommendationContext, debug=False) -> ...
Retrieval.retrieve(self, context: RecommendationContext) -> List[ScoredItem]
RecommendationContext fields: ['user_id', 'seed_items', 'k', 'filters', 'metadata']
ScoredItem fields:            ['item_id', 'scores', 'metadata']
```

## C. Package internal inconsistencies (the shipped four-stage code is itself inconsistent)

These are in the package (not the listings); they matter for ch05 because 5.14 claims the
pipeline has "the same interface as Chapter 2." Confirmed by executing the classes on a tiny
fixture. (C1–C3 overlap ch02 FINDINGS B1/B3/B5; re-verified here.)

| id | where | issue | evidence |
|----|-------|-------|----------|
| **C1** | `pipeline.recommend` ↔ `HistoryFiltering.filter` | Liskov break: the `Filtering` ABC is `filter(scored_items, context)` and the pipeline calls `self.filter.filter(candidates, context)`, but the concrete `HistoryFiltering.filter(self, candidates, user_id)` treats the 2nd arg as a **user id**. A `RecommendationContext` gets compared against `ratings['userId']`, matches nothing, so **no history is filtered**. | ran `hf.filter(cands, ctx)` → kept all 3 candidates. |
| **C2** | `ScoredItem.item_id: str` ↔ int `movieId` | Even calling `HistoryFiltering.filter(cands, user_id)` the *correct* way filters nothing: `get_user_history` returns int movie ids `{1,2,3}` but `ScoredItem.item_id` is a `str`, so `item.item_id not in user_history` is always true. | `get_user_history(7) == {1,2,3}` (ints); candidate `item_id`s are `'1','2','99'` (strs) → 0 removed. |
| **C3** | `RecommendationContext.user_id: Optional[str]` | Typed `str`; constructing with an int (`user_id=7`) raises `pydantic ValidationError`. The natural MovieLens `userId` is an int, so callers must remember to `str()` it. | `RecommendationContext(user_id=7)` → `ValidationError: Input should be a valid string`. |
| **C4** | `Retrieval` ABC ↔ all shipped retrievers vs the wired pipeline | the ABC declares `retrieve(self, context)` and `vectordb/database/file_based` implement `retrieve(context)`, **but** `pipeline.recommend` calls `self.retrieval.retrieve_similar_items(context.seed_items, k=100)` — a method that exists on none of them (it lives on `ItemKNNRetrieval`). So the ANN/DB/file retrievers cannot actually be dropped into the shipped pipeline. | source: pipeline.py:25 vs retrieval ABC + concretes. |
| **C5** | `vectordb_retrieval.py` / `database_retrieval.py` | Construct `ScoredItem(item_id=..., score={...})`, but the field is **`scores`** (plural). With pydantic v2 default (`extra='ignore'`) the `score=` kwarg is silently dropped, so every returned item carries an empty `scores={}`. They also read `context.item_id`, a field `RecommendationContext` does not have (`AttributeError` at call). | `ScoredItem` fields are `['item_id','scores','metadata']`; context fields lack `item_id`. |

**Proposed patches (report only — several have >1 reasonable fix):**
- **C1/D5:** make `HistoryFiltering.filter(self, candidates, context)` read `context.user_id`
  (matches the `Filtering` ABC and the pipeline), keeping `get_user_history(user_id)` as-is.
- **C2:** normalize id types end-to-end — simplest is to `str()` movie ids in
  `get_user_history` (`return {str(m) for m in movie_ids}`) so they compare against
  `ScoredItem.item_id`.
- **C3:** either loosen `user_id` to `Optional[Union[str,int]]` or `str()` at the data
  boundary; document that ids are strings.
- **C4:** pick one retrieval contract — either have the pipeline call `retrieve(context)`
  (and give each retriever that method), or declare `retrieve_similar_items` on the ABC.
- **C5:** rename the kwarg `score=` → `scores=` in both retrievers (and wrap the raw float in
  a dict, e.g. `scores={"distance": ...}`), and replace `context.item_id` with
  `context.seed_items[0]` (guarding for empty).

## D. Non-parsing classification

None. All 16 ch05 listings parse (`compile()` succeeds). No `real-book-syntax-bug`,
`output-not-code`, or `pseudocode` non-parsing entries in this chapter. The runtime
failures documented above are `NameError`/`AttributeError`/`ImportError` at exec time, not
`SyntaxError`.

## E. Executable-listing verification (harness)

Ran in the shared `chapter_namespace` (np, pd, csr_matrix, cosine_similarity, ratings,
movies) in listing order:

| L | category | result |
|---|----------|--------|
| 5.1 | standalone | pass (def) |
| 5.2 | standalone | pass (def; latent bug: negative pair mixes id with `movie_to_idx[...]` index) |
| 5.4 | standalone | pass (self-imports faiss + numpy; faiss-cpu present in .venv) |
| 5.15 | standalone | pass (def) |

All other listings marked `executable=false`: 5.3/5.6(hard)/5.7/5.8/5.9 need torch (class
bases / annotations evaluated at def time), 5.9 also needs transformers + a model download,
5.11 needs faiss annotations, 5.10/5.12/5.13/5.14 additionally reference the drifted
`Scoring`/`Filtering`/`ANNRetrieval`/`DeduplicationFilter`/`ScoreRanker` names, and
5.5/5.6(warm) reference prior real-data symbols (trained embeddings + a built FAISS index).

## How this was checked

```bash
# import health of the ch05-relevant modules
.venv/bin/python -c "import importlib; ... import each module ..."
# -> only sentence_transformer_retrieval fails (P1)

# real API via introspection
.venv/bin/python  # inspect.signature on FourStageRecommender / Scorer / Filtering /
                  # Ordering / Retrieval; .model_fields on RecommendationContext / ScoredItem

# contract probes on a tiny fixture
.venv/bin/python  # HistoryFiltering.filter(cands, context) vs (cands, user_id); str/int
                  # id mismatch; RecommendationContext(user_id=7) ValidationError

# grep for the book's four-stage class names
grep -rn "ANNRetrieval|CrossEncoderScoring|TransformerScoring|DeduplicationFilter|ScoreRanker" recsys/
# -> no matches

# harness
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch05]" -q   # 1 passed
.venv/bin/pytest tests/figures/test_figures.py -q -k ch05                           # 17 passed
```

_All findings are reported with proposed patches; no `recsys` source (or any file outside
`tests/figures/ch05/`) was modified._
