# Chapter 2 — figure ↔ package consistency findings

Goal: verify the shipped `recsys` package works correctly with the book's chapter-2
listings, and surface inconsistencies. All items below are **verified by executing the
installed package** (`pip install -e .`) against the listings, not by inspection alone.

## A. Book listing ↔ package drift (printed code doesn't match the library)

| # | Where | Issue | Reality in package |
|---|-------|-------|--------------------|
| A1 | L2.15 | imports `recsys.retrievals` / `recsys.filters` / `recsys.scorers` / `recsys.rankers` | modules don't exist; real paths are `recsys.fourstage_recsys.{retrieval,filtering,scoring,ordering}.*` — **ImportError** |
| A2 | L2.15 | imports classes `ItemKNNRetrieval`, `HistoryFilter`, `PopularityScorer`, `WeightedRanker` | class names are `HistoryFiltering`, `PopularityScoring` (retrieval/ranker names OK) |
| A3 | L2.15 | `recommender.recommend(user_id='123', k=10)` | real signature is `recommend(context: RecommendationContext, debug=False)` — no `user_id`/`k` kwargs; **TypeError** |
| A4 | L2.3, L2.18 (inline) | retrieval returns `list[dict]` with keys `movie_id` / `similarity` / `content_similarity` | package retrieval returns `list[ScoredItem]` with `item_id` / `scores{}` — a reader moving from the inline code to the framework meets a different data shape |

The chapter-2 **notebooks** import the correct `recsys.fourstage_recsys.*` paths, so the
drift is specific to the printed listing 2.15.

## B. Package internal inconsistencies (the "main code" does not work correctly)

| # | Where | Issue | Evidence |
|---|-------|-------|----------|
| B1 | `pipeline.recommend` ↔ `HistoryFiltering.filter` | pipeline calls `self.filter.filter(candidates, context)`, but `HistoryFiltering.filter(candidates, user_id)` expects a **user id**. A `RecommendationContext` is compared against `ratings['userId']`, matches nothing, so **no watched items are filtered**. | `filter(cands, context)` kept all 10 candidates incl. watched {3,5,7,8,9,11,12}; `filter(cands, user_id)` correctly kept only {10,1,4}. |
| B2 | `Retrieval` ABC ↔ `ItemKNNRetrieval` | abstract base declares `retrieve(self, context)`; `ItemKNNRetrieval` never implements it — it defines `retrieve_similar_items(seed_ids, k)`, which is what the pipeline actually calls. The ABC contract is unsatisfied/misleading. | `ItemKNNRetrieval` instantiates despite not implementing the abstract `retrieve`. |
| B3 | `RecommendationContext.seed_items: List[str]` ↔ `ScoredItem.item_id: str` ↔ integer `movieId` | Type drift across the core models. With **int** movie ids, `ItemKNNRetrieval.retrieve_similar_items` raises `pydantic.ValidationError` (item_id expects str, got `np.int64`). With **str** seeds, nothing matches the int-keyed `movie_to_idx`, so retrieval returns `[]`. Either way the wired pipeline yields **0 recommendations out of the box**. | reproduced both branches. |
| B4 | `ItemKNNRetrieval.retrieve_similar_items` | return annotation says `-> list[dict]` but it returns `list[ScoredItem]`. | source + runtime. |
| B5 | `Filtering.filter` ABC ↔ `HistoryFiltering.filter` | Liskov mismatch: ABC signature is `filter(scored_items, context: RecommendationContext)`; override is `filter(candidates, user_id)` (untyped, different meaning). Root cause of B1. | signatures differ. |
| B6 | packaging | `recsys.data.loaders` imports `requests`, which is not declared in `requirements.txt`. | import fails on a clean env. |

## How this was checked

```bash
uv venv .venv
uv pip install --python .venv -e . --no-deps
uv pip install --python .venv pandas scipy scikit-learn pydantic pytest
.venv/bin/pytest tests/figures        # compile + execution harness
# plus the introspection/execution probes recorded in this report
```

_These are reported, not fixed — they belong to the author's code and several have more
than one reasonable resolution (e.g. B1/B5 could be fixed on the pipeline side or the
filter side)._
