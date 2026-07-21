# Patches applied on `with-figures-fixes`

This branch applies the **package-side** proposed patches from `CONSISTENCY_REPORT.md`
(the P-series). The book-listing patches (B-series) are edits to `chapters/*.md`, which
live outside this git repo, so they are **not** included here — see the per-chapter
`FINDINGS.md` for those.

All fixes were verified: `37/37` recsys modules import, the assembled four-stage
pipeline returns real, correctly-filtered recommendations, `tests/test_semantic_ids.py`
is `7 passed`, and the figure harness stays green (`126 passed, 17 xfailed`).

| id | file | fix |
|----|------|-----|
| P1 | `retrieval/sentence_transformer_retrieval.py` | `RecommenderContext` → `RecommendationContext` (import + annotation) — module now imports |
| P2 | `models/recsys_speaking_LLM/multi_token_prediction.py` | removed stray ```` ``` ```` fence, added `np`/`torch`/`F` imports, guarded the demo under `__main__` — module now imports |
| P3/P7 | `fourstage_recsys/filtering/history_filtering.py` | `filter(candidates, context)` now reads `context.user_id` (accepts a `RecommendationContext` or a raw id) so the pipeline actually filters watched items |
| P5 | `retrieval/itemknn_retrieval.py`, `filtering/history_filtering.py`, `scoring/popularity.py` | normalized item/user ids to `str` end-to-end so `ScoredItem.item_id: str` validates and history/popularity lookups match; pipeline no longer returns `[]`/raises |
| P6 | `retrieval/itemknn_retrieval.py` | return annotation `-> list[dict]` → `-> list[ScoredItem]` |
| P8 | `requirements.txt` | added `requests`, `openai` |
| P9 | `retrieval/vectordb_retrieval.py`, `retrieval/database_retrieval.py` | `ScoredItem(score=…)` → `scores=…`; `context.item_id` → `context.seed_items[0]` |
| P10 | `semantic_ids/evaluations.py` | dedented the summary/`return` out of the `for` loop in `test_semantic_coherence` (was returning after 1 of 6 pairs) |
| P11 | `semantic_ids/training.py` | added the missing 5th `title`; unpack `run_pipeline_debug` into 3 values |
| P12 | `tests/test_semantic_ids.py` | replaced undefined `HAS_PYTEST`/`pytest.warn` with `warnings.warn` |
| P13 | `agentic/guardrails.py` | `GroundingValidator` now strips the year on both sides so real catalog titles are no longer flagged as hallucinations |
| P15 | `models/recsys_speaking_LLM/sampled_softmax_loss.py` | restored the `embedding` constructor arg; defined `target_positions` |

Not applied (deliberately):
- **P4** (`Retrieval` ABC declares `retrieve` but retrievers implement `retrieve_similar_items`) — left as an interface-design decision for the author; the pipeline works via `retrieve_similar_items` regardless.
- **B-series** book-listing edits — they belong in `chapters/*.md` (outside this repo).
