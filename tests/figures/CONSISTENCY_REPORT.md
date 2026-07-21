# recsys ↔ book-figures consistency report

**Goal:** verify the shipped `recsys` package works correctly with the book's code
listings, and surface inconsistencies. The package is the thing under test; the
extracted "figure" files (one per printed Listing) are the vehicle. **Every finding
below was verified by executing the shipped code** in a full venv — not by inspection.
Findings are reported with **proposed patches**; no `recsys` source has been changed.

## Scope & status

- **134 listings** extracted across 9 drafted chapters → `tests/figures/chNN/listing_*.py`
- Per-chapter deep audit **complete for all 9 chapters** → each has its own
  `chNN/FINDINGS.md` with detailed evidence + proposed patches.
- Test harness green: **126 passed, 17 xfailed** (`.venv/bin/pytest tests/figures`).

| chapter | listings | executable | expected-fail | non-parsing | FINDINGS.md |
|---------|:--------:|:----------:|:-------------:|:-----------:|:-----------:|
| ch01 Concepts        |  7 |  7 | 0 | 0 | ✓ |
| ch02 Simple model    | 22 | 14 | 1 | 0 | ✓ |
| ch04 Evaluation      |  8 |  2 | 0 | 2 | ✓ |
| ch05 Retrieval/rerank| 16 |  4 | 0 | 0 | ✓ |
| ch06 Sequential      | 11 |  5 | 0 | 1 | ✓ |
| ch08 Semantic IDs    | 17 |  2 | 0 | 2 | ✓ |
| ch09 Generative      | 26 | 11 | 0 | 3 | ✓ |
| ch10 RAG/agentic     | 16 | 10 | 0 | 1 | ✓ |
| ch13 Bandits         | 11 |  1 | 0 | 8 | ✓ |
| **total**            | **134** | **56** | **1** | **17** | 9/9 |

Reproduce:
```bash
uv venv .venv
uv pip install --python .venv -e . -r requirements.txt \
    faiss-cpu implicit mlflow requests plotly transformers pytest
.venv/bin/pytest tests/figures            # compile + cumulative-execution harness
```

**One cross-cutting result up front:** the chapter **notebooks import the package
correctly** everywhere; the import/API drift is confined to the **printed listings**.
But the package itself has genuine bugs independent of the book — including two modules
that don't import at all.

---

## 1. Package modules that don't import (2 of 37)

| id | module | error | proposed patch |
|----|--------|-------|----------------|
| **P1** | `fourstage_recsys/retrieval/sentence_transformer_retrieval.py` (lines 10 & 117) | `ImportError: cannot import name 'RecommenderContext'` — the class is `RecommendationContext` | rename in both the import and the `retrieve` annotation |
| **P2** | `models/recsys_speaking_LLM/multi_token_prediction.py` (line 58) | `SyntaxError` — a stray markdown ```` ``` ```` fence; also never imports `np`/`torch`/`F`, and lines 44–57 are an unguarded top-level demo using undefined names | delete the fence, add the 3 imports, guard/remove the demo block |

## 2. Package runtime / logic bugs (the "main code" is wrong, independent of the book)

| id | where | issue (verified) | proposed patch |
|----|-------|------------------|----------------|
| **P3** | `pipeline.recommend` ↔ `HistoryFiltering.filter` | pipeline calls `filter.filter(candidates, context)`; override expects `user_id`, so a `RecommendationContext` is compared to `userId` → **nothing is filtered** (watched items leak) | make `HistoryFiltering.filter(self, candidates, context)` read `context.user_id` (align to the `Filtering` ABC) |
| **P4** | `Retrieval` ABC ↔ all retrievers | abstract `retrieve(self, context)` is never implemented; pipeline actually calls `retrieve_similar_items` | pick one contract: declare `retrieve_similar_items` on the ABC, or make retrievers implement `retrieve(context)` |
| **P5** | `RecommendationContext.seed_items: List[str]` / `ScoredItem.item_id: str` / int `movieId` | type drift: int ids → `pydantic.ValidationError`; str seeds → no match on int-keyed maps → pipeline returns `[]`. `RecommendationContext(user_id=7)` also raises (typed `str`) | choose one id type end-to-end; cast at the data boundary |
| **P6** | `ItemKNNRetrieval.retrieve_similar_items` | annotated `-> list[dict]` but returns `list[ScoredItem]` | fix annotation |
| **P7** | `Filtering` ABC ↔ `HistoryFiltering` | Liskov mismatch (`context` vs untyped `user_id`) — root cause of P3 | resolved by P3 |
| **P8** | `requirements.txt` | `data/loaders.py` imports `requests` (undeclared); `agentic/llm_client.py` default backend imports `openai` (undeclared) | add `requests` and `openai` (or make the openai backend degrade gracefully) |
| **P9** | `retrieval/vectordb_retrieval.py`, `retrieval/database_retrieval.py` | build `ScoredItem(score=…)` (real field is `scores`) and read a nonexistent `context.item_id` → both retrievers raise when used | use `scores={...}`; read `context.seed_items` |
| **P10** | `semantic_ids/evaluations.py::test_semantic_coherence` | summary block + `return` indented **inside** the `for` loop → returns after only the **first** of six test pairs (reproduced `len(results)==1`); corrupts `evaluate_semantic_ids` | dedent the return out of the loop |
| **P11** | `semantic_ids/training.py` `__main__` | demo `simple_data` has mismatched array lengths (`ValueError`); `run_pipeline_debug` returns 3 values unpacked into 2 | fix the demo fixture + unpack |
| **P12** | `tests/test_semantic_ids.py::test_no_pycache_in_semantic_ids` | fails with `NameError: HAS_PYTEST` (and `pytest.warn` isn't a real API); the 6 core VQ/RQVAE tests pass | define the flag / use `warnings.warn` |
| **P13** | `agentic/guardrails.py::GroundingValidator` | `_extract_titles()` strips the year but catalog titles keep it, so titles never match → **every** recommendation flagged `hallucinated`, `grounding_rate` always `0.0`; `AgentEvaluator`/`llm_as_judge` inherit it | normalize both sides (strip year consistently) before matching |
| **P14** | `agentic/llm_client.py::LLMClient` | default `backend="api"` raises `ImportError` at construction (`openai` absent) — not a graceful degrade | see P8; or fall back to local backend with a warning |
| **P15** | `models/recsys_speaking_LLM/sampled_softmax_loss.py::SampledSoftmaxLoss` | out-of-sync with the (correct) printed Listing 7.26: `__init__` drops the `embedding` param that `forward` reads (`AttributeError`), and references undefined `target_positions` | restore the `embedding` param + `target_positions = torch.arange(len(target_ids))` |

*Also:* `evaluation/metrics.py` and both `recsys_speaking_LLM/*` modules are **orphans**
(nothing imports them; `evaluation/__init__.py` is empty) — worth wiring up or noting.

## 3. Book-listing ↔ package drift (printed code that can't run against the library)

| id | listing | issue | proposed patch |
|----|---------|-------|----------------|
| **B1** | ch02 L2.15 | imports `recsys.retrievals/filters/scorers/rankers` (don't exist), `FourStageRecommender` from top-level (isn't), classes `HistoryFilter`/`PopularityScorer` (real: `…Filtering`/`…Scoring`), and calls `recommend(user_id=,k=)` (real: `recommend(context)`) | correct to real paths + build a `RecommendationContext`, or add re-export shims |
| **B2** | ch05 L5.10–5.14 | base `Scoring` (real: `Scorer`); `ranker`/`ranker.rank` (real: `ordering`/`order(candidates,context,debug)`); `context.seed_movie_id` (real: `seed_items`); `score()` arg count varies; classes `DeduplicationFilter`/`ScoreRanker`/`ANNRetrieval`/`CrossEncoderScoring`/`TransformerScoring` don't exist | rewrite listings to the shipped API or implement the missing classes |
| **B3** | ch06 L6.3 | `gbce_loss` printed without the `pos_mask` arg the shipped 4-arg version requires; the book **contradicts itself** (L6.5 calls the 4-arg form) | reprint 6.3 with `pos_mask` |
| **B4** | ch06 L6.7 | stray trailing `:` on the `super().__init__(...)` call → `SyntaxError`; also reintroduces the all-padding-NaN via direct `self.transformer(...)` | remove the `:`; use the base class's layer loop |
| **B5** | ch08 defaults | undocumented default drift: `RQVAE.codebook_sizes` book `[16,32,128]` vs pkg `[8,64,512]`; `SemanticIDPipeline.internal_dim` prose says 512 but ships **64**; `usage_loss_weight` 2.0 vs 10.0; `train.epochs` 500 vs 100 | reconcile prose/listings with shipped defaults |
| **B6** | ch10 L10.3 | `from recsys.retrieval import ContentRetriever, CollaborativeRetriever` — module + classes don't exist (closest: `agentic.hybrid_retriever.HybridRetriever`, different contract) | rewrite to shipped agentic API or add the classes |
| **B7** | ch10 L10.15 | `from recsys.generative import generate_recommendations, SemanticDecoder` — no such module (the ch09 generative model is never packaged) | package the generative model or mark as ch09-notebook code |
| **B8** | ch10 L10.6/10.7 | `build_recommendation_prompt` returns undefined `system_prompt` (`NameError`); L10.7 calls `retriever.hybrid_search` (real method: `search`) | define `system_prompt`; use `search` |
| **B9** | ch04 L4.8 | broken reimplementation diverging from the (correct) `evaluation/metrics.py`: `IndentationError`, AP accumulation dedented out of the loop, AP divides by hit-count not `min(len(rel),k)`, NDCG mis-orders `ndcg_score` args | replace with the package's metric implementations |
| **B10** | ch01 L1.2/1.6/1.7/1.5 | `catalog` vs `catalogue` → `NameError`; item-CF `np.dot(matrix.T[i], matrix.T)` shape mismatch (should dot against `matrix`); `np` never imported; L1.5's printed output is wrong (`Thorgal` shouldn't appear) | fix variable names, the dot operand, add the import, correct the stated output |
| **B11** | ch13 L15.7–15.11 | real book bugs: L15.9 dangling `/` + `Self` typo + mis-indented `return`; L15.11 tie-break bug + hardcoded `arm_index=1`; L15.7/15.8/15.10 indentation/typos (`availble_arms`, capital `For`) | correct each (details + patches in `ch13/FINDINGS.md`) |

## 4. Formatting / numbering issues to flag to the author

- **ch13 L15.3** — the pivotal `EpsilonGreedyBandit` class is printed as **un-fenced,
  escaped text**, so it can't be extracted/run and is the root cause of L15.5's
  `NameError`. This is a manuscript formatting bug, not just a code bug.
- **Duplicate / missing listing numbers:** ch04 (two "4.3", no "4.4"), ch05 (two "5.6"),
  ch06 (two "6.5"), ch10 (two "10.3"; missing 10.2/10.9/10.13), ch13 (two "15.1").
- **17 listings don't parse** as Python — a mix of real syntax bugs (B4, B11),
  output-not-code blocks, and illustrative pseudocode. Classified per chapter.
- **Internal chapter numbering drift:** ch09 is internally numbered 7.x, ch13 is 15.x;
  cross-references are stale.

## 5. Per-chapter detail

Each chapter's `chNN/FINDINGS.md` contains the full evidence (runnable reproductions)
and a concrete proposed patch for every row above. See also `ch02/FINDINGS.md` for the
worked template and `README.md` for the harness/category scheme.
