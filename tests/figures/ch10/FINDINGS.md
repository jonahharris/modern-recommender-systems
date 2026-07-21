# Chapter 10 — figure ↔ package consistency findings

Chapter 10 is **RAG & Agentic Recommender Systems** (ReAct agent, RAG chain, query
rewriting / HyDE, tools, memory, guardrails). Goal: verify the shipped `recsys.agentic`
package works with the chapter-10 listings and surface inconsistencies. Every item below
is **verified by importing/executing the installed package** (`./.venv`, editable
`recsys`), not by inspection alone.

Chapter numbering is messy: no 10.2 / 10.9 / 10.13; **two listings numbered 10.3**
(explanations wrapper + unified retrieval); 10.7/10.8 rendered as bold headers.

16 listings extracted. **10 executable** (all pure `def`/class defs → `expected=pass`),
**6 non-executable** (2 api-drift imports, 3 need a live agent/LLM/tools, 1 is an output
trace), **1 non-parsing** (10.14). **0 expected-fail.**

## A. Book listing ↔ package drift (printed code doesn't match / can't run against the library)

| # | Where | Issue | Reality in package | Proposed patch |
|---|-------|-------|--------------------|----------------|
| A1 | **L10.3 "Unified retrieval"** | `from recsys.retrieval import ContentRetriever` / `CollaborativeRetriever` | module `recsys.retrieval` does **not** exist (`ModuleNotFoundError`), and neither `ContentRetriever` nor `CollaborativeRetriever` is implemented anywhere in `recsys`. Verified. | The shipped equivalent is `recsys.agentic.hybrid_retriever.HybridRetriever`, but it has a **completely different contract** (see A2). Simplest fix: rewrite the listing to import and use the real class, or drop the two phantom imports and present `HybridRetriever` from `recsys.agentic.hybrid_retriever` as-is. If the book intends the two-retriever composition shown, add real `ContentRetriever`/`CollaborativeRetriever` classes to a `recsys/retrieval.py` shim. |
| A2 | **L10.3 "Unified retrieval"** (`HybridRetriever`) | book class is `HybridRetriever(content_retriever, collab_retriever)` with `search(query, user_id, k)` delegating to `content.search` / `collab.recommend` and a dict-based `_merge` | shipped `recsys.agentic.hybrid_retriever.HybridRetriever(faiss_index, embeddings, movies_df, item_knn=None, encoder_name="all-MiniLM-L6-v2")`; retrieval is FAISS + `SentenceTransformer` with `_reciprocal_rank_fusion`. Same *idea* (RRF, k=60), totally different constructor/dependencies. | Reconcile the printed class with the shipped one, or clearly label the listing as a simplified illustration distinct from `recsys.agentic`. |
| A3 | **L10.15 "Generative model as an agent tool"** | `from recsys.generative import generate_recommendations, SemanticDecoder` | module `recsys.generative` does **not** exist (`ModuleNotFoundError`). The Chapter 7/9 generative (semantic-ID) model is **never packaged** under `recsys`; there is no `generate_recommendations`/`SemanticDecoder` anywhere. Verified. | Either package the ch07/09 generative model as `recsys/generative/` exposing `generate_recommendations` + `SemanticDecoder`, or annotate the listing as depending on chapter-7/9 notebook code that is not in the installed package. Closest shipped agent-facing retriever is `HybridRetriever` (content/collaborative, not generative). |
| A4 | **L10.7 `recommend_with_rag`** | calls `retriever.hybrid_search(query, user_id=..., k=15)` | shipped `HybridRetriever` exposes **`search(query, user_id=None, k=10, filters=None)`** — there is no `hybrid_search` method → `AttributeError` at call time. Verified via `inspect`. | Change the listing to `retriever.search(...)`, or add a `hybrid_search` alias to `HybridRetriever`. |
| A5 | **L10.11 `MovieRecommenderAgent`** | book reprints a **simplified** agent: `run(self, user_query, user_id=None)`, dict-tools only, no error handling | shipped `recsys.agentic.agent.MovieRecommenderAgent.run(self, user_query, user_id=None, debug=False)` also accepts `Tool` objects, wraps LLM errors, detects an implicit final answer, truncates long observations, and adds `get_trace()`. Signatures are call-compatible; the package is a superset. | No fix required — note in the text that the shipped class extends the printed one. `Tool` (constructor `Tool(name, description, function, parameters=None)`) is only in the package, not shown in the listing. |
| A6 | **L10.18 `UserProfiler`** | book `__init__(self, llm, profile_store)` (store **required**) and keys the store by the **raw** `user_id` | shipped `UserProfiler(self, llm, profile_store=None)` defaults to `{}` and coerces keys with `str(user_id)` in both `get_profile` and `update_profile` (adds `save()`/`load()`). Verified: `update_profile(7, ...)` stores under key `"7"`. | Minor; align the listing's key type with the package (`str(user_id)`) or note the difference. A reader who instantiates `UserProfiler(llm)` (no store) works against the package but fails against the printed signature. |

The chapter-10 **notebooks import the correct paths** (`recsys.agentic.hybrid_retriever`,
`recsys.agentic.agent`, `recsys.agentic.memory`, `recsys.agentic.guardrails`,
`recsys.fourstage_recsys.retrieval.itemknn_retrieval`); the drift is confined to the
printed listings 10.3 (unified) and 10.15.

## B. Book-internal bugs in the listings (independent of the package)

| # | Where | Issue | Evidence | Proposed patch |
|---|-------|-------|----------|----------------|
| B1 | **L10.6 `build_recommendation_prompt`** | `return system_prompt, user_message` but `system_prompt` is never a parameter and never assigned in the function. The chapter shows the system prompt only as **prose** ("### System Prompt for RAG recommendation chain") above the listing. | Calling the function raises `NameError: name 'system_prompt' is not defined` (verified). The def itself compiles/executes, so it passes the harness but breaks on first call — and L10.7 calls it. | Add `system_prompt = "..."` (the prose block) inside the function or as a parameter/module constant, then `return system_prompt, user_message`. |
| B2 | **L10.14** | printed as a code listing but it is a rendered agent execution **trace** (`[thought]`/`[action]`/`[observation]`/`[final_answer]`), not Python. | `compile()` → `SyntaxError: invalid decimal literal` (line 6). | Present it as an output block / figure, not a Listing (category `output`). |

## C. Package internal inconsistency (the shipped code is wrong)

| # | Where | Issue | Evidence | Proposed patch |
|---|-------|-------|----------|----------------|
| C1 | `recsys/agentic/guardrails.py` — **`GroundingValidator`** | The catalog set is built from **full titles incl. year** (`"toy story (1995)"`), but `_extract_titles()` captures the title with the **year stripped** (regex group ends before `(\d{4})`), yielding `"toy story"`. The two can never match, so *every* recommendation is classified `hallucinated` and `grounding_rate` collapses to `0.0` — even for real catalog items. | Verified: `GroundingValidator(df={"Toy Story (1995)","Heat (1995)"}).validate("**Toy Story (1995)**")` → `{'grounded': [], 'hallucinated': ['Toy Story'], 'grounding_rate': 0.0}`. The same year-stripping mismatch hits the optional `candidates` comparison. `AgentEvaluator`/`llm_as_judge` reuse this, so grounding metrics are unreliable. | Normalize both sides identically. Simplest: strip the trailing `(YYYY)` from catalog titles when building `self.titles` (and from `candidate_titles`), so both are year-less. Alternatively keep the year in `_extract_titles` (capture through `\(\d{4}\)`). |

## D. `LLMClient` behavior without an API key (and packaging gap)

- Default backend is `"api"`. `LLMClient(backend="api")` (no key) **does not** silently
  degrade — it raises **`ImportError: Install openai: pip install openai`** because the
  `openai` package is **not installed** in the venv and is **not declared** in
  `requirements.txt`. Verified. With `openai` present it would construct an `OpenAI`
  client with `api_key=None` and only fail on the first network call.
- `backend="local"` uses `transformers`+`torch` (both present) and needs no key, but
  downloads/loads a HF model (`microsoft/Phi-3-mini-4k-instruct`); heavy, not run here.
- `LLMClient(backend="bogus")` correctly raises `ValueError: Unknown backend: bogus`.
- **Proposed patch:** add `openai` to `requirements.txt` (the documented default backend
  can't be constructed without it), or default `backend="local"` / raise a clearer
  "no API key configured" message.

## E. What ran, and how each class behaves on toy inputs

Verified by instantiating the shipped classes with a scripted **fake LLM**
(`generate`/`generate_json` returning canned strings):

- **`ConversationMemory(llm, max_recent_turns=2, summary_threshold=3)`** — after 5
  `add_turn` calls, keeps 3 turns and populates `summary` via `_compress()`; sliding
  window + summarization behave exactly as printed in L10.17.
- **`UserProfiler(llm)`** — `get_profile` → `"No profile yet."`; `update_profile(7, …)`
  returns the LLM output and stores it under key `"7"` (str-coerced). (see A6)
- **`GroundingValidator`** — instantiates and runs, but is **buggy** (C1).
- **`LoopDetector().is_looping(<4 repeated Action responses>)`** → `True`.
- **`MovieRecommenderAgent`** — with a `Tool` and a 2-step fake LLM script
  (Action then Final Answer), `run()` executes the tool, parses the final answer, and
  records a 2-entry trace. Also accepts the L10.10-style **dict-tools** form.
- **`HybridRetriever`** — **not instantiated** on toy inputs: its `__init__` requires a
  trained FAISS index + item embeddings + `movies_df` and constructs a
  `SentenceTransformer` (model download). `sentence_transformers`, `faiss`, `torch` are
  all installed, so it is importable; full instantiation is a `needs-real-data`/training
  concern out of scope for the shared harness namespace.

## How this was checked

```bash
# from repo root, using the provided venv
.venv/bin/python  <import-resolution, inspect signatures, fake-LLM instantiation probes>
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch10]" -q   # green
.venv/bin/pytest "tests/figures/test_figures.py" -q -k ch10                          # 16 passed, 1 xfailed
```

Probes performed:
1. `import recsys.agentic` (OK; exports the 6 documented classes) and attempted
   `import recsys.retrieval` / `import recsys.generative` → both `ModuleNotFoundError`
   (confirms A1, A3).
2. `inspect.signature` on every class constructor + `run`/`search`/`add_turn`/
   `get_context`/`update_profile` (confirms A2, A4, A5, A6).
3. Instantiated `ConversationMemory`, `UserProfiler`, `GroundingValidator`,
   `LoopDetector`, `MovieRecommenderAgent` (+`Tool`) with a fake LLM (confirms C1, E).
4. Constructed `LLMClient(backend="api")` with no key (confirms D).
5. Executed L10.6's function to trigger its `NameError` (confirms B1); `compile()` of
   L10.14 to confirm it does not parse (B2).

_These are reported, not fixed — they belong to the author's package/listings and several
have more than one reasonable resolution. Per task scope, no `recsys` source or file
outside `tests/figures/ch10/` was modified._
