# Book listing figures — test harness

Every numbered **Listing** printed in the book chapters is extracted here as an
individual, **verbatim** `.py` "figure" so it can be checked in isolation. This lets
us catch listings that don't parse, don't run, or have drifted away from the actual
`recsys` package — before readers hit them.

## Layout

```
tests/figures/
├── conftest.py          # synthetic MovieLens fixtures + seeded execution namespace
├── test_figures.py      # the harness (compile check + cumulative execution check)
├── chNN/
│   ├── manifest.json    # one entry per listing: category, expected outcome, notes
│   └── listing_NN_X_<slug>.py
└── ...
```

Currently populated: **ch02** (pilot, 22 listings). Other chapters to follow with
the same extractor.

## What each figure file contains

The exact code as printed in the chapter (indentation preserved). The only change is
that trailing `#A`/`#B` annotation *markers* are stripped from code lines and moved
into the header comment as an annotation legend, together with source provenance:

```python
# Figure — Listing 2.1: Creating the user-item matrix
# Source: chapters/ch02.md lines 94-107
# Chapter: 2
# Category: needs-fixture  (executable=True, expected=pass)
# Verbatim from book (only trailing #A annotation markers stripped).
# Annotations:
#   #A Create mappings from IDs to matrix indices
#   ...
```

Nothing is "fixed" — the printed bugs are preserved on purpose so the harness reports
them.

## Categories (in `manifest.json`)

| category          | executable | meaning |
|-------------------|:----------:|---------|
| `standalone`      | yes | runs with only stdlib + scientific imports (usually a pure `def`) |
| `needs-fixture`   | yes | runs against the small synthetic MovieLens fixture |
| `needs-prior`     | yes | depends on symbols defined by earlier listings in the same chapter |
| `needs-real-data` | no  | hardcodes real MovieLens IDs / needs the full dataset — compile-checked only |
| `pseudocode`      | no  | illustrative skeleton; helpers undefined by design — compile-checked only |
| `api-drift`       | no  | imports names that don't match the real `recsys` package — compile-checked only |

`expected` is `pass`, `fail`, or `skip`. `fail` means the listing is *expected* to
raise when run (a documented book bug); the harness asserts it does, so the suite stays
green while recording the defect.

## Running

```bash
uv venv .venv
uv pip install -e ".[dev]" pandas scikit-learn scipy   # or: pip install -e ".[dev]"
uv run pytest tests/figures -v
```

The execution test prints a per-chapter report of each listing's observed outcome vs.
its expected outcome.

## Known findings surfaced by the ch02 pilot

- **L2.10** calls `add_popularity_scores()` — never defined in the chapter (only
  `score_popularity` at L2.8). `expected: fail`.
- **L2.11 / L2.19** call the same undefined `add_popularity_scores` at runtime; L2.11
  also filters the wrong variable and ranks a list that was never scored.
- **L2.13** calls undefined `reorder_candidates_by_seed_similarity` and passes a `k=`
  arg to `get_user_history`, whose L2.5 signature takes none.
- **L2.15** imports `recsys.retrievals` / `recsys.filters` / `recsys.scorers` /
  `recsys.rankers` with classes `ItemKNNRetrieval` / `HistoryFilter` / `PopularityScorer`
  / `WeightedRanker`; the real package is `recsys.fourstage_recsys.retrieval.*` with
  `HistoryFiltering` / `PopularityScoring`. Book↔code drift.
- **L2.21** has a `return` before `final_score` is computed (dead code); calling it
  raises `KeyError('final_score')`.
- **L2.22** shadows `candidates` with `Candidates` and calls `filter_watched(candidates)`
  without the required `user_id`.
