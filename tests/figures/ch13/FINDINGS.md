# Chapter 13 — figure ↔ code consistency findings

Chapter 13 ("Multi-armed bandits in recommender systems"; internally numbered **15.x** —
listings are labelled 15.1, 15.2, …). The source (`chapters/ch13.md`) is a rough **draft**
with author TODOs, duplicate section numbers ("15.1" appears twice) and unfinished
sentences.

**No `recsys` package involvement.** This chapter imports **no** `recsys` modules — the
only dependencies the listings assume are `numpy` (as `np`), `random`, and
`scipy.stats.beta`. There is therefore essentially **no package/API drift** to audit; the
findings below are all **book-internal code correctness** issues. There are also **no
in-repo notebooks** for this chapter — the text points readers to an external Colab
("RecSys2025 Tutorial: Thompson sampling.ipynb"), which is not in the repository, so it
could not be checked.

Everything below was **verified by executing the extracted listings** with
`.venv/bin/python` in a namespace matching the harness (`np, pd, csr_matrix,
cosine_similarity, ratings, movies` — note **`random` and `scipy.stats.beta` are NOT
provided**), plus by building minimally-corrected versions to confirm each proposed patch
actually runs.

## Snapshot

| listings | executable=true | expected pass | expected skip | non-parsing (compiles=false) |
|:--------:|:---------------:|:-------------:|:-------------:|:----------------------------:|
| 11 | 1 (L15.4) | 1 | 10 | 8 (15.1, 15.2, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11) |

Of the 11 listings, **8 do not parse**, **1 is an empty extraction** (L15.3), **1 is
blocked by that empty extraction** (L15.5), and only **1 runs end-to-end** (L15.4).

## A. Book-internal code issues (each verified by running)

| # | Listing | Exact bug | Evidence from running | PROPOSED PATCH (not applied) |
|---|---------|-----------|-----------------------|------------------------------|
| A1 | L15.1 A/B experiment | `return` used at module level (fragment shown without an enclosing `def`) | `compile()` → `SyntaxError: 'return' outside function @ line 11` | Wrap in a function, e.g. `def ab_route():` then indent the `if/else`; note `random` must be imported. |
| A2 | L15.2 ε-greedy algorithm | `return` at module level (bare select-body fragment) | `SyntaxError: 'return' outside function @ line 12` | Present it inside a `def select_arm(self, epsilon):` (as L15.3 does) rather than as loose statements. |
| A3 | **L15.3 ε-greedy bandit class** | **Figure is EMPTY** — the `EpsilonGreedyBandit` class (ch13.md 164-194) was printed as **un-fenced, escaped plain text** (`\_\_init\_\_`, `\=`, `\[0.0\]`), not a code block, so the extractor captured nothing | empty file compiles as a no-op; provides nothing → breaks L15.5 | Re-typeset the class as a proper fenced code block so it is captured; it is the pivotal listing of the section. Classification: **output-not-code / extraction artifact**, not a syntax bug. |
| A4 | L15.5 Bandit simulator | Not itself buggy, but instantiates `EpsilonGreedyBandit`, which is undefined because L15.3 was lost (A3) | run → `NameError: name 'EpsilonGreedyBandit' is not defined` | No patch to the listing itself; fixing A3 (fencing L15.3) makes this run. |
| A5 | L15.6 ε-greedy w/ recsys | `return` at module level (cold-start fragment) | `SyntaxError: 'return' outside function @ line 12` | Wrap in a `def`; also `select_a_cold_start_item()` / `recsys_model` are illustrative placeholders. |
| A6 | L15.7 Bayesian bandit | `def update_estimates` is **over-indented by one space** (3 spaces vs the 2-space `__init__` / `select_arm`) | `SyntaxError: unindent does not match any outer indentation level @ line 32` | Dedent `update_estimates` (and its body) by one space to align at 2 spaces. **Also** `beta.rvs` needs `from scipy.stats import beta` (not imported, not in harness ns). Corrected class was built and `select_arm`/`update_estimates` run. |
| A7 | L15.8 Abstract bandit | (1) `For arm in ...` — **capital-F `For`**; (2) loop/method bodies **tab-indented** amid space indentation; (3) spelling: `context.availble_arms` | `SyntaxError: invalid syntax @ line 21` (the `For`) | `For`→`for`; convert tabs→spaces; `availble_arms`→`available_arms`. Corrected abstract class defines cleanly. (It is illustrative: `arm.calc_value`/`context` are undefined by design.) |
| A8 | L15.9 UCB bandit | **Three** bugs: (1) line ends with a **dangling `/` operator** (`... ) /` then a blank line then a continuation) — illegal outside brackets; (2) **`Self.n_values`** capital-S typo; (3) **`return arm` mis-indented** to 2 spaces, i.e. at class-body level after `select_arm`, so `select_arm` returns nothing | `SyntaxError: invalid syntax @ line 52` (the dangling `/`); fixing it alone still yields `return outside function` and a `select_arm` that returns `None` | Join into `... ) / self.n_values[chosen_arm]`; lowercase `self`; indent `return arm` to 4 spaces inside `select_arm`. Corrected `UCBAlgorithm` ran a 20-step simulation successfully. |
| A9 | L15.10 LinUCB disjoint arm | **Inconsistent indentation**: `def __init__` at 3 spaces vs `def calc_UCB`/`def reward_update` at 2 spaces; inside `calc_UCB`, `x`/`p`/`return` at 7/14/7 spaces | `SyntaxError: unindent does not match any outer indentation level @ line 30` | Normalise to 2-space class body / 4-space method body. Corrected `linucb_disjoint_arm.calc_UCB` returns the expected (1,1) UCB value. |
| A10 | L15.11 LinUCB policy | (1) **Indent bug**: `def __init__` at 4 spaces vs `def select_arm` at 2 spaces; (2) **tie-breaker logic bug**: `if arm_ucb == highest_ucb` is nested *inside* the `if arm_ucb > highest_ucb` block right after `highest_ucb = arm_ucb`, so it is always true and only **duplicates the current arm**; real ties on other arms are never collected; (3) every arm built with hardcoded `arm_index = 1` (loop var `i` ignored) | `SyntaxError: unindent ... @ line 30`. After fixing indent, demonstrated: 3 tied arms → `candidate_arms = [0, 0]` (book) vs the correct `[0, 1, 2]` | Normalise indentation; change the nested `if arm_ucb == highest_ucb` to a sibling **`elif`** at the same level as the `>` test; use `arm_index=i`. |

## B. Non-parsing classification (8 of 11 listings)

Every `compiles=false` listing was classified per the harness convention:

| Listing | Class | Root cause |
|---------|-------|------------|
| L15.1, L15.2, L15.6 | **pseudocode** (code fragment) | `return` outside a function — printed as a loose 4-line snippet, meant to live inside a `def`. |
| L15.7 | **real-book-syntax-bug** | `update_estimates` over-indented by 1 space. |
| L15.8 | **real-book-syntax-bug** | capital `For`, tab/space mix (on an illustrative abstract base). |
| L15.9 | **real-book-syntax-bug** | dangling `/`, `Self` typo, mis-indented `return`. |
| L15.10 | **real-book-syntax-bug** | inconsistent method/body indentation. |
| L15.11 | **real-book-syntax-bug** | inconsistent method indentation (plus a separate algorithmic tie-breaker bug). |

`L15.3` is a distinct case: it **compiles** but is **empty** — an **output-not-code /
extraction artifact** because the class was typeset as un-fenced escaped text. It is the
reason L15.5 cannot run.

## C. Package / notebook drift

**None.** No listing imports `recsys`, so there is no package-drift to report (the sole
`recsys`-looking token is the placeholder variable `recsys_model.recs()` in L15.6). No
`recsys` bandit module exists in the repo, and there is no in-repo notebook for this
chapter — the referenced "RecSys2025 Tutorial: Thompson sampling" is an external Colab
link only.

## How this was checked

```bash
# from repo root
grep -rn "import" tests/figures/ch13/*.py          # -> no imports in any listing
.venv/bin/python scratch/probe.py                  # compile + cumulative exec per listing
.venv/bin/python scratch/fixes.py                  # build minimally-corrected versions, confirm they run
.venv/bin/pytest "tests/figures/test_figures.py::test_chapter_execution[ch13]" -q
```

Observed execution outcomes (harness namespace: `np, pd, csr_matrix, cosine_similarity,
ratings, movies`; **no `random`, no `scipy.stats.beta`**):

```
L15.1  SyntaxError: 'return' outside function @ line 11
L15.2  SyntaxError: 'return' outside function @ line 12
L15.3  compiles OK (empty file, no-op)
L15.4  RUN pass
L15.5  RUN fail  (NameError: EpsilonGreedyBandit)
L15.6  SyntaxError: 'return' outside function @ line 12
L15.7  SyntaxError: unindent does not match any outer indentation level @ line 32
L15.8  SyntaxError: invalid syntax @ line 21   (capital `For`)
L15.9  SyntaxError: invalid syntax @ line 52   (dangling `/`)
L15.10 SyntaxError: unindent does not match any outer indentation level @ line 30
L15.11 SyntaxError: unindent does not match any outer indentation level @ line 30
```

_All findings are **reported, not fixed** — the figures are kept verbatim so the harness
records the defects. Patches above are proposals for the author._
