# Book figures that need updating

Derived from the faithful figures + the consistency audit. Each row is a printed
Listing whose code and/or callouts should be revised in `chapters/*.md`. Reason codes:

- `syntax-error` — the printed code does not parse
- `runtime-bug` / `code-bug(noted)` — printed code runs wrong / has a flagged logic bug
- `api-drift` / `code<->pkg-drift` — printed code doesn't match the shipped `recsys` API
- `callout-not-placed` — a #X explanation exists but no code line carries #X
- `marker-no-explanation` — a code line has #X but no explanation is listed
- `duplicate-callout` / `callout-out-of-order` — legend letters repeat or are unordered

**57 listings need updating.** Full per-issue detail + proposed fixes are in each chapter's `FINDINGS.md`.

| Chapter | Listing | Reasons |
|---------|---------|---------|
| ch01 | L1.2 | code-bug(noted) |
| ch01 | L1.6 | code-bug(noted) |
| ch01 | L1.7 | code-bug(noted) |
| ch02 | L2.1 | callout-not-placed |
| ch02 | L2.8 | callout-not-placed |
| ch02 | L2.10 | runtime-bug, code-bug(noted) |
| ch02 | L2.11 | code-bug(noted), callout-not-placed |
| ch02 | L2.13 | code-bug(noted) |
| ch02 | L2.15 | api-drift, code<->pkg-drift |
| ch02 | L2.16 | callout-not-placed |
| ch02 | L2.19 | code-bug(noted), callout-not-placed, callout-out-of-order |
| ch02 | L2.20 | callout-not-placed |
| ch02 | L2.21 | code-bug(noted), callout-not-placed |
| ch02 | L2.22 | code-bug(noted) |
| ch04 | L4.2 | syntax-error |
| ch04 | L4.7 | code-bug(noted) |
| ch04 | L4.8 | syntax-error, duplicate-callout, callout-out-of-order |
| ch05 | L5.2 | code-bug(noted) |
| ch05 | L5.10 | api-drift, code<->pkg-drift |
| ch05 | L5.12 | api-drift, code<->pkg-drift |
| ch05 | L5.13 | api-drift, code<->pkg-drift |
| ch05 | L5.14 | api-drift, code<->pkg-drift |
| ch06 | L6.3 | api-drift, code<->pkg-drift |
| ch06 | L6.5 | callout-not-placed |
| ch06 | L6.5 | callout-not-placed |
| ch06 | L6.6 | callout-not-placed |
| ch06 | L6.7 | syntax-error, code-bug(noted) |
| ch06 | L6.10 | callout-not-placed |
| ch08 | L8.3 | code-bug(noted) |
| ch08 | L8.4 | syntax-error |
| ch08 | L8.5 | code<->pkg-drift |
| ch08 | L8.7 | code<->pkg-drift |
| ch08 | L8.10 | code<->pkg-drift |
| ch08 | L8.11 | code<->pkg-drift |
| ch08 | L8.17 | syntax-error |
| ch09 | L7.1 | syntax-error |
| ch09 | L7.2 | syntax-error |
| ch09 | L7.5 | code-bug(noted) |
| ch09 | L7.13 | syntax-error |
| ch09 | L7.26 | code-bug(noted), code<->pkg-drift |
| ch10 | L10.3 | api-drift, code<->pkg-drift |
| ch10 | L10.3 | api-drift, code<->pkg-drift |
| ch10 | L10.6 | code-bug(noted) |
| ch10 | L10.7 | code<->pkg-drift |
| ch10 | L10.14 | syntax-error |
| ch10 | L10.15 | api-drift, code<->pkg-drift |
| ch10 | L10.18 | code<->pkg-drift |
| ch13 | L15.1 | syntax-error |
| ch13 | L15.2 | syntax-error |
| ch13 | L15.4 | marker-no-explanation |
| ch13 | L15.5 | duplicate-callout |
| ch13 | L15.6 | syntax-error |
| ch13 | L15.7 | syntax-error |
| ch13 | L15.8 | syntax-error, duplicate-callout |
| ch13 | L15.9 | syntax-error, marker-no-explanation |
| ch13 | L15.10 | syntax-error, marker-no-explanation |
| ch13 | L15.11 | syntax-error |
