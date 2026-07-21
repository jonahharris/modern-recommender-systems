"""Test harness for extracted book listings ("figures").

Two layers of checking per chapter:

1. test_listing_compiles  — every extracted listing must at least parse. A syntax
   error here is a real defect in the printed listing.

2. test_chapter_execution — runs the listings the manifest marks `executable` in
   source order, inside a single cumulative namespace seeded by `chapter_namespace`
   (mirroring how the reader runs cells top-to-bottom). Each listing's observed
   outcome is compared against its manifest `expected` value:
       expected="pass" -> executing the listing must NOT raise
       expected="fail" -> executing the listing MUST raise (documents a book bug)
   Listings that are not executable (needs-real-data / pseudocode / api-drift) are
   skipped here and covered by the compile test only.

The suite is GREEN when every listing behaves exactly as the manifest says — so a
newly-broken listing (or a book fix that makes a known-bad listing start working)
turns the suite red and points at the exact listing.
"""
import json
from pathlib import Path

import pytest

FIG_ROOT = Path(__file__).parent
CHAPTERS = sorted(p.name for p in FIG_ROOT.iterdir()
                  if p.is_dir() and (p / "manifest.json").exists())


def _load(chapter):
    manifest = json.loads((FIG_ROOT / chapter / "manifest.json").read_text())
    for entry in manifest:
        entry["_code"] = (FIG_ROOT / chapter / entry["file"]).read_text()
        entry["_chapter"] = chapter
    return manifest


ALL_ENTRIES = [e for ch in CHAPTERS for e in _load(ch)]


@pytest.mark.parametrize(
    "entry", ALL_ENTRIES,
    ids=[f"{e['_chapter']}-L{e['number']}" for e in ALL_ENTRIES],
)
def test_listing_compiles(entry):
    """Every printed listing should parse as valid Python. Listings the manifest
    already records as non-parsing (compiles=False) are xfailed — they are known
    findings (a real syntax bug in the book, or an output/pseudocode block), not
    surprises."""
    try:
        compile(entry["_code"], entry["file"], "exec")
        compiled = True
    except SyntaxError:
        compiled = False
    if entry.get("compiles", True) is False:
        if compiled:
            pytest.fail("manifest says compiles=False but the listing now parses "
                        "— update the manifest")
        pytest.xfail("listing does not parse (recorded finding)")
    assert compiled, "listing failed to parse but manifest expected it to"


@pytest.mark.parametrize("chapter", CHAPTERS)
def test_chapter_execution(chapter, chapter_namespace, request):
    manifest = _load(chapter)
    ns = chapter_namespace
    report = []
    mismatches = []
    for entry in manifest:
        if not entry.get("executable"):
            report.append((entry["number"], "skipped (non-executable)", entry["expected"]))
            continue
        try:
            exec(compile(entry["_code"], entry["file"], "exec"), ns)
            got = "pass"
        except Exception as e:  # noqa: BLE001 - we are probing arbitrary book code
            got = f"fail ({type(e).__name__}: {e})"
        expected = entry["expected"]
        ok = (expected == "pass" and got == "pass") or \
             (expected == "fail" and got.startswith("fail"))
        report.append((entry["number"], got, expected))
        if not ok:
            mismatches.append((entry["number"], expected, got))

    lines = [f"\n=== Chapter {chapter} execution report ==="]
    for num, got, exp in report:
        lines.append(f"  L{num:<5} expected={exp:<5} -> {got}")
    print("\n".join(lines))

    assert not mismatches, (
        "Listings deviated from expected runnability:\n"
        + "\n".join(f"  L{n}: expected {e}, got {g}" for n, e, g in mismatches)
    )
