"""Verify that the claims in README.md are literally true.

Documentation drifts. This executes the README's examples against the real
library, so a claim that stops being true fails a test instead of quietly
misleading every visitor.

    python tests/test_readme_claims.py
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from citeseal import Answer, Citation, Engine, Record, Store, UncitedRecordError  # noqa: E402

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []
_TMP: list[Path] = []
_STORES: list[Store] = []

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test(fn):
    def wrapper():
        try:
            fn()
            PASSED.append(fn.__name__)
            print(f"  PASS  {fn.__name__}")
        except Exception:  # noqa: BLE001
            FAILED.append((fn.__name__, traceback.format_exc()))
            print(f"  FAIL  {fn.__name__}")
    wrapper.__name__ = fn.__name__
    return wrapper


def tmp_store(name: str) -> Store:
    tmp = Path(tempfile.mkdtemp(prefix=f"citeseal-readme-{name}-"))
    _TMP.append(tmp)
    store = Store(tmp / "corpus.sqlite3")
    _STORES.append(store)
    return store


# --- the opening example ---------------------------------------------------

@test
def test_readme_opening_example_runs_verbatim():
    store = tmp_store("opening")
    store.upsert(Record(
        key="northwind-aurelia-kettle",
        kind="recall",
        title="Northwind Recalls Aurelia Kettle Due to Burn Hazard",
        body="The kettle's lid can detach during pouring, releasing scalding water.",
        citation=Citation(
            url="https://example.gov/recalls/northwind-aurelia-kettle",
            source_name="Consumer Product Safety Commission",
        ),
    ))
    engine = Engine(store)

    got = engine.verify("kettle burn hazard").results[0]["source_url"]
    assert got == "https://example.gov/recalls/northwind-aurelia-kettle", got

    negative = engine.verify("lithium battery fire")
    assert negative.found is False, negative.as_dict()
    assert negative.reason == "no record in the store matches this query", negative.reason


@test
def test_readme_claims_the_negative_is_an_explicit_reason_not_a_wildcard():
    """'a weaker system would have returned the kettle anyway, because it is the
    only thing in the corpus' -- so an unrelated query must not return it."""
    store = tmp_store("onlything")
    store.upsert(Record(
        key="northwind-aurelia-kettle",
        kind="recall",
        title="Northwind Recalls Aurelia Kettle Due to Burn Hazard",
        body="The kettle's lid can detach during pouring, releasing scalding water.",
        citation=Citation(
            url="https://example.gov/recalls/northwind-aurelia-kettle",
            source_name="Consumer Product Safety Commission",
        ),
    ))
    for unrelated in ("lithium battery fire", "allergen labelling", "toy choke hazard"):
        answer = Engine(store).verify(unrelated)
        assert not answer.found, f"the only record was returned for {unrelated!r}"


# --- the type-level claims -------------------------------------------------

@test
def test_readme_type_guarantee_table_is_accurate():
    """Each row of the 'enforced, not promised' table must actually raise."""
    cases = [
        (lambda: Citation(url="", source_name="X"), UncitedRecordError),
        (lambda: Citation(url="ftp://x/y", source_name="X"), UncitedRecordError),
        (lambda: Record(key="k", kind="n", title="t", body="b",
                        citation={"url": "https://x.gov"}), UncitedRecordError),
        (lambda: Answer(found=True, query="q", results=({},)), UncitedRecordError),
        (lambda: Answer(found=False, query="q", results=({"source_url": "https://x.gov"},)),
         ValueError),
    ]
    for index, (call, expected) in enumerate(cases, 1):
        try:
            call()
        except expected:
            continue
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(
                f"README row {index} raised {type(exc).__name__}, expected {expected.__name__}"
            ) from exc
        raise AssertionError(f"README row {index} did not raise at all")


# --- the conformance output block -----------------------------------------

@test
def test_readme_conformance_output_matches_reality():
    """The pasted report claims six checks and a specific corpus line."""
    store = tmp_store("conformance")
    store.upsert(Record(key="a", kind="recall", title="Recall A", body="alpha",
                        citation=Citation(url="https://x.gov/a", source_name="X")))
    store.upsert(Record(key="b", kind="recall", title="Recall B", body="beta",
                        citation=Citation(url="https://x.gov/b", source_name="X")))
    store.upsert(Record(key="c", kind="recall", title="Recall C", body="gamma",
                        citation=Citation(url="https://x.gov/c", source_name="X")))
    store.upsert(Record(key="d", kind="definition", title="Def A", body="delta",
                        citation=Citation(url="https://x.gov/d", source_name="X")))
    store.upsert(Record(key="e", kind="definition", title="Def B", body="epsilon",
                        citation=Citation(url="https://x.gov/e", source_name="X")))
    store.upsert(Record(key="f", kind="definition", title="Def C", body="zeta",
                        citation=Citation(url="https://x.gov/f", source_name="X")))

    from citeseal import run_conformance
    report = run_conformance(store)
    rendered = report.render()

    assert report.passed, rendered
    assert len(report.checks) == 6, len(report.checks)
    # The README shows exactly these six codes, in this order.
    assert [c.code for c in report.checks] == ["G1", "G2", "G3", "G4", "G5", "G6"]
    for code, label in [
        ("G1", "every record carries a resolvable citation"),
        ("G2", "no match returns an explicit negative"),
        ("G3", "results are stored records, not generated text"),
        ("G4", "a positive answer cites every result"),
        ("G5", "re-ingesting unchanged data emits no events"),
        ("G6", "a changed record is superseded, not overwritten"),
    ]:
        assert f"{code}  {label}" in rendered, f"README label drifted for {code}"
    assert "RESULT: PASS" in rendered


# --- structural claims -----------------------------------------------------

@test
def test_readme_claims_zero_runtime_dependencies():
    import ast
    stdlib_ok = {"sqlite3", "json", "math", "re", "os", "sys", "threading",
                 "hashlib", "dataclasses", "datetime", "pathlib", "typing",
                 "contextlib", "collections", "urllib", "importlib", "argparse",
                 "tempfile", "gzip", "shutil", "time", "warnings", "textwrap",
                 "functools", "itertools", "logging", "base64", "uuid", "abc",
                 "pyproject", "citeseal", "__future__"}
    offenders: list[str] = []
    for path in sorted((ROOT / "citeseal").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in stdlib_ok:
                        offenders.append(f"{path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    root = node.module.split(".")[0]
                    if root not in stdlib_ok:
                        offenders.append(f"{path.name}: from {node.module}")
    assert not offenders, f"README claims zero dependencies, found: {offenders}"


@test
def test_readme_claims_unicode_aware_tokeniser():
    from citeseal.matching import tokenize
    assert tokenize("リチウム電池"), "CJK produced no tokens"
    assert tokenize("литиевая"), "Cyrillic produced no tokens"
    assert tokenize("بطارية"), "Arabic produced no tokens"
    assert tokenize("Lithiumbatterie"), "Latin produced no tokens"


@test
def test_readme_mentions_the_real_test_counts():
    """If the README states a count, it must match the suite."""
    for name, path in (
        ("guarantees", ROOT / "tests" / "test_guarantees.py"),
        ("adversarial", ROOT / "tests" / "test_adversarial.py"),
        ("stress", ROOT / "tests" / "test_stress.py"),
        ("real_corpus", ROOT / "tests" / "test_real_corpus.py"),
        ("readme_claims", ROOT / "tests" / "test_readme_claims.py"),
    ):
        source = path.read_text(encoding="utf-8")
        # @guarded counts too: it is @test plus a graceful skip when the fixture
        # is absent, and missing it made the real-corpus suite look empty.
        count = len(re.findall(r"^@(?:test|guarded)\b", source, re.M))
        # The README pads the number into a column, so allow extra spaces.
        claim = re.search(rf"test_{name}\.py\s+#\s*(\d+)\s+checks", README)
        assert claim, f"README no longer states a check count for {name}"
        assert int(claim.group(1)) == count, (
            f"README says {claim.group(1)} {name} checks, suite has {count}"
        )


@test
def test_readme_states_a_total_that_matches():
    counts = {}
    for path in (ROOT / "tests").glob("test_*.py"):
        counts[path.stem] = len(
            re.findall(r"^@(?:test|guarded)\b", path.read_text(encoding="utf-8"), re.M)
        )
    total = sum(counts.values())
    claim = re.search(r"(\d+) checks across five suites", README)
    assert claim, "README no longer states a total"
    assert int(claim.group(1)) == total, (
        f"README says {claim.group(1)} checks; the suites hold {total}: {counts}"
    )
    assert len(counts) == 5, f"expected five suites, found {sorted(counts)}"


def main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    print(f"running {len(tests)} README claim tests\n")
    for fn in tests:
        fn()

    print()
    for name, tb in FAILED:
        print(f"--- {name} ---")
        print(tb.encode("ascii", "backslashreplace").decode("ascii"))

    for store in _STORES:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
    for tmp in _TMP:
        shutil.rmtree(tmp, ignore_errors=True)

    total = len(PASSED) + len(FAILED)
    print(f"OK ({len(PASSED)} passed)" if not FAILED
          else f"FAILED ({len(FAILED)} of {total})")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
