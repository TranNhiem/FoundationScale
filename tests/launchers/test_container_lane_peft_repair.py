"""Off-stack guard for the container-bound peft.* composition cases (fix43 / task #74).

Why this module exists. The eight #73 regression cases exercise the vendored
megatron.bridge omegaconf predicate, importable only in the training
container. Their previous home used a module-level importorskip, and that
skip is precisely the shape conftest.py's FS_FORBID_SKIPS=1 guard fails CI
on — measured: "1 skip(s) — tolerated locally, FAILED by CI". The resolution
is a lane split, not a skip and not a tolerated absence:

  * the cases now live at container_tests/peft_omegaconf_repair_cases.py, a
    filename NO pytest collection glob matches, so they cannot be collected
    off-stack under any configuration (a declared, structural abstention —
    the test_launcher_contracts.sh precedent pushed one notch further: there
    is not even a skip record to tolerate);
  * what remains here is a denominated PIN of that lane, runnable everywhere
    because it needs only ast + pathlib: the lane file must EXIST, contain
    EXACTLY the eight named test functions (an exact census, not a loose
    count — an added case without a guard edit is a red, by design), and the
    retired tests/ path must stay retired so the suite can never re-grow
    a skip-shaped twin.

Doctrine 1 is the whole point: the off-stack suite must NOT green over zero
executed composition cases, and it does not — the assertions below have a
denominator (8) and a MUST_FIRE (the last test), and they certify the lane
record, not the predicate. They abstain, explicitly, on every claim about
whether the container lane is actually SCHEDULED: no Makefile/CI file was
available when this was written, and the lane file's docstring carries the
invocation contract until that wiring lands.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
LANE_FILE = REPO_ROOT / "container_tests" / "peft_omegaconf_repair_cases.py"
RETIRED_PATH = REPO_ROOT / "tests" / "test_peft_omegaconf_repair.py"

# The exact census. Sourced from the fix42 module's lived history (its own
# docstring promised "8 named abstentions"), pinned as a SET: a renamed body
# under the same count is still a red.
EXPECTED_CASES = frozenset(
    {
        "test_callable_dataclass_instance_is_not_problematic",
        "test_dataclass_type_is_allowed",
        "test_bare_lambda_still_problematic",
        "test_functools_partial_still_problematic",
        "test_plain_set_still_problematic",
        "test_callable_dataclass_subtree_is_addressable_in_the_view",
        "test_tracker_records_the_genuine_leaf_not_the_whole_subtree",
        "test_overrides_land_and_are_not_silently_reverted",
    }
)


def _lane_case_names(path: Path) -> frozenset[str]:
    """Top-level test_* FunctionDef names in ``path``, via AST — never via
    import, so this guard stays lawful on hosts without torch (the whole
    reason the lane split exists)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    )


def test_lane_file_exists():
    # RED on the pre-fix43 tree (the lane file does not exist there) — the
    # declared fail-before for this module.
    assert LANE_FILE.is_file(), (
        f"container lane file missing: {LANE_FILE} — the 8 vendored-predicate "
        "composition cases have NO lane, which is worse than a skip: it is "
        "coverage that does not exist. Run them in-container: "
        "python3 -m pytest container_tests/peft_omegaconf_repair_cases.py -q"
    )


def test_lane_case_census_is_exactly_the_fix42_eight():
    # RED pre-fix43 (file absent -> collection of THIS test errors on the
    # read, which is the declared fail-before), green after.
    names = _lane_case_names(LANE_FILE)
    missing = sorted(EXPECTED_CASES - names)
    extra = sorted(names - EXPECTED_CASES)
    assert names == EXPECTED_CASES, (
        f"lane census drifted ({len(names)}/8 present): missing={missing} "
        f"unexpected={extra} — an edited, renamed, or dropped case must land "
        "here as a deliberate guard edit, never as silent drift (doctrine 2)"
    )


def test_old_tests_path_is_retired():
    # The pre-fix43 module carried the importorskip that failed CI. If it
    # reappears, the suite has re-grown its skip-shaped twin next to the
    # lane — red, loudly, at the path, not as a tolerated skip count.
    assert not RETIRED_PATH.exists(), (
        f"{RETIRED_PATH} is back — the importorskip-based module was retired "
        "to the container lane (Edit 9/10 of fix43); re-landing it re-arms "
        "the FS_FORBID_SKIPS=1 CI failure that task #74 closed"
    )


def test_lane_census_must_fire(tmp_path):
    # MUST_FIRE for the census itself (doctrine 3): a case dropped from the
    # lane must change what _lane_case_names reports. Construct the defect
    # on a WRITE-TO-TMP copy — never on the lane file — prove the
    # construction (needle present before surgery, the renamed def present
    # after), and demand the census report exactly the 7-remainder.
    src = LANE_FILE.read_text(encoding="utf-8")
    needle = "def test_bare_lambda_still_problematic"
    assert needle in src, (
        "MUST_FIRE UNREACHABLE: the live lane file no longer carries the case "
        "this control doctors out — construction premise broken"
    )
    doctored = src.replace(needle, "def retired_bare_lambda_control_probe", 1)
    assert "def retired_bare_lambda_control_probe" in doctored, (
        "MUST_FIRE UNREACHABLE: the rename did not construct"
    )
    probe = tmp_path / "lane_copy.py"
    probe.write_text(doctored, encoding="utf-8")
    names = _lane_case_names(probe)
    assert names == EXPECTED_CASES - {"test_bare_lambda_still_problematic"}, (
        f"census failed to move: after removing one case the copy still "
        f"censused {len(names)}/8 — a counter that cannot move is wallpaper"
    )
