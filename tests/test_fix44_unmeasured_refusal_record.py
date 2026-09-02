"""fix44 / #77-B3: every UNMEASURED exit of tools/live_save_gate.py must
leave a classified refusal record on disk.

Measured incidence (jobs 1787517960364 and 1787518637847): the launcher told
the operator "the adapter-prefix abstention is recorded on disk at
$ART_REPORT" while fs_gate/ contained only resolved-train-config.json -- a
claim about a file that was never produced, the artifact gate's only durable
record asserting evidence that did not exist. These tests pin both halves of
the repair: (i) every unmeasured exit writes its refusal, classified by
cause, to --json when one is given; (ii) the classification vocabulary names
exactly the causes the launcher demultiplexes -- the calibrated
adapter-prefix abstention versus everything else, which is the rc-92 class.

Fail-before accounting (house rule): on the pre-fix44 tree _refusal_class
does not exist and main() writes no record on the exit-3 paths, so all 5
tests below are red there (import error / absent record assertions) and green
only with the patch. The must-not-skip guard (FS_FORBID_SKIPS=1) is honored:
nothing here skips; a broken import fails, by name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The tools/ directory sits at the repository root beside src/; the suite's
# other tests of this tool resolve it the same way. Path setup is defensive
# and idempotent so the file also runs standalone.
REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.live_save_gate import (  # noqa: E402 -- repo-local import, path set above
    _refusal_class,
    main,
)


def _rungate_args(tmp_path: Path, report: Path) -> list[str]:
    """argv for an unmeasured run that needs NO artifacts and NO torch: a
    missing base-model dir raises GateUnmeasured before any artifact is
    touched, so the refusal path is exercised in pure python."""
    return [
        str(tmp_path / "iter_0000020"),
        "--event",
        "save",
        "--run-kind",
        "lora",
        "--base-model-dir",
        str(tmp_path / "no-such-base-model"),
        "--json",
        str(report),
    ]


def test_refusal_class_names_the_calibrated_abstention() -> None:
    msg = (
        "--adapter-prefix was not pinned for a lora adjudication (exit 3 "
        "-- a refused measurement, not a checkpoint verdict): ..."
    )
    assert _refusal_class(msg) == "adapter_prefix_unpinned", (
        "the launcher calibrates exactly this member of the exit-3 class to "
        "rc 0; if the tool's raise text and the classifier drift, the "
        "calibrated abstention silently moves to the rc-92 class (loud, "
        "but a false chain-stop) or an unchosen cause moves to rc 0 (the "
        "#77-B2 defect in reverse)"
    )


def test_refusal_class_names_an_unreadable_checkpoint() -> None:
    msg = (
        "checkpoint unreadable: /x/iter_0000020: torch.distributed.checkpoint"
        " is unavailable; cannot read DCP (path='/x/iter_0000020')"
    )
    assert _refusal_class(msg) == "checkpoint_unreadable", (
        "the measured cause of both PROBE runs' exit 3s (a torch-less host "
        "python reading a healthy DCP, #77-B1) must be named as what it is, "
        "never filed under the prefix abstention by default"
    )


def test_refusal_class_everything_else_is_the_rc92_class() -> None:
    other1 = _refusal_class("base model dir not found: /nope")
    other2 = _refusal_class("unexpected failure above (a tool bug is not a checkpoint verdict)")
    assert other1 == "other_unmeasured" and other2 == "other_unmeasured", (
        f"got {other1!r}/{other2!r}: every cause the calibration does not "
        "name must default AWAY from the calibrated arm -- an unknown "
        "member of the class inherits rc-92, never rc 0 (fail closed)"
    )


def test_main_writes_a_classified_refusal_record_on_exit_3(
    tmp_path: Path,
) -> None:
    report = tmp_path / "fs_gate" / "report-lora.json"
    rc = main(_rungate_args(tmp_path, report))
    assert rc == 3, f"rc={rc}: an unmeasurable estate must stay UNMEASURED"
    assert report.is_file(), (
        "rc 3 without an on-disk refusal record is exactly #77-B3: the "
        "launcher's old narration then claims a report that was never "
        "written. Denominator: 1 report expected, "
        f"{int(report.is_file())} found"
    )
    record = json.loads(report.read_text(encoding="utf-8"))
    assert record["verdict"] == "UNMEASURED"
    assert record["exit_code"] == 3
    assert record["refusal_class"] == "other_unmeasured"
    assert "base model dir not found" in record["refusal"]
    assert record["gates_exercised"] == "0 of 3"
    assert record["controls_exercised"] == "0 of 3", (
        "a refusal precedes any verdict, so the record must name the honest "
        "denominators: 0 of 3 gates and 0 of 3 controls exercised -- never "
        "a silent omission the launcher could read as coverage"
    )


def test_main_refusal_record_write_failure_stays_loud(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "a-plain-file"
    blocker.write_text("x", encoding="utf-8")
    # --json points under a path whose parent cannot be created: best-effort
    # writing must degrade to a LOUD stderr line and keep rc 3 -- never to a
    # silent skip, and never to a fabricated success.
    report = blocker / "report-lora.json"
    rc = main(_rungate_args(tmp_path, report))
    assert rc == 3, (
        f"rc={rc}: a refusal that cannot be persisted must not change the "
        "adjudication outcome in either direction"
    )
    err = capsys.readouterr().err
    assert "could NOT record its own refusal report" in err, (
        "an unwritable refusal record must be LOUD so the launcher-side "
        "claim-vs-disk verification can indict it (doctrine 4); a silent "
        "skip would re-create #77-B3 one layer down"
    )
