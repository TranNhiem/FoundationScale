"""Adjudicate T0-10: is the effective topology READ from the runtime, or echoed?

Pure verdict over a payload dict. No I/O, no torch, no Slurm, so `python3 -S` on
a laptop and on a tray reach the same answer.

The row claims the framework derives the topology it ACTUALLY got rather than
repeating the declaration back. Until #489 that claim rested on unit tests that
set ``WORLD_SIZE`` in ``os.environ`` by hand, which measures the parser and not
the plane. The payload this scores comes from four real launches of the shipped
entry point across sixteen GB200 GPUs.

WHY FOUR ARMS. A single green run is compatible with an implementation that
reports a CONSTANT, and a single refusal is compatible with a launch that was
simply broken. So:

  misdeclared   declared 4x4 (dp=16), launched with EIGHT ranks. THE DECIDING
                ARM: an implementation that echoed the config would agree with
                itself here and sail through. It must report EIGHT and stop.
  truthful_4    declared 1x4, launched 4. Must train.
  truthful_16   declared 4x4, launched 16. Must train, and must report a
                DIFFERENT number from truthful_4 -- otherwise the reading is a
                constant that happens to be right once.
  no_torchrun   declared 4x4, launched with no torchrun at all. Must say the
                comparison was SKIPPED. An absent reading must not read as a
                matching one; that is the failure mode where "no disagreement
                detected" gets published as "agreement verified".

WHY THE EXIT CODE IS NOT THE DECIDING SIGNAL, AND THIS IS DELIBERATE. The
deciding arm's srun returned 1. The framework returned 5 -- torchrun collapses
every worker's exit code into its own, so a correct refusal and a segfault leave
by the same door. Scoring rc here would therefore score TORCHRUN. The verdict is
taken from what the run SAID and from what it WROTE, and there is a control
below asserting that a correct arm with an uninformative rc still passes.

WHY THE ARTIFACT IS SCORED TOO. The console is what an operator reads once; the
manifest is what survives. Both are checked, because the T0-9 measurement this
same week found a subsystem that recorded the truth perfectly and never said it
aloud, and the mirror-image defect -- saying it aloud and recording nothing --
would be just as invisible six months later.
"""
from __future__ import annotations

import argparse
import json
import re
import sys

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ARMS = ("truthful_4", "truthful_16")
CONTROL_ARMS = ("misdeclared_16_launched_8", "no_torchrun")
WORLD_RE = re.compile(r"WORLD_SIZE=(\d+)")
OVERRIDE_CODE = "topology.effective_overrides_"


def _arm(payload: dict, name: str) -> dict | None:
    """Arms and control arms live in two sections; the scorer needs neither."""
    for section in ("arms", "controls"):
        block = payload.get(section)
        if isinstance(block, dict) and isinstance(block.get(name), dict):
            return block[name]
    return None


def _marker(arm: dict, step: str) -> str | None:
    """The first marker line for a step, or None if the step never reported."""
    for line in arm.get("markers") or []:
        if isinstance(line, str) and line.startswith(f"[fs:train:{step}]"):
            return line
    return None


def _reported_world(arm: dict) -> int | None:
    """The world size the CONSISTENCY step says it observed, if it says one."""
    line = _marker(arm, "consistency")
    if not line:
        return None
    hit = WORLD_RE.search(line)
    return int(hit.group(1)) if hit else None


def _trained(arm: dict) -> bool:
    """Did this arm finish a real training run?

    A dry run also reports DONE with the word PASS in it, and the no_torchrun
    control IS a dry run -- so a naive "PASS in the done line" test would score
    that control as a completed training run and hide the one thing it exists
    to detect.
    """
    line = _marker(arm, "done")
    return bool(line) and "PASS" in line and "dry-run" not in line


def _override_findings(arm: dict) -> list[str]:
    return [
        line
        for line in arm.get("finding_lines") or []
        if line.startswith("[block]") and OVERRIDE_CODE in line
    ]


def t0_10_verdict(payload: dict) -> tuple[int, str]:  # noqa: PLR0911, PLR0912
    """GREEN / RED / UNMEASURED for "the effective topology is read, not echoed"."""
    # --- 0. the instrument, before anything it measures ---------------------
    # Every judgement below is made on the presence or absence of a line. An
    # extractor that cannot match a line would make every arm look silent, and
    # silence is the shape of the defect -- so the extractor is checked first.
    selftest = (payload.get("controls") or {}).get("marker_selftest")
    if not isinstance(selftest, dict):
        return UNMEASURED, "no marker self-test in the payload; the extractor is unverified"
    for key, want_hits in (
        ("positive_hits", True),
        ("negative_hits", False),
        ("finding_positive_hits", True),
        ("finding_negative_hits", False),
    ):
        hits = selftest.get(key)
        if not isinstance(hits, list):
            return UNMEASURED, f"marker self-test has no {key}; the extractor is unverified"
        if bool(hits) is not want_hits:
            return UNMEASURED, (
                f"the line extractor is broken: {key} = {hits!r}. Every verdict below reads "
                "presence or absence of a line, so a broken extractor reports the defect's "
                "exact signature and must abstain instead"
            )

    # --- 1. all four arms must have actually run ----------------------------
    arms = {}
    for name in (*ARMS, *CONTROL_ARMS):
        arm = _arm(payload, name)
        if arm is None or "rc" not in arm:
            return UNMEASURED, f"arm {name!r} is absent from the payload"
        if arm.get("timed_out"):
            return UNMEASURED, (
                f"arm {name!r} timed out after {arm.get('elapsed_s')}s. A hang is not a refusal "
                "and must not be scored as one"
            )
        arms[name] = arm

    # --- 2. the truthful arms settle FIRST ----------------------------------
    # They are what makes the deciding arm readable. If a truthful launch is
    # itself broken, the deciding arm's refusal could be the same breakage
    # wearing a useful-looking costume, and the right answer is to abstain.
    worlds = {}
    for name in ARMS:
        arm = arms[name]
        launched = (arm.get("launched") or {}).get("world")
        got = _reported_world(arm)
        if not _trained(arm):
            return UNMEASURED, (
                f"truthful arm {name!r} did not complete a training run (rc={arm.get('rc')}). "
                "These arms exist to prove the deciding arm's refusal is a decision and not a "
                "broken launch; with one of them broken, that cannot be told apart"
            )
        if got is None:
            return RED, (
                f"truthful arm {name!r} trained without ever reporting an observed world size: "
                "nothing was read from the runtime at all"
            )
        if got != launched:
            return RED, (
                f"truthful arm {name!r} launched {launched} ranks and reported {got}. The "
                "reported figure does not come from the launch"
            )
        worlds[name] = got
    if worlds["truthful_4"] == worlds["truthful_16"]:
        return RED, (
            f"both truthful arms reported {worlds['truthful_4']} despite launching "
            f"{(arms['truthful_4'].get('launched') or {}).get('world')} and "
            f"{(arms['truthful_16'].get('launched') or {}).get('world')} ranks. A reading that "
            "does not change when the launch changes is a constant, not a reading"
        )

    # --- 3. the absent-reading control --------------------------------------
    no_tr = arms["no_torchrun"]
    line = _marker(no_tr, "consistency")
    if line is None:
        return RED, (
            "the no-torchrun arm reported no consistency step at all. An absent comparison must "
            "be ANNOUNCED as absent; a step that simply does not speak is indistinguishable "
            "from one that compared and agreed"
        )
    if _reported_world(no_tr) is not None:
        return RED, (
            f"the no-torchrun arm reported a world size with no torchrun running: {line!r}. "
            "That figure cannot have come from a runtime that does not exist"
        )
    if "skip" not in line.lower():
        return RED, (
            f"the no-torchrun arm did not say the comparison was skipped: {line!r}. "
            "'No disagreement detected' is not 'agreement verified'"
        )

    # --- 4. the deciding arm ------------------------------------------------
    bad = arms["misdeclared_16_launched_8"]
    declared_dp = (bad.get("declared") or {}).get("dp")
    launched = (bad.get("launched") or {}).get("world")
    got = _reported_world(bad)
    if got is None:
        return RED, (
            "the deciding arm reported no observed world size. Declared "
            f"dp={declared_dp}, launched {launched} ranks, and the run said neither"
        )
    if got == declared_dp and got != launched:
        return RED, (
            f"the deciding arm declared dp={declared_dp}, launched {launched} ranks, and "
            f"reported {got} -- the DECLARATION. This is the echo the row exists to rule out"
        )
    if got != launched:
        return RED, (
            f"the deciding arm launched {launched} ranks and reported {got}, which is neither "
            f"the launch nor the declaration ({declared_dp})"
        )
    if _trained(bad):
        return RED, (
            f"the deciding arm read the disagreement ({launched} launched vs dp={declared_dp} "
            "declared) and trained anyway. Reading it and not acting on it is the same "
            "published number being wrong, one step later"
        )
    if _marker(bad, "blocked") is None:
        return RED, (
            f"the deciding arm read {got} against a declared dp={declared_dp} and did not "
            "block. It stopped, but nothing says the disagreement is why"
        )
    overrides = _override_findings(bad)
    if not overrides:
        blocks = [ln for ln in bad.get("finding_lines") or [] if ln.startswith("[block]")]
        return RED, (
            "the deciding arm blocked, but no blocking finding names an effective-topology "
            f"override; it blocked for {len(blocks)} other reason(s): {blocks[:2]}. A run that "
            "stops for the wrong reason passes this arm by accident"
        )

    # --- 5. the artifact, not just the console ------------------------------
    if bad.get("manifest_stage") != "blocked":
        return RED, (
            f"the console blocked but the written manifest records stage="
            f"{bad.get('manifest_stage')!r}. The manifest is what survives the terminal"
        )
    recorded = bad.get("manifest_blocking") or ""
    if OVERRIDE_CODE not in str(recorded):
        return RED, (
            f"the manifest records that the run was blocked but not by what: {recorded!r}. "
            "A durable record of a stop with no reason is the T0-9 defect in mirror image -- "
            "said aloud once, written down never"
        )

    codes = sorted({ln.split(":")[0].replace("[block] ", "") for ln in overrides})
    return GREEN, (
        f"the effective topology is read, not echoed. Declared dp={declared_dp} against "
        f"{launched} launched ranks, the run reported {got} -- the launch -- blocked on "
        f"{', '.join(codes)} before importing torch, and recorded both the stage and the "
        f"codes in the manifest. The two truthful arms trained and reported "
        f"{worlds['truthful_4']} and {worlds['truthful_16']}, so the figure tracks the launch "
        "rather than being a constant, and the no-torchrun arm announced its comparison as "
        "skipped rather than passing an absent reading off as a matching one"
    )


# --- self-test ---------------------------------------------------------------
# The baseline is the REAL measurement (#489), trimmed to the fields the verdict
# reads. Every control is that payload with one thing changed, so a control that
# fires names exactly one cause.

_BASE = {
    "row": "T0-10",
    "controls": {
        "marker_selftest": {
            "positive_hits": ["[fs:train:consistency]  WORLD_SIZE=16: declared-vs-effective"],
            "negative_hits": [],
            "finding_positive_hits": ["[block] topology.effective_overrides_dp: declared dp=16"],
            "finding_negative_hits": [],
        },
        "misdeclared_16_launched_8": {
            "rc": 1,
            "timed_out": False,
            "declared": {"nodes": 4, "gpus_per_node": 4, "dp": 16},
            "launched": {"nnodes": 2, "nproc_per_node": 4, "world": 8},
            "markers": [
                "[fs:train:topology]      topology: 16 GPUs = 4 nodes x 4 GPUs/node",
                "[fs:train:consistency]   WORLD_SIZE=8: declared-vs-effective compared",
                "[fs:train:blocked]       2/5 finding(s) block; stopping BEFORE an allocation "
                "is burned -- no GPU touched, no torch imported",
            ],
            "finding_lines": [
                "[block] topology.effective_overrides_dp: declared dp=16 but the runtime "
                "constructed dp=8",
                "[block] topology.effective_overrides_nodes: declared nodes=4 but the runtime "
                "constructed nodes=2",
                "[warn] topology.master_addr_unrecorded: nodes=4 but no master_addr",
            ],
            "manifest_stage": "blocked",
            "manifest_blocking": (
                "['topology.effective_overrides_dp', 'topology.effective_overrides_nodes']"
            ),
        },
        "no_torchrun": {
            "rc": 0,
            "timed_out": False,
            "declared": {"nodes": 4, "gpus_per_node": 4, "dp": 16},
            "launched": {"nnodes": None, "nproc_per_node": None, "world": None},
            "markers": [
                "[fs:train:consistency]   no torchrun runtime (WORLD_SIZE unset); "
                "declared-vs-effective comparison skipped on the driver",
                "[fs:train:done]          dry-run PASS: full validation prologue ran, "
                "0 GPUs touched",
            ],
            "finding_lines": ["[warn] topology.master_addr_unrecorded: nodes=4"],
            "manifest_stage": "dry_run",
            "manifest_blocking": None,
        },
    },
    "arms": {
        "truthful_4": {
            "rc": 0,
            "timed_out": False,
            "declared": {"nodes": 1, "gpus_per_node": 4, "dp": 4},
            "launched": {"nnodes": 1, "nproc_per_node": 4, "world": 4},
            "markers": [
                "[fs:train:consistency]   WORLD_SIZE=4: declared-vs-effective compared",
                "[fs:train:done]          PASS",
            ],
            "finding_lines": ["[   ok] topology.validate_summary: 5 checks, 0 blocking"],
            "manifest_stage": "done",
            "manifest_blocking": None,
        },
        "truthful_16": {
            "rc": 0,
            "timed_out": False,
            "declared": {"nodes": 4, "gpus_per_node": 4, "dp": 16},
            "launched": {"nnodes": 4, "nproc_per_node": 4, "world": 16},
            "markers": [
                "[fs:train:consistency]   WORLD_SIZE=16: declared-vs-effective compared",
                "[fs:train:done]          PASS",
            ],
            "finding_lines": ["[warn] topology.master_addr_unrecorded: nodes=4"],
            "manifest_stage": "done",
            "manifest_blocking": None,
        },
    },
}


def _mk(**overrides: object) -> dict:
    """Deep-copy the baseline, then apply `section__arm__field=value` overrides.

    A sentinel of ``None`` for a FIELD deletes it, because "the payload does not
    carry this key" and "it carries None" are different measurements and some
    controls need the first one.
    """
    payload = json.loads(json.dumps(_BASE))
    for dotted, value in overrides.items():
        section, arm, field = dotted.split("__")
        target = payload[section][arm]
        if value == "__DELETE__":
            target.pop(field, None)
        else:
            target[field] = value
    return payload


_CONSISTENCY_16 = "[fs:train:consistency]   WORLD_SIZE=16: declared-vs-effective compared"
_DONE_PASS = "[fs:train:done]          PASS"

CONTROLS: list[tuple[str, int, dict, str]] = [
    ("real_measurement", GREEN, _mk(), "the payload as measured on sixteen GB200 GPUs"),
    (
        "deciding_arm_echoes_the_declaration",
        RED,
        _mk(
            controls__misdeclared_16_launched_8__markers=[
                _CONSISTENCY_16,
                "[fs:train:blocked]       2/5 finding(s) block",
            ]
        ),
        "the whole row in one control: declared 16, launched 8, reported 16. An implementation "
        "that echoes the config produces exactly this and nothing else changes",
    ),
    (
        "deciding_arm_trains_anyway",
        RED,
        _mk(
            controls__misdeclared_16_launched_8__markers=[
                "[fs:train:consistency]   WORLD_SIZE=8: declared-vs-effective compared",
                _DONE_PASS,
            ]
        ),
        "reading the disagreement and training through it is the same wrong number one step "
        "later, and it is the likelier defect once the reading works",
    ),
    (
        "deciding_arm_blocks_for_another_reason",
        RED,
        _mk(
            controls__misdeclared_16_launched_8__finding_lines=[
                "[block] memory.weights_do_not_fit: 2.88 GiB needed, 0.4 GiB free"
            ]
        ),
        "a run that stops for an unrelated reason passes the deciding arm by accident; the "
        "blocking finding must NAME the override",
    ),
    (
        "deciding_arm_stops_but_says_nothing",
        RED,
        _mk(
            controls__misdeclared_16_launched_8__markers=[
                "[fs:train:consistency]   WORLD_SIZE=8: declared-vs-effective compared"
            ]
        ),
        "stopping without a blocked marker leaves the operator unable to tell a refusal from a "
        "crash -- which is precisely what the exit code cannot tell them either",
    ),
    (
        "reading_is_a_constant",
        RED,
        _mk(arms__truthful_4__markers=[_CONSISTENCY_16, _DONE_PASS]),
        "both truthful arms reporting 16 while launching 4 and 16 means the figure never "
        "changes; one green run is compatible with a hard-coded answer, which is why there "
        "are two",
    ),
    (
        "absent_reading_reported_as_agreement",
        RED,
        _mk(
            controls__no_torchrun__markers=[
                "[fs:train:consistency]   declared-vs-effective compared: no disagreement",
                "[fs:train:done]          dry-run PASS",
            ]
        ),
        "the quietest failure available: with no runtime to read, 'nothing disagreed' gets "
        "published as 'they agree'",
    ),
    (
        "absent_reading_invents_a_world_size",
        RED,
        _mk(
            controls__no_torchrun__markers=[
                _CONSISTENCY_16,
                "[fs:train:done]          dry-run PASS",
            ]
        ),
        "a world size reported with no torchrun running cannot have been read from anywhere "
        "except the config",
    ),
    (
        "no_torchrun_arm_silent",
        RED,
        _mk(controls__no_torchrun__markers=["[fs:train:done]          dry-run PASS"]),
        "a consistency step that simply does not speak is indistinguishable from one that "
        "compared and agreed; absence must be announced",
    ),
    (
        "console_blocked_manifest_did_not",
        RED,
        _mk(controls__misdeclared_16_launched_8__manifest_stage="done"),
        "the mirror image of the T0-9 defect measured the same week: said aloud once, written "
        "down never. The manifest is what survives the terminal",
    ),
    (
        "manifest_blocked_without_a_reason",
        RED,
        _mk(controls__misdeclared_16_launched_8__manifest_blocking="[]"),
        "a durable record that a run stopped, with nothing recording why, cannot be audited "
        "six months later by anyone who was not in the terminal",
    ),
    (
        "uninformative_rc_on_a_correct_arm",
        GREEN,
        _mk(controls__misdeclared_16_launched_8__rc=1),
        "torchrun collapses every worker exit code into its own, so a correct refusal and a "
        "segfault leave by the same door. Scoring rc would score torchrun -- this asserts the "
        "verdict does not",
    ),
    (
        "truthful_arm_did_not_train",
        UNMEASURED,
        _mk(arms__truthful_16__markers=[_CONSISTENCY_16]),
        "with a truthful launch broken, the deciding arm's refusal might be the same breakage; "
        "that is an abstention, never a RED against the subject",
    ),
    (
        "truthful_arm_misreports_its_world",
        RED,
        _mk(
            arms__truthful_4__markers=[
                "[fs:train:consistency]   WORLD_SIZE=7: declared-vs-effective compared",
                _DONE_PASS,
            ]
        ),
        "a figure matching neither the launch nor the declaration is not a reading, and it "
        "would otherwise hide behind the deciding arm being right",
    ),
    (
        "deciding_arm_timed_out",
        UNMEASURED,
        _mk(controls__misdeclared_16_launched_8__timed_out=True),
        "a hang looks exactly like a refusal from the outside -- no output, no artifacts, "
        "nonzero rc -- and must never be scored as one",
    ),
    (
        "deciding_arm_absent",
        UNMEASURED,
        _mk(controls__misdeclared_16_launched_8__rc="__DELETE__"),
        "a missing arm is a measurement that did not happen",
    ),
    (
        "extractor_cannot_match_a_marker",
        UNMEASURED,
        _mk(controls__marker_selftest__positive_hits=[]),
        "every verdict here reads presence or absence of a line, so a blind extractor reports "
        "the defect's exact signature on a healthy subject",
    ),
    (
        "extractor_matches_everything",
        UNMEASURED,
        _mk(controls__marker_selftest__negative_hits=["  0%| 0/20 [00:00<?, ?it/s]"]),
        "an over-matching extractor turns progress-bar noise into markers, and the arm checks "
        "become unfalsifiable rather than false",
    ),
    (
        "finding_extractor_blind",
        UNMEASURED,
        _mk(controls__marker_selftest__finding_positive_hits=[]),
        "the deciding arm's content is two [block] lines; an extractor that cannot see them "
        "would report 'blocked for no named reason' on a perfectly healthy run",
    ),
    (
        "empty_payload",
        UNMEASURED,
        {},
        "nothing to score, and the 95 door is the only honest exit",
    ),
]


def _self_test() -> int:
    bad = 0
    for name, want, payload, why in CONTROLS:
        got, reason = t0_10_verdict(payload)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {name} -> {got:>2} (want {want:>2})   {why}")
        if not ok:
            print(f"       got: {reason[:180]}")
    total = len(CONTROLS)
    print(f"{'CLEAR' if not bad else 'FAILED'}: {total - bad} of {total}")
    return GREEN if not bad else RED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--payload")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    if not args.payload:
        print("REFUSE: --payload is required without --self-test", file=sys.stderr)
        return REFUSE
    try:
        # An unreadable payload is a MEASUREMENT that did not happen, not a row
        # that failed, so it leaves by the 95 door rather than as a traceback.
        payload = json.loads(open(args.payload).read())  # noqa: PTH123, SIM115
    except Exception as exc:  # noqa: BLE001
        print(f"UNMEASURED: payload unreadable: {type(exc).__name__}: {exc}")
        return UNMEASURED
    code, reason = t0_10_verdict(payload)
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}[code]
    print(f"{label}: {reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
