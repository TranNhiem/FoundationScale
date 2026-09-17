"""Adjudicate T0-9: `code.status is a real commit`, and WHICH half of it fails.

Pure verdict over a payload dict. No I/O, no torch, no git, so `python3 -S` on a
laptop and on a tray reach the same answer.

The row's declared control arm is compound and its two halves are independent:

  (a) a COPY must RECORD status=not_a_repository
  (b) and the run must SAY SO LOUDLY, not proceed silently

A bare "RED" cannot tell a reader which one failed, and the difference matters:
(a) failing would mean the manifest ATTRIBUTES a run to a commit it is not
running, which is the original sin this whole provenance subsystem was built to
prevent. (b) failing means the record is truthful but only a reader who opens
the manifest learns it. The first is a lie; the second is a quiet truth. This
file refuses to collapse them.

It is also written to be able to RETRACT. If the run stream does mention the
absence, the verdict is GREEN and the row's asserted RED was wrong -- which has
happened three times this week on this matrix, so it is not a hypothetical.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

GREEN, RED, UNMEASURED, REFUSE = 0, 5, 95, 96

HEX40 = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_ARMS = ("capture_on_copy", "dry_run_on_copy")
REQUIRED_CONTROLS = (
    "capture_on_real_repo",
    "copy_parent_not_repo",
    "detector_selftest",
    "dry_run_on_real_repo",
    "expected_head",
)


def t0_9_verdict(payload: dict[str, Any]) -> tuple[int, str]:
    """Return (exit_code, one-paragraph reason).

    Order, and every step of it is a real failure mode seen on this matrix:

     1. a required arm is missing                         -> UNMEASURED 95
     2. a required control is missing                     -> UNMEASURED 95
     3. the stream detector cannot fire                   -> REFUSE 96
     4. the stream detector fires on anything             -> REFUSE 96
     5. the "copy" is inside an enclosing repository      -> REFUSE 96
     6. the real checkout did not read as a repository    -> REFUSE 96
     7. the real capture's commit is not 40-hex           -> REFUSE 96
     8. no independently-read HEAD to check it against    -> REFUSE 96
     9. the capture named a commit that is NOT HEAD       -> RED 5
    10. the control dry-run did not succeed               -> UNMEASURED 95
    11. the control wrote no readable manifest            -> UNMEASURED 95
    12. the control's MANIFEST is itself wrong            -> REFUSE 96
    13. the CONTROL run printed the tokens too            -> REFUSE 96
    14. the copy's capture could not be read back         -> UNMEASURED 95
    15. the copy did NOT record not_a_repository          -> RED 5
    16. the copy recorded a commit anyway                 -> RED 5
    17. the copy wrote no readable manifest               -> UNMEASURED 95
    18. the copy's MANIFEST disagrees with its capture    -> RED 5
    19. the copy's run stream DID say so                  -> GREEN 0
    20. the copy's run failed AND said nothing            -> UNMEASURED 95
    21. recorded truthfully, said nothing                 -> RED 5

    Steps 7-9 and 11-12, 17-18 were added after the first run of this row. The
    first version checked the CAPTURE FUNCTION only, and a 40-hex test alone
    cannot tell a real commit from a hard-coded one -- hence the independently
    read HEAD at step 8. Steps 11-12 and 17-18 move the question to where the
    claim actually lives: a reader trusts run_manifest.json, not the function
    that fed it, and the two are different code paths that can disagree. Step 13
    was added after the FIX for this row was written: the deciding arm is a
    run that speaks, and a fix that made every run speak would have passed it.
    """
    arms = payload.get("arms") or {}
    controls = payload.get("controls") or {}

    for name in REQUIRED_ARMS:
        if name not in arms:
            return UNMEASURED, (
                f"the payload carries no '{name}' arm, so the deciding half of this row was "
                "never exercised; a row cannot be scored on the arms that happened to run"
            )
    for name in REQUIRED_CONTROLS:
        if name not in controls:
            return UNMEASURED, (
                f"control '{name}' is missing. Every control here exists to stop a specific "
                "false reading, and an absent control is an unproven assumption, not a pass"
            )

    det = controls["detector_selftest"]
    if not det.get("hits"):
        return REFUSE, (
            "the stream detector did not fire on a synthetic line written to contain its own "
            "tokens, so it cannot witness anything. Reporting 'the run said nothing' with a "
            "detector that has never been shown to fire would be reporting the detector"
        )
    if det.get("negative_line_hits"):
        return REFUSE, (
            "the stream detector fired on an ordinary training line "
            f"({det['negative_line_hits']}), so a hit carries no information and a GREEN from "
            "it would be an artefact of the pattern rather than of the framework"
        )

    parent = controls["copy_parent_not_repo"]
    if parent.get("inside_work_tree") or parent.get("dot_git_present"):
        return REFUSE, (
            f"the supposedly-unversioned copy at {parent.get('copy')!r} is itself inside a git "
            "work tree (git rev-parse walks UPWARD), so it is not a copy of anything and every "
            "arm below it would have measured the enclosing repository instead"
        )

    real = controls["capture_on_real_repo"]
    if real.get("status") == "not_a_repository":
        return REFUSE, (
            "capture on a REAL git checkout also came back not_a_repository, so the capture "
            "path answers the same way for everything and cannot distinguish the two cases. "
            "Without this control, a function that always said not_a_repository would have "
            "scored as a correct detection of the copy"
        )
    commit = real.get("commit")
    if not isinstance(commit, str) or not HEX40.match(commit):
        return REFUSE, (
            f"capture on the real checkout produced commit {commit!r}, which is not a 40-hex "
            "object id, so the positive side of the control was never established and the "
            "negative side means nothing on its own"
        )

    want_head = (controls["expected_head"] or {}).get("commit")
    if not isinstance(want_head, str) or not HEX40.match(want_head):
        return REFUSE, (
            f"the probe could not read the checkout's HEAD independently (got {want_head!r}), "
            "so there is nothing to check the recorded commit AGAINST and a hard-coded sha "
            "would satisfy the 40-hex test forever"
        )
    if commit != want_head:
        return RED, (
            f"the capture recorded commit {commit!r} while the checkout's actual HEAD is "
            f"{want_head!r}. A manifest naming the wrong commit is worse than one naming none"
        )

    ctl_run = controls["dry_run_on_real_repo"]
    if ctl_run.get("rc") != 0:
        return UNMEASURED, (
            f"the control dry-run from the real checkout exited {ctl_run.get('rc')} rather than "
            "0, so the command itself is not in a state where the copy's behaviour can be "
            "attributed to the copy"
        )

    ctl_man = ctl_run.get("manifest_code")
    if not isinstance(ctl_man, dict):
        return UNMEASURED, (
            "the control run wrote no readable manifest "
            f"({ctl_run.get('manifest_error') or 'no code section'}), so the artifact path this "
            "row is actually about was never exercised on the positive side"
        )
    if ctl_man.get("status") == "not_a_repository" or ctl_man.get("commit") != want_head:
        return REFUSE, (
            f"the manifest written from the REAL checkout records status="
            f"{ctl_man.get('status')!r} commit={ctl_man.get('commit')!r}, which does not match "
            f"HEAD {want_head!r}. The artifact side of the control is not established, so the "
            "copy's manifest cannot be read as a contrast"
        )

    # The hole this closes: the deciding evidence below is "the copy's run
    # MENTIONED it". A framework that printed the same words on EVERY run --
    # including this control, which is perfectly attributable -- would satisfy
    # that and be a new defect wearing the old one's costume, because a line
    # printed unconditionally is a line nobody reads. It is also the shape a
    # careless FIX for this row would take. So the control must be silent, and
    # its silence is checked before the copy's speech is allowed to mean
    # anything. A token appearing here for an unrelated reason -- "provenance"
    # inside a path, say -- lands in the same trap and is caught by the same gate.
    if ctl_run.get("hits"):
        return REFUSE, (
            f"the control run, from a real checkout with commit {want_head[:12]}, printed "
            f"{sorted(ctl_run['hits'])!r} -- the same vocabulary the copy is scored on. A run "
            "that is perfectly attributable is saying it is not, or the detector is matching "
            "something incidental; either way a hit on the copy would not be evidence about "
            "the copy"
        )

    cap = arms["capture_on_copy"]
    if cap.get("parse_error") or "status" not in cap:
        return UNMEASURED, (
            "the copy's capture could not be read back "
            f"({cap.get('parse_error') or 'no status field'}), so nothing about the recording "
            "half was measured"
        )

    if cap.get("status") != "not_a_repository":
        return RED, (
            f"a tree with no git metadata was recorded as status={cap.get('status')!r} rather "
            "than not_a_repository. This is the severe half of the row: the manifest is the "
            "artifact a reader trusts to say WHICH CODE RAN, and here it is not saying 'I do "
            "not know'"
        )
    if cap.get("commit") is not None:
        return RED, (
            f"the copy has no git metadata yet the manifest recorded commit {cap['commit']!r}. "
            "That is the exact failure this provenance subsystem was built to prevent -- a run "
            "attributed to a commit it is not running -- and it is worse than silence"
        )

    run = arms["dry_run_on_copy"]
    man = run.get("manifest_code")
    if not isinstance(man, dict):
        return UNMEASURED, (
            "the copy's run wrote no readable manifest "
            f"({run.get('manifest_error') or 'no code section'}), so the recording half was not "
            "measured where the claim actually lives"
        )
    if man.get("status") != "not_a_repository" or man.get("commit") is not None:
        return RED, (
            f"the WRITTEN manifest for an unversioned tree records status="
            f"{man.get('status')!r} commit={man.get('commit')!r}. The capture function answered "
            f"correctly ({cap.get('status')!r}/{cap.get('commit')!r}) and the artifact does not, "
            "so the defect is in the path between them -- and the artifact is the thing a reader "
            "trusts"
        )
    hits = run.get("hits") or []
    if hits:
        return GREEN, (
            "both halves hold, and this RETRACTS the row's asserted RED. A tree with no git "
            f"metadata is recorded as status=not_a_repository with commit null, AND the run "
            f"itself says so on its own output stream (matched {hits}), so a user who never "
            "opens the manifest still cannot mistake an unversioned run for a versioned one"
        )
    if run.get("rc") != 0:
        return UNMEASURED, (
            f"the copy's run exited {run.get('rc')} while saying nothing about provenance, and "
            f"its last line was {run.get('tail')!r}. A run that failed for an unrelated reason "
            "cannot be used to show that a SUCCESSFUL run would have been silent"
        )
    return RED, (
        "RED, and precisely on the second half of the claim. The recording half is CORRECT and "
        "is not in dispute: a tree with no git metadata is recorded as status=not_a_repository "
        "with commit null, so the manifest never attributes the run to a commit it is not "
        f"running. But the run proceeded to a clean exit {run.get('rc')} over "
        f"{run.get('stream_lines')} lines of output without one of them mentioning it (none of "
        "the probed tokens appeared, and the detector was shown to fire on a line that contains "
        "them). Nothing outside the provenance module reads code.status, so there is no site "
        "that could have spoken. The practical shape of the defect: a user who unpacks a "
        "tarball, trains, and reads only the console has no signal at all that the run is "
        "unattributable -- the truth is in the manifest and only in the manifest. That is a "
        "quiet truth rather than a lie, which is why this row is RED on (b) and not on (a), and "
        "the difference belongs in the published table rather than being collapsed into one word"
    )


# --------------------------------------------------------------------------
# controls
# --------------------------------------------------------------------------
def _mk(**over: Any) -> dict[str, Any]:
    """A payload in the shape the probe emits, RED-by-default on half (b)."""
    p: dict[str, Any] = {
        "controls": {
            "capture_on_real_repo": {
                "rc": 0, "status": "clean", "commit": "f" * 40, "dirty_files": 0,
            },
            "copy_parent_not_repo": {
                "copy": "/x/copy", "inside_work_tree": False, "dot_git_present": False,
            },
            "detector_selftest": {
                "line": "... not_a_repository ...",
                "hits": ["not_a_repository"],
                "negative_line_hits": [],
            },
            "dry_run_on_real_repo": {
                "rc": 0, "stream_lines": 40, "hits": [], "tail": "ok",
                "manifest_code": {"status": "not_captured", "commit": "f" * 40},
                "manifest_error": None,
            },
            "expected_head": {"rc": 0, "commit": "f" * 40},
        },
        "arms": {
            "capture_on_copy": {
                "rc": 0, "status": "not_a_repository", "commit": None, "dirty_files": 0,
            },
            "dry_run_on_copy": {
                "rc": 0, "stream_lines": 40, "hits": [], "tail": "dry-run PASS",
                "manifest_code": {"status": "not_a_repository", "commit": None},
                "manifest_error": None,
            },
        },
    }
    for dotted, value in over.items():
        section, name, field = dotted.split("__", 2)
        if value is _DROP:
            p[section].pop(name, None)
        else:
            p[section][name][field] = value
    return p


class _Drop:
    pass


_DROP = _Drop()

CONTROLS: list[tuple[str, int, dict[str, Any], str]] = [
    ("baseline_red_on_half_b", RED, _mk(),
     "the expected shipped shape: recorded truthfully, said nothing. RED on (b) only"),
    ("retraction_is_reachable", GREEN,
     _mk(arms__dry_run_on_copy__hits=["not_a_repository"]),
     "if the run DOES say so the verdict must be GREEN -- an adjudicator that cannot reach "
     "green for the row it is scoring is a rubber stamp for the status already written down"),
    ("missing_capture_arm", UNMEASURED, _mk(arms__capture_on_copy__x=_DROP),
     "no deciding arm means no measurement, never a pass"),
    ("missing_run_arm", UNMEASURED, _mk(arms__dry_run_on_copy__x=_DROP),
     "half (b) cannot be scored without the run"),
    ("missing_detector_control", UNMEASURED, _mk(controls__detector_selftest__x=_DROP),
     "an absent control is an unproven assumption"),
    ("missing_parent_control", UNMEASURED, _mk(controls__copy_parent_not_repo__x=_DROP),
     "without it, a copy inside a repo would be scored as a copy"),
    ("missing_real_repo_control", UNMEASURED, _mk(controls__capture_on_real_repo__x=_DROP),
     "without it, a capture that always says not_a_repository scores as correct"),
    ("missing_control_run", UNMEASURED, _mk(controls__dry_run_on_real_repo__x=_DROP),
     "without it, a broken command scores as a silent framework"),
    ("detector_cannot_fire", REFUSE, _mk(controls__detector_selftest__hits=[]),
     "a detector that does not match its own planted line measures its own vocabulary"),
    ("detector_fires_on_anything", REFUSE,
     _mk(controls__detector_selftest__negative_line_hits=["provenance"]),
     "a hit that an ordinary training line also produces carries no information"),
    ("copy_is_inside_a_work_tree", REFUSE,
     _mk(controls__copy_parent_not_repo__inside_work_tree=True),
     "git rev-parse walks UPWARD, so a copy under any repo measures the enclosing tree"),
    ("copy_kept_a_dot_git", REFUSE,
     _mk(controls__copy_parent_not_repo__dot_git_present=True),
     "a stray .git fragment makes the copy a repository again"),
    ("real_repo_reads_as_non_repo", REFUSE,
     _mk(controls__capture_on_real_repo__status="not_a_repository"),
     "the capture path answering identically for both cases cannot distinguish them"),
    ("real_repo_has_no_commit", REFUSE, _mk(controls__capture_on_real_repo__commit=None),
     "the positive side of the control was never established"),
    ("real_repo_commit_not_hex40", REFUSE,
     _mk(controls__capture_on_real_repo__commit="HEAD"),
     "a branch name is not an object id; it would pass a truthiness check and prove nothing"),
    ("control_run_failed", UNMEASURED, _mk(controls__dry_run_on_real_repo__rc=96),
     "if the command does not work from the real checkout, the copy's behaviour is not the "
     "copy's fault"),
    ("copy_capture_unreadable", UNMEASURED,
     _mk(arms__capture_on_copy__parse_error="JSONDecodeError"),
     "an unparsed arm is unmeasured, not a defect"),
    ("copy_recorded_a_wrong_status", RED,
     _mk(arms__capture_on_copy__status="clean"),
     "the severe half: an unversioned tree recorded as clean"),
    ("copy_recorded_a_commit", RED,
     _mk(arms__capture_on_copy__commit="a" * 40),
     "the original sin -- attributing a run to a commit it is not running"),
    ("copy_run_failed_and_was_silent", UNMEASURED,
     _mk(arms__dry_run_on_copy__rc=1),
     "a run that died for an unrelated reason cannot witness that a successful one is silent"),
    ("missing_expected_head_control", UNMEASURED, _mk(controls__expected_head__x=_DROP),
     "without an independently read HEAD there is nothing to check the recorded commit against"),
    ("head_unreadable", REFUSE, _mk(controls__expected_head__commit=None),
     "a hard-coded sha would satisfy the 40-hex test forever; this is what stops it"),
    ("capture_named_the_wrong_commit", RED,
     _mk(controls__capture_on_real_repo__commit="b" * 40),
     "a manifest naming the WRONG commit is worse than one naming none -- and a 40-hex check "
     "alone cannot see it"),
    ("control_wrote_no_manifest", UNMEASURED,
     _mk(controls__dry_run_on_real_repo__manifest_code=None),
     "the artifact path was never exercised on the positive side"),
    ("control_manifest_disagrees_with_head", REFUSE,
     _mk(controls__dry_run_on_real_repo__manifest_code={"status": "clean", "commit": "c" * 40}),
     "if the written manifest does not match HEAD on the REAL checkout, the copy's manifest is "
     "not a contrast against anything"),
    ("control_shouted_too", REFUSE,
     _mk(controls__dry_run_on_real_repo__hits=["not_a_repository"]),
     "a warning printed on EVERY run, attributable or not, satisfies the deciding arm while "
     "being the same defect as silence: nobody reads a line that is always there. This is also "
     "the likeliest shape of a careless fix for this very row"),
    ("copy_wrote_no_manifest", UNMEASURED,
     _mk(arms__dry_run_on_copy__manifest_code=None),
     "the recording half must be measured where the claim lives, in the artifact"),
    ("artifact_disagrees_with_capture", RED,
     _mk(arms__dry_run_on_copy__manifest_code={"status": "clean", "commit": "d" * 40}),
     "the capture function answering correctly while the WRITTEN manifest does not is a defect "
     "in the path between them, and the artifact is what a reader trusts"),
    ("copy_run_failed_but_spoke", GREEN,
     _mk(arms__dry_run_on_copy__rc=96, arms__dry_run_on_copy__hits=["not a repository"]),
     "a LOUD refusal is the best possible outcome for half (b) and must not be scored as a "
     "failure merely because the exit code is non-zero"),
]


def _self_test() -> int:
    bad = 0
    for name, want, payload, why in CONTROLS:
        got, reason = t0_9_verdict(payload)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'} {name} -> {got:>2} (want {want:>2})   {why}")
        if not ok:
            print(f"       got: {reason[:160]}")
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
        # Same idiom as every other shipped adjudicator, and for the same reason:
        # an unreadable payload is a MEASUREMENT that did not happen, not a row
        # that failed, so it must leave by the 95 door rather than as a traceback.
        payload = json.loads(open(args.payload).read())  # noqa: PTH123, SIM115
    except Exception as exc:  # noqa: BLE001
        print(f"UNMEASURED: payload unreadable: {type(exc).__name__}: {exc}")
        return UNMEASURED
    code, reason = t0_9_verdict(payload)
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}[code]
    print(f"{label}: {reason}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
