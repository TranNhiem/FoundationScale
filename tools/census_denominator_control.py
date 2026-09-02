#!/usr/bin/env python3
"""MUST_FIRE / MUST_PASS / WIRING controls for
tools/count_census_modules.py (#88).

Usage: python3 tools/census_denominator_control.py [REAL_CENSUS_JSON]

Two payload modes, because doctrine 3 needs both halves:

  no argument      -- build a shape-exact replica of the writer payload
                      (same wrapper keys, same record structure, 168
                      records, tonight's population; any count > 2
                      carries the leg) and run every leg against it. This
                      is the doctored input doctrine 3 demands a
                      MUST_FIRE be observed on: the red stays
                      reproducible any time, with no launch-time artifact
                      at hand.
  REAL_CENSUS_JSON -- run every leg against an actual launch-time census
                      artifact, the wrapped {"adapter_modules", "source"}
                      payload launchers/lora_target_census.py writes to
                      --out. A fixture proves the code runs; the real
                      artifact proves the SEAM. If the real payload's
                      shape cannot show the defect firing (a bare list),
                      the fire leg is UNMEASURED and this control exits
                      nonzero -- never PASS on an unobserved observation
                      (doctrines 1+3).

Legs -- each prints its denominator (doctrine 2):

  WIRING     the launcher still invokes THIS counter by name. A control
             over a counter nothing calls is decoration.
  MUST_FIRE  the pre-#88-fix logic -- len() of the parsed JSON root,
             transcribed verbatim below -- is EXECUTED against the
             payload and OBSERVED printing the root count while the
             census declares many, and that print is OBSERVED matching
             the launcher's numeric guard, i.e. passing SILENTLY.
  MUST_PASS  the SHIPPED counter, executed as a subprocess exactly the
             way the launcher invokes it -- never a re-typed paraphrase
             -- on the same payload: it must print ONLY the true count on
             stdout. Then four garbage roots must each be REFUSED:
             nonzero status, stderr naming the shape found, nothing
             countable on stdout (doctrine 4).
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve()
COUNTER = HERE.with_name("count_census_modules.py")
LAUNCHER = HERE.parent.parent / "launchers" / "launch_g4e4b_lora_1tray.sh"
# The launcher's numeric guard -- the one 2 silently passed pre-#88. The
# guard-match check belongs ONLY to the MUST_FIRE leg, where passing the
# guard IS the defect; for the shipped counter a bare-integer stdout is
# the contract (zero is honestly printable; the launcher's vacuity arm
# refuses 0 by name -- refusing it HERE would be doctrine 5's symmetric
# false red on documented behavior).
NUMERIC_GUARD = re.compile(r"^[1-9][0-9]*$")
PURE_INT = re.compile(r"^(0|[1-9][0-9]*)$")

GARBAGE_ROOTS = {
    ("bare name->record mapping (the pre-#88 comment's imaginary 'flat' contract)"): json.dumps(
        {
            "module.decoder.layers.0.self_attn.linear_qkv": {
                "fqn": "module.decoder.layers.0.self_attn.linear_qkv",
                "out_features": 3072,
                "in_features": 2048,
            },
            "module.decoder.layers.0.self_attn.linear_proj": {
                "fqn": "module.decoder.layers.0.self_attn.linear_proj",
                "out_features": 2048,
                "in_features": 2048,
            },
        }
    ),
    "object missing 'adapter_modules'": json.dumps(
        {
            "module_inventory": ["a", "b", "c"],
            "producer": "not the census writer",
        }
    ),
    "wrapped object whose 'adapter_modules' member is not a list": json.dumps(
        {"adapter_modules": {"count": 168}, "source": "garbage"}
    ),
    "scalar root": "17",
}


def pre_fix_len_of_root(d: object) -> int:
    """The counter EXACTLY as the launcher shipped it pre-#88-fix: len()
    of the parsed root, dict or list alike -- a dict counting its KEYS.
    Transcribed from the deleted inline block, not paraphrased: any edit
    here silently un-tests the defect this control exists to keep dead.
    Never 'repair' this function; it is the red half of the control."""
    if isinstance(d, (dict, list)):
        return len(d)
    raise TypeError(f"census root is {type(d).__name__}, not an object/array of module records")


def true_declared_count(obj: object, path: Path) -> int:
    """The control's OWN derivation of the truth off the artifact --
    independent of the shipped counter, or the comparison asserts
    nothing."""
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        seq = obj.get("adapter_modules")
        if isinstance(seq, list):
            return len(seq)
    raise SystemExit(
        f"CONTROL UNMEASURED: {path} is not a census shape this control "
        f"can derive truth from (wrapped 'adapter_modules' list or bare "
        f"list). A control that cannot state its denominator asserts "
        f"nothing (doctrines 1+2)."
    )


def build_writer_shape_fixture(path: Path) -> None:
    """Shape-exact replica of the lora_target_census.py payload at its
    lines ~340-352: same wrapper keys, same record structure, 168 records
    (tonight's population; any count > 2 carries the leg). The doctored
    input that keeps the red observable without tonight's artifact."""
    entries = [
        {
            "fqn": f"module.decoder.layers.{i}.self_attn.linear_qkv",
            "out_features": 4096,
            "in_features": 4096,
        }
        for i in range(168)
    ]
    payload = {
        "adapter_modules": entries,
        "source": (
            "launchers/lora_target_census.py launch-time live-module "
            "census (#78) -- BLOCKER #88 control fixture, shape-replica "
            "of the writer payload"
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    args = sys.argv[1:]
    if len(args) > 1:
        raise SystemExit(
            "usage: census_denominator_control.py [REAL_CENSUS_JSON] -- "
            "no argument: shape-exact writer-payload fixture (perpetual "
            "doctored-input red); with an argument: all legs grade the "
            "real artifact, the seam itself"
        )
    with tempfile.TemporaryDirectory(prefix="census-ctl-") as td:
        if args:
            census = Path(args[0])
            payload_kind = f"REAL artifact {census}"
        else:
            census = Path(td) / "writer_shape_fixture.json"
            build_writer_shape_fixture(census)
            payload_kind = (
                "built fixture (shape-exact replica of the "
                "lora_target_census.py payload, 168 records)"
            )
        try:
            obj = json.loads(census.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"CONTROL UNMEASURED: census payload {census} is not "
                f"readable JSON ({type(exc).__name__}: {exc}) -- a "
                f"control that cannot parse its input cannot state a "
                f"denominator; UNMEASURED, never PASS (doctrines 2+4)."
            ) from exc
        true_n = true_declared_count(obj, census)

        # WIRING leg -- the production path must BE the controlled path.
        if not COUNTER.is_file():
            raise SystemExit(
                f"CONTROL FAIL (wiring): {COUNTER} missing -- the "
                f"launcher counts with a counter that does not exist."
            )
        if not LAUNCHER.is_file():
            raise SystemExit(
                f"CONTROL UNMEASURED (wiring): launcher {LAUNCHER} not "
                f"found -- cannot prove the controlled counter is the "
                f"production counter."
            )
        if "count_census_modules.py" not in LAUNCHER.read_text(encoding="utf-8"):
            raise SystemExit(
                f"CONTROL FAIL (wiring): {LAUNCHER.name} never names "
                f"count_census_modules.py -- production drifted from the "
                f"controlled counter; a control over dead code is "
                f"decoration."
            )
        print(
            f"WIRING observed (denominator: 1 launcher): {LAUNCHER.name} "
            f"invokes {COUNTER.name} -- the controlled counter IS the "
            f"production counter."
        )

        # MUST_FIRE leg -- the defect, observed.
        try:
            old_n = pre_fix_len_of_root(obj)
        except TypeError:
            raise SystemExit(
                f"CONTROL UNMEASURED (MUST_FIRE): root of {census} is "
                f"not dict/list, so the pre-fix logic would have ERRORED "
                f"here rather than miscounted -- the #88 silent form is "
                f"not observable on this payload (doctrine 3)."
            ) from None
        if old_n == true_n:
            raise SystemExit(
                f"CONTROL UNMEASURED (MUST_FIRE): pre-fix len(root) "
                f"prints the TRUE count {true_n} on this payload, so the "
                f"#88 miscount cannot be OBSERVED here. Supply the "
                f"wrapped census lora_target_census.py writes, or run "
                f"with no argument for the fixture. An unobserved fire "
                f"leg is UNMEASURED, never PASS (doctrines 1+3)."
            )
        if not (isinstance(obj, dict) and NUMERIC_GUARD.fullmatch(str(old_n))):
            raise SystemExit(
                f"CONTROL UNMEASURED (MUST_FIRE): pre-fix count {old_n} "
                f"vs true {true_n}, but the misfire is not the measured "
                f"#88 form (wrapped dict whose print passes the launcher "
                f"guard); the defect being controlled for is not firing "
                f"as measured."
            )
        print(
            f"MUST_FIRE red observed (denominator: {true_n} module "
            f"records declared off {payload_kind}): the pre-fix "
            f"len(root) counter printed {old_n} -- the root-KEY count of "
            f"the wrapped object -- and '{old_n}' matches the launcher "
            f"guard ^[1-9][0-9]*$, so the false denominator sailed "
            f"through SILENTLY (#88 reproduced: {old_n} root keys vs "
            f"{true_n} declared records, over 1 file)."
        )

        # MUST_PASS leg 1 -- the shipped counter on the same payload.
        cp = subprocess.run(
            [sys.executable, str(COUNTER), str(census)],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            raise SystemExit(
                f"CONTROL FAIL (MUST_PASS): the shipped counter REFUSED "
                f"the {payload_kind} (rc={cp.returncode}, stderr="
                f"{cp.stderr!r}) -- a healthy census must parse; that is "
                f"doctrine 5's symmetric defect, a false red costing "
                f"what a false green costs; repair the counter, never "
                f"this arm."
            )
        got = cp.stdout.strip()
        if not PURE_INT.fullmatch(got) or int(got) != true_n:
            raise SystemExit(
                f"CONTROL FAIL (MUST_PASS): stdout={cp.stdout!r} but the "
                f"payload declares {true_n} module records -- the count "
                f"still does not read the contracted shape, or stdout is "
                f"impure (the launcher's $(...) capture needs the bare "
                f"count)."
            )
        print(
            f"MUST_PASS count green (denominator: {true_n} module "
            f"records off {payload_kind}): shipped counter returned "
            f"{got}, stdout bare as the launcher capture requires."
        )

        # MUST_PASS leg 2 -- garbage roots refuse loudly and count nothing.
        refused = 0
        total = 0
        for label, text in GARBAGE_ROOTS.items():
            total += 1
            junk = Path(td) / f"garbage_{total}.json"
            junk.write_text(text, encoding="utf-8")
            cp = subprocess.run(
                [sys.executable, str(COUNTER), str(junk)],
                capture_output=True,
                text=True,
            )
            if cp.returncode != 0 and not PURE_INT.fullmatch(cp.stdout.strip()):
                refused += 1
            else:
                raise SystemExit(
                    f"CONTROL FAIL (MUST_PASS refusal): garbage root "
                    f"[{label}] was ACCEPTED (rc={cp.returncode}, "
                    f"stdout={cp.stdout.strip()!r}) -- fail-open "
                    f"counting is the #88 defect reborn (doctrine 4)."
                )
        print(
            f"MUST_PASS refusal green (denominator: {total} garbage "
            f"roots): refused {refused}/{total} with nonzero status, "
            f"stderr naming the shape found, nothing countable on "
            f"stdout -- unrecognised is UNMEASURED and BLOCKS "
            f"(doctrine 4)."
        )
    print(
        f"CONTROL PASS ({payload_kind}): 4 legs observed -- wiring, "
        f"MUST_FIRE, MUST_PASS count, MUST_PASS refusal; denominators "
        f"printed per leg above (doctrine 2)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
