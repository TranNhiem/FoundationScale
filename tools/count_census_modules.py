#!/usr/bin/env python3
"""Count what a census --out artifact DECLARES (#88): the one production
implementation of the LoRA census denominator. Invoked inline by
launchers/launch_g4e4b_lora_1tray.sh; exercised by
tools/census_denominator_control.py.

The contract, measured against the writer (launchers/lora_target_census.py):

    {"adapter_modules": [<module record>, ...], "source": "<provenance>"}

-- a WRAPPED object whose module records live UNDER the named key
"adapter_modules". A bare JSON list of records is the other accepted shape.
A record is a non-empty string module stem, or a dict carrying a non-empty
string 'fqn' (dims, duplicates and per-entry adjudication are the gate's
business -- _load_adapter_modules; this counter only takes the denominator).

EVERYTHING ELSE IS REFUSED, with a message naming the shape actually found:
an unrecognised root is UNMEASURED and BLOCKS (doctrine 4). Never a guess
between shapes -- #88 was exactly that guess: len(root) over the wrapped
payload printed 2 (the wrapper-KEY count) on a real census declaring 168,
and 2 cleared the launcher's numeric guard SILENTLY, handing a false
denominator to every downstream --adapter-modules claim (doctrine 2).

Success prints ONLY the bare count on stdout -- the launcher's
$(... 2>&1) capture plus its ^[1-9][0-9]*$ guard rely on stdout carrying
nothing else. Zero is printable here on purpose: an empty census is counted
honestly as 0 and the launcher's vacuity arm refuses it by name (doctrine
1). Refusals go to stderr with a nonzero exit.

Stdlib only: the host python on this path has no torch (fix44).
"""

import json
import pathlib
import sys

WRAPPED_KEY = "adapter_modules"

CONTRACT = (
    "a JSON object carrying an 'adapter_modules' list of module records "
    "(the wrapped payload launchers/lora_target_census.py writes to "
    "--out), or a bare JSON list of module records"
)


def is_record(entry: object) -> bool:
    """One countable module record: a non-empty string stem, or a dict
    with a non-empty string 'fqn'. Counting anything else would print a
    denominator over guesses -- refuse instead (doctrine 4)."""
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, dict):
        fqn = entry.get("fqn")
        return isinstance(fqn, str) and bool(fqn.strip())
    return False


def extract_entries(obj: object, path: str) -> list:
    """Read the contracted shapes BY NAME; refuse every other root,
    naming the shape found."""
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if WRAPPED_KEY not in obj:
            keys = sorted(obj)
            shown = ", ".join(repr(k) for k in keys[:8])
            if len(keys) > 8:
                shown += f", ... ({len(keys)} keys total)"
            raise SystemExit(
                f"census root in {path} is a JSON object WITHOUT an "
                f"'{WRAPPED_KEY}' key -- found a bare mapping with keys "
                f"[{shown}] (the pre-#88 comment's imaginary 'flat "
                f"name->record' contract; the real writer wraps). "
                f"Expected {CONTRACT}. This counter reads module records "
                f"BY NAME and never guesses between shapes: an "
                f"unrecognised root is UNMEASURED and BLOCKS (doctrine 4)."
            )
        seq = obj[WRAPPED_KEY]
        if not isinstance(seq, list):
            raise SystemExit(
                f"census '{WRAPPED_KEY}' in {path} is a "
                f"{type(seq).__name__}, not a list of module records. "
                f"Expected {CONTRACT}. Refusing to count a shape this "
                f"counter cannot read -- UNMEASURED, BLOCK (doctrine 4)."
            )
        return seq
    raise SystemExit(
        f"census root in {path} is a JSON {type(obj).__name__}, not an "
        f"object/array of module records. Expected {CONTRACT}. An "
        f"unrecognised root is UNMEASURED and BLOCKS (doctrine 4)."
    )


def main(argv: list) -> int:
    if len(argv) != 2:
        raise SystemExit(
            "usage: count_census_modules.py CENSUS_JSON -- exactly one "
            "argument, the census artifact path"
        )
    path = argv[1]
    try:
        with pathlib.Path(path).open(encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"census {path} is not readable JSON ({type(exc).__name__}: "
            f"{exc}) -- unreadable is not empty and missing is not zero "
            f"(doctrine 4); a census the launcher cannot parse states no "
            f"denominator (doctrine 2) and BLOCKS."
        ) from exc
    entries = extract_entries(obj, path)
    bad = sum(1 for e in entries if not is_record(e))
    if bad:
        raise SystemExit(
            f"census {path} carries {bad} of {len(entries)} entries that "
            f"are not module records (a non-empty string stem, or a dict "
            f"with a non-empty string 'fqn') -- counting non-records "
            f"prints a denominator over guesses, so this BLOCKS "
            f"(doctrine 4)."
        )
    print(len(entries))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
