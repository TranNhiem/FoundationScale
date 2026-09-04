#!/usr/bin/env python3
"""Stage for the model-root plane: extract h100/gen/fs_model_root.py and its test.

`h100/gate133.json` is the raw kimi-fanout response. Extracting the module and test by
hand once and editing them in place is precisely the drift this build exists to prevent:
the fix ends up in the file you read and not in the file that runs, or a re-extraction
silently reverts it. So both files become generated artifacts of the plane, produced
here atomically and patched by later stages, exactly like the entrypoint.

The response payload is JSON-in-JSON: the fan-out envelope's `content` is itself a JSON
object carrying BOTH `module` and `test`. Both layers are validated, because a truncated
generation deserialises into a shorter-but-valid string and would otherwise be written
out as a plausible-looking half file.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from fs_estate_pat import estate_blocklist, estate_ident_pat


BLOCKLIST_PATTERN = estate_blocklist(strict_token=True)

# Generation garbage the model emitted inside test_t10_determinism. It is a hard
# SyntaxError (`del` on a name bound by a walrus in an assert), so py_compile would
# catch it -- but silently dropping lines we did not ask about is how drift starts, so
# each strip asserts an exact occurrence count before touching anything.
JUNK_LINES = ("    assert get := None\n", "    del get\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    source = root / "h100" / "gate133.json"
    module_target = root / "h100" / "gen" / "fs_model_root.py"
    test_target = root / "h100" / "gen" / "test_fs_model_root.py"

    gates: list[tuple[str, bool, str]] = []

    envelope = json.loads(source.read_text(encoding="utf-8"))
    gates.append(
        ("C1 envelope holds exactly one task result",
         isinstance(envelope, list) and len(envelope) == 1, f"{len(envelope)} result(s)")
    )
    if not gates[-1][1]:
        return _fail(gates)

    task = envelope[0]
    gates.append(("C2 task reported ok", bool(task.get("ok")), str(task.get("error"))[:80]))
    # A truncated generation is the failure mode that looks most like a success: the
    # JSON still parses, the source still starts with a docstring, and only the tail is
    # missing. finish_reason is the only signal that distinguishes them.
    gates.append(
        ("C3 generation ran to a stop, not a length cap",
         task.get("finish_reason") == "stop", str(task.get("finish_reason")))
    )
    if not (gates[-2][1] and gates[-1][1]):
        return _fail(gates)

    payload = json.loads(task["content"])
    gates.append(
        ("C4 payload carries both 'module' and 'test'",
         "module" in payload and "test" in payload, ",".join(payload)[:80])
    )
    if not gates[-1][1]:
        return _fail(gates)

    module_text = payload["module"]
    test_text = payload["test"]
    if not module_text.endswith("\n"):
        module_text += "\n"
    if not test_text.endswith("\n"):
        test_text += "\n"

    # Strip the two junk lines, but only after proving each occurs exactly once: a
    # blind replace would pass silently over a regenerated file whose garbage has moved
    # or multiplied, and the test would then be checked against a contract it no longer
    # satisfies.
    for junk in JUNK_LINES:
        count = test_text.count(junk)
        if count != 1:
            gates.append((f"strip {junk.strip()!r}: exactly 1 occurrence",
                          False, f"{count} occurrence(s)"))
            return _fail(gates)
        test_text = test_text.replace(junk, "")

    module_lines = module_text.count("\n")
    test_lines = test_text.count("\n")
    gates.append(
        ("C5 both sources are plausible (>=150 lines each)",
         module_lines >= 150 and test_lines >= 150,
         f"module {module_lines}, test {test_lines} lines")
    )
    test_fns = test_text.count("\ndef test_")
    gates.append(
        ("C6 module and test expose the agreed contract",
         "def resolve_model_root" in module_text
         and "class ModelRootError" in module_text
         and "from fs_model_root import" in test_text
         and test_fns >= 10, f"{test_fns} test function(s)")
    )
    module_hits = BLOCKLIST_PATTERN.findall(module_text)
    test_hits = BLOCKLIST_PATTERN.findall(test_text)
    gates.append(
        ("C7 no estate literal in either output",
         not module_hits and not test_hits,
         f"{len(module_hits)}+{len(test_hits)} hit(s)")
    )

    if not all(ok for _, ok, _ in gates):
        return _fail(gates)

    module_target.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for target, text in ((module_target, module_text), (test_target, test_text)):
        fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
        paths.append(str(target))

    # Subprocess rather than py_compile.compile(): the in-process form needs a writable
    # cfile (3.14 refuses os.devnull outright with FileExistsError) and raises OSError
    # subclasses that are not PyCompileError, so a narrow except would let an unverified
    # artifact survive.
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", *paths], capture_output=True, text=True
    )
    gates.append(("C8 py_compile clean on both outputs",
                  proc.returncode == 0, proc.stderr.strip()[:120]))
    if proc.returncode != 0:
        module_target.unlink(missing_ok=True)
        test_target.unlink(missing_ok=True)
        _report(gates)
        print("removed both artifacts; unverified files are not left in place")
        return 4

    _report(gates)
    print(
        f"extracted {module_target} ({module_lines} lines) and {test_target} "
        f"({test_lines} lines, {test_fns} tests, 2 junk lines stripped); "
        "py_compile: clean"
    )
    return 0


def _fail(gates: list[tuple[str, bool, str]]) -> int:
    _report(gates)
    return 2


def _report(gates: list[tuple[str, bool, str]]) -> None:
    print("gate table:")
    for label, ok, note in gates:
        print(f"  {label}: {'PASS' if ok else 'FAIL'}{(' (' + note + ')') if note else ''}")


if __name__ == "__main__":
    raise SystemExit(main())
