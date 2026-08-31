#!/usr/bin/env python3
"""Stage 0 for the training entrypoint: extract h100/gen/fs_train.fixed.py.

`h100/fs_train.json` is the raw generator response. Extracting it by hand once and
editing the result in place is precisely the drift this build exists to prevent: the
fix ends up in the file you read and not in the file that runs, or a re-extraction
silently reverts it. So the entrypoint becomes a third generated artifact of the plane,
produced here and patched by later stages, exactly like the launcher and the backend.

The response payload is JSON-in-JSON: the fan-out envelope's `content` is itself a JSON
object whose `file` key holds the module source. Both layers are validated, because a
truncated generation deserialises into a shorter-but-valid string and would otherwise
be written out as a plausible-looking half file.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
from fs_estate_pat import estate_blocklist

BLOCKLIST = estate_blocklist(strict_token=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = pathlib.Path(args.root)
    source = root / "h100" / "fs_train.json"
    target = root / "h100" / "gen" / "fs_train.fixed.py"

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
    # JSON still parses, the source still starts with a shebang, and only the tail is
    # missing. finish_reason is the only signal that distinguishes them.
    gates.append(
        ("C3 generation ran to a stop, not a length cap",
         task.get("finish_reason") == "stop", str(task.get("finish_reason")))
    )
    if not (gates[-2][1] and gates[-1][1]):
        return _fail(gates)

    payload = json.loads(task["content"])
    gates.append(("C4 payload carries a 'file' key", "file" in payload, ",".join(payload)[:80]))
    if not gates[-1][1]:
        return _fail(gates)

    text = payload["file"]
    if not text.endswith("\n"):
        text += "\n"

    lines = text.count("\n")
    gates.append(("C5 source is a plausible module (>=500 lines)", lines >= 500, f"{lines} lines"))
    gates.append(("C6 starts with a shebang", text.startswith("#!"), text[:16]))
    # Fail closed on truncation: a module whose last statement is half-written parses
    # as a SyntaxError below, but one truncated at a statement boundary would not.
    gates.append(
        ("C7 defines the CLI entrypoint contract",
         '__name__ == "__main__"' in text or "__name__ == '__main__'" in text, "")
    )
    hits = BLOCKLIST.findall(text)
    gates.append(("C8 no estate literal in the generated source", not hits, f"{len(hits)} hit(s)"))

    if not all(ok for _, ok, _ in gates):
        return _fail(gates)

    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, target)

    # Subprocess rather than py_compile.compile(): the in-process form needs a writable
    # cfile (3.14 refuses os.devnull outright) and raises OSError subclasses that are not
    # PyCompileError, so a narrow except would let an unverified artifact survive.
    proc = subprocess.run(
        [sys.executable, "-m", "py_compile", str(target)], capture_output=True, text=True
    )
    gates.append(("C9 py_compile clean", proc.returncode == 0, proc.stderr.strip()[:120]))
    if proc.returncode != 0:
        target.unlink(missing_ok=True)
        _report(gates)
        print("removed the artifact; an unverified entrypoint is not left in place")
        return 4

    _report(gates)
    print(f"extracted {target}: {lines} lines, {len(text)} bytes; py_compile: clean")
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
