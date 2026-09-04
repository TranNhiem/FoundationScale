#!/usr/bin/env python3
"""Gated, idempotent patch stage for the generated fs_container_backend.bound.sh.

Replaces the weak fs117 `[[ -r "$p" ]]` probe (issue #132) with one that also
resolves symlinks BENEATH each declared bind destination, so config-overlay /
HF-snapshot layouts can no longer hang a 0-byte payload behind a passing read
test. Refuses closed on any gate failure; never leaves an unverified artifact.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
from typing import List, Tuple
from fs_estate_pat import estate_blocklist

REL_TARGET = os.path.join("h100", "gen", "fs_container_backend.bound.sh")

ANCHOR = r"""  local fs117_verify_script='k=0; for p do [[ -r "$p" ]] && k=$((k+1)); done; printf "%s\n" "fs117: $k of $# declared bind paths readable in-container"; [[ $k -eq $# ]]'"""

REPLACEMENT = r"""  # fs117 (R4), post-#132. `[[ -r "$p" ]]` on the declared destination is
  # necessary but NOT sufficient: a config-overlay model root (patched
  # config.json + a symlink to weights in another tree) passes it while the
  # payload dangles, and every HuggingFace cache snapshot has exactly that
  # shape. So the probe also resolves the symlinks BENEATH each declared
  # destination and refuses on any that do not resolve, naming examples and
  # their raw targets so the operator learns which root is missing from
  # FS_BIND_PATHS rather than just that something is broken.
  local fs117_verify_script='
cap=${FS_BIND_SCAN_CAP:-20000}
k=0; seen=0; dangle=0; trunc=0; ex=""
for p do
  [[ -r "$p" ]] && k=$((k+1))
  [[ -d "$p" ]] || continue
  n=0
  while IFS= read -r l; do
    n=$((n+1)); seen=$((seen+1))
    if [[ ! -e "$l" ]]; then
      dangle=$((dangle+1))
      [[ $dangle -le 3 ]] && ex="$ex
    dangling: $l -> $(readlink "$l" 2>/dev/null)"
    fi
    [[ $n -ge $cap ]] && { trunc=1; break; }
  done < <(find "$p" -type l -print 2>/dev/null)
done
printf "fs117: %s of %s declared bind paths readable in-container; %s symlinks beneath them resolved, %s DANGLING (scan cap %s, truncated=%s)%s\n" \
  "$k" "$#" "$seen" "$dangle" "$cap" "$trunc" "$ex"
if [[ $trunc -ne 0 ]]; then
  printf "fs117: symlink scan hit its cap, so the dangling count is UNMEASURED, not zero; raise FS_BIND_SCAN_CAP deliberately if this tree is genuinely that large.\n" >&2
  exit 1
fi
[[ $k -eq $# && $dangle -eq 0 ]]'"""

MARKER = "cap=${FS_BIND_SCAN_CAP:-20000}"
CALLSITE = '-c "$fs117_verify_script" fs-declared-mounts'
BANNED = estate_blocklist(strict_token=True)


def apply_patch(text: str) -> Tuple[str, bool, int, int]:
    """Return (text, already_applied, anchor_count, anchor_lineno_1based)."""
    count = text.split("\n").count(ANCHOR)
    if MARKER in text:
        return text, True, count, -1
    if count != 1:
        return text, False, count, -1
    lines = text.split("\n")
    idx = lines.index(ANCHOR)
    lines[idx:idx + 1] = REPLACEMENT.split("\n")
    return "\n".join(lines), False, count, idx + 1


def write_atomic(path: str, data: str) -> None:
    """Write data to a temp file in the target's directory, then os.replace."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".",
                               prefix=".fs117.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(data)
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def print_gates(gates: List[Tuple[str, str]]) -> None:
    print("gate table:")
    for label, verdict in gates:
        print(f"  {label}: {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch fs117 to fail closed on dangling bind payloads.")
    ap.add_argument("--root", default=".", help="tree root containing h100/gen/ (default: cwd)")
    args = ap.parse_args()
    path = os.path.join(args.root, REL_TARGET)

    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            original = fh.read()
    except OSError as exc:
        print(f"FATAL: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    patched, already, count, lineno = apply_patch(original)
    if already:
        print(f"already applied: {path} carries the FS_BIND_SCAN_CAP probe; left untouched")
        return 0

    gates: List[Tuple[str, str]] = []
    gates.append((f"C1 anchor occurs exactly once (found {count})",
                  "PASS" if count == 1 else "FAIL"))
    if count != 1:
        print_gates(gates)
        print(f"FATAL: anchor line found {count} time(s), need exactly 1; refusing", file=sys.stderr)
        return 2

    mk = patched.count(MARKER)
    cs = patched.count(CALLSITE)
    repatched, realready, _, _ = apply_patch(patched)
    gates.append((f"C2 FS_BIND_SCAN_CAP sentinel occurs exactly once (found {mk})",
                  "PASS" if mk == 1 else "FAIL"))
    gates.append((f"C3 verify-script call sites unchanged (found {cs}, need 2)",
                  "PASS" if cs == 2 else "FAIL"))
    gates.append(("C5 inserted text carries no estate path literal",
                  "PASS" if not BANNED.search(REPLACEMENT) else "FAIL"))
    gates.append(("C6 second in-process application is a no-op exit-0 path",
                  "PASS" if realready and repatched == patched else "FAIL"))
    if any(v != "PASS" for _, v in gates):
        gates.append(("C4 bash -n clean", "n/a (earlier gate failed; file untouched)"))
        gates.sort(key=lambda g: g[0])
        print_gates(gates)
        print("FATAL: gate failure; file left untouched", file=sys.stderr)
        return 5

    write_atomic(path, patched)
    try:
        proc = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
    except FileNotFoundError:
        write_atomic(path, original)
        gates.insert(3, ("C4 bash -n clean", "UNMEASURED (bash unavailable)"))
        print_gates(gates)
        print("FATAL: bash unavailable -> syntax UNMEASURED; original bytes restored",
              file=sys.stderr)
        return 3
    if proc.returncode != 0:
        write_atomic(path, original)
        gates.insert(3, ("C4 bash -n clean", "FAIL"))
        print_gates(gates)
        sys.stderr.write(proc.stderr)
        print("FATAL: bash -n rejected the patched file; original bytes restored",
              file=sys.stderr)
        return 4
    gates.insert(3, ("C4 bash -n clean", "PASS"))

    print_gates(gates)
    delta = len(patched.encode("utf-8")) - len(original.encode("utf-8"))
    print(f"patched {path}: replaced anchor line {lineno}; byte delta {delta:+d}; bash -n: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())