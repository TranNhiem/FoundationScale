#!/usr/bin/env python3
"""Gated build stage: wire the fs_model_root resolver into fs_train.fixed.py.

Defect #133 -- "the model root" meant three different things on one estate.
The training entrypoint bound every loader to the operator-declared path:
for a commit-SHA-nested checkpoint the config is one level down, and for a
config-overlay checkpoint the payload symlinks escape the declared root. The
resolver (h100/gen/fs_model_root.py) existed and was tested but never called.

This stage patches h100/gen/fs_train.fixed.py so that load_artifacts() calls
resolve_model_root() first and points EVERY loader (config, processor,
tokenizer, and both downstream model_class.from_pretrained readers) at the
resolver's config_dir instead of the declared root.

Stage contract:
  * paths derive from pathlib.Path(__file__).resolve().parent -- no hardcoded
    build-host absolute path (#123/#136);
  * each anchor must occur EXACTLY ONCE before substitution (0 = upstream
    changed, 2+ = ambiguous; both refuse);
  * re-applying the stage to its own output is a byte-exact no-op;
  * py_compile must be clean on the patched source;
  * the old unpatched read must be fully gone (0 remaining);
  * resolve_model_root must be CALLED, not merely imported;
  * exit 0 only when every gate is green; on any red gate nothing is written.
"""

from __future__ import annotations

import os
import py_compile
import re
import sys
import tempfile
from pathlib import Path

BUILD_ROOT = Path(__file__).resolve().parent
GEN_DIR = BUILD_ROOT / "h100" / "gen"
ENTRYPOINT = GEN_DIR / "fs_train.fixed.py"
RESOLVER = GEN_DIR / "fs_model_root.py"

OLD_READ = "_require_local_dir(config.model_path.value"
CALL_RE = re.compile(r"\bresolve_model_root\s*\(")

# ---------------------------------------------------------------------------
# Patch anchors. Each is byte-verbatim from the unpatched entrypoint and must
# occur EXACTLY ONCE before substitution.
# ---------------------------------------------------------------------------

# P1: sibling import of the resolver. Placed at module scope, anchored on the
# first top-level definition the entrypoint owns (_require_local_dir, kept for
# the dataset directory); the anchor line is re-emitted at the end of the
# replacement so the function itself is untouched.
ANCHOR_IMPORT = "def _require_local_dir(path_value: Any, label: str) -> Path:"
REPLACEMENT_IMPORT = '''\
# --- defect #133: mandatory sibling model-root resolver ---------------------
# fs_model_root.py is generated into the SAME directory as this entrypoint and
# is imported at module scope. There is deliberately NO fallback: if the import
# fails, the entrypoint must fail loudly, because silently reverting to the
# operator-declared root reinstates defect #133 while every downstream gate
# stays green.
try:
    from fs_model_root import (
        ModelRoot,
        ModelRootError,
        resolve_model_root,
    )
except ImportError as exc:
    raise ImportError(
        "fs_model_root.py must be co-located with fs_train.fixed.py; the "
        "model-root resolver is mandatory and no silent fallback to the "
        "operator-declared root is permitted"
    ) from exc
# ----------------------------------------------------------------------------


def _require_local_dir(path_value: Any, label: str) -> Path:'''
MARKER_IMPORT = "from fs_model_root import ("

ANCHOR_DATACLASS = '''\
@dataclass(frozen=True)
class LoadArtifacts:
    """Bundle architecture evidence chosen from declarations rather than a name table."""

    model_path: Path
    config: Any
    feature_adapter: Any
    tokenizer: Any'''
REPLACEMENT_DATACLASS = '''\
@dataclass(frozen=True)
class LoadArtifacts:
    """Bundle architecture evidence chosen from declarations rather than a name table."""

    # defect #133: model_path deliberately holds the resolver's config_dir --
    # the single directory a loader must be pointed at -- NOT the operator-
    # declared root. The two downstream readers
    #     model_class.from_pretrained(str(artifacts.model_path), ...)
    # therefore load from the resolved config dir with no change at their call
    # sites, consistent with _load_config / AutoProcessor / AutoTokenizer in
    # load_artifacts(). The full resolution is surfaced alongside for the
    # launcher and downstream consumers.
    model_path: Path
    model_root: ModelRoot  # complete resolution evidence from fs_model_root
    config_dir: Path  # == model_path; named for resolver-aware consumers
    layout: str  # one of fs_model_root.LAYOUT_*
    bind_closure: tuple[str, ...]  # roots that MUST be mounted, root first
    config: Any
    feature_adapter: Any
    tokenizer: Any'''
MARKER_DATACLASS = "bind_closure: tuple[str, ...]  # roots that MUST be mounted, root first"

ANCHOR_LOAD = '''\
    model_path = _require_local_dir(config.model_path.value, "model_path")
    config_obj = _load_config(model_path)'''
REPLACEMENT_LOAD = '''\
    # defect #133: resolve the declared root BEFORE any load. ModelRootError's
    # message carries the config-candidate denominator; forward it verbatim
    # through the entrypoint's own OperationFailure vocabulary -- never swallow
    # it, never fall back to the declared root.
    try:
        model_root = resolve_model_root(config.model_path.value)
    except ModelRootError as exc:
        raise OperationFailure("load", "model_root", str(exc)) from exc
    # Every loader below (_load_config, AutoProcessor, AutoTokenizer) and both
    # downstream model_class.from_pretrained readers of artifacts.model_path
    # bind the resolved config dir, never the declared root.
    model_path = Path(model_root.config_dir)
    config_obj = _load_config(model_path)'''
MARKER_LOAD = "model_root = resolve_model_root(config.model_path.value)"

ANCHOR_RETURN = '''\
    return LoadArtifacts(
        model_path=model_path,
        config=config_obj,
        feature_adapter=feature_adapter,
        tokenizer=tokenizer,
    )'''
REPLACEMENT_RETURN = '''\
    return LoadArtifacts(
        model_path=model_path,
        model_root=model_root,
        config_dir=model_path,
        layout=model_root.layout,
        bind_closure=model_root.bind_closure,
        config=config_obj,
        feature_adapter=feature_adapter,
        tokenizer=tokenizer,
    )'''
MARKER_RETURN = "bind_closure=model_root.bind_closure,"

PATCHES = [
    (
        "A1",
        "import the sibling resolver (fail loudly, no fallback)",
        ANCHOR_IMPORT,
        REPLACEMENT_IMPORT,
        MARKER_IMPORT,
    ),
    (
        "A2",
        "surface ModelRoot/config_dir/layout/bind_closure on LoadArtifacts",
        ANCHOR_DATACLASS,
        REPLACEMENT_DATACLASS,
        MARKER_DATACLASS,
    ),
    (
        "A3",
        "resolve_model_root(...) replaces the declared-root read in load_artifacts",
        ANCHOR_LOAD,
        REPLACEMENT_LOAD,
        MARKER_LOAD,
    ),
    (
        "A4",
        "LoadArtifacts(...) return carries the resolution evidence",
        ANCHOR_RETURN,
        REPLACEMENT_RETURN,
        MARKER_RETURN,
    ),
]


def _py_compile_check(source: str) -> tuple[bool, str]:
    """py_compile the patched source in a scratch file; clean up both artefacts."""
    fd, name = tempfile.mkstemp(suffix=".py", prefix="patch_fs_train_model_root_")
    os.close(fd)
    tmp = Path(name)
    cfile = name + ".pyc"
    try:
        tmp.write_text(source, encoding="utf-8")
        py_compile.compile(str(tmp), cfile=cfile, doraise=True)
        return True, "py_compile clean"
    except py_compile.PyCompileError as exc:
        message = (exc.msg or str(exc)).strip().splitlines()
        return False, f"py_compile failed: {message[0][:140] if message else 'unknown'}"
    finally:
        tmp.unlink(missing_ok=True)
        Path(cfile).unlink(missing_ok=True)


def main() -> int:
    gates: list[tuple[str, str, bool, str]] = []

    def gate(gid: str, desc: str, ok: bool, detail: str) -> None:
        gates.append((gid, desc, bool(ok), detail))

    rel_entry = ENTRYPOINT.relative_to(BUILD_ROOT)
    rel_resolver = RESOLVER.relative_to(BUILD_ROOT)

    # -- PRE gates -----------------------------------------------------------
    text: str | None = None
    try:
        text = ENTRYPOINT.read_text(encoding="utf-8")
        gate("PRE-1", f"entrypoint {rel_entry} exists and is readable", True, f"{len(text)} bytes")
    except OSError as exc:
        gate("PRE-1", f"entrypoint {rel_entry} exists and is readable", False, str(exc))

    gate(
        "PRE-2",
        f"resolver {rel_resolver} is co-located with the entrypoint",
        RESOLVER.is_file(),
        "found"
        if RESOLVER.is_file()
        else "missing -- the patched entrypoint would fail loudly at import; the stage refuses to ship that state",
    )

    patched = text
    changed = False

    if text is None:
        for gid, desc, _, _, _ in PATCHES:
            gate(gid, desc, False, "not evaluated: entrypoint unreadable")
        gate("IDEM", "re-application is a byte-exact no-op", False, "not evaluated")
        gate("POST-1", "py_compile clean on patched entrypoint", False, "not evaluated")
        gate("POST-2", "old declared-root read fully removed", False, "not evaluated")
        gate("POST-3", "resolve_model_root is called, not merely imported", False, "not evaluated")
        gate("POST-4", "_require_local_dir retained for the dataset directory", False, "not evaluated")
    else:
        # -- Anchor/substitution gates --------------------------------------
        for gid, desc, anchor, replacement, marker in PATCHES:
            assert patched is not None
            if marker in patched:
                gate(gid, desc, True, "already applied (idempotent state detected)")
                continue
            count = patched.count(anchor)
            if count == 1:
                patched = patched.replace(anchor, replacement, 1)
                changed = True
                gate(gid, desc, True, "anchor unique; substituted")
            elif count == 0:
                gate(gid, desc, False, "anchor occurs 0 times and patch marker is absent -- upstream changed; refusing")
            else:
                gate(gid, desc, False, f"anchor occurs {count} times -- substitution would be ambiguous; refusing")

        # -- Idempotence gate ------------------------------------------------
        assert patched is not None
        if changed:
            idem_ok = all(marker in patched for _, _, _, _, marker in PATCHES)
            gate(
                "IDEM",
                "re-application is a byte-exact no-op",
                idem_ok,
                "all patch markers present in output; a re-run substitutes nothing and writes 0 bytes"
                if idem_ok
                else "a substitution failed to install its idempotence marker",
            )
        else:
            gate(
                "IDEM",
                "re-application is a byte-exact no-op",
                True,
                "already-patched state detected on entry; no rewrite performed",
            )

        # -- POST gates -------------------------------------------------------
        compile_ok, compile_detail = _py_compile_check(patched)
        gate("POST-1", "py_compile clean on patched entrypoint", compile_ok, compile_detail)

        remaining_old = patched.count(OLD_READ)
        gate(
            "POST-2",
            "old declared-root read (_require_local_dir(config.model_path.value...) fully removed",
            remaining_old == 0,
            f"{remaining_old} occurrence(s) remain",
        )

        call_sites = len(CALL_RE.findall(patched))
        gate(
            "POST-3",
            "resolve_model_root is actually CALLED, not merely imported",
            call_sites >= 1,
            f"{call_sites} call site(s) found",
        )

        require_defs = patched.count("def _require_local_dir(")
        gate(
            "POST-4",
            "_require_local_dir retained (still used for the dataset directory)",
            require_defs == 1,
            f"{require_defs} definition(s) present",
        )

    # -- Gate table + verdict -------------------------------------------------
    for gid, desc, ok, detail in gates:
        print(f"  {gid} {desc}: {'PASS' if ok else 'FAIL'} ({detail})")

    if not all(ok for _, _, ok, _ in gates):
        print(f"VERDICT: FAIL -- refusing to write; {rel_entry} left at last good state")
        return 1

    if changed:
        assert patched is not None
        ENTRYPOINT.write_text(patched, encoding="utf-8")
        print(
            f"VERDICT: PASS -- wrote {rel_entry}: resolve_model_root wired into "
            "load_artifacts, every loader bound to the resolved config dir (defect #133)"
        )
    else:
        print(f"VERDICT: PASS -- {rel_entry} already patched; byte-exact no-op, nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
