"""Header support statements for tools.preflight (bootstrap/constants)."""

from __future__ import annotations

import sys
from pathlib import Path

# Console-script installs ship foundationscale; a bare ``python -m
# tools.preflight`` from a checkout may not have ``src`` on sys.path.
# Bootstrap it rather than refuse: the tool must run from the login node with
# zero installation ceremony. Path.resolve() plays abspath()'s role here — a
# bare checkout invocation may hand us a relative __file__ — and additionally
# follows a symlinked script back to the true checkout, which is the tree whose
# src/ belongs on sys.path.
#
# These two are bound OUTSIDE the except so the arithmetic is testable. The
# except carries ``pragma: no cover``, so while _SRC lived inside it no test
# could see it — and this module moved one directory deeper than the script it
# was split from, which silently changed what ``.parent`` means. A path
# computed only on the failure branch of an import is a path nothing measures.
_HERE = Path(__file__).resolve().parent  # <repo>/tools/preflight
_SRC = _HERE.parent.parent / "src"  # <repo>/src

# The `as` aliases are the explicit-re-export form, and they are load-bearing
# rather than stylistic. `_base` does not itself USE Coverage or Verdict -- the
# twelve modules split out of the old single file do, via `from .._base import
# Coverage, Verdict` -- so plain imports read to F401 as dead, and `ruff check
# --fix` duly DELETED the second import, the one on the bootstrap branch. That
# autofix is silent and it disarms the branch: on a bare checkout the sys.path
# insert would succeed and the names would never bind. Aliasing states the
# re-export, which is true, and takes the line out of the autofixer's reach.
try:
    from foundationscale.gates.core import Coverage as Coverage
    from foundationscale.gates.core import Verdict as Verdict
except ImportError:  # pragma: no cover - depends on invocation environment
    if _SRC.is_dir():
        sys.path.insert(0, str(_SRC))
        from foundationscale.gates.core import Coverage as Coverage
        from foundationscale.gates.core import Verdict as Verdict
    else:  # foundationscale is a hard dependency; degrading to local enums
        # would silently fork Verdict's meaning, which requirement A forbids.
        raise


TOOL_VERSION = 1

EXIT_CLEAR = 0

EXIT_BLOCKED = 1

EXIT_TOOL_ERROR = 2


_CHUNK = 1 << 16


# Byte widths for safetensors dtype tags. Deliberately mirrors
# foundationscale.provenance.manifest._KNOWN_DTYPE_WIDTHS and
# gates.checkpoint_gates._DTYPE_BYTES: three small tables that must each
# stay honest beats an import edge that drags gate registration side
# effects onto a login node.
_SAFETENSORS_DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


# Distinguishes "key absent" from "key present and set to None". A config that
# explicitly pins a value to null is making a statement; a config that omits the
# key is not, and collapsing the two would let an unset denominator read as a
# deliberate one.
_MISSING = object()
