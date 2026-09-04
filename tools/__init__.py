"""Command-line adjudicators, executed as scripts and imported by the suite.

Why this file exists at all — it declares nothing and exports nothing, and that
is deliberate. Before it existed, ``tools/`` was a bare directory, so
``from tools.live_save_gate import ...`` resolved only through PEP 420 implicit
namespace packages. That mechanism has a property the suite was silently
relying on being untrue: a namespace portion is a *fallback*. When the import
machinery scans ``sys.path`` it records directories with no ``__init__.py`` as
candidate portions and keeps scanning; if a regular package or module of the
same name turns up anywhere later on the path, the regular one wins outright.
Path order does not save you. Putting the repository root first on ``sys.path``
— which every affected test already did — changes nothing.

MEASURED, 2026-08-30, on the H100 estate inside
``nemo-automodel-26-04.sif``: that image ships an unrelated real package at
``/usr/local/lib/python3.12/dist-packages/tools/__init__.py``. With the
repository root inserted at ``sys.path[0]``, ``import tools`` resolved to the
container's package, and collection of
``tests/test_fix44_unmeasured_refusal_record.py`` died with
``ModuleNotFoundError: No module named 'tools.live_save_gate'`` — aborting the
entire run at collection time, so the suite reported an error rather than a
verdict. The same tree collects and passes on a laptop whose environment
happens to ship no such package. That difference is environmental luck, not a
property of this repository, and a suite that depends on it cannot certify
anything about the estate it runs on.

Declaring ``tools`` a regular package removes the dependency on that luck: a
regular package found earlier on ``sys.path`` beats one found later, so the
repository's own adjudicators win wherever the suite runs.

Kept empty of imports on purpose. ``tools/live_save_gate.py``,
``tools/preflight/`` and ``tools/emit_run_manifest.py`` are also entry points
run directly as scripts on control-plane nodes with no ML stack installed, and
several of them are targets of the mutation battery. Re-exporting their symbols
here would (a) force an import of every adjudicator whenever any one of them is
named, dragging their lazy ``torch``-free import discipline into a package
``__init__`` where it cannot be enforced, and (b) give the battery a second file
whose text must stay in step with the modules it mutates.
"""

from __future__ import annotations
