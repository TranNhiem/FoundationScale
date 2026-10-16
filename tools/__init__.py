"""Command-line adjudicators, executed as scripts and imported by the suite.

Declares nothing and exports nothing, deliberately. This package exists to make
``tools`` a *regular* package: without an ``__init__.py``, ``tools`` resolves via
PEP 420 implicit namespace packages, and a namespace portion is a *fallback* —
any regular package or module of the same name found later on ``sys.path`` wins
outright, regardless of path order (repository root at ``sys.path[0]`` does not
help). Some container images (e.g. ``nemo-automodel-26-04.sif``, which ships an
unrelated package at
``/usr/local/lib/python3.12/dist-packages/tools/__init__.py``) shadow this tree
that way, turning imports like ``tools.live_save_gate`` into
``ModuleNotFoundError`` at collection time. Making ``tools`` a regular package
guarantees the repository's own adjudicators win wherever the suite runs.

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
