"""``python -m foundationscale.train`` -- finding #246.

``docs/deliverables/B2_scaling.md:196`` documents
``entrypoint: "foundationscale.train"``. Before this file existed, the plain
import succeeded -- the package did exist -- and the documented invocation
failed with ``No module named foundationscale.train.__main__``. That is the
most misleading half-truth a documented entrypoint can carry: the package
shipped ``__init__.py``, ``cli.py`` and ``loop.py``, the only runnable form
was the ``foundationscale-train`` console script (#224's fix), and a reader
copying the documented line got an error claiming the package cannot be
EXECUTED rather than one admitting the documented command did not exist. An
error of the second kind is diagnosed in seconds; an error of the first kind
sends the reader debugging their interpreter.

The fix is deliberately this thin. The body below forwards to
``foundationscale.train.cli.main`` and nothing else -- no duplicated
``ArgumentParser``, no additional flags, no behaviour this module owns. Two
entry points that CAN drift will drift, and the rift is discovered by the
person furthest from the repo; doctrine 4 applied to entrypoints is that
there must be exactly one argument-parsing site. The console script from
``[project.scripts]`` and ``python -m foundationscale.train`` are therefore
the same entry point by CONSTRUCTION -- they call the same function -- not
by a convention someone must remember to honour. #224's lesson applies
again: a second copy of the parser written "to keep this module
self-contained" would be prose that happens to execute until the day it
doesn't.

``raise SystemExit(main())`` is load-bearing, not idiomatic decoration.
``main`` returns the verdict -- 0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE --
and the process exit code must BE that verdict, because a launcher wrapping
this entrypoint reads the exit code and nothing else. Barely calling
``main()`` would exit 0 on every verdict, laundering RED into green at
exactly the boundary this package exists to guard.

A test executes the module form in a subprocess (``python -m
foundationscale.train --help``, exit code asserted) so the two spellings
cannot silently diverge: if this module is ever deleted, emptied, or given
its own parser, the control fires there rather than in a user's terminal.

**This file reads 0% in the coverage report, and that number is an artifact,
not a gap.** ``tests/test_train_module_entrypoint.py`` exercises every line
below -- four legs, including a MUST_FIRE leg that deletes this file and
demands red -- but it does so in a CHILD interpreter, which the parent's
coverage plugin does not instrument. Reading the 0% as "untested" would be
the repository's own error run backwards: a number quoted without its
denominator. The denominator here is "lines executed in THIS process", and
the entry point is, by construction, only ever executed in another one.
"""

from __future__ import annotations

from foundationscale.train.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
