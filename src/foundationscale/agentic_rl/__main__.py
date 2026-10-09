"""``python -m foundationscale.agentic_rl`` -- the module-form spelling of the
``foundationscale-agentic-rl`` console script, the same way
``foundationscale.train.__main__`` is the module-form spelling of
``foundationscale-train`` (see that module's docstring for the full doctrine:
exactly one argument-parsing site, so the console script and this module form
are the same entry point by CONSTRUCTION, not by a convention someone must
remember to honour).

``raise SystemExit(main())`` is load-bearing: ``main`` returns the verdict --
0 PASS, 5 RED, 95 UNMEASURED, 96 REFUSE -- and the process exit code must BE
that verdict.

This file reads untested in an in-process coverage report for the same reason
``train/__main__.py`` does (see its docstring): the module form is only ever
executed in a CHILD interpreter (``tests/agentic_rl/test_cli.py`` runs
``python -m foundationscale.agentic_rl --dry-run`` in a subprocess so the
dry-run's own no-torch claim can be checked via that child's
``sys.modules``), which the parent process's coverage instrumentation does
not see. The denominator is "lines executed in THIS process"; this entry
point is, by construction, only ever executed in another one.
"""

from __future__ import annotations

from foundationscale.agentic_rl.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
