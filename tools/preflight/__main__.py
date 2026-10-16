"""``python -m tools.preflight`` entry point.

Keep this file minimal: the package front door imports ``_cli``, so any logic
here would run twice under ``-m``.
"""

from __future__ import annotations

from ._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
