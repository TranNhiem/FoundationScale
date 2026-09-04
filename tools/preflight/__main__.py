"""``python -m tools.preflight`` entry point.

Kept to three lines on purpose: importing the package front door pulls in
``_cli``, so any logic living here would be executed twice under ``-m``
(once as ``tools.preflight._cli``, once as ``__main__``).
"""

from __future__ import annotations

from ._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
