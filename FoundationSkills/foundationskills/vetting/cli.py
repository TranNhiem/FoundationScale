"""CLI wiring for the FoundationSkills vetting gate (A6.1).

Adds the ``vet`` subcommand: run every vetting check over one skill tree, write the
report and print a short summary.  The advisory (description-vs-body consistency) is
enabled only when both ``--advisory-base-url`` and ``--advisory-model`` are supplied;
it talks to any OpenAI-compatible chat backend.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from .gate import run_gate, write_report


def add_parser(sub: Any) -> None:
    """Add the ``vet`` subcommand to the CLI's subparsers."""
    p: ArgumentParser = sub.add_parser("vet", help="vet a skill package against the lock")
    p.add_argument("skill_dir", type=Path, help="path to the skill directory")
    p.add_argument("--lock", type=Path, required=True, help="path to skills.lock")
    p.add_argument("--name", default=None, help="skill name (default: directory basename)")
    p.add_argument("--out", type=Path, default=None, help="report output path")
    p.add_argument("--install-to", type=Path, default=None, dest="install_to", help="install target directory")
    p.add_argument("--project-root", type=Path, default=None, dest="project_root", help="project root directory")
    p.add_argument("--advisory-base-url", default=None, dest="advisory_base_url", help="advisory LLM base URL")
    p.add_argument("--advisory-model", default=None, dest="advisory_model", help="advisory LLM model name")
    p.add_argument(
        "--advisory-api-key-env",
        default=None,
        dest="advisory_api_key_env",
        help="name of the environment variable holding the advisory API key",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="advisory timeout in seconds")
    p.set_defaults(func=cmd_vet)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences (```...```) from an LLM JSON response."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl == -1:
            return t
        t = t[nl + 1 :]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


def _build_advisory(
    base_url: str,
    model: str,
    api_key_env: str | None,
    timeout: float,
) -> Any:
    """Return a callable ``(description, body) -> {"consistent": bool, "reason": str}``."""
    from foundationskills.skills.data_engine.llm_backend import make_backend

    cfg: dict[str, Any] = {
        "kind": "openai_compatible",
        "base_url": base_url,
        "model": model,
        "timeout_s": timeout,
    }
    if api_key_env:
        cfg["api_key_env"] = api_key_env

    backend = make_backend(cfg, "vetting.advisory")

    def advisory(description: str, body: str) -> dict[str, Any]:
        system_msg = (
            "You are a vetting assistant. Compare the given skill description with the "
            "skill body and decide whether they are consistent. Respond with JSON only: "
            '{"consistent": true/false, "reason": "<explanation>"}'
        )
        user_msg = f"Description:\n{description}\n\nBody:\n{body}"
        response = backend.complete(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=4096,
            seed=0,
        )
        if response.error and not response.content:
            raise RuntimeError(f"advisory backend error: {response.error}")
        raw = response.content
        return json.loads(_strip_fences(str(raw)))

    return advisory


def cmd_vet(args: Namespace) -> int:
    """Run the vetting gate, write the report and return the exit code."""
    name: str = args.name if args.name else args.skill_dir.name
    out: Path = args.out if args.out else Path.cwd() / f"vetting_report.{name}.json"

    advisory = None
    base_url = getattr(args, "advisory_base_url", None)
    model = getattr(args, "advisory_model", None)
    if base_url and model:
        advisory = _build_advisory(
            base_url,
            model,
            getattr(args, "advisory_api_key_env", None),
            getattr(args, "timeout", 120.0),
        )

    report = run_gate(
        args.skill_dir,
        args.lock,
        name,
        advisory=advisory,
        install_to=getattr(args, "install_to", None),
        project_root=getattr(args, "project_root", None),
    )

    write_report(report, out)

    failing = [c["id"] for c in report["checks"] if c["status"] != "PASS"]
    summary = {
        "status": report["status"],
        "exit_code": report["exit_code"],
        "out": str(out),
        "failing": failing,
    }
    print(json.dumps(summary, sort_keys=True))
    return report["exit_code"]
