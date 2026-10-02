
"""``fskills`` -- the FoundationSkills command line.

Exit codes are only the contract codes:

* 0 PASS, 5 RED (unexpected crash -> ``[fskills:cli:red]`` + traceback),
* 95 UNMEASURED (a skill result's own verdict passes through),
* 96 REFUSED (argparse errors -- a declaration that did not parse -- and any
  refusal, including a missing skill/recipe/catalog or a confirmation gate).

Skills come from ``foundationskills.skills.register_builtin_skills()``,
imported lazily so skill-free subcommands (probe/hash/launch) work even while
other parts of the package are being built in parallel.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from foundationskills.core import SkillContext, get_skill
from foundationskills.core.orchestrator import ConfirmationRequired, plan_hash
from foundationskills.interfaces.fs.launch import LaunchRefused


class _Parser(argparse.ArgumentParser):
    """argparse whose usage errors exit 96 (REFUSED), never 2."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"[fskills:cli:refuse] declaration rejected: {message}", file=sys.stderr)
        raise SystemExit(96)


def _load_payload(path: str) -> dict[str, Any]:
    """Read a JSON file; unwrap an artifact envelope to its payload if present."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    if "payload" in data and "type" in data and isinstance(data["payload"], dict):
        return dict(data["payload"])
    return data


def _ensure_skills() -> None:
    from foundationskills.skills import register_builtin_skills

    register_builtin_skills()


def _execute(skill_name: str, request: dict[str, Any], workdir: str) -> int:
    from foundationskills.interfaces.fs.capabilities import probe

    _ensure_skills()
    skill = get_skill(skill_name)
    ctx_workdir = Path(workdir)
    ctx_workdir.mkdir(parents=True, exist_ok=True)
    result = skill.execute(request, SkillContext(workdir=ctx_workdir, capabilities=probe()))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return int(result.exit_code)


def _cmd_probe(args: argparse.Namespace) -> int:
    # A completed probe returns 0 even when FS is unavailable: "available" is
    # measured data in the payload, not a verdict on the command.
    from foundationskills.interfaces.fs.capabilities import probe

    caps = probe(deep=bool(args.deep))
    data = caps.to_dict()
    if args.json:
        print(json.dumps(data, sort_keys=True))
    else:
        print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _cmd_skills(args: argparse.Namespace) -> int:
    _ensure_skills()
    from foundationskills.core import REGISTRY

    if args.skills_command == "list":
        print(json.dumps(REGISTRY.names(), indent=2))
        return 0
    try:
        skill = REGISTRY.get(args.name)
    except KeyError:
        print(f"[fskills:cli:refuse] missing: skill {args.name!r} is not registered", file=sys.stderr)
        return 96
    description = {
        "name": skill.name,
        "version": skill.version,
        "description": skill.description,
        "scope": skill.scope.to_dict(),
        "consumes": list(skill.consumes),
        "produces": list(skill.produces),
        "rules": [
            {"rule_id": r.rule_id, "description": r.description, "severity": r.severity.value, "phase": r.phase}
            for r in skill.rules
        ],
        "fs_interface": {
            "entries": list(skill.fs_interface.entries),
            "emits": list(skill.fs_interface.emits),
            "apis": list(skill.fs_interface.apis),
            "notes": skill.fs_interface.notes,
        },
    }
    print(json.dumps(description, indent=2, sort_keys=True))
    return 0


def _cmd_data(args: argparse.Namespace) -> int:
    with open(args.request, "r", encoding="utf-8") as handle:
        request = json.load(handle)
    if not isinstance(request, dict):
        raise ValueError(f"{args.request} is not a JSON object")
    return _execute("data_engine", request, args.workdir or "fskills-data")


def _cmd_plan(args: argparse.Namespace) -> int:
    request: dict[str, Any] = {"goal": _load_payload(args.goal)}
    if args.readiness:
        request["readiness"] = _load_payload(args.readiness)
    if args.hardware:
        request["hardware_id"] = args.hardware
    return _execute("training.planner", request, args.workdir or "fskills-plan")


def _cmd_emit(args: argparse.Namespace) -> int:
    request: dict[str, Any] = {
        "plan": _load_payload(args.plan),
        "dataset": _load_payload(args.dataset),
        "model": args.model,
        "output_root": args.output_root,
        "nodes": int(args.nodes),
        "gpus_per_node": int(args.gpus_per_node),
        "hardware_id": args.hardware or "local",
        "run_prefix": "fskills",
    }
    return _execute("training.emit", request, args.workdir or "fskills-emit")


def _cmd_launch(args: argparse.Namespace) -> int:
    from foundationskills.interfaces.fs.launch import launch

    spec = _load_payload(args.spec)
    try:
        result = launch(spec, confirm=args.confirm, submit=not args.no_submit,
                        measurement_only=bool(getattr(args, "measurement_only", False)))
    except ConfirmationRequired as exc:
        print(f"[fskills:cli:refuse] {exc}", file=sys.stderr)
        return 96
    except LaunchRefused as exc:
        print(f"[fskills:cli:refuse] {exc}", file=sys.stderr)
        return 96
    print(json.dumps(result, indent=2, sort_keys=True))
    rc = result.get("returncode")
    if rc is None:  # submitted to the scheduler: the verdict arrives with the job
        return 0
    return int(rc) if int(rc) in (0, 5, 95, 96) else 5


def _cmd_eval(args: argparse.Namespace) -> int:
    request = _eval_request(args)
    if getattr(args, "dry_run", False):
        return _eval_dry_run(request)
    workdir = args.workdir or str(Path(args.out).parent)
    return _execute("evaluation", request, workdir)


def _eval_request(args: argparse.Namespace) -> dict[str, Any]:
    request: dict[str, Any] = {"benchmarks": [b for b in args.benchmarks.split(",") if b], "policy": args.policy,
                               "eval_cache": args.eval_cache, "out": args.out}
    for key in ("checkpoint", "base", "run_manifest", "baseline_cache", "num_fewshot", "seed", "gen_kwargs",
                "limit", "dtype", "batch_size", "device", "include_path"):
        if getattr(args, key) is not None:
            request[key] = getattr(args, key)
    if args.parallelize:
        request["parallelize"] = True
    return request


def _eval_dry_run(request: dict[str, Any]) -> int:
    """Input checks only (the launch prologue): 0 when the run could start, 96 with the refusal otherwise."""
    from foundationskills.core.schema import SchemaError, assert_valid
    from foundationskills.skills.evaluation.skill import EvalSkill

    skill = EvalSkill()
    try:
        assert_valid(request, skill.input_schema, "evaluation request")
    except SchemaError as exc:
        print(f"[fskills:eval:refuse] invalid request: {str(exc).split('; ', 1)[0]}", file=sys.stderr)
        return 96
    blockers = [f for f in skill.check_inputs(request, SkillContext(workdir=Path(request["out"]).parent))
                if f.severity.name == "BLOCK"]
    for f in blockers:
        print(f"[fskills:eval:refuse] {f.rule_id}: {f.message}", file=sys.stderr)
    if not blockers:
        print("[fskills:eval:dry-run] inputs PASS")
    return 96 if blockers else 0


def _cmd_eval_emit(args: argparse.Namespace) -> int:
    from foundationskills.interfaces.fs.emit_eval import emit_eval

    spec = emit_eval(_eval_request(args), nodes=args.nodes, gpus_per_node=args.gpus_per_node,
                     hardware_id=args.hardware, python=args.python, run_name=args.run_name)
    target = Path(args.spec_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"spec": str(target), "executable": spec["executable"], "confirm": plan_hash(spec),
                      "missing": spec["missing"]}, indent=2, sort_keys=True))
    if not spec["executable"]:
        print(f"[fskills:eval:refuse] {spec['missing']}", file=sys.stderr)
        return 96
    return 0


def _cmd_hash(args: argparse.Namespace) -> int:
    print(plan_hash(_load_payload(args.spec)))
    return 0


def _cmd_routing_eval(args: argparse.Namespace) -> int:
    from foundationskills.agent import routing_eval as re_
    from foundationskills.skills.data_engine.llm_backend import LLMOpError, make_backend

    try:
        index = re_.load_index()
        if args.routing_command == "render":
            report = json.loads(Path(args.report).read_text(encoding="utf-8"))
            for path in re_.write_benchmarks(report, index):
                print(path)
            return 0
        cfg: dict[str, Any] = {"kind": "openai_compatible", "base_url": args.base_url, "model": args.model,
                               "timeout_s": args.timeout}
        if args.api_key_env:
            cfg["api_key_env"] = args.api_key_env
        if args.extra_body:
            cfg["extra_body"] = json.loads(args.extra_body)
        backend = make_backend(cfg, "routing_eval")
        report = re_.run_routing_eval(index, backend, arms=tuple(args.arms.split(",")), reps=args.reps,
                                      workers=args.workers, temperature=args.temperature,
                                      max_tokens=args.max_tokens, threshold=args.threshold)
    except (re_.RoutingEvalRefused, LLMOpError, OSError, json.JSONDecodeError) as exc:
        print(f"[fskills:routing-eval:refuse] {exc}", file=sys.stderr)
        return 96
    re_.write_report(report, Path(args.out))
    if args.write_benchmarks and report["verdict"] != "unmeasured":
        for path in re_.write_benchmarks(report, index):
            print(path, file=sys.stderr)
    summary = {arm: report["results"][arm]["accuracy"] for arm in report["arms"]}
    print(json.dumps({"verdict": report["verdict"], "accuracy": summary, "uplift": report["uplift"],
                      "reasons": report["reasons"], "report": str(Path(args.out).resolve())}, indent=2))
    return int(report["exit_code"])


def _cmd_recipes(args: argparse.Namespace) -> int:
    from foundationskills.skills.training.knowledge import load_recipes

    recipes = load_recipes()
    if args.recipes_command == "list":
        out = []
        for recipe in recipes:
            index = dict(recipe.raw.get("index") or {})
            if args.stage and index.get("stage") != args.stage:
                continue
            if args.goal and index.get("goal") != args.goal:
                continue
            if args.family and index.get("family") != args.family:
                continue
            out.append({"id": recipe.raw.get("id"), "title": recipe.raw.get("title"), "index": index})
        print(json.dumps(out, indent=2, sort_keys=True))
        return 0
    matches = [r for r in recipes if r.raw.get("id") == args.id]
    if not matches:
        print(f"[fskills:cli:refuse] missing: recipe {args.id!r}", file=sys.stderr)
        return 96
    print(json.dumps(matches[0].raw, indent=2, sort_keys=True))
    return 0


def _cmd_datasets(args: argparse.Namespace) -> int:
    import yaml
    from importlib import resources

    rel = resources.files("foundationskills.skills.data_engine").joinpath("catalog", "public_datasets.yaml")
    try:
        with rel.open("r", encoding="utf-8") as handle:
            catalog = yaml.safe_load(handle)
    except FileNotFoundError:
        print(
            "[fskills:cli:refuse] missing: data_engine catalog public_datasets.yaml",
            file=sys.stderr,
        )
        return 96
    entries = catalog or []
    out = []
    for entry in entries:
        if args.goal and args.goal not in (entry.get("goals") or []):
            continue
        if args.stage and args.stage not in (entry.get("formats") or []):
            continue
        if args.domain and args.domain not in (entry.get("domains") or []):
            continue
        out.append(entry)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(prog="fskills", description="FoundationSkills command line (exit 0/5/95/96).")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    p = sub.add_parser("probe", help="probe the installed FoundationScale")
    p.add_argument("--deep", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_probe)

    p = sub.add_parser("skills", help="inspect registered skills")
    skills_sub = p.add_subparsers(dest="skills_command", required=True, parser_class=_Parser)
    p_list = skills_sub.add_parser("list")
    p_list.set_defaults(func=_cmd_skills)
    p_describe = skills_sub.add_parser("describe")
    p_describe.add_argument("name")
    p_describe.set_defaults(func=_cmd_skills)

    p = sub.add_parser("data", help="data engine operations")
    data_sub = p.add_subparsers(dest="data_command", required=True, parser_class=_Parser)
    p_run = data_sub.add_parser("run")
    p_run.add_argument("--request", required=True)
    p_run.add_argument("--workdir", default=None)
    p_run.set_defaults(func=_cmd_data)

    p = sub.add_parser("plan", help="build a training plan from a goal")
    p.add_argument("--goal", required=True)
    p.add_argument("--readiness", default=None)
    p.add_argument("--hardware", default=None)
    p.add_argument("--workdir", default=None)
    p.set_defaults(func=_cmd_plan)

    p = sub.add_parser("emit", help="emit FS launch specs from a plan")
    p.add_argument("--plan", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-root", dest="output_root", required=True)
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--gpus-per-node", dest="gpus_per_node", type=int, default=1)
    p.add_argument("--hardware", default=None)
    p.add_argument("--workdir", default=None)
    p.set_defaults(func=_cmd_emit)

    p = sub.add_parser("launch", help="launch a confirmed fs_launch_spec")
    p.add_argument("--spec", required=True)
    p.add_argument("--confirm", default=None)
    p.add_argument("--no-submit", dest="no_submit", action="store_true")
    p.add_argument("--measurement-only", dest="measurement_only", action="store_true",
                   help="run a spec whose only gap is the hand-off (e.g. FS RL saves no checkpoint)")
    p.set_defaults(func=_cmd_launch)

    p = sub.add_parser("eval", help="evaluate a checkpoint against its base (lm-eval, offline)")
    eval_sub = p.add_subparsers(dest="eval_command", required=True, parser_class=_Parser)
    p_run = eval_sub.add_parser("run")
    p_run.add_argument("--checkpoint", default=None)
    p_run.add_argument("--base", default=None)
    p_run.add_argument("--run-manifest", dest="run_manifest", default=None)
    p_run.add_argument("--benchmarks", required=True, help="comma-separated policy task names")
    p_run.add_argument("--policy", required=True)
    p_run.add_argument("--eval-cache", dest="eval_cache", required=True,
                       help="pre-staged offline HF_HOME; nothing is downloaded")
    p_run.add_argument("--out", required=True, help="path of eval_report.json")
    p_run.add_argument("--baseline-cache", dest="baseline_cache", default=None)
    p_run.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=None)
    p_run.add_argument("--seed", type=int, default=None)
    p_run.add_argument("--gen-kwargs", dest="gen_kwargs", default=None)
    p_run.add_argument("--limit", type=int, default=None, help="smoke slice; the verdict can never be PASS")
    p_run.add_argument("--dtype", default=None)
    p_run.add_argument("--batch-size", dest="batch_size", default=None)
    p_run.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: the harness's)")
    p_run.add_argument("--parallelize", action="store_true")
    p_run.add_argument("--include-path", dest="include_path", default=None)
    p_run.add_argument("--workdir", default=None)
    p_run.add_argument("--dry-run", dest="dry_run", action="store_true",
                       help="check inputs only (the launch prologue); nothing is evaluated")
    p_run.set_defaults(func=_cmd_eval)

    p_emit = eval_sub.add_parser("emit", help="emit an fs_launch_spec for a Slurm eval job; nothing is submitted")
    for action in p_run._actions:
        if action.dest not in ("help", "workdir", "dry_run"):
            p_emit._add_action(action)
    p_emit.add_argument("--spec-out", dest="spec_out", required=True, help="path of the fs_eval_launch_spec JSON")
    p_emit.add_argument("--nodes", type=int, default=1)
    p_emit.add_argument("--gpus", dest="gpus_per_node", type=int, default=1)
    p_emit.add_argument("--hardware", default="gb200", help="hardware id; 'local' emits no sbatch")
    p_emit.add_argument("--python", default="python", help="interpreter the job runs (its lm_eval is version-checked)")
    p_emit.add_argument("--run-name", dest="run_name", default="fskills-eval")
    p_emit.set_defaults(func=_cmd_eval_emit)

    p_hash = eval_sub.add_parser("hash", help="print the confirmation hash of an eval spec")
    p_hash.add_argument("--spec", required=True)
    p_hash.set_defaults(func=_cmd_hash)

    p_launch = eval_sub.add_parser("launch", help="launch a confirmed eval spec (dry-run first)")
    p_launch.add_argument("--spec", required=True)
    p_launch.add_argument("--confirm", default=None)
    p_launch.add_argument("--no-submit", dest="no_submit", action="store_true")
    p_launch.set_defaults(func=_cmd_launch)

    p = sub.add_parser("hash", help="print the confirmation hash of a spec/plan")
    p.add_argument("--spec", required=True)
    p.set_defaults(func=_cmd_hash)

    p = sub.add_parser("routing-eval", help="measure agent routing to skills on each package's evals/evals.json")
    routing_sub = p.add_subparsers(dest="routing_command", required=True, parser_class=_Parser)
    p_run = routing_sub.add_parser("run", help="ask an OpenAI-compatible model to route every eval case")
    p_run.add_argument("--base-url", required=True)
    p_run.add_argument("--model", required=True)
    p_run.add_argument("--api-key-env", default=None, help="NAME of the env var holding the key (never the key)")
    p_run.add_argument("--extra-body", default=None, help="JSON merged into each request body")
    p_run.add_argument("--arms", default="full,description,names_only")
    p_run.add_argument("--reps", type=int, default=3)
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--temperature", type=float, default=0.6)
    p_run.add_argument("--max-tokens", type=int, default=2048)
    p_run.add_argument("--threshold", type=float, default=0.9)
    p_run.add_argument("--timeout", type=float, default=300.0)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--write-benchmarks", action="store_true",
                       help="write BENCHMARK.md per package unless unmeasured")
    p_run.set_defaults(func=_cmd_routing_eval)
    p_render = routing_sub.add_parser("render", help="write BENCHMARK.md per package from an existing report")
    p_render.add_argument("--report", required=True)
    p_render.set_defaults(func=_cmd_routing_eval)

    p = sub.add_parser("recipes", help="browse recipe knowledge")
    recipes_sub = p.add_subparsers(dest="recipes_command", required=True, parser_class=_Parser)
    p_list = recipes_sub.add_parser("list")
    p_list.add_argument("--stage", default=None)
    p_list.add_argument("--goal", default=None)
    p_list.add_argument("--family", default=None)
    p_list.set_defaults(func=_cmd_recipes)
    p_show = recipes_sub.add_parser("show")
    p_show.add_argument("id")
    p_show.set_defaults(func=_cmd_recipes)

    p = sub.add_parser("datasets", help="discover public datasets")
    datasets_sub = p.add_subparsers(dest="datasets_command", required=True, parser_class=_Parser)
    p_discover = datasets_sub.add_parser("discover")
    p_discover.add_argument("--goal", required=True)
    p_discover.add_argument("--stage", required=True)
    p_discover.add_argument("--domain", default=None)
    p_discover.set_defaults(func=_cmd_datasets)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        if exc.code is None or exc.code == 0:
            return 0
        return int(exc.code) if isinstance(exc.code, int) else 96
    try:
        return int(args.func(args))
    except ConfirmationRequired as exc:
        print(f"[fskills:cli:refuse] {exc}", file=sys.stderr)
        return 96
    except LaunchRefused as exc:
        print(f"[fskills:cli:refuse] {exc}", file=sys.stderr)
        return 96
    except Exception as exc:  # noqa: BLE001 - the contract has no code for "crashed"
        traceback.print_exc(file=sys.stderr)
        print(f"[fskills:cli:red] unhandled {type(exc).__name__}: {exc}", file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
