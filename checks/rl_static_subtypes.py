#!/usr/bin/env python3
"""RL protocol subtype gate: discovery is static, truth is mypy-only.

House contract:
  rc 0  = clean
  rc 5  = subtype violation, empty discovery, shrunken discovery, or positive
          control failed to fire
  rc 95 = mypy is not installed: UNMEASURED, never silently green
  rc 96 = cannot measure: no usable repo mypy config, no committed baseline
          for completeness, or mypy itself failed fatally

This file imports no torch and imports no foundationscale modules. It parses
the tree with stdlib ast only to find CLAIMS, then writes a temporary probe
module and lets mypy decide whether each claim is a real subtype.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
RL = SRC / "foundationscale" / "rl"
BASELINE = Path(__file__).with_suffix(".baseline.json")

ALGORITHM_MODULE = "foundationscale.rl.algorithm"
INTERFACES_MODULE = "foundationscale.rl.interfaces"
REGISTRY_PATH = RL / "registry.py"

CONFIG_CANDIDATES = ("mypy.ini", ".mypy.ini", "setup.cfg", "pyproject.toml")


@dataclass(frozen=True, order=True)
class ProtocolInfo:
    module: str
    name: str
    members: tuple[str, ...]


@dataclass(frozen=True, order=True)
class Pair:
    key: str
    protocol_module: str
    protocol: str
    concrete_module: str
    concrete: str
    kind: str  # class | factory_call | structural_class
    source: str


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC).with_suffix("")
    return ".".join(rel.parts)


def _module_path(module: str) -> Path:
    return SRC.joinpath(*module.split(".")).with_suffix(".py")


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_protocol_base(base: ast.expr) -> bool:
    if isinstance(base, ast.Name):
        return base.id in {"Protocol", "TypedDict"}
    if isinstance(base, ast.Attribute):
        return base.attr == "Protocol"
    if isinstance(base, ast.Subscript):
        return _is_protocol_base(base.value)
    return False


def _class_members(node: ast.ClassDef) -> tuple[str, ...]:
    members: set[str] = set()
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name.startswith("__") and item.name != "__call__":
                continue
            members.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            members.add(item.target.id)
    return tuple(sorted(members))


def _discover_protocols() -> dict[str, ProtocolInfo]:
    found: dict[str, ProtocolInfo] = {}
    for path in sorted(RL.glob("*.py")):
        module = _module_name(path)
        tree = _parse(path)
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(_is_protocol_base(base) for base in node.bases):
                continue
            members = _class_members(node)
            if members:
                found[f"{module}.{node.name}"] = ProtocolInfo(module, node.name, members)
    return found


def _local_import_map(tree: ast.Module) -> dict[str, str]:
    # ast.walk, not tree.body: registry.py imports every binding INSIDE
    # _install_default_algorithms() to break a module-import cycle, so a
    # top-level-only walk reads an empty map -- and an empty map is
    # indistinguishable from "this module imports nothing", which is what
    # makes the miss silent rather than loud.
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None or not node.module.startswith("foundationscale.rl."):
            continue
        for alias in node.names:
            if alias.name == "*":
                continue
            out[alias.asname or alias.name] = node.module
    return out


def _function_returns_concrete(tree: ast.Module, fn_name: str) -> str | None:
    imported: dict[str, str] = {}
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported[alias.asname or alias.name] = node.module or ""
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != fn_name:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Return):
                continue
            value = sub.value
            func: ast.expr | None = value.func if isinstance(value, ast.Call) else value
            if isinstance(func, ast.Name) and func.id in classes:
                return func.id
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                return func.attr
        return None
    return None


def _registry_binding_symbol(tree: ast.Module, key: str) -> str | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        targets = node.targets
        if not isinstance(value, (ast.Name, ast.Attribute)):
            continue
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == "_REGISTRY"
                and isinstance(target.slice, ast.Constant)
                and target.slice.value == key
            ):
                continue
            if isinstance(value, ast.Name):
                return value.id
            return value.attr
    return None


def _discover_algorithm_pairs() -> set[Pair]:
    pairs: set[Pair] = set()
    if not REGISTRY_PATH.exists():
        return pairs
    tree = _parse(REGISTRY_PATH)
    import_map = _local_import_map(tree)
    registry_keys: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "_REGISTRY"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                registry_keys.append(target.slice.value)
    for key in sorted(set(registry_keys)):
        symbol = _registry_binding_symbol(tree, key)
        if symbol is None:
            pairs.add(
                Pair(
                    key=f"registry:{key}",
                    protocol_module=ALGORITHM_MODULE,
                    protocol="Algorithm",
                    concrete_module=REGISTRY_PATH.stem,
                    concrete="<unresolved>",
                    kind="factory_call",
                    source=f"registry _REGISTRY[{key!r}] unresolved",
                )
            )
            continue
        if symbol not in import_map:
            # Defaulting the defining module to ALGORITHM_MODULE here would
            # emit a probe importing a name that module does not export, and
            # mypy's attr-defined error would read as a SUBTYPE failure. A
            # symbol whose module cannot be resolved is UNRESOLVED, which is
            # a different verdict from "not a subtype".
            pairs.add(
                Pair(
                    key=f"registry:{key}",
                    protocol_module=ALGORITHM_MODULE,
                    protocol="Algorithm",
                    concrete_module=REGISTRY_PATH.stem,
                    concrete="<unresolved>",
                    kind="factory_call",
                    source=(
                        f"registry _REGISTRY[{key!r}] -> {symbol}; defining "
                        f"module not resolvable from registry imports"
                    ),
                )
            )
            continue
        module = import_map[symbol]
        source = f"registry _REGISTRY[{key!r}] -> {symbol} from {module}"
        path = _module_path(module)
        kind = "class"
        concrete = symbol
        if path.exists():
            other = _parse(path)
            class_names = {n.name for n in other.body if isinstance(n, ast.ClassDef)}
            fn_names = {n.name for n in other.body if isinstance(n, ast.FunctionDef)}
            if symbol in fn_names and symbol not in class_names:
                returned = _function_returns_concrete(other, symbol)
                if returned:
                    concrete = returned
                    kind = "class"
                    source += f"; factory returns {returned}"
                else:
                    kind = "factory_call"
                    source += "; concrete return not statically resolvable"
        pairs.add(
            Pair(
                key=f"registry:{key}",
                protocol_module=ALGORITHM_MODULE,
                protocol="Algorithm",
                concrete_module=module,
                concrete=concrete,
                kind=kind,
                source=source,
            )
        )
    return pairs


def _discover_structural_pairs(protocols: Mapping[str, ProtocolInfo]) -> set[Pair]:
    """Conservative claims: LossFn candidates must live in rl/losses.py.

    This is discovery of claims only. A class is never accepted here; it is
    merely written into a probe that mypy can reject.
    """
    pairs: set[Pair] = set()
    wanted = protocols.get(f"{INTERFACES_MODULE}.LossFn")
    if wanted is None:
        return pairs
    losses = RL / "losses.py"
    if not losses.exists():
        return pairs
    module = _module_name(losses)
    tree = _parse(losses)
    required = set(wanted.members)
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        if any(_is_protocol_base(base) for base in node.bases):
            continue
        if required.issubset(set(_class_members(node))):
            pairs.add(
                Pair(
                    key=f"lossfn:{node.name}",
                    protocol_module=wanted.module,
                    protocol=wanted.name,
                    concrete_module=module,
                    concrete=node.name,
                    kind="structural_class",
                    source=f"{module}.{node.name} names every LossFn member {sorted(required)}",
                )
            )
    return pairs


def discover_pairs() -> tuple[set[Pair], tuple[str, ...]]:
    protocols = _discover_protocols()
    notes = tuple(
        f"protocol {name} members={list(info.members)}"
        for name, info in sorted(protocols.items())
        if name in {f"{ALGORITHM_MODULE}.Algorithm", f"{INTERFACES_MODULE}.LossFn"}
    )
    pairs = _discover_algorithm_pairs() | _discover_structural_pairs(protocols)
    return pairs, notes


def _display(path: Path) -> str:
    # A --baseline pointed outside the repo has no repo-relative rendering,
    # and relative_to() RAISES rather than abstaining -- an unreadable input
    # would then surface as a traceback instead of the 96 the contract owes.
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def _read_baseline(path: Path) -> tuple[dict[str, str] | None, str]:
    if not path.exists():
        return None, f"baseline absent at {_display(path)}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"baseline unreadable: {exc}"
    if not isinstance(raw, dict) or not isinstance(raw.get("pairs"), list):
        return None, "baseline must be {'pairs': [key, ...]}"
    pairs: dict[str, str] = {}
    for item in raw["pairs"]:
        if not isinstance(item, dict) or not isinstance(item.get("key"), str):
            return None, "baseline contains a non-string pair key"
        pairs[item["key"]] = str(item.get("fingerprint", ""))
    return pairs, "baseline loaded"


def _fingerprint(pair: Pair) -> str:
    return (
        f"{pair.protocol_module}.{pair.protocol}"
        f"<-{pair.concrete_module}.{pair.concrete}:{pair.kind}"
    )


def _write_baseline(pairs: Iterable[Pair], path: Path) -> None:
    payload = {
        "contract": "rl-static-subtypes-v1",
        "pairs": [{"fingerprint": _fingerprint(p), "key": p.key} for p in sorted(pairs)],
    }
    # Written to `path`, not to BASELINE: --write-baseline --baseline OTHER that
    # silently wrote the default would make the doctoring lever unusable, and the
    # gate would then re-read a file the operator never asked it to produce.
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_config() -> Path | None:
    for name in CONFIG_CANDIDATES:
        path = REPO / name
        if not path.exists():
            continue
        if name == "pyproject.toml" and "[tool.mypy]" not in path.read_text(encoding="utf-8"):
            continue
        if name == "setup.cfg" and "[mypy]" not in path.read_text(encoding="utf-8"):
            continue
        return path
    return None


def _run_mypy(probe_dir: Path) -> tuple[int, str]:
    if importlib.util.find_spec("mypy") is None:
        return UNMEASURED, "mypy is not installed"
    config = _find_config()
    if config is None:
        return REFUSE, f"no repo mypy config among {CONFIG_CANDIDATES}"
    env = dict(os.environ)
    env["MYPYPATH"] = str(SRC) + (os.pathsep + env["MYPYPATH"] if env.get("MYPYPATH") else "")
    cmd = [
        sys.executable,
        "-m",
        "mypy",
        "--config-file",
        str(config),
        "--no-color-output",
        "--hide-error-context",
        "--show-error-codes",
        str(probe_dir),
    ]
    try:
        proc = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return REFUSE, f"mypy cannot be measured: {exc}"
    output = (proc.stdout + proc.stderr).strip().replace("\n", " | ")
    if proc.returncode == 0:
        return GREEN, output or "mypy clean"
    if proc.returncode == 1:
        return RED, output or "mypy reported errors"
    return REFUSE, output or f"mypy fatal rc={proc.returncode}"


def _probe_source(pairs: Sequence[Pair]) -> str:
    lines = [
        "from __future__ import annotations",
        f"from {ALGORITHM_MODULE} import Algorithm as _P_Algorithm",
        f"from {INTERFACES_MODULE} import LossFn as _P_LossFn",
        "",
    ]
    for i, pair in enumerate(pairs):
        proto_alias = "_P_Algorithm" if pair.protocol == "Algorithm" else "_P_LossFn"
        if pair.kind == "factory_call":
            lines.append(f"from {pair.concrete_module} import {pair.concrete} as _F{i}")
            lines.append(f"def _probe_{i}() -> {proto_alias}:")
            lines.append(f"    return _F{i}()")
        else:
            lines.append(f"from {pair.concrete_module} import {pair.concrete} as _C{i}")
            lines.append(f"def _probe_{i}(v: _C{i}) -> {proto_alias}:")
            lines.append("    return v")
        lines.append(f"# {pair.key} :: {pair.source}")
        lines.append("")
    return "\n".join(lines)


def _must_fire() -> tuple[int, str]:
    """Plant a wrong return signature and demand this gate map mypy to rc 5."""
    if importlib.util.find_spec("mypy") is None:
        return UNMEASURED, "mypy is not installed"
    if _find_config() is None:
        return REFUSE, "positive control cannot run without repo mypy config"
    with tempfile.TemporaryDirectory(prefix="rl_static_subtypes_mustfire_", dir=REPO) as td:
        probe = Path(td) / "probe_bad.py"
        probe.write_text(
            "from __future__ import annotations\n"
            f"from {ALGORITHM_MODULE} import Algorithm as _P_Algorithm\n"
            "def _probe_bad(v: int) -> _P_Algorithm:\n"
            "    return v\n",
            encoding="utf-8",
        )
        rc, detail = _run_mypy(Path(td))
    if rc == RED:
        return GREEN, f"positive control fired rc5: {detail}"
    return RED, f"positive control did NOT fire: expected rc5, got rc{rc}: {detail}"


def _verdict(rc: int, message: str) -> int:
    label = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "CANNOT-MEASURE"}[rc]
    print(f"RL-STATIC-SUBTYPES {label}: {message}")
    return rc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False, prog="rl_static_subtypes.py")
    parser.add_argument("--write-baseline", action="store_true")
    # The baseline is this gate's ONLY doctorable input, and without a way to
    # point at another copy the only control a suite could run is the gate's
    # own --self-test -- the same instrument twice, which measures nothing new.
    # checks/mutation_scope.py carries --mutate-py for exactly this reason.
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    # --self-test is the name every other checks/ gate uses for "run your own
    # controls"; --must-fire is kept as the alias this gate was authored with,
    # because the Makefile convention pairs a self-test with the real run and a
    # gate the aggregate cannot invoke by the usual name is an orphan (#293).
    parser.add_argument("--must-fire", "--self-test", action="store_true", dest="must_fire")
    parser.add_argument("-h", "--help", action="store_true")
    args = parser.parse_args(argv)
    if args.help:
        return _verdict(
            REFUSE,
            "usage: rl_static_subtypes.py [--baseline PATH] "
            "[--write-baseline] | [--self-test|--must-fire]",
        )
    if args.must_fire:
        rc, detail = _must_fire()
        return _verdict(rc, "MUST-FIRE " + detail)

    pairs, notes = discover_pairs()
    keys = {pair.key for pair in pairs}
    detail_bits = [
        f"pairs={len(pairs)}",
        "algorithm-keys="
        + ",".join(sorted(k for k in keys if k.startswith("registry:")) or ["none"]),
        "lossfn="
        + ",".join(sorted(p.concrete for p in pairs if p.protocol == "LossFn") or ["none"]),
    ]
    if not pairs:
        return _verdict(RED, "; ".join(detail_bits + ["empty pair set is refused"]))
    if args.write_baseline:
        _write_baseline(pairs, args.baseline)
        return _verdict(
            GREEN, "; ".join(detail_bits + [f"baseline written with {len(pairs)} pairs"])
        )

    baseline, why = _read_baseline(args.baseline)
    if baseline is None:
        return _verdict(REFUSE, "; ".join(detail_bits + ["completeness " + why]))
    missing = sorted(set(baseline) - keys)
    if missing:
        return _verdict(
            RED, "; ".join(detail_bits + [f"discovered pair set shrank by missing={missing}"])
        )

    with tempfile.TemporaryDirectory(prefix="rl_static_subtypes_", dir=REPO) as td:
        probe_dir = Path(td)
        (probe_dir / "probe_claims.py").write_text(_probe_source(sorted(pairs)), encoding="utf-8")
        rc, mypy_detail = _run_mypy(probe_dir)
    if rc == GREEN:
        return _verdict(GREEN, "; ".join(detail_bits + list(notes) + [f"baseline={len(baseline)}"]))
    return _verdict(rc, "; ".join(detail_bits + [mypy_detail]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
