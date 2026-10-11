"""Dockerfile-subset builder producing enroot ``.sqsh`` images.

Stdlib only. Every external command goes through the injectable ``Runner``
provided by :mod:`foundationscale.agentic_rl.sandbox.enroot`.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from foundationscale.agentic_rl.sandbox.enroot import (
    EnrootSandbox,
    Runner,
    SandboxBuildFailed,
    SandboxRefusal,
    SandboxSpec,
    SubprocessRunner,
)

_STDERR_TAIL = 2000
_IGNORED_INSTRUCTIONS = frozenset(
    {"EXPOSE", "CMD", "ENTRYPOINT", "LABEL", "HEALTHCHECK", "VOLUME", "STOPSIGNAL"}
)
_SUPPORTED_INSTRUCTIONS = (
    frozenset({"FROM", "ARG", "ENV", "WORKDIR", "RUN", "COPY", "ADD", "USER"})
    | _IGNORED_INSTRUCTIONS
)


@dataclass(frozen=True)
class DockerfileStep:
    lineno: int
    instruction: str
    argument: str


@dataclass(frozen=True)
class BuildReport:
    image: str
    steps_applied: int
    seconds: float | None
    env: Mapping[str, str]
    workdir: str


def parse_dockerfile(text: str) -> tuple[DockerfileStep, ...]:
    """Parse a Dockerfile into logical steps (comments and blanks dropped)."""
    if not isinstance(text, str):
        raise SandboxRefusal(f"text: expected str, got {type(text).__name__}")
    steps: list[DockerfileStep] = []
    logical = ""
    start_lineno = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip()
        if not logical:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            start_lineno = lineno
            logical = line
        else:
            logical = logical + "\n" + line
        if logical.rstrip().endswith("\\"):
            logical = logical.rstrip()[:-1]
            continue
        joined = " ".join(part.strip() for part in logical.splitlines())
        logical = ""
        parts = joined.split(None, 1)
        if not parts:
            continue
        instruction = parts[0].upper()
        argument = parts[1].strip() if len(parts) > 1 else ""
        steps.append(
            DockerfileStep(lineno=start_lineno, instruction=instruction, argument=argument)
        )
    if logical:
        joined = " ".join(part.strip() for part in logical.splitlines())
        parts = joined.split(None, 1)
        if parts:
            steps.append(
                DockerfileStep(
                    lineno=start_lineno,
                    instruction=parts[0].upper(),
                    argument=parts[1].strip() if len(parts) > 1 else "",
                )
            )
    return tuple(steps)


def build_image(
    dockerfile: str | Path,
    context_dir: str | Path,
    out_sqsh: str | Path,
    *,
    data_root: str,
    image_store: str,
    scratch_root: str,
    runner: Runner | None = None,
    build_args: Mapping[str, str] | None = None,
    timeout_s: float = 3600.0,
) -> BuildReport:
    """Build a ``.sqsh`` image from a Dockerfile subset inside ``context_dir``."""
    df_path = Path(dockerfile)
    if not df_path.is_file():
        raise SandboxRefusal(f"dockerfile: expected existing file, got {df_path}")
    context = Path(context_dir)
    if not context.is_dir():
        raise SandboxRefusal(f"context_dir: expected existing directory, got {context}")
    out_path = Path(out_sqsh)
    if out_path.suffix != ".sqsh":
        raise SandboxRefusal(f"out_sqsh: expected a *.sqsh path, got {out_sqsh!r}")
    if not Path(data_root).is_absolute():
        raise SandboxRefusal(f"data_root: expected absolute path, got {data_root!r}")
    if not Path(image_store).is_absolute():
        raise SandboxRefusal(f"image_store: expected absolute path, got {image_store!r}")
    if not Path(scratch_root).is_absolute():
        raise SandboxRefusal(f"scratch_root: expected absolute path, got {scratch_root!r}")
    if build_args is not None:
        for key, value in build_args.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise SandboxRefusal(
                    f"build_args: expected Mapping[str, str], got {key!r}: {value!r}"
                )
    if timeout_s <= 0:
        raise SandboxRefusal(f"timeout_s: expected > 0, got {timeout_s!r}")

    steps = parse_dockerfile(df_path.read_text(encoding="utf-8"))
    if not steps:
        raise SandboxRefusal(
            f"dockerfile: expected at least one instruction, got none in {df_path}"
        )

    from_steps = [s for s in steps if s.instruction == "FROM"]
    if len(from_steps) != 1:
        raise SandboxRefusal(f"FROM: expected exactly one, got {len(from_steps)}")
    from_step = from_steps[0]
    # ARGs declared before FROM are in scope for the FROM line itself (Dockerfile semantics).
    pre_from_args: dict[str, str] = {}
    for step in steps:
        if step.lineno >= from_step.lineno:
            break
        if step.instruction == "ARG":
            _apply_arg(step, _substitute(step.argument, pre_from_args), pre_from_args, build_args)
    from_text = _substitute(from_step.argument, pre_from_args)
    from_arg = from_text.split()[0] if from_text else ""
    if not from_arg or from_arg.lower() == "scratch":
        raise SandboxRefusal(f"FROM: expected an image reference, got {from_step.argument!r}")
    if from_arg.startswith("docker://") or from_arg.endswith(".sqsh"):
        base_image = from_arg
    else:
        base_image = "docker://" + from_arg

    args: dict[str, str] = {}
    if build_args:
        args.update(build_args)
    env: dict[str, str] = {}
    workdir = "/"
    applied = 0
    started = time.monotonic()

    digest = hashlib.sha256(
        f"{df_path.resolve()}|{context.resolve()}|{sorted((build_args or {}).items())}".encode()
    ).hexdigest()[:16]
    spec = SandboxSpec(
        name=f"build-{digest}",
        image=base_image,
        data_root=data_root,
        image_store=image_store,
        scratch_root=scratch_root,
        network="public",
    )
    _validate_steps(steps, dict(build_args or {}), build_args, context)
    sandbox = EnrootSandbox(spec, runner=runner if runner is not None else SubprocessRunner())
    sandbox.start()
    try:
        for step in steps:
            instruction = step.instruction
            if instruction == "FROM":
                applied += 1
                continue
            if instruction not in _SUPPORTED_INSTRUCTIONS:
                raise SandboxRefusal(
                    f"line {step.lineno}: expected a supported instruction, got {instruction!r}"
                )
            argument = _substitute(step.argument, args)
            if instruction == "ARG":
                _apply_arg(step, argument, args, build_args)
            elif instruction == "ENV":
                env.update(_parse_env(step, argument))
            elif instruction == "WORKDIR":
                workdir = _parse_workdir(step, argument)
                # Docker's WORKDIR creates the directory; every later exec does `cd <workdir>`.
                sandbox.host_path(workdir).mkdir(parents=True, exist_ok=True)
            elif instruction == "RUN":
                remaining = timeout_s - (time.monotonic() - started)
                if remaining <= 0:
                    raise SandboxBuildFailed(
                        f"line {step.lineno}: RUN: build budget of {timeout_s}s exhausted "
                        f"before this step"
                    )
                _run_step(sandbox, step, argument, env, workdir, remaining)
            elif instruction in ("COPY", "ADD"):
                _copy_step(sandbox, step, instruction, argument, context)
            elif instruction == "USER":
                _check_user(step, argument)
            applied += 1

        metadata = {"env": dict(env), "workdir": workdir}
        meta_path = sandbox.rootfs() / "etc" / "fs-sandbox.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        export_runner: Runner = runner if runner is not None else SubprocessRunner()
        result = export_runner.run(
            ["enroot", "export", "-o", str(out_path), spec.name],
            env={"ENROOT_DATA_PATH": data_root},
            timeout_s=timeout_s,
        )
        if result.returncode != 0:
            raise SandboxBuildFailed(
                f"enroot export failed (exit {result.returncode}): {result.stderr[-_STDERR_TAIL:]}"
            )
    finally:
        sandbox.stop(delete=True)

    return BuildReport(
        image=from_arg,
        steps_applied=applied,
        seconds=time.monotonic() - started,
        env=MappingProxyType(dict(env)),
        workdir=workdir,
    )


def _validate_steps(
    steps: tuple[DockerfileStep, ...],
    args: dict[str, str],
    build_args: Mapping[str, str] | None,
    context: Path,
) -> None:
    """Refuse every unsupported or unsafe line BEFORE a container exists (cheap and leak-free)."""
    for step in steps:
        instruction = step.instruction
        if instruction == "FROM":
            continue
        if instruction not in _SUPPORTED_INSTRUCTIONS:
            raise SandboxRefusal(
                f"line {step.lineno}: expected a supported instruction, got {instruction!r}"
            )
        argument = _substitute(step.argument, args)
        if instruction == "ARG":
            _apply_arg(step, argument, args, build_args)
        elif instruction == "ENV":
            _parse_env(step, argument)
        elif instruction == "WORKDIR":
            _parse_workdir(step, argument)
        elif instruction in ("COPY", "ADD"):
            _copy_plan(step, instruction, argument, context)
        elif instruction == "USER":
            _check_user(step, argument)


def _substitute(text: str, args: Mapping[str, str]) -> str:
    out = text
    for key, value in args.items():
        out = out.replace("${" + key + "}", value)
        out = out.replace("$" + key, value)
    return out


def _apply_arg(
    step: DockerfileStep,
    argument: str,
    args: dict[str, str],
    build_args: Mapping[str, str] | None,
) -> None:
    if not argument:
        raise SandboxRefusal(f"line {step.lineno}: ARG: expected a name, got empty argument")
    name, sep, default = argument.partition("=")
    name = name.strip()
    if not name or any(c.isspace() for c in name):
        raise SandboxRefusal(f"line {step.lineno}: ARG: expected a single name, got {argument!r}")
    if build_args is not None and name in build_args:
        args[name] = build_args[name]
    elif sep:
        args[name] = default.strip().strip("'\"")
    else:
        args.setdefault(name, "")


def _parse_env(step: DockerfileStep, argument: str) -> dict[str, str]:
    if not argument:
        raise SandboxRefusal(f"line {step.lineno}: ENV: expected assignments, got empty argument")
    result: dict[str, str] = {}
    tokens = _split_env_tokens(argument)
    if "=" in tokens[0]:
        for token in tokens:
            if "=" not in token:
                raise SandboxRefusal(f"line {step.lineno}: ENV: expected K=V form, got {token!r}")
            key, _, value = token.partition("=")
            key = key.strip()
            if not key:
                raise SandboxRefusal(
                    f"line {step.lineno}: ENV: expected a non-empty key, got {token!r}"
                )
            result[key] = value
    else:
        if len(tokens) < 2:
            raise SandboxRefusal(f"line {step.lineno}: ENV: expected 'K V' form, got {argument!r}")
        result[tokens[0]] = " ".join(tokens[1:])
    return result


def _split_env_tokens(argument: str) -> list[str]:
    lexer = shlex.shlex(argument, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _parse_workdir(step: DockerfileStep, argument: str) -> str:
    if not argument or any(c.isspace() for c in argument):
        raise SandboxRefusal(
            f"line {step.lineno}: WORKDIR: expected a single path, got {argument!r}"
        )
    return argument


def _check_user(step: DockerfileStep, argument: str) -> None:
    if argument not in ("root", "0"):
        raise SandboxRefusal(f"line {step.lineno}: USER: expected 'root' or '0', got {argument!r}")


def _run_step(
    sandbox: EnrootSandbox,
    step: DockerfileStep,
    argument: str,
    env: Mapping[str, str],
    workdir: str,
    timeout_s: float,
) -> None:
    command = _run_command(step, argument)
    result = sandbox.exec(command, cwd=workdir, env=env, timeout_s=timeout_s)
    if result.return_code != 0:
        tail = result.stderr[-_STDERR_TAIL:]
        raise SandboxBuildFailed(
            f"line {step.lineno}: RUN {argument!r} failed (exit {result.return_code}): {tail}"
        )


def _run_command(step: DockerfileStep, argument: str) -> str:
    if not argument:
        raise SandboxRefusal(f"line {step.lineno}: RUN: expected a command, got empty argument")
    stripped = argument.lstrip()
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except ValueError as exc:
            raise SandboxRefusal(
                f"line {step.lineno}: RUN: expected valid JSON exec form, got {argument!r}"
            ) from exc
        if (
            not isinstance(parsed, list)
            or not parsed
            or not all(isinstance(part, str) for part in parsed)
        ):
            raise SandboxRefusal(
                f"line {step.lineno}: RUN: expected a non-empty JSON array of strings, "
                f"got {argument!r}"
            )
        return shlex.join(parsed)
    return argument


_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz", ".zip")


def _copy_plan(
    step: DockerfileStep, instruction: str, argument: str, context: Path
) -> tuple[list[Path], list[str], str]:
    """Validate a COPY/ADD line without touching any container: (sources, raw sources, target)."""
    if not argument:
        raise SandboxRefusal(f"line {step.lineno}: {instruction}: expected arguments, got empty")
    tokens = _split_env_tokens(argument)
    flags = [t for t in tokens if t.startswith("--")]
    for flag in flags:
        if flag.startswith("--from"):
            raise SandboxRefusal(f"line {step.lineno}: {instruction}: --from is not supported")
        raise SandboxRefusal(f"line {step.lineno}: {instruction}: unsupported flag {flag!r}")
    paths = [t for t in tokens if not t.startswith("--")]
    if len(paths) < 2:
        raise SandboxRefusal(
            f"line {step.lineno}: {instruction}: expected sources and a target, got {argument!r}"
        )
    sources, target = paths[:-1], paths[-1]
    if instruction == "ADD":
        for source in sources:
            if "://" in source:
                raise SandboxRefusal(
                    f"line {step.lineno}: ADD: URL sources are not supported, got {source!r}"
                )
            if source.lower().endswith(_ARCHIVE_SUFFIXES):
                # Docker auto-extracts local archives on ADD; copying the archive instead
                # would silently build a different environment.
                raise SandboxRefusal(
                    f"line {step.lineno}: ADD: archive auto-extraction is not supported, "
                    f"got {source!r}"
                )
    resolved: list[Path] = []
    for source in sources:
        resolved.extend(_resolve_context_paths(step, instruction, source, context))
    if not resolved:
        raise SandboxRefusal(
            f"line {step.lineno}: {instruction}: expected at least one source, got {argument!r}"
        )
    if not target.startswith("/"):
        raise SandboxRefusal(
            f"line {step.lineno}: {instruction}: expected an absolute target, got {target!r}"
        )
    return resolved, sources, target


def _copy_step(
    sandbox: EnrootSandbox,
    step: DockerfileStep,
    instruction: str,
    argument: str,
    context: Path,
) -> None:
    resolved, sources, target = _copy_plan(step, instruction, argument, context)
    if len(resolved) == 1 and not _is_glob(sources[0]) and not target.endswith("/"):
        source = resolved[0]
        if source.is_dir():
            sandbox.upload_dir(source, target)
        else:
            sandbox.upload_file(source, target)
        return
    for source in resolved:
        if source.is_dir():
            sandbox.upload_dir(source, target.rstrip("/") + "/" + source.name)
        else:
            sandbox.upload_file(source, target)


def _is_glob(source: str) -> bool:
    return any(ch in source for ch in "*?[")


def _resolve_context_paths(
    step: DockerfileStep,
    instruction: str,
    source: str,
    context: Path,
) -> list[Path]:
    if not source or source.startswith("/"):
        raise SandboxRefusal(
            f"line {step.lineno}: {instruction}: expected a relative context path, got {source!r}"
        )
    if _is_glob(source):
        matches = sorted(p for p in context.glob(source) if not p.is_symlink())
        checked: list[Path] = []
        for match in matches:
            checked.append(_contain(step, instruction, match, context))
        return checked
    candidate = context / source
    return [_contain(step, instruction, candidate, context)]


def _contain(step: DockerfileStep, instruction: str, path: Path, context: Path) -> Path:
    resolved_context = context.resolve()
    resolved = path.resolve()
    if resolved != resolved_context and resolved_context not in resolved.parents:
        raise SandboxRefusal(
            f"line {step.lineno}: {instruction}: expected a path inside {resolved_context}, "
            f"got {path} resolving to {resolved}"
        )
    return resolved
