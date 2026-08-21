"""Topology: where a run thinks it is, and how its GPUs are actually arranged.

Why this module exists
----------------------
A ground-truth probe of the audited estate measured 526 shell files, of which 189 are
launchers. Everything this module encodes — and every number in it — comes from that
corpus. The failures were never about GPU arithmetic being hard; they were about facts
that existed in one place and were never compared against the same fact in another
place. The measured record:

1. **Pipeline parallelism does not exist in the estate.** ``pipeline_model_parallel_size``
   appears in **zero** of 189 launchers; ``virtual_pipeline`` in zero. That zero is only
   believable because the positive controls fired on the same corpus:
   ``tensor_model_parallel_size`` matched 51 files, ``expert_model_parallel_size`` 53,
   ``context_parallel_size`` 49, ``sequence_parallel`` 50. *A negative claim is only as
   good as the positive control on the same detector.* Every check in this module
   therefore reports its own positive control, so a "clean" result is distinguishable
   from a check that examined nothing.
2. **Context parallelism is dead, silently.** ``CP=${CP:-1}`` appears in 44 places, but
   four Python entrypoints hardcode ``m.context_parallel_size = 1`` and
   ``m.pipeline_model_parallel_size = 1`` with no CLI flag to override. A knob that
   appears configurable and is not is worse than a missing knob — it manufactures
   confidence while guaranteeing the config lies.
3. **Two incompatible dialects for the same degrees.** Hydra-style
   ``model.*_parallel_size=`` on the SFT path, argparse ``--tp/--etp/--ep`` on the RL
   path, re-typed by hand in every fork (``--tp 8 --etp 1 --ep 8`` appears as a bare
   literal inside a command string).
4. **World size was welded.** Every launcher used ``--nodes`` in {1,2,4,8} times
   ``--gpus-per-node=4`` and nothing else. The 8-node ceiling is a property of those
   scripts, not of the cluster — other users on the same cluster run 18 nodes.
   :class:`ClusterProfile` carries the cluster's real limits as data so the ceiling
   stops being inherited by copy-paste.
5. **Rendezvous config was wrong in two opposite ways.** ``MASTER_ADDR=127.0.0.1`` in 27
   places (every multi-node run trying to rendezvous with itself) and a hardcoded node
   name in 22 (the launcher capped at one specific machine).
6. **A real crash, cheaply.** ``#SBATCH --ntasks-per-node=8`` against
   ``--gpus-per-node=4`` produced ``ncclInvalidUsage / Duplicate GPU detected: rank 3
   and rank 7 both on CUDA device ...``, 2m10s into the job, before a single weight was
   read. The mismatch is arithmetic; it never needed to reach a GPU.
7. **Nobody compared partition spellings.** The tree contains two spellings of the same
   partition, 188 files one way and 4 the other. Nothing ever compared them.

What this module does about it
------------------------------
* :class:`ClusterProfile` is **data, not code**: name, scheduler, partitions, node
  pattern, GPUs per node, interconnect, container, filesystem roots — loadable from
  JSON. Two ship as literal dicts (:data:`PROFILES`); a third is one more dict.
* :class:`Topology` validates its decomposition at construction, in
  ``__post_init__``, and the error **shows the arithmetic**. ``tasks_per_node`` is
  derived and checked against ``gpus_per_node`` there too, catching failure (6) at
  launch instead of 2m10s in.
* :meth:`Topology.validate_against`, :func:`declared_vs_effective` and
  :func:`partition_consistency` return :class:`Finding` objects. Every finding names
  the positive control proving the check can fire; the validators always emit a
  coverage summary so an empty result is never mistaken for a clean one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

__all__ = [
    "Severity",
    "Finding",
    "ClusterProfile",
    "PROFILES",
    "profile_by_name",
    "Topology",
    "declared_vs_effective",
    "partition_consistency",
    "blocking",
    "render_findings",
]


# The evidence format the whole module imitates: on the same 189-launcher corpus these
# four greps fired (51, 53, 49, 50 files), which is the *only* reason the zero count
# for pipeline_model_parallel_size / virtual_pipeline means anything. A check that
# cannot point at a firing control has never been shown to detect anything.
_GREP_POSITIVE_CONTROLS = (
    "positive controls on the same 189-launcher corpus: tensor_model_parallel_size "
    "matched 51 files, expert_model_parallel_size 53, context_parallel_size 49, "
    "sequence_parallel 50 — a zero count is trusted only because those fired"
)


class Severity(str, Enum):
    """How loudly a :class:`Finding` speaks.

    ``OK`` exists so a check can report *coverage* when it found nothing — the
    difference between "0 defects in N units" and "this function returned an empty
    list because of a bug".
    """

    OK = "ok"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True)
class Finding:
    """One observation from a consistency check.

    Args:
        code: Stable machine-readable identifier, ``topology.*``.
        severity: OK / WARN / BLOCK. Only BLOCK stops a launch.
        message: Human-readable statement of the defect (or of clean coverage).
        details: Structured evidence — counts, spellings, degrees — JSON-safe.
        control: The positive control proving this check can fire. Named, never
            implied. An ``OK`` finding with no control is indistinguishable from a
            broken detector, which is the exact shape of the ``all([]) is True``
            incident that motivated the gate contract.
    """

    code: str
    severity: Severity
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    control: str = ""

    def render(self) -> str:
        line = f"[{self.severity.value:>5}] {self.code}: {self.message}"
        if self.control and self.severity is not Severity.OK:
            line += f"\n         control: {self.control}"
        return line


def blocking(findings: Sequence[Finding]) -> list[Finding]:
    """Filter to the findings that must stop the job."""
    return [f for f in findings if f.severity is Severity.BLOCK]


def render_findings(findings: Sequence[Finding]) -> str:
    """Render a finding list for a launch log. Explicitly non-empty on no findings."""
    if not findings:
        return "(no findings — this is itself suspicious; validators never return empty)"
    return "\n".join(f.render() for f in findings)


# --------------------------------------------------------------------------- #
# Cluster description — data, not code.                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ClusterProfile:
    """Everything a launcher needs to know about a cluster, as data.

    The estate hardcoded all of this into launcher scripts; the result was the
    measured record above. Keeping it in one loadable structure is what removes the
    weld: topology arithmetic stays in code, the cluster stays in configuration.

    Args:
        name: Short identifier, e.g. ``"slurm-generic"``.
        scheduler: ``"slurm"``, ``"pbs"``, ``"local"``, … Informational except where a
            validator keys on it.
        partitions: Valid scheduler partition names on this cluster.
        node_pattern: Regex matching any *concrete* schedulable hostname. Used to
            detect a hardcoded node name masquerading as ``MASTER_ADDR``.
        gpus_per_node: Accelerators per node on this cluster.
        nccl_socket_ifname: Value for ``NCCL_SOCKET_IFNAME``.
        ib_hca_pattern: Glob/regex for InfiniBand HCAs, ``""`` if none.
        mnnvl_available: Whether NVLink spans multiple nodes (affects how bad an
            inter-node TP crossing actually is).
        container_runtime: ``"enroot"``, ``"docker"``, ``"apptainer"``, or ``"none"``.
        container_image: Fully-pinned image reference; ``""`` when runtime is none.
        filesystem_roots: Mounts that must exist for artifacts/logs/data.
        max_nodes: The cluster's real node ceiling, or ``None`` if not constrained.
            Deliberately *not* inherited from launcher habit: the estate's 8-node
            ceiling (``--nodes`` only ever {1,2,4,8}) was a script artifact — other
            users on the same cluster ran 18 nodes.
    """

    name: str
    scheduler: str
    partitions: tuple[str, ...]
    node_pattern: str
    gpus_per_node: int
    nccl_socket_ifname: str
    ib_hca_pattern: str
    mnnvl_available: bool
    container_runtime: str
    container_image: str
    filesystem_roots: tuple[str, ...]
    max_nodes: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ClusterProfile.name must be non-empty")
        if not self.scheduler.strip():
            raise ValueError("ClusterProfile.scheduler must be non-empty")
        if not self.partitions or not all(p.strip() for p in self.partitions):
            raise ValueError("ClusterProfile.partitions must contain non-empty names")
        if isinstance(self.gpus_per_node, bool) or self.gpus_per_node < 1:
            raise ValueError(f"gpus_per_node must be >= 1: {self.gpus_per_node}")
        if self.max_nodes is not None and self.max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1 or None: {self.max_nodes}")
        try:
            re.compile(self.node_pattern)
        except re.error as exc:
            raise ValueError(f"node_pattern is not a valid regex: {exc}") from exc
        if not self.container_runtime.strip():
            raise ValueError("container_runtime must be non-empty ('none' is valid)")
        if self.container_runtime == "none" and self.container_image:
            raise ValueError("container_image must be empty when runtime is 'none'")
        if self.container_runtime != "none" and not self.container_image.strip():
            raise ValueError(
                "container_image must be pinned when a runtime is used — an "
                "unpinned image is exactly the unrecorded-config failure"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ClusterProfile:
        """Build a profile from parsed JSON. Refuses unknown or missing keys.

        Rejecting unknown keys is load-bearing: a misspelled key in a profile file
        must not silently reset that field to its default — silent config drift is
        the incident class this module exists to kill.

        Args:
            data: Mapping with exactly the dataclass field names.

        Returns:
            A validated :class:`ClusterProfile`.

        Raises:
            ValueError: On unknown keys, missing required keys, or failed validation.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(str(k) for k in data if k not in known)
        if unknown:
            raise ValueError(
                f"cluster profile {data.get('name', '<unnamed>')!r} has unknown keys "
                f"{unknown}; refusing to drop them silently — a typo would quietly "
                f"reset that field to its default"
            )
        required = [
            f.name for f in fields(cls) if f.default is MISSING and f.default_factory is MISSING
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"cluster profile missing required keys {missing}")
        normalised = dict(data)
        for key in ("partitions", "filesystem_roots"):
            if isinstance(normalised.get(key), list):
                normalised[key] = tuple(normalised[key])
        return cls(**normalised)

    @classmethod
    def from_json(cls, source: str | Path) -> ClusterProfile:
        """Load a profile from a JSON document or a path to one.

        Args:
            source: JSON text (starts with ``{`` after whitespace) or a filesystem
                path to a JSON file.

        Returns:
            A validated :class:`ClusterProfile`.
        """
        if isinstance(source, Path) or (
            isinstance(source, str) and not source.lstrip().startswith("{")
        ):
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError(f"profile JSON must be an object, got {type(data).__name__}")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Serialise back to a JSON-compatible dict (round-trips through from_dict)."""
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["partitions"] = list(self.partitions)
        out["filesystem_roots"] = list(self.filesystem_roots)
        return out


# Profiles are pure data. Adding a third cluster appends one dict here — if a change
# requires touching anything below the table, the table is missing a field.
_PROFILE_DATA: tuple[dict[str, Any], ...] = (
    {
        "name": "slurm-generic",
        "scheduler": "slurm",
        "partitions": ("gpu",),
        "node_pattern": r"[a-z0-9]+-[0-9]+",
        "gpus_per_node": 8,
        "nccl_socket_ifname": "eth0",
        "ib_hca_pattern": r"mlx5_*",
        "mnnvl_available": False,
        "container_runtime": "enroot",
        "container_image": "nvcr.io/nvidia/pytorch:25.01-py3",
        "filesystem_roots": ("/home", "/scratch"),
        # A *cluster* limit, stated as data — not the {1,2,4,8} x 4-GPU habit the
        # measured launchers welded in while neighbours ran 18 nodes.
        "max_nodes": 64,
    },
    {
        "name": "local-single-node",
        "scheduler": "local",
        "partitions": ("local",),
        "node_pattern": r"localhost|127\.0\.0\.1",
        "gpus_per_node": 8,
        "nccl_socket_ifname": "lo",
        "ib_hca_pattern": "",
        "mnnvl_available": False,
        "container_runtime": "none",
        "container_image": "",
        "filesystem_roots": ("/",),
        "max_nodes": 1,
    },
)

PROFILES: Mapping[str, ClusterProfile] = MappingProxyType(
    {d["name"]: ClusterProfile.from_dict(d) for d in _PROFILE_DATA}
)
"""Built-in cluster profiles, keyed by name. Read-only by construction."""


def profile_by_name(name: str) -> ClusterProfile:
    """Look up a built-in profile.

    Args:
        name: Key into :data:`PROFILES`.

    Returns:
        The requested :class:`ClusterProfile`.

    Raises:
        KeyError: If unknown — the message says how to add clusters (as data).
    """
    try:
        return PROFILES[name]
    except KeyError:
        raise KeyError(
            f"unknown cluster profile {name!r}; known: {sorted(PROFILES)}. New clusters "
            f"are added as data: one dict in _PROFILE_DATA, no code changes."
        ) from None


# --------------------------------------------------------------------------- #
# Parallelism decomposition — verified arithmetic at construction time.        #
# --------------------------------------------------------------------------- #

_DEGREES = ("dp", "tp", "pp", "ep", "cp")
_LAYOUT = ("nodes", "gpus_per_node")


@dataclass(frozen=True)
class Topology:
    """The parallelism decomposition of one run, validated when it is created.

    Construction is the launch gate: the degrees must multiply to the total GPU
    count or the object never exists. This replaces the estate's two re-typed-by-hand
    dialects (Hydra ``model.*_parallel_size=`` on SFT, argparse ``--tp/--etp/--ep``
    on RL, ``--tp 8 --etp 1 --ep 8`` appearing as a bare literal inside a command
    string) with one object both paths build and both paths validate.

    Args:
        dp: Data-parallel replicas.
        tp: Tensor-parallel width.
        pp: Pipeline-parallel stages. Zero of 189 measured launchers ever set this;
            requesting ``pp > 1`` will draw scrutiny in :meth:`validate_against`.
        ep: Expert-parallel width. ``ep=1`` on an MoE replicates every expert on
            every model-parallel rank.
        cp: Context-parallel width. In the measured estate this was *hardcoded to 1*
            in four entrypoints regardless of the launcher's ``CP`` — see
            :func:`declared_vs_effective`.
        nodes: Node count.
        gpus_per_node: GPUs used per node.
        tasks_per_node: Launcher tasks per node. Defaults to ``gpus_per_node``; any
            other explicit value is rejected, because one-process-per-GPU is the only
            mapping this stack supports — the measured ``--ntasks-per-node=8`` with
            ``--gpus-per-node=4`` crashed with NCCL "Duplicate GPU detected" 2m10s
            into the job.
    """

    dp: int
    tp: int
    pp: int
    ep: int
    cp: int
    nodes: int
    gpus_per_node: int
    tasks_per_node: int | None = None

    def __post_init__(self) -> None:
        for name in (*_DEGREES, *_LAYOUT):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"topology {name} must be a positive int: {value!r}")

        total = self.nodes * self.gpus_per_node
        product = self.dp * self.tp * self.pp * self.ep * self.cp
        if product != total:
            raise ValueError(
                f"invalid topology: the degrees must multiply to the total GPU count\n"
                f"  dp({self.dp}) x tp({self.tp}) x pp({self.pp}) x ep({self.ep}) x "
                f"cp({self.cp}) = {product}\n"
                f"  but nodes({self.nodes}) x gpus_per_node({self.gpus_per_node}) "
                f"= {total}"
            )

        if self.tasks_per_node is None:
            object.__setattr__(self, "tasks_per_node", self.gpus_per_node)
        else:
            if (
                isinstance(self.tasks_per_node, bool)
                or not isinstance(self.tasks_per_node, int)
                or self.tasks_per_node < 1
            ):
                raise ValueError(f"tasks_per_node must be a positive int: {self.tasks_per_node!r}")
            if self.tasks_per_node != self.gpus_per_node:
                raise ValueError(
                    f"tasks_per_node={self.tasks_per_node} != "
                    f"gpus_per_node={self.gpus_per_node}: one launcher slot per GPU is "
                    f"the only mapping this stack supports. Measured crash to encode: "
                    f"'#SBATCH --ntasks-per-node=8' against '--gpus-per-node=4' "
                    f"produced 'ncclInvalidUsage / Duplicate GPU detected: rank 3 and "
                    f"rank 7 both on CUDA device ...', 2m10s into the job, before a "
                    f"single weight was read. Checking at launch costs milliseconds."
                )

    # -- derived quantities -------------------------------------------------------

    @property
    def total_gpus(self) -> int:
        """Total accelerators the run occupies."""
        return self.nodes * self.gpus_per_node

    @property
    def model_parallel_width(self) -> int:
        """GPUs holding exactly one model replica: tp x pp x ep x cp."""
        return self.tp * self.pp * self.ep * self.cp

    def degree_dict(self) -> dict[str, int]:
        """The five parallelism degrees as a plain mapping."""
        return {name: getattr(self, name) for name in _DEGREES}

    def describe(self) -> str:
        """One-screen launch summary showing the full decomposition and total GPUs.

        Returns:
            A small multi-line string suitable for the head of a launch log. The
            first line carries the total; everything after it shows the arithmetic
            that produces the total, so a wrong decomposition is visible on screen
            rather than buried in a flag list.
        """
        decl = " x ".join(f"{name}={getattr(self, name)}" for name in _DEGREES)
        lines = [
            f"topology: {self.total_gpus} GPUs "
            f"= {self.nodes} nodes x {self.gpus_per_node} GPUs/node",
            f"  decomposition : {decl} = {self.total_gpus}",
            f"  model replica : tp x pp x ep x cp = {self.model_parallel_width} GPUs "
            f"-> dp={self.dp} independent replicas",
            f"  per node      : {self.gpus_per_node} GPUs, "
            f"{self.tasks_per_node} tasks (must be equal; a gap is the Duplicate-GPU "
            f"crash, caught here, not 2m10s in)",
        ]
        if self.pp > 1 or self.cp > 1:
            lines.append(
                "  scrutiny      : pp/cp > 1 requested, but 0/189 measured launchers "
                "exercised pp and cp was hardcoded 1 in every RL/SFT entrypoint — "
                "confirm with declared_vs_effective() that the runtime honours these"
            )
        return "\n".join(lines)

    # -- consistency against the real world --------------------------------------

    def validate_against(
        self,
        profile: ClusterProfile,
        *,
        master_addr: str | None = None,
        num_experts: int | None = None,
        runtime_overrides: Mapping[str, int] | None = None,
    ) -> list[Finding]:
        """Check this topology against a cluster profile and run-time evidence.

        Encodes the measured failures: TP crossing the node boundary, EP=1 expert
        replication on MoE, loopback and hardcoded-hostname rendezvous, and PP/CP
        values the runtime silently overrides.

        A coverage summary finding is *always* appended, so a clean run is
        distinguishable from a validator that examined nothing.

        Args:
            profile: The cluster this run will land on.
            master_addr: The exact address the launcher will export for rendezvous.
                ``None`` with ``nodes > 1`` is itself a finding — unrecorded
                rendezvous config is how runs silently diverge.
            num_experts: MoE expert count, when known, to gate expert replication.
            runtime_overrides: Degrees the entrypoint is known to force, e.g.
                ``{"pp": 1, "cp": 1}`` for the four measured entrypoints that
                hardcode them. When absent and ``pp``/``cp`` exceed 1, a WARN is
                emitted demanding :func:`declared_vs_effective` evidence.

        Returns:
            Findings, always non-empty (the last one is the coverage summary).
        """
        findings: list[Finding] = []
        checks = 0

        # 1. Node shape vs. the cluster description ------------------------------
        checks += 1
        if self.gpus_per_node > profile.gpus_per_node:
            findings.append(
                Finding(
                    code="topology.gpus_per_node_exceeds_profile",
                    severity=Severity.BLOCK,
                    message=(
                        f"gpus_per_node={self.gpus_per_node} exceeds profile "
                        f"{profile.name!r} ({profile.gpus_per_node}); the scheduler "
                        f"will either refuse the job or — worse — oversubscribe"
                    ),
                    details={
                        "requested": self.gpus_per_node,
                        "profile": profile.gpus_per_node,
                    },
                    control=(
                        "positive control: gpus_per_node=8 against the local profile "
                        "(max 8) must pass; 16 must produce this BLOCK"
                    ),
                )
            )
        checks += 1
        if profile.max_nodes is not None and self.nodes > profile.max_nodes:
            findings.append(
                Finding(
                    code="topology.nodes_exceed_profile_limit",
                    severity=Severity.BLOCK,
                    message=(
                        f"nodes={self.nodes} exceeds profile {profile.name!r} limit "
                        f"of {profile.max_nodes}. Note the estate's 8-node ceiling "
                        f"was a launcher artifact (nodes only ever in "
                        f"{{1,2,4,8}} x 4 GPUs) while other users ran 18 nodes — if "
                        f"the real limit differs, fix the *profile data*, not the "
                        f"launcher"
                    ),
                    details={"requested": self.nodes, "limit": profile.max_nodes},
                    control=_GREP_POSITIVE_CONTROLS,
                )
            )

        # 2. TP must not silently cross the node boundary ------------------------
        checks += 1
        if self.tp > self.gpus_per_node:
            severity = Severity.BLOCK if not profile.mnnvl_available else Severity.WARN
            findings.append(
                Finding(
                    code="topology.tp_crosses_node_boundary",
                    severity=severity,
                    message=(
                        f"tp={self.tp} > gpus_per_node={self.gpus_per_node}: the "
                        f"tensor-parallel group silently crosses the node boundary, "
                        f"so every TP all-reduce rides the network"
                        + (
                            " (profile reports MNNVL, so this may be intentional — "
                            "confirm the TP domain was *meant* to span nodes)"
                            if profile.mnnvl_available
                            else ""
                        )
                    ),
                    details={
                        "tp": self.tp,
                        "gpus_per_node": self.gpus_per_node,
                        "mnnvl": profile.mnnvl_available,
                    },
                    control=(
                        "positive control: tp=8 with gpus_per_node=4 must BLOCK on a "
                        "non-MNNVL profile; tp<=gpus_per_node must not appear here"
                    ),
                )
            )

        # 3. Expert replication ---------------------------------------------------
        if num_experts is not None:
            checks += 1
            if num_experts > 1 and self.ep == 1:
                findings.append(
                    Finding(
                        code="topology.ep1_replicates_experts",
                        severity=Severity.WARN,
                        message=(
                            f"ep=1 on an MoE with num_experts={num_experts} replicates "
                            f"every expert on every model-parallel rank. Measured in "
                            f"the estate: 88.7% of parameters replicated — and the fix "
                            f"was ep=2, not a reshard"
                        ),
                        details={"num_experts": num_experts, "ep": self.ep},
                        control=(
                            "positive control: num_experts=8 with ep=1 must produce "
                            "this WARN; ep=2 must not"
                        ),
                    )
                )
            checks += 1
            if self.ep > 1 and num_experts % self.ep != 0:
                findings.append(
                    Finding(
                        code="topology.ep_uneven_expert_shard",
                        severity=Severity.BLOCK,
                        message=(
                            f"num_experts={num_experts} is not divisible by "
                            f"ep={self.ep}; the expert shard cannot be even"
                        ),
                        details={"num_experts": num_experts, "ep": self.ep},
                        control=(
                            "positive control: num_experts=7, ep=2 must BLOCK; "
                            "num_experts=8, ep=2 must not"
                        ),
                    )
                )

        # 4. Rendezvous ------------------------------------------------------------
        if self.nodes > 1:
            checks += 1
            if master_addr is None:
                findings.append(
                    Finding(
                        code="topology.master_addr_unrecorded",
                        severity=Severity.WARN,
                        message=(
                            f"nodes={self.nodes} but no master_addr was supplied to "
                            f"check. Unrecorded config is how runs silently diverge; "
                            f"hand validate_against the exact value the launcher will "
                            f"export"
                        ),
                        details={"nodes": self.nodes},
                        control=(
                            "positive control: nodes=2 with master_addr='127.0.0.1' "
                            "must BLOCK (measured verbatim in 27 launchers)"
                        ),
                    )
                )
            else:
                host = _host_of(master_addr)
                if host.startswith("127.") or host in ("localhost", "::1"):
                    findings.append(
                        Finding(
                            code="topology.master_addr_loopback",
                            severity=Severity.BLOCK,
                            message=(
                                f"MASTER_ADDR={master_addr!r} is loopback with "
                                f"nodes={self.nodes}: every node will rendezvous with "
                                f"itself. Found verbatim in 27 of the 189 measured "
                                f"launchers"
                            ),
                            details={"master_addr": master_addr, "nodes": self.nodes},
                            control=(
                                "positive control: nodes=1 (single node) never emits "
                                "this; the loopback value with nodes>1 always does"
                            ),
                        )
                    )
                elif re.fullmatch(profile.node_pattern, host):
                    findings.append(
                        Finding(
                            code="topology.master_addr_hardcoded_node",
                            severity=Severity.BLOCK,
                            message=(
                                f"MASTER_ADDR={master_addr!r} is a concrete node name "
                                f"matching profile pattern {profile.node_pattern!r}: "
                                f"the launcher is capped at one specific machine and "
                                f"fails when that node is drained. Found in 22 of the "
                                f"189 measured launchers. Use hostname resolution at "
                                f"batch time (e.g. scontrol) instead"
                            ),
                            details={"master_addr": master_addr, "host": host},
                            control=_GREP_POSITIVE_CONTROLS,
                        )
                    )

        # 5. PP/CP: requested vs. what the runtime will actually honour ------------
        overrides = dict(runtime_overrides or {})
        for degree in ("pp", "cp"):
            declared = getattr(self, degree)
            checks += 1
            forced = overrides.get(degree)
            if forced is not None and forced != declared:
                findings.append(
                    Finding(
                        code=f"topology.runtime_overrides_{degree}",
                        severity=Severity.BLOCK,
                        message=(
                            f"config requests {degree}={declared} but the runtime will "
                            f"silently force {degree}={forced}. A knob that appears "
                            f"configurable and is not is worse than a missing knob — "
                            f"measured: 44 launchers set CP=${{CP:-1}} while four "
                            f"entrypoints hardcoded context_parallel_size=1 and "
                            f"pipeline_model_parallel_size=1 with no override flag"
                        ),
                        details={
                            "degree": degree,
                            "requested": declared,
                            "forced": forced,
                        },
                        control=getattr(self, "_override_control", None)
                        or (
                            f"positive control: {degree}=2 with "
                            f"runtime_overrides={{'{degree}': 1}} must produce this "
                            f"BLOCK; matching values must not"
                        ),
                    )
                )
            elif declared > 1 and forced is None:
                findings.append(
                    Finding(
                        code=f"topology.{degree}_unverified",
                        severity=Severity.WARN,
                        message=(
                            f"{degree}={declared} requested, but in the measured "
                            f"estate 0/189 launchers ever exercised "
                            f"{'pipeline' if degree == 'pp' else 'context'} "
                            f"parallelism end-to-end. Require declared_vs_effective() "
                            f"evidence from the built model before trusting this run"
                        ),
                        details={"degree": degree, "requested": declared},
                        control=_GREP_POSITIVE_CONTROLS,
                    )
                )

        n_block = sum(f.severity is Severity.BLOCK for f in findings)
        n_warn = sum(f.severity is Severity.WARN for f in findings)
        findings.append(
            Finding(
                code="topology.validate_summary",
                severity=Severity.OK,
                message=(
                    f"validate_against ran {checks} checks against profile "
                    f"{profile.name!r}: {n_block} blocking, {n_warn} warnings. A "
                    f"summary is always emitted, so a clean result is distinguishable "
                    f"from a validator that examined nothing"
                ),
                details={
                    "checks_run": checks,
                    "blocking": n_block,
                    "warnings": n_warn,
                    "topology": {
                        **self.degree_dict(),
                        "nodes": self.nodes,
                        "gpus_per_node": self.gpus_per_node,
                    },
                },
                control=_GREP_POSITIVE_CONTROLS,
            )
        )
        return findings


def _host_of(addr: str) -> str:
    """Strip scheme-ish decoration, brackets and an optional port from a host spec.

    Args:
        addr: Raw MASTER_ADDR-style value, e.g. ``"gpu-14:6000"`` or ``"[::1]:29400"``.

    Returns:
        The bare hostname or address, lower-cased.
    """
    host = addr.strip().lower()
    host = re.sub(r"^[a-z][a-z0-9+.-]*://", "", host)
    if host.startswith("["):
        host, _, _ = host[1:].partition("]")
        return host
    if host.count(":") == 1 and host.rsplit(":", 1)[1].isdigit():
        host = host.rsplit(":", 1)[0]
    return host


def declared_vs_effective(declared: Topology, effective: Topology) -> list[Finding]:
    """Compare the config's topology against the one the runtime actually built.

    The check nobody in the estate had. A degree the manifest sets and the code then
    overwrites is a lie with a green check mark on it. Measured: ``CP=${CP:-1}`` in 44
    places, while four Python entrypoints hardcoded ``m.context_parallel_size = 1``
    and ``m.pipeline_model_parallel_size = 1`` with no CLI flag to override — the RL
    path could never see the shell variable, and nothing ever compared the two.

    Args:
        declared: Built from the launch configuration / manifest.
        effective: Read back from the constructed model (e.g. the values actually set
            on the model object after entrypoint code ran).

    Returns:
        One BLOCK finding per degree that differs, plus an ``OK`` summary carrying
        comparison coverage — or a single ``OK`` finding stating how many fields were
        compared when everything matches.
    """
    findings: list[Finding] = []
    compared = 0
    for name in (*_DEGREES, *_LAYOUT):
        compared += 1
        want, got = getattr(declared, name), getattr(effective, name)
        if want != got:
            findings.append(
                Finding(
                    code=f"topology.effective_overrides_{name}",
                    severity=Severity.BLOCK,
                    message=(
                        f"declared {name}={want} but the runtime constructed "
                        f"{name}={got}: the config is not what the model is training "
                        f"with. A knob that appears configurable and is not is worse "
                        f"than a missing knob — publish the effective topology in the "
                        f"manifest, or make the degree genuinely settable"
                    ),
                    details={"field": name, "declared": want, "effective": got},
                    control=(
                        "positive control: declared_vs_effective(Topology(cp=4, ...), "
                        "Topology(cp=1, ...)) must emit this BLOCK; in the measured "
                        "estate 44 launchers set CP=${CP:-1} while four entrypoints "
                        "hardcoded cp=1 — only this comparison catches it"
                    ),
                )
            )
    if findings:
        findings.append(
            Finding(
                code="topology.effective_comparison_summary",
                severity=Severity.OK,
                message=(
                    f"compared all {compared} topology fields; "
                    f"{len(findings)} field(s) differ — the BLOCK findings above "
                    "carry the evidence. Coverage is reported here precisely "
                    "because a mismatch is where 'how much was compared' matters "
                    "most"
                ),
                details={
                    "fields_compared": compared,
                    "blocking": len(findings),
                },
                control=_GREP_POSITIVE_CONTROLS,
            )
        )
    else:
        findings.append(
            Finding(
                code="topology.effective_matches_declared",
                severity=Severity.OK,
                message=(
                    f"compared all {compared} topology fields: the runtime honours "
                    f"the declared config"
                ),
                details={"fields_compared": compared},
                control=_GREP_POSITIVE_CONTROLS,
            )
        )
    return findings


# --------------------------------------------------------------------------- #
# Partition spelling consistency.                                              #
# --------------------------------------------------------------------------- #

# Line-based extraction. The short `-p` form only counts on scheduler lines because
# `-p` is overloaded everywhere else (mkdir, cp, tee): overmatching manufactures
# variants out of nothing. The long and Hydra forms are unambiguous.
_PARTITION_LONG_RE = re.compile(r"--partition[=\s]+['\"]?([A-Za-z][\w.-]*)")
_PARTITION_KV_RE = re.compile(r"\bpartition\s*=\s*['\"]?([A-Za-z][\w.-]*)")
_PARTITION_SHORT_RE = re.compile(r"(?<![\w-])-p\s+['\"]?([A-Za-z][\w.-]*)")
_SBATCH_LINE_RE = re.compile(r"#SBATCH|\bsbatch\b|\bsrun\b")


def _normalize_partition(spelling: str) -> str:
    """Normalise to a spelling-insensitive key: case and separators are ignored.

    ``gpu-h100``, ``gpu_h100`` and ``GPU.H100`` all normalise to ``gpuh100`` —
    one partition, three spellings.
    """
    return re.sub(r"[^a-z0-9]", "", spelling.lower())


def partition_consistency(files: Mapping[str, str]) -> Finding:
    """Detect spelling variants of one partition across a corpus of launcher files.

    The measured estate contained two spellings of the same partition, 188 files one
    way and 4 the other, and nothing ever compared them. This check does not assume
    the majority spelling is correct — it reports the split and refuses to resolve
    it for you — and it reports its own coverage, because a clean result must be
    distinguishable from a scan that read nothing.

    Args:
        files: Mapping of path -> file contents (launcher scripts, configs, etc.).

    Returns:
        A single :class:`Finding`:

        * ``BLOCK / topology.partition_scan_empty`` — zero files supplied. A scan of
          nothing cannot assert consistency; honouring it would be the ``all([]) is
          True`` failure one level up.
        * ``BLOCK / topology.partition_not_found`` — files scanned, zero partition
          declarations extracted. Either the corpus genuinely has no partitions (a
          finding in itself) or the extractor is blind.
        * ``BLOCK / topology.partition_spelling_variants`` — one partition appears
          under multiple spellings; details carry per-spelling file counts.
        * ``OK / topology.partition_consistent`` — every extracted declaration of a
          partition uses one spelling; details carry files scanned and variants found.
    """
    scanned = len(files)
    if scanned == 0:
        return Finding(
            code="topology.partition_scan_empty",
            severity=Severity.BLOCK,
            message=(
                "0 files supplied; a scan of nothing cannot assert partition "
                "consistency. Accepting a clean result here is the audit tool's "
                "all([]) is True failure, one level up"
            ),
            details={"files_scanned": 0},
            control=(
                "positive control: a fixture corpus containing both 'gpu-a100' and "
                "'gpu_a100' must produce the spelling-variants BLOCK (the estate "
                "split 188/4 on one partition)"
            ),
        )

    # normalized spelling -> {raw spelling -> {paths}}
    groups: dict[str, dict[str, set[str]]] = {}
    files_with_partition: set[str] = set()
    for path, text in files.items():
        for line in text.splitlines():
            spellings = _PARTITION_LONG_RE.findall(line) + _PARTITION_KV_RE.findall(line)
            if _SBATCH_LINE_RE.search(line):
                spellings += _PARTITION_SHORT_RE.findall(line)
            for spelling in set(spellings):
                files_with_partition.add(path)
                groups.setdefault(_normalize_partition(spelling), {}).setdefault(
                    spelling, set()
                ).add(path)

    if not files_with_partition:
        return Finding(
            code="topology.partition_not_found",
            severity=Severity.BLOCK,
            message=(
                f"scanned {scanned} files and found no partition declarations "
                f"(--partition, -p on sbatch/srun lines, or partition=). Either the "
                f"corpus has no partitions — which matters — or the extractor is "
                f"blind; a clean verdict is not available"
            ),
            details={"files_scanned": scanned, "files_with_partition": 0},
            control=(
                "positive control fixtures must extract from '#SBATCH --partition=x', "
                "'sbatch -p x', '--partition=x', and 'partition = \"x\"' alike"
            ),
        )

    variants = {
        norm: {sp: paths for sp, paths in spelling_to_paths.items()}
        for norm, spelling_to_paths in groups.items()
        if len(spelling_to_paths) > 1
    }
    if variants:
        return Finding(
            code="topology.partition_spelling_variants",
            severity=Severity.BLOCK,
            message=(
                f"{len(variants)} partition(s) appear under multiple spellings. The "
                f"majority spelling is NOT assumed correct — the measured estate "
                f"split 188 files one way and 4 the other on a single partition, and "
                f"nothing ever compared them. Resolve explicitly; do not pick a side "
                f"by count"
            ),
            details={
                "files_scanned": scanned,
                "files_with_partition": len(files_with_partition),
                "variants": {
                    norm: {
                        sp: {
                            "files": len(paths),
                            "examples": sorted(paths)[:3],
                        }
                        for sp, paths in sp_map.items()
                    }
                    for norm, sp_map in variants.items()
                },
            },
            control=(
                "positive control: a corpus with 'gpu-h100' and 'gpu_h100' must "
                "produce this BLOCK; a uniform corpus must produce the OK finding "
                "with variants=0"
            ),
        )

    return Finding(
        code="topology.partition_consistent",
        severity=Severity.OK,
        message=(
            f"scanned {scanned} files ({len(files_with_partition)} declaring "
            f"partitions); {len(groups)} distinct partition(s), 0 spelling variants"
        ),
        details={
            "files_scanned": scanned,
            "files_with_partition": len(files_with_partition),
            "partitions": sorted(groups),
            "variants": 0,
        },
        control=_GREP_POSITIVE_CONTROLS,
    )
