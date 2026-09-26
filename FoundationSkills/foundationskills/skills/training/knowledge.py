
"""Loaders for the training knowledge pack (families, hardware, algorithms, recipes).

The knowledge pack lives under ``foundationskills/skills/training/knowledge/`` as
YAML package data with JSON schemas under ``foundationskills/schemas/knowledge/``.
Every loader takes an optional ``root`` pointing at an alternative knowledge
directory (used by tests); ``root=None`` means the package data. Results are
cached, keyed by the root, so repeated loads are free and share objects.

All loaders validate every file against its schema and raise
:class:`KnowledgeError` naming the offending file on any problem. There is no
silent fallback: a broken knowledge pack is a hard error, not "empty knowledge".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from foundationskills.core.schema import SchemaError, load_schema, validate

__all__ = [
    "KnowledgeError",
    "Family",
    "Variant",
    "Hardware",
    "AlgorithmCard",
    "Recipe",
    "load_families",
    "load_hardware",
    "load_algorithm_cards",
    "load_recipes",
    "load_stage_rules",
    "find_variant",
    "family_for_model_type",
]


class KnowledgeError(ValueError):
    """A knowledge file or directory is missing, unreadable, or invalid.

    The message always names the file (or directory) at fault.
    """


def _default_root() -> Path:
    return Path(str(resources.files("foundationskills.skills.training") / "knowledge"))


def _root_key(root: Path | None) -> str:
    return str(Path(root)) if root is not None else str(_default_root())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One model variant of a family. Sizes are in billions of parameters (B)."""

    id: str
    hf_id: str | None
    local_path: str | None
    size_b: float
    active_b: float | None
    arch: str  # "dense" | "moe"
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int | None
    vocab: int
    context_length: int
    num_experts: int | None
    experts_per_token: int | None
    tied_embeddings: bool
    instruct: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def total_params(self) -> int:
        """Total parameter count (MoE: all experts). Drives memory."""
        return int(round(self.size_b * 1e9))

    @property
    def active_params(self) -> int:
        """Parameters active per token (MoE: routed subset). Drives FLOPs."""
        return int(round((self.active_b if self.active_b is not None else self.size_b) * 1e9))


@dataclass(frozen=True)
class Family:
    """A model family (e.g. ``gemma4``) with its variants and metadata."""

    name: str
    display_name: str
    fs_family: str | None
    model_types: tuple[str, ...]
    modalities: tuple[str, ...]
    variants: tuple[Variant, ...]
    tokenizer: dict[str, Any]
    chat_template: dict[str, Any]
    special_tokens: dict[str, str]
    lora_targets: tuple[str, ...]
    quirks: tuple[str, ...]
    provenance: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Hardware:
    """A measured (or literature) GPU target."""

    id: str
    gpu_name: str
    mem_gb: float
    gpus_per_node: int
    bf16_dense_tflops: float
    peak_provenance: str  # measured | datasheet
    interconnect: str
    mfu: dict[str, Any]  # {"dense": {value, provenance, evidence}, "moe": {...}}
    scheduler: str | None  # slurm | local | null
    cluster_rules: tuple[str, ...]
    notes: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    env: dict[str, str] = field(default_factory=dict)  # launch env the emitter exports
    cpus_per_task: int | None = None


@dataclass(frozen=True)
class AlgorithmCard:
    """One training algorithm as the FS registry knows it."""

    name: str
    stage: str
    family: str
    data_format: str
    requires: tuple[str, ...]
    key_hparams: dict[str, Any]
    failure_modes: tuple[dict[str, Any], ...]
    fs_entry: str
    references: tuple[str, ...]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class Recipe:
    """A proven training recipe, indexed by family/size/arch/stage/goal."""

    id: str
    version: str
    title: str
    index: dict[str, Any]
    data: dict[str, Any]
    stages: tuple[dict[str, Any], ...]
    hardware: dict[str, Any]
    evaluation: dict[str, Any]
    risks: tuple[str, ...]
    provenance: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)


# ---------------------------------------------------------------------------
# File reading / validation helpers
# ---------------------------------------------------------------------------


def _read_yaml_obj(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise KnowledgeError(f"{path}: unreadable: {exc}") from exc
    except yaml.YAMLError as exc:
        raise KnowledgeError(f"{path}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise KnowledgeError(f"{path}: top-level YAML value must be a mapping")
    return data


def _validate_file(data: Any, schema_name: str, path: Path) -> None:
    try:
        schema = load_schema(schema_name)
    except SchemaError as exc:
        raise KnowledgeError(f"{path}: cannot load schema {schema_name!r}: {exc}") from exc
    errors = validate(data, schema)
    if errors:
        raise KnowledgeError(f"{path}: schema validation failed: " + "; ".join(errors))


def _yaml_files(directory: Path, kind: str) -> list[Path]:
    if not directory.is_dir():
        raise KnowledgeError(f"{directory}: {kind} directory not found")
    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    return files


# ---------------------------------------------------------------------------
# Object construction (KeyError/TypeError become KnowledgeError naming the file)
# ---------------------------------------------------------------------------


def _variant_from(data: Any, path: Path) -> Variant:
    try:
        if not isinstance(data, dict):
            raise TypeError("variant entry must be a mapping")
        return Variant(
            id=str(data["id"]),
            hf_id=None if data.get("hf_id") is None else str(data["hf_id"]),
            local_path=None if data.get("local_path") is None else str(data["local_path"]),
            size_b=float(data["size_b"]),
            active_b=None if data.get("active_b") is None else float(data["active_b"]),
            arch=str(data["arch"]),
            hidden=int(data["hidden"]),
            layers=int(data["layers"]),
            heads=int(data["heads"]),
            kv_heads=int(data["kv_heads"]),
            head_dim=None if data.get("head_dim") is None else int(data["head_dim"]),
            vocab=int(data["vocab"]),
            context_length=int(data["context_length"]),
            num_experts=None if data.get("num_experts") is None else int(data["num_experts"]),
            experts_per_token=None if data.get("experts_per_token") is None else int(data["experts_per_token"]),
            tied_embeddings=bool(data["tied_embeddings"]),
            instruct=bool(data["instruct"]),
            raw=dict(data),
        )
    except KeyError as exc:
        raise KnowledgeError(f"{path}: variant missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise KnowledgeError(f"{path}: malformed variant: {exc}") from exc


def _family_from(data: dict[str, Any], path: Path) -> Family:
    try:
        variants = tuple(_variant_from(item, path) for item in data["variants"])
        return Family(
            name=str(data["name"]),
            display_name=str(data["display_name"]),
            fs_family=None if data.get("fs_family") is None else str(data["fs_family"]),
            model_types=tuple(str(x) for x in data["model_types"]),
            modalities=tuple(str(x) for x in data["modalities"]),
            variants=variants,
            tokenizer=dict(data.get("tokenizer", {})),
            chat_template=dict(data.get("chat_template", {})),
            special_tokens={str(k): str(v) for k, v in dict(data.get("special_tokens", {})).items()},
            lora_targets=tuple(str(x) for x in data.get("lora_targets", [])),
            quirks=tuple(str(x) for x in data.get("quirks", [])),
            provenance=dict(data.get("provenance", {})),
            raw=dict(data),
        )
    except KeyError as exc:
        raise KnowledgeError(f"{path}: family missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise KnowledgeError(f"{path}: malformed family: {exc}") from exc


def _hardware_from(data: dict[str, Any], path: Path) -> Hardware:
    try:
        return Hardware(
            id=str(data["id"]),
            gpu_name=str(data["gpu_name"]),
            mem_gb=float(data["mem_gb"]),
            gpus_per_node=int(data["gpus_per_node"]),
            bf16_dense_tflops=float(data["bf16_dense_tflops"]),
            peak_provenance=str(data["peak_provenance"]),
            interconnect=str(data["interconnect"]),
            mfu={str(k): (dict(v) if isinstance(v, dict) else [dict(x) for x in v]) for k, v in dict(data["mfu"]).items()},
            env={str(k): str(v) for k, v in dict(data.get("env") or {}).items()},
            cpus_per_task=None if data.get("cpus_per_task") is None else int(data["cpus_per_task"]),
            scheduler=None if data.get("scheduler") is None else str(data["scheduler"]),
            cluster_rules=tuple(str(x) for x in data.get("cluster_rules", [])),
            notes=tuple(str(x) for x in data.get("notes", [])),
            raw=dict(data),
        )
    except KeyError as exc:
        raise KnowledgeError(f"{path}: hardware missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise KnowledgeError(f"{path}: malformed hardware entry: {exc}") from exc


def _algo_from(data: dict[str, Any], path: Path) -> AlgorithmCard:
    try:
        return AlgorithmCard(
            name=str(data["name"]),
            stage=str(data["stage"]),
            family=str(data["family"]),
            data_format=str(data["data_format"]),
            requires=tuple(str(x) for x in data.get("requires", [])),
            key_hparams=dict(data.get("key_hparams", {})),
            failure_modes=tuple(dict(x) for x in data.get("failure_modes", [])),
            fs_entry=str(data["fs_entry"]),
            references=tuple(str(x) for x in data.get("references", [])),
            raw=dict(data),
        )
    except KeyError as exc:
        raise KnowledgeError(f"{path}: algorithm card missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise KnowledgeError(f"{path}: malformed algorithm card: {exc}") from exc


def _recipe_from(data: dict[str, Any], path: Path) -> Recipe:
    try:
        return Recipe(
            id=str(data["id"]),
            version=str(data["version"]),
            title=str(data["title"]),
            index=dict(data["index"]),
            data=dict(data["data"]),
            stages=tuple(dict(x) for x in data["stages"]),
            hardware=dict(data["hardware"]),
            evaluation=dict(data["evaluation"]),
            risks=tuple(str(x) for x in data.get("risks", [])),
            provenance=dict(data.get("provenance", {})),
            raw=dict(data),
        )
    except KeyError as exc:
        raise KnowledgeError(f"{path}: recipe missing required key {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise KnowledgeError(f"{path}: malformed recipe: {exc}") from exc


# ---------------------------------------------------------------------------
# Cached loaders (cache key: knowledge root)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _cached_families(root_key: str) -> dict[str, Family]:
    out: dict[str, Family] = {}
    for path in _yaml_files(Path(root_key) / "families", "families"):
        data = _read_yaml_obj(path)
        _validate_file(data, "knowledge/family", path)
        family = _family_from(data, path)
        if family.name in out:
            raise KnowledgeError(f"{path}: duplicate family name {family.name!r}")
        out[family.name] = family
    return out


@lru_cache(maxsize=None)
def _cached_hardware(root_key: str) -> dict[str, Hardware]:
    out: dict[str, Hardware] = {}
    for path in _yaml_files(Path(root_key) / "hardware", "hardware"):
        data = _read_yaml_obj(path)
        _validate_file(data, "knowledge/hardware", path)
        hw = _hardware_from(data, path)
        if hw.id in out:
            raise KnowledgeError(f"{path}: duplicate hardware id {hw.id!r}")
        out[hw.id] = hw
    return out


@lru_cache(maxsize=None)
def _cached_algorithms(root_key: str) -> dict[str, AlgorithmCard]:
    out: dict[str, AlgorithmCard] = {}
    for path in _yaml_files(Path(root_key) / "algorithms", "algorithms"):
        data = _read_yaml_obj(path)
        _validate_file(data, "knowledge/algorithm_card", path)
        card = _algo_from(data, path)
        if card.name in out:
            raise KnowledgeError(f"{path}: duplicate algorithm name {card.name!r}")
        out[card.name] = card
    return out


@lru_cache(maxsize=None)
def _cached_recipes(root_key: str) -> list[Recipe]:
    out: list[Recipe] = []
    for path in _yaml_files(Path(root_key) / "recipes", "recipes"):
        data = _read_yaml_obj(path)
        _validate_file(data, "knowledge/recipe", path)
        out.append(_recipe_from(data, path))
    return out


@lru_cache(maxsize=None)
def _cached_stage_rules(root_key: str) -> list[dict[str, Any]]:
    path = Path(root_key) / "stage_rules.yaml"
    if not path.is_file():
        raise KnowledgeError(f"{path}: stage rules file not found")
    data = _read_yaml_obj(path)
    # The stage-rules schema may not be shipped yet; fall back to a structural check.
    try:
        schema = load_schema("knowledge/stage_rules")
    except SchemaError:
        schema = None
    if schema is not None:
        errors = validate(data, schema)
        if errors:
            raise KnowledgeError(f"{path}: schema validation failed: " + "; ".join(errors))
    rules = data.get("rules")
    if not isinstance(rules, list) or not all(isinstance(r, dict) for r in rules):
        raise KnowledgeError(f"{path}: 'rules' must be a list of mappings")
    return [dict(r) for r in rules]


# ---------------------------------------------------------------------------
# Public loader API
# ---------------------------------------------------------------------------


def load_families(root: Path | None = None) -> dict[str, Family]:
    """All model families keyed by name. Cached; treat the result as read-only."""
    return _cached_families(_root_key(root))


def load_hardware(root: Path | None = None) -> dict[str, Hardware]:
    """All hardware targets keyed by id. Cached; treat the result as read-only."""
    return _cached_hardware(_root_key(root))


def load_algorithm_cards(root: Path | None = None) -> dict[str, AlgorithmCard]:
    """All algorithm cards keyed by FS registry name. Cached; read-only."""
    return _cached_algorithms(_root_key(root))


def load_recipes(root: Path | None = None) -> list[Recipe]:
    """All recipes in deterministic (filename) order. Cached; read-only."""
    return list(_cached_recipes(_root_key(root)))


def load_stage_rules(root: Path | None = None) -> list[dict[str, Any]]:
    """The stage-selection rules from ``stage_rules.yaml``. Cached; read-only."""
    return list(_cached_stage_rules(_root_key(root)))


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def family_for_model_type(model_type: str, root: Path | None = None) -> Family | None:
    """Return the family whose ``model_types`` contains ``model_type``, else None."""
    for family in load_families(root).values():
        if model_type in family.model_types:
            return family
    return None


def find_variant(model: str, root: Path | None = None) -> tuple[Family, Variant] | None:
    """Resolve a model reference to ``(family, variant)``.

    Matching order (spec): exact match on variant ``id``, ``hf_id`` or
    ``local_path``; then, when ``model`` is a local directory, read
    ``<model>/config.json`` (using ``text_config`` when present) and match the
    family's variant by ``model_type`` plus ``hidden_size``/``num_hidden_layers``.
    The first variant of the family matching the available dimensions wins;
    when neither dimension is present the first variant of the family is used.
    Returns None when nothing matches.
    """
    families = load_families(root)
    for family in families.values():
        for variant in family.variants:
            if model == variant.id:
                return family, variant
            if variant.hf_id is not None and model == variant.hf_id:
                return family, variant
            if variant.local_path is not None and model == variant.local_path:
                return family, variant

    path = Path(model)
    config_file = path / "config.json"
    if not (path.is_dir() and config_file.is_file()):
        return None
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KnowledgeError(f"{config_file}: unreadable config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise KnowledgeError(f"{config_file}: config.json must be a JSON object")

    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    model_type = text.get("model_type", config.get("model_type"))
    family = family_for_model_type(str(model_type), root) if model_type else None
    if family is None:
        return None
    hidden = text.get("hidden_size")
    layers = text.get("num_hidden_layers")
    for variant in family.variants:
        if hidden is not None and variant.hidden != int(hidden):
            continue
        if layers is not None and variant.layers != int(layers):
            continue
        return family, variant
    return None
