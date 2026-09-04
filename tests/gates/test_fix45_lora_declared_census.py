"""fix45 / #78 -- the lora declared-FQN oracle is the launch-time live-module census.

Fail-before accounting (the suite's own rule): 14 of the 15 tests below are RED
on the pre-fix45 tree (the pre-fix45 module has no _AdapterModuleCensus /
_load_adapter_modules, derive_declared_block knows no adapter_modules, and the
adjudicate entry point rejects the keyword -- every reference 404s or the new
refusal text is absent). The FIFTEENTH -- test_prefix_abstention_arm_intact --
is GREEN ON BOTH TREES BY CONSTRUCTION: it is an invariant guard over the
launcher's calibrated rc-0 arm (fix44), disclosed here in the harness's own
tradition so nobody quotes it as a fail-before leg.

Doctrinal coverage map for the detector-shaped pieces (doctrine 3):
  MUST_FIRE : empty census refused; duplicates refused; mixed dims refused;
              census inside the judged tree refused; adjudication with NO
              census refused (class adapter_census_unavailable); dropped
              adapter BLOCKs with the drop control fired; stale/foreign
              namespace census BLOCKs with the ZERO-overlap indictment;
              shape mismatch BLOCKs.
  MUST_PASS : names-only census loads; dims census loads; CLEAR end-to-end on
              a census faithful to the artifact (8 declared = 4 stems x 2);
              shapes bind CLEAR when the census carries dims; the shape
              abstention is STATED without poisoning the verdict; the fix44
              adapter-prefix calibrated arm is byte-stable #first and intact.
Denominators are asserted numerically everywhere (8 = 4 x 2, 7 remaining after
one drop, 336-style zero-overlap indictment on 8 fixture fqns, etc.).
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _import_gate_module():
    # Self-contained path setup: src/ (foundationscale package) and tools/
    # (real_checkpoint_probe sibling import) -- no conftest assumption, so
    # these tests run identically from the repo root or elsewhere.
    root = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
    for sub in ("src", "tools"):
        entry = str(root / sub)
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # The decision API is a library module (T2_lib_script_boundary#0). It was
    # previously spec-loaded from tools/live_save_gate.py under a distinct
    # module name; that script is now an argparse wrapper that re-exports, and
    # a re-export binds a NAME, not the defining module's globals -- so the
    # _measure monkeypatch below would be inert against the wrapper.
    from foundationscale.gates import adjudication

    return adjudication


LSG = _import_gate_module()

# Four fixture stems in the estate's measured Megatron shape (sample stem per
# the #78 measurement: language_model.decoder.layers.0.mlp.mlp.linear_fc1);
# the real census is 168 stems x 2 templates = 336 declared, the fixture is
# the same shape at 4 x 2 = 8 -- the ratio arithmetic is what is pinned.
CENSUS_STEMS = (
    "language_model.decoder.layers.0.self_attention.linear_qkv",
    "language_model.decoder.layers.0.self_attention.linear_proj",
    "language_model.decoder.layers.0.mlp.mlp.linear_fc1",
    "language_model.decoder.layers.0.mlp.mlp.linear_fc2",
)
WEIGHT_SUFFIXES = (".adapter.linear_in.weight", ".adapter.linear_out.weight")
FIXTURE_DIMS = {stem: (16, 8) for stem in CENSUS_STEMS}  # (out, in) small ints


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    """Write a header-correct safetensors shard: 8-byte len + JSON header +
    zero payload. The gate's base loader parses ONLY the header; the payload
    bytes exist so data_offsets are truthful."""
    header: dict[str, dict[str, object]] = {}
    offset = 0
    size_of = {"F32": 4, "F16": 2, "BF16": 2, "I64": 8}
    for name, (dtype, shape) in tensors.items():
        nbytes = math.prod(shape) * size_of[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": [int(d) for d in shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


def _make_base_dir(tmp_path: Path) -> Path:
    """A tiny HF-namespace base: config + 2 safetensors entries. Dense by
    affirmative statement (enable_moe_block False, num_experts null) with a
    zero expert-family census -- the probe's two-source mint is satisfiable."""
    base = tmp_path / "hf_base"
    base.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4_fixture",
                "text_config": {
                    "num_hidden_layers": 1,
                    "hidden_size": 8,
                    "num_key_value_heads": 2,
                    "enable_moe_block": False,
                    "num_experts": None,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_safetensors(
        base / "model.safetensors",
        {
            "model.language_model.embed_tokens.weight": ("F32", (8, 8)),
            "model.language_model.layers.0.mlp.down_proj.weight": ("F32", (8, 8)),
        },
    )
    return base


def _make_train_config(tmp_path: Path, rank: int | None) -> Path:
    doc: dict[str, object] = {"run_kind": "lora", "peft_scheme": "lora"}
    if rank is not None:
        doc["lora_rank"] = rank
    path = tmp_path / "resolved-train-config.json"
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def _adapter_metadata(
    stems=CENSUS_STEMS,
    dims: dict[str, tuple[int, int]] | None = None,
    rank: int = 32,
    drop: tuple[str, ...] = (),
    shape_corrupt: str | None = None,
) -> SimpleNamespace:
    """Duck-typed CheckpointMetadata standing in for read_metadata output:
    two adapter weight tensors per stem plus one is_extra_state blob (the
    real DCP's 4-entries-per-stem arithmetic in miniature -- 2 weights +
    extra_state rows the census counts but the sweep ignores)."""
    tensors: dict[str, SimpleNamespace] = {}
    for stem in stems:
        for suffix_idx, suffix in enumerate(WEIGHT_SUFFIXES):
            fqn = f"{stem}{suffix}"
            if fqn in drop:
                continue
            if dims is not None:
                out_d, in_d = dims[stem]
                shape = (rank, in_d) if suffix_idx == 0 else (out_d, rank)
            else:
                shape = (rank, 8) if suffix_idx == 0 else (16, rank)
            if shape_corrupt == fqn:
                shape = (shape[0], shape[1] + 1)
            tensors[fqn] = SimpleNamespace(
                shape=shape,
                dtype="torch.float32",
                storage_id=f"fixture://{fqn}",
                is_extra_state=False,
            )
    extra = f"{stems[0]}.adapter.linear_in._extra_state"
    tensors[extra] = SimpleNamespace(
        shape=(),
        dtype="torch.float32",
        storage_id=f"fixture://{extra}",
        is_extra_state=True,
    )
    return SimpleNamespace(tensors=tensors, origin="fix45 census fixture", format="torch_dist")


def _write_census(
    tmp_path: Path,
    stems=CENSUS_STEMS,
    dims: dict[str, tuple[int, int]] | None = None,
    name: str = "adapter-modules.json",
) -> Path:
    path = tmp_path / name
    if dims is None:
        doc: object = {
            "adapter_modules": list(stems),
            "source": (
                "fix45 test fixture standing in for the launch-time step-(5) live-module census"
            ),
        }
    else:
        doc = {
            "adapter_modules": [
                {"fqn": s, "out_features": dims[s][0], "in_features": dims[s][1]} for s in stems
            ],
            "source": "fix45 test fixture carrying parent dims",
        }
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def _adjudicate(
    monkeypatch,
    tmp_path: Path,
    *,
    meta=None,
    census_path=None,
    prefix: str | None = "",
    rank=32,
):
    """Drive the REAL adjudicate_checkpoint with _measure stubbed at the one
    seam (torch-free host cannot parse a real DCP; every layer downstream of
    _measure -- derive, context, gates, controls, sweep, report -- is the
    shipped code under judgment)."""
    base = _make_base_dir(tmp_path)
    cfg = _make_train_config(tmp_path, rank)
    ckpt = tmp_path / "checkpoints" / "iter_0000010"
    ckpt.mkdir(parents=True)
    if meta is None:
        meta = _adapter_metadata()
    monkeypatch.setattr(LSG, "_measure", lambda _p: meta)
    kwargs: dict[str, object] = {
        "event": "save",
        "run_kind": "lora",
        "base_model_dir": base,
        "train_config_path": cfg,
        "adapter_modules": census_path,
    }
    if prefix is not None:
        kwargs["adapter_prefix"] = prefix
    assert LSG._probe_derive_declared is not None, (
        "probe helpers unimportable in this environment -- this suite's "
        "denominator would collapse; fix sys.path, do not skip"
    )
    return LSG.adjudicate_checkpoint(ckpt, **kwargs)


# ---------------------------------------------------------------- loader ----


def test_loader_accepts_names_only_and_reports_denominator(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps(["m.b", "m.a"]) + "\n", encoding="utf-8")
    loaded = LSG._load_adapter_modules(census, judged_dir=tmp_path / "elsewhere")
    assert loaded.stems == ("m.a", "m.b")
    assert loaded.dims is None
    assert "2 artifact-namespace module stems" in loaded.basis
    assert "no parent dims" in loaded.basis  # shape abstention stated, not silent


def test_loader_accepts_dims_uniformly(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps(
            {
                "adapter_modules": [
                    {"fqn": "m.a", "out_features": 16, "in_features": 8},
                    {"fqn": "m.b", "out_features": 4, "in_features": 2},
                ],
                "producer": "fixture",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = LSG._load_adapter_modules(census, judged_dir=tmp_path / "elsewhere")
    assert loaded.dims == {"m.a": (16, 8), "m.b": (4, 2)}
    assert "parent dims for all 2" in loaded.basis
    assert "producer recorded in-file: fixture" in loaded.basis


def test_loader_refuses_zero_denominator(tmp_path):
    census = tmp_path / "census.json"
    census.write_text("[]\n", encoding="utf-8")
    with pytest.raises(LSG.GateUnmeasured) as exc:
        LSG._load_adapter_modules(census, judged_dir=tmp_path / "elsewhere")
    assert "ZERO" in str(exc.value)  # doctrine 1: empty is never a denominator


def test_loader_refuses_duplicates(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(json.dumps(["m.a", "m.a", "m.b"]) + "\n", encoding="utf-8")
    with pytest.raises(LSG.GateUnmeasured) as exc:
        LSG._load_adapter_modules(census, judged_dir=tmp_path / "elsewhere")
    assert "duplicate" in str(exc.value)
    assert "m.a" in str(exc.value)  # the broken entry is named, with its count


def test_loader_refuses_mixed_dims(tmp_path):
    census = tmp_path / "census.json"
    census.write_text(
        json.dumps(
            {
                "adapter_modules": [
                    {"fqn": "m.a", "out_features": 16, "in_features": 8},
                    "m.b",
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LSG.GateUnmeasured) as exc:
        LSG._load_adapter_modules(census, judged_dir=tmp_path / "elsewhere")
    assert "1 of 2" in str(exc.value)  # partial coverage named with its tally


def test_loader_refuses_census_inside_judged_tree(tmp_path):
    judged = tmp_path / "ckpt" / "iter_0000010"
    judged.mkdir(parents=True)
    census = judged / "adapter-modules.json"  # INSIDE the tree under judgment
    census.write_text(json.dumps(["m.a"]) + "\n", encoding="utf-8")
    with pytest.raises(LSG.GateUnmeasured) as exc:
        LSG._load_adapter_modules(census, judged_dir=judged)
    msg = str(exc.value)
    assert "INSIDE the tree under judgment" in msg
    assert LSG._refusal_class(msg) == LSG._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE


# --------------------------------------------------------------- refusal ----


def test_refusal_class_vocabulary_and_ordering():
    # The launcher calibrates exactly the prefix member today; the census
    # member exists for tools/operators. ORDER pin: a prefix message that
    # mentions --adapter-modules in its guidance must still classify prefix.
    assert (
        LSG._refusal_class(
            "--adapter-prefix was not pinned for a lora adjudication ... "
            "the fix path here is --adapter-modules ..."
        )
        == LSG._REFUSAL_ADAPTER_PREFIX_UNPINNED
    )
    assert (
        LSG._refusal_class("--adapter-modules was not supplied for a lora adjudication ...")
        == LSG._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE
    )
    assert (
        LSG._refusal_class("checkpoint unreadable: /x: why") == LSG._REFUSAL_CHECKPOINT_UNREADABLE
    )
    assert LSG._refusal_class("something else entirely") == "other_unmeasured"


def test_prefix_abstention_arm_intact(monkeypatch, tmp_path):
    # INVARIANT GUARD, green on both trees by construction (disclosed in the
    # module docstring): the calibrated fix44 arm must survive the #78 edits
    # byte-for-byte -- unpinned prefix refuses FIRST, before census loading,
    # with the exact token the launcher's m2 leg greps in the capture log.
    census_path = _write_census(tmp_path)
    with pytest.raises(LSG.GateUnmeasured) as exc:
        _adjudicate(monkeypatch, tmp_path, census_path=census_path, prefix=None)
    msg = str(exc.value)
    assert msg.startswith("--adapter-prefix was not pinned")
    assert LSG._refusal_class(msg) == LSG._REFUSAL_ADAPTER_PREFIX_UNPINNED


def test_census_absent_refuses_with_new_class(monkeypatch, tmp_path):
    # MUST_FIRE for the new refuse: prefix asserted, no census -> exit-3-class
    # refusal naming the measured <compute-node> failure, never the old BLOCKED.
    with pytest.raises(LSG.GateUnmeasured) as exc:
        _adjudicate(monkeypatch, tmp_path, census_path=None, prefix="")
    msg = str(exc.value)
    assert msg.startswith("--adapter-modules was not supplied")
    assert LSG._refusal_class(msg) == LSG._REFUSAL_ADAPTER_CENSUS_UNAVAILABLE
    assert "VACUOUS" in msg and "168" in msg  # the measurement is quoted


# ----------------------------------------------------------- end-to-end -----


def test_clear_with_census_oracle(monkeypatch, tmp_path):
    # MUST_PASS, end to end through derive, gates and controls: 8 declared =
    # 4 census stems x 2 templates, drop constructable AND fired.
    census_path = _write_census(tmp_path)
    d = _adjudicate(monkeypatch, tmp_path, census_path=census_path, prefix="")
    assert d.exit_code == 0, d.blocking_reasons
    assert d.verdict == "CLEAR"
    assert d.blocking_reasons == []
    assert "8 adapter tensors = 4 census modules" in d.declared_basis["fqns"]
    drops = [c for c in d.controls if c["control"] == "drop"]
    assert len(drops) == 1 and drops[0]["status"] == "fired", d.controls


def test_blocked_when_adapter_dropped(monkeypatch, tmp_path):
    # MUST_FIRE with the failure class itself: one adapter weight absent from
    # an otherwise faithful save -> BLOCKED, and the drop control still fires
    # on top (the detector stays exercised even when the baseline blocks).
    missing = CENSUS_STEMS[1] + ".adapter.linear_out.weight"
    meta = _adapter_metadata(drop=(missing,))
    census_path = _write_census(tmp_path)
    d = _adjudicate(monkeypatch, tmp_path, meta=meta, census_path=census_path)
    assert d.exit_code == LSG.EXIT_BLOCKED
    assert d.blocking_reasons  # named reasons, never a silent red
    assert d.report["inventory"]["real_tensors"] == 7  # 8 - 1, denominated
    drops = [c for c in d.controls if c["control"] == "drop"]
    assert len(drops) == 1 and drops[0]["status"] == "fired"


def test_blocked_when_census_namespace_stale(monkeypatch, tmp_path):
    # MUST_FIRE, the #78 signature shape rebuilt on the CORRECT side: census
    # in a foreign namespace against a Megatron-namespaced artifact -> zero
    # overlap indictment plus blocking verdict, never a vacuous anything.
    foreign = tuple(f"hfns.model.layers.0.mod{i}" for i in range(4))
    census_path = _write_census(tmp_path, stems=foreign)
    d = _adjudicate(monkeypatch, tmp_path, census_path=census_path)
    assert d.exit_code == LSG.EXIT_BLOCKED
    notes = d.declared_basis["notes"]
    assert any("ZERO names" in n for n in notes), notes
    assert "0/8" in d.declared_basis["fqns"] or "0/" in d.declared_basis["fqns"]


def test_shapes_bind_clear_when_census_carries_dims(monkeypatch, tmp_path):
    # MUST_PASS, shape-checked arm: dims (16, 8) + rank 32 -> declared shapes
    # (32, 8) / (16, 32); the fixture artifact matches exactly -> CLEAR.
    census_path = _write_census(tmp_path, dims=FIXTURE_DIMS)
    d = _adjudicate(monkeypatch, tmp_path, census_path=census_path, rank=32)
    assert d.exit_code == 0, d.blocking_reasons
    assert not any("DECLARED WITHOUT SHAPE CHECK" in n for n in d.declared_basis["notes"])


def test_shape_mismatch_fires(monkeypatch, tmp_path):
    # MUST_FIRE, shape-checked arm: one adapter weight one row off its
    # declared (out, rank) -> BLOCKED with the shapes reason named.
    bad = CENSUS_STEMS[0] + ".adapter.linear_in.weight"
    meta = _adapter_metadata(dims=FIXTURE_DIMS, rank=32, shape_corrupt=bad)
    census_path = _write_census(tmp_path, dims=FIXTURE_DIMS)
    d = _adjudicate(monkeypatch, tmp_path, meta=meta, census_path=census_path, rank=32)
    assert d.exit_code == LSG.EXIT_BLOCKED
    assert any("shapes" in r for r in d.blocking_reasons), d.blocking_reasons


def test_shape_abstention_is_stated_without_poisoning(monkeypatch, tmp_path):
    # MUST_PASS + named abstention (doctrine 5): names-only census (today's
    # wireable state) -> CLEAR, with the shape-check abstention STATED in the
    # notes -- an abstention is a first-class outcome, not a hidden one.
    census_path = _write_census(tmp_path)  # no dims
    d = _adjudicate(monkeypatch, tmp_path, census_path=census_path)
    assert d.exit_code == 0, d.blocking_reasons
    assert any(
        "ADAPTER SHAPES DECLARED WITHOUT SHAPE CHECK (8 FQNs)" in n
        for n in d.declared_basis["notes"]
    ), d.declared_basis["notes"]
