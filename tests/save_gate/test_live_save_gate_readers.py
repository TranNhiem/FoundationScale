"""Reader and config units that need no artifact: base model, train spec, MoE override.

Split verbatim from tests/test_live_save_gate.py; shared helpers live in
tests/_live_save_gate_fixtures.py. No line here differs from the original.
"""

from __future__ import annotations

from _live_save_gate_fixtures import (
    DENSE_CFG,
    MOE_CFG,
    _dense_full_tensors,
    _moe_full_tensors,
    _probe_declared_or_calibrate,
    _write_safetensors,
    json,
    lsg,
    pytest,
    struct,
)


class TestBaseModelReader:
    def test_sharded_base_index_is_honored(self, tmp_path):
        """[PASSES-BEFORE] Red if: the idx.is_file() branch is inverted."""
        tensors = _dense_full_tensors()
        items = sorted(tensors.items())
        shards = [dict(items[:6]), dict(items[6:])]
        base = tmp_path / "sharded-base"
        base.mkdir()
        weight_map = {}
        for i, shard in enumerate(shards):
            name = f"model-{i + 1:05d}-of-00002.safetensors"
            _write_safetensors(base / name, shard)
            for fqn in shard:
                weight_map[fqn] = name
        (base / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
        (base / "config.json").write_text(json.dumps(DENSE_CFG))
        loaded = lsg.BaseModel.load(base)
        assert len(loaded.tensors) == 12
        assert "2 shards" in loaded.tensors_source

    def test_empty_weight_map_is_unmeasured(self, tmp_path):
        """[PASSES-BEFORE] Red if: the `if not weight_map` guard is deleted."""
        base = tmp_path / "b"
        base.mkdir()
        (base / "config.json").write_text("{}")
        (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
        with pytest.raises(lsg.GateUnmeasured, match="empty weight_map"):
            lsg.BaseModel.load(base)

    def test_unknown_dtype_is_unmeasured_not_coerced(self, tmp_path):
        """[PASSES-BEFORE] Byte pricing on a guessed dtype prices wrong; the
        tool refuses. Red if: the `dtype is None` raise becomes a default."""
        path = tmp_path / "odd.safetensors"
        blob = json.dumps(
            {"x.w": {"dtype": "FP8_E4M3", "shape": [2, 2], "data_offsets": [0, 4]}}
        ).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * 4)
        with pytest.raises(lsg.GateUnmeasured, match="unrecognized safetensors"):
            lsg._read_safetensors_header(path)


class TestTrainSpecResolution:
    def test_key_value_dump_is_accepted(self, tmp_path):
        """[PASSES-BEFORE] env-dump configs are first-class inputs. Red if:
        the KEY=VALUE fallback parser is deleted."""
        p = tmp_path / "resolved.env"
        p.write_text('declare -x PEFT_SCHEME="lora"\ndeclare -x LORA_RANK="8"\n')
        cfg, source = lsg._load_train_config(p)
        assert cfg["PEFT_SCHEME"] == "lora" and "KEY=VALUE" in source

    def test_rank_is_coerced_from_string(self, tmp_path):
        """[PASSES-BEFORE] Red if: the int() coercion is deleted (rank None
        downstream -> lora derivation abstains where it should not)."""
        spec = lsg.resolve_train_spec(
            {"peft_scheme": "lora", "lora_rank": "8"}, "test://cfg", "auto", None
        )
        assert spec.run_kind == "lora" and spec.lora_rank == 8

    def test_missing_kind_key_defers_with_stated_basis(self, tmp_path):
        """[PASSES-BEFORE] Deferral is a STATED abstention, not a silent full.
        Red if: the kbasis message loses the word 'inferred'."""
        spec = lsg.resolve_train_spec({}, "test://cfg", "auto", None)
        assert spec.run_kind == "auto" and "inferred" in spec.kind_basis


class TestMoeOverride:
    """The Gemma-4 dense declaration, post-bridge (fix25): the override is
    now the probe's two-source mint, never a local one
    (MINT_ZERO_ONLY_IN_PROBE). Doctrine 3 for the bridge: one MUST_PASS of
    the measured estate shape, two MUST_FIREs -- a self-contradicting config,
    and a frozen-regex laundering attempt against a real MoE base."""

    def test_enable_moe_block_false_zeroes_expert_denominator(self, tmp_path):
        """[FAILS-BEFORE on the corroboration and census assertions; the
        first two assertions pass on the current tree -- kept deliberately,
        see below] STRENGTHENED, never weakened. The author's intent, per the
        pre-patch docstring, was that an explicit enable_moe_block=false must
        flip the denominator to zero WITH the flip on the record. The
        pre-patch fixture undercut that intent: it paired the flag with a
        POSITIVE count (num_experts=8), a self-contradicting config whose
        count the single-source override silently overwrote -- and its first
        assertion (`decl.num_experts == 0`) survived the contract change
        without noticing, because the override minted regardless of basis.

        The fixture is now the MEASURED estate shape (gemma-4-E4B-it):
        enable_moe_block is False and num_experts is present-but-null. The
        test proves strictly more than the old one: the mint happens, the
        affirmative statement is quoted in the basis by the probe, the
        corroboration language is present, AND the census denominator
        (0 of 12 base-header names) travels in the notes. Red-makers on the
        current tree: the un-bridged call supplies no census, so the probe's
        basis reads "no artifact census was supplied to corroborate it" --
        containing "corroborate" but never "corroborated"/"two independent
        sources" -- and no census note exists (asserts 4-6 fail). Asserts
        1-2 pass pre-patch via the illegitimate mint and pass post-patch via
        the legitimate one: they are kept as the pin that the mint itself
        still happens."""
        cfg = {
            "model_type": "calibration-gemma4-dense",
            "text_config": {"num_moe_layers": 2, "enable_moe_block": False, "num_experts": None},
        }
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", None)
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=cfg,
            tensors={k: (v[0], "float32") for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic",
        )
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts == 0
        assert "enable_moe_block=false" in decl.experts_basis
        assert "corroborated" in decl.experts_basis
        assert "two independent sources agree" in decl.experts_basis
        assert any("expert-family census: 0 of 12" in n for n in decl.notes)

    def test_flag_false_beside_a_positive_count_abstains_loudly(self, tmp_path):
        """[FAILS-BEFORE] MUST_FIRE on the OLD test's exact fixture shape,
        preserved rather than discarded: enable_moe_block=false NEXT TO
        num_experts=8 is a self-contradicting config. Pre-patch the override
        adjudicated it FOR the config (num_experts minted to 0, the note
        branch skipped because the probe had abstained) -- precisely the
        mint-without-basis the verbatim fix25 failure reported. Post-patch
        the probe refuses to pick a winner and abstains; UNKNOWN and
        gates-block are named in the basis (doctrine 4)."""
        cfg = {
            **MOE_CFG,
            "text_config": {"num_experts": 8, "num_moe_layers": 2, "enable_moe_block": False},
        }
        _probe_declared_or_calibrate(MOE_CFG, 8, 2)
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", None)
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=cfg,
            tensors={k: (v[0], "float32") for k, v in _dense_full_tensors().items()},
            tensors_source="test://synthetic",
        )
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts is None
        assert "contradicts itself" in decl.experts_basis
        assert "UNKNOWN" in decl.experts_basis and "gates block" in decl.experts_basis

    def test_frozen_regex_cannot_launder_an_moe_base_into_dense(self, tmp_path):
        """[FAILS-BEFORE] MUST_FIRE for the census trap (fix25-s4): a config
        that affirmatively -- and WRONGLY -- declares dense against a REAL
        MoE base, plus a --frozen-regex that matches the expert stem and
        empties the in-scope expert set. A census computed over the
        frozen-filtered population would read 0 and corroborate the lie: the
        founding incident wearing a user regex. The census is taken over the
        UNFILTERED base header, finds 32 expert-family names (8 experts x
        2 layers x 2 projections), and the contradiction blocks. Pre-patch
        the override mints 0 despite the experts sitting in the header --
        red on the first assertion."""
        cfg = {
            "model_type": "calibration-laundering-attempt",
            "text_config": {"enable_moe_block": False, "num_experts": None},
        }
        spec = lsg.resolve_train_spec({}, "test://cfg", "full", r"\.experts\.")
        base = lsg.BaseModel(
            model_dir=tmp_path,
            config=cfg,
            tensors={k: (v[0], "float32") for k, v in _moe_full_tensors().items()},
            tensors_source="test://synthetic",
        )
        decl = lsg.derive_declared_block(base, spec, set(), "", r"\.(lora_[AB])$")
        assert decl.num_experts is None
        assert "CONTRADICTION" in decl.experts_basis
        assert "32" in decl.experts_basis
