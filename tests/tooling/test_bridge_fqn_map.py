"""The Bridge --fqn-map producer: what it counts, and what it must not drop.

Megatron is not importable here, so the sharded types are stand-ins passed in the
way main() passes the real ones. Every measured fact these pin came from the
gemma-4-26B-A4B checkpoints named in the tool's docstring."""

from __future__ import annotations

from tools import bridge_fqn_map as bfm


class _Keyed:
    def __init__(self, key: str) -> None:
        self.key = key


# Three unrelated types, as in Megatron: none is a subclass of another, so an
# isinstance check on one never matches the others.
class _Tensor(_Keyed):
    """Stands in for ShardedTensor."""


class _Factory(_Keyed):
    """Stands in for ShardedTensorFactory: a gated fc1 that merges to one key."""


class _Object(_Keyed):
    """Stands in for ShardedObject: non-tensor state such as TE _extra_state."""


class _Chunk:
    def __init__(self, ssd: dict, n_params: int = 2) -> None:
        self._ssd = ssd
        self._n = n_params

    def sharded_state_dict(self) -> dict:
        return self._ssd

    def parameters(self):
        return iter(range(self._n))


TENSORS = (_Tensor, _Factory)
OBJECTS = (_Object,)


def _ssd() -> dict:
    return {
        "decoder.layers.0.self_attention.linear_qkv.weight": _Tensor(
            "language_model.decoder.layers.0.self_attention.linear_qkv.weight"
        ),
        "decoder.layers.0.mlp": {
            "linear_fc1.weight": _Factory("language_model.decoder.layers.0.mlp.linear_fc1.weight"),
            "linear_fc2.weight": _Tensor("language_model.decoder.layers.0.mlp.linear_fc2.weight"),
        },
        "decoder.layers.0.self_attention.linear_qkv._extra_state": _Object(
            "language_model.decoder.layers.0.self_attention.linear_qkv._extra_state"
        ),
        "nested": [(_Tensor("vision_tower.patch_embedder.input_proj.weight"),)],
    }


def test_factory_tensors_are_declared_not_dropped() -> None:
    tensors, objects = bfm.collect_keys(_ssd(), TENSORS, OBJECTS)
    assert "language_model.decoder.layers.0.mlp.linear_fc1.weight" in tensors
    assert len(tensors) == 4 and len(objects) == 1


def test_leaving_out_the_factory_type_loses_the_gated_fc1() -> None:
    # The first hardware run made exactly this mistake: 868 declared, 928 on disk.
    tensors, _ = bfm.collect_keys(_ssd(), (_Tensor,), OBJECTS)
    assert "language_model.decoder.layers.0.mlp.linear_fc1.weight" not in tensors


def test_map_excludes_extra_state_and_records_its_source() -> None:
    doc = bfm.build_map([_Chunk(_ssd())], TENSORS, OBJECTS, recipe="r_cfg", hf_path="/m")
    assert doc["declared_fqns"] == sorted(doc["declared_fqns"])
    assert not any("_extra_state" in f for f in doc["declared_fqns"])
    assert len(doc["declared_fqns"]) == 4 and doc["declared_objects"] == 1
    src = doc["source"]
    assert src["recipe"] == "r_cfg" and src["hf_config"] == "/m/config.json"
    assert src["parallel_layout_built"] == "tp=pp=ep=etp=cp=1"
    assert src["model_parameters"] == 2


def test_chunks_are_unioned_so_layout_does_not_change_the_map() -> None:
    whole = bfm.build_map([_Chunk(_ssd())], TENSORS, OBJECTS, recipe="r", hf_path="/m")
    items = list(_ssd().items())
    split = [_Chunk(dict(items[:2]), 1), _Chunk(dict(items[2:]), 1)]
    parts = bfm.build_map(split, TENSORS, OBJECTS, recipe="r", hf_path="/m")
    assert parts["declared_fqns"] == whole["declared_fqns"]
    assert parts["source"]["model_parameters"] == 2


def test_an_empty_model_yields_an_empty_map_for_main_to_refuse() -> None:
    doc = bfm.build_map([_Chunk({})], TENSORS, OBJECTS, recipe="r", hf_path="/m")
    assert doc["declared_fqns"] == []
    assert bfm.EXIT_UNMEASURED == 95
