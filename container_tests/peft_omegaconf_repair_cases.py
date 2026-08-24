"""fix42 / #73 regression cases: callable dataclass instances and the override view.

LANE CONTRACT (fix43, task #74). These cases exercise the REAL vendored
`megatron.bridge.training.utils.omegaconf_utils`, importable only inside the
training container (that module imports torch/hydra/omegaconf at module
scope; the off-stack checkout cannot provide it, and the fix42-era
module-level importorskip converted that fact into 1 pytest skip that CI
rightly fails under FS_FORBID_SKIPS=1). Two decisions keep this honest:

1. The filename deliberately does NOT match pytest's `test_*.py` /
   `*_test.py` collection globs, so no pytest configuration — any rootdir,
   any testpaths — can collect it off-stack. Run it BY NAME, in-container:
       python3 -m pytest container_tests/peft_omegaconf_repair_cases.py -q
   Its off-stack abstention is therefore not a runtime verdict pytest could
   launder; it is structural, and it is pinned with a denominator off-stack
   by tests/test_container_lane_peft_repair.py (existence, exact 8-case AST
   census, retirement of the old path, plus that census's own MUST_FIRE).
2. The import below is a PLAIN import. In the lane where the vendored stack
   is promised, an ImportError is a real defect: it surfaces as a pytest
   collection ERROR, i.e. RED — never a skip and never a vacuous green.
   This is strictly stronger than the importorskip it replaces.

Founding incident for this file: `_is_omegaconf_problematic` tested
`callable(val)` before `dataclasses.is_dataclass(val)`. A PEFT transform
(LoRA) is BOTH — a dataclass of ints/floats/strings that implements
`__call__(model)` — so the whole peft subtree was classified "callable
OmegaConf cannot serialize", dropped from the override view (measured
24 -> 23 fields), AND tracked whole in `excluded_callables`, which
`_restore_excluded_fields` setattr()s back AFTER overrides.

That second-order path was FEARED to cause a silent revert. It does not,
and the correction is recorded here rather than quietly dropped, because a
scary story that survives its own refutation is how a suite acquires tests
nobody can read. MEASURED on <compute-node> 2026-08-24 (probe_revert_semantics.py,
both controls green): `excluded['root.peft'] IS cfg.peft` — the tracker holds
a REFERENCE, and `_apply_overrides` MUTATES that same instance in place, so
the "restore" writes back the already-mutated object and dim=96 survives even
pre-repair. A revert needs REPLACEMENT semantics, which this code does not
use. The real and only defect was the view-drop: Hydra refused `peft.dim`
before any of this ran. The repair is one reorder; these cases pin the
view-drop, the tracker's granularity, and the preserved side, against the
REAL vendored module — no textual restatement, no copy.

RED/GREEN ledger (container lane): on the pre-Edit-1 vendored tree
(`MBRIDGE/` predicate reordered only by hand, per the fix42/fix43
orchestration) —
MEASURED, not predicted — both halves run on the tray with the tree's md5
printed in each (UNREPAIRED 6dea9ac5 -> 3 failed/5 passed; REPAIRED
8c577b11 -> 8 passed):
  RED  test_callable_dataclass_instance_is_not_problematic
  RED  test_callable_dataclass_subtree_is_addressable_in_the_view
  RED  test_tracker_records_the_genuine_leaf_not_the_whole_subtree
  GREEN both before and after (controls — the exclusion machinery has a
  real job the repair must not disarm): the lambda, partial, plain-set, and
  dataclass-TYPE tests, AND test_overrides_land_and_are_not_silently_reverted
  (see its own comment: it was authored as a 4th RED and measured green on
  both trees; every assertion in it is true, so it is kept as a control and
  relabelled rather than deleted).
  DENOMINATOR: 8 cases = 3 that discriminate the repair + 5 controls. A
  reader who takes all 8 as evidence FOR the reorder overcounts by 5.
"""

import dataclasses
import functools

from megatron.bridge.training.utils import omegaconf_utils as ocu


@dataclasses.dataclass
class _CallableTransform:
    """The measured LoRA shape: a dataclass whose instances also implement __call__.

    Every field type below is load-bearing: the scalars are the addressable
    knobs (dim/alpha/dropout), the list is target_modules, and params_to_save
    is a set — genuinely OmegaConf-unserializable — which must stay excluded
    and restored as a LEAF while its parent becomes addressable.
    """

    dim: int = 32
    alpha: int = 64
    dropout: float = 0.0
    target_modules: list = dataclasses.field(default_factory=lambda: ["linear_qkv", "linear_proj"])
    params_to_save: set = dataclasses.field(default_factory=lambda: {"weight_a"})

    def __call__(self, model):
        return model


@dataclasses.dataclass
class _Container:
    """Minimal ConfigContainer stand-in: one plain field, one callable-dataclass field,
    one genuine bare callable field (the control the tracker must keep tracking)."""

    lr: float = 1e-4
    peft: _CallableTransform = dataclasses.field(default_factory=_CallableTransform)
    collate_fn: object = None


def test_callable_dataclass_instance_is_not_problematic():
    # RED pre-fix (callable() won the ordering accident), GREEN after.
    assert ocu._is_omegaconf_problematic(_CallableTransform()) is False


def test_dataclass_type_is_allowed():
    # Control: dataclass CLASSES are types, allowed by the isinstance(val, type)
    # branch both before and after the repair — untroubled by the reorder.
    assert ocu._is_omegaconf_problematic(_CallableTransform) is False


def test_bare_lambda_still_problematic():
    # Control: the exclusion machinery's actual job must survive the repair.
    assert ocu._is_omegaconf_problematic(lambda: None) is True


def test_functools_partial_still_problematic():
    assert ocu._is_omegaconf_problematic(functools.partial(int, base=2)) is True


def test_plain_set_still_problematic():
    # Control for the leaf mechanics: params_to_save must keep being excluded.
    assert ocu._is_omegaconf_problematic({"weight_a"}) is True


def test_callable_dataclass_subtree_is_addressable_in_the_view():
    # RED pre-fix: peft was absent from the view entirely (24 -> 23 measured).
    cfg = _Container(collate_fn=lambda batch: batch)
    omega, _ = ocu.create_omegaconf_dict_config(cfg)
    assert "peft" in omega
    assert omega.peft.dim == 32
    assert list(omega.peft.target_modules) == ["linear_qkv", "linear_proj"]
    # The unserializable leaf is excluded from the view even though its parent
    # is addressable — leaf granularity, not subtree granularity.
    assert "params_to_save" not in omega.peft


def test_tracker_records_the_genuine_leaf_not_the_whole_subtree():
    # RED pre-fix: tracker held 'root.peft' (the whole transform) and therefore
    # did NOT hold the leaf path; post-fix it must hold the leaf and not the
    # parent. This is the second-order trap made into a tripwire.
    cfg = _Container(collate_fn=lambda batch: batch)
    excluded = ocu._track_excluded_fields(cfg, "root")
    assert "root.peft" not in excluded
    assert "root.peft.params_to_save" in excluded
    assert excluded["root.peft.params_to_save"] == {"weight_a"}
    # Control: the genuine bare callable is still tracked whole.
    assert excluded["root.collate_fn"] is cfg.collate_fn


def test_overrides_land_and_are_not_silently_reverted():
    # CONTROL (green on BOTH trees) — authored as a 4th RED and measured
    # otherwise. The original claim was: _apply_overrides sets dim=96, then
    # _restore_excluded_fields setattr()s the ORIGINAL transform back over
    # 'peft', net effect 32, loud nowhere. MEASURED false on the real tree:
    # the tracker holds a REFERENCE to the very object _apply_overrides
    # mutates in place, so the restore writes back the already-mutated
    # instance and 96 survives pre-repair too. The silent-revert hazard is
    # real only under REPLACEMENT semantics, which this code does not use.
    # Kept because every assertion below is true and worth pinning; NOT
    # counted as evidence that the reorder works — cases 1, 6 and 7 are.
    cfg = _Container(collate_fn=lambda batch: batch)
    original_peft = cfg.peft
    _, excluded = ocu.create_omegaconf_dict_config(cfg)
    ocu.apply_overrides(cfg, {"peft": {"dim": 96}}, excluded)
    assert cfg.peft.dim == 96
    # Identity control: composition mutates the instance, it must not replace
    # it with a dict-shaped stand-in (the rehearsal's MUST_PASS, made cheap).
    assert cfg.peft is original_peft
    assert dataclasses.is_dataclass(cfg.peft) and callable(cfg.peft)
    # The unserializable leaf was leaf-restored, not lost.
    assert cfg.peft.params_to_save == {"weight_a"}
    # The genuine bare callable survived composition untouched.
    assert callable(cfg.collate_fn)
