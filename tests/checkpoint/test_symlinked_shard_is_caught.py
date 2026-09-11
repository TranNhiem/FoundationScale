"""Detector and control arms for symlinked checkpoint shards (#343 / row T1-6).

The defect measured on the cluster was an indirection the checkpoint plane could
not see: the fresh checkpoint's only tensor payload was a symbolic link to the
base model's shard. Every filesystem predicate that mattered used a
link-following API, so :meth:`pathlib.Path.exists` and
:meth:`pathlib.Path.stat` described the base model's bytes and credited the
fresh checkpoint with data it never wrote. The detector must instead classify
the link inode itself with ``Path.is_symlink()`` or another
``follow_symlinks=False`` / ``lstat`` path.

Both arms are required. A run arm that merely raises with ``assert raises`` is
indistinguishable from a detector that rejects every checkpoint. A control arm
that merely passes leaves open the possibility that the run arm was malformed
rather than detected. Only the five/zero pair shows that the inode type, not
the bytes or layout, drives the verdict:

* run arm: ``checkpoint/model-....safetensors`` is a symlink to a valid shard
  outside the checkpoint, and must be RED (5);
* control arm: the identical bytes are copied to the same relative shard path
  as a regular file, and must remain GREEN (0);
* accounting arm: while the run arm is measured, no link-following ``stat``
  may record the target's size; the only recorded shard size may be the
  symlink inode's own ``lstat`` size.

The fixture uses safetensors because a complete shard can be written with the
standard library alone, preserving the checkpoint metadata module's torch-free
contract. There is deliberately no skip for symlink creation: under
``FS_FORBID_SKIPS=1``, an environment that cannot construct the measured run
arm is a failure, not an UNMEASURED row. Neither fixture is CANNOT-MEASURE;
the run artifact is a concrete bad checkpoint and therefore RED, not a refusal.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from foundationscale.checkpoint import dcp_meta
from foundationscale.checkpoint.dcp import CheckpointFormatError, SafetensorsReader

_GREEN = 0
_RED = 5

_TENSOR_KEY = "model.layers.0.experts.0.weight"
_SHARD_NAME = "model-00001-of-00001.safetensors"
_PAYLOAD_BYTES = 64 * 1024
_SYLINK_RE = r"(?i)(symbolic[ -]?link|symlink)"


@dataclass(frozen=True)
class _CheckpointArms:
    """The measured and control checkpoints, kept layout-identical."""

    tensor_key: str
    shard_name: str
    payload_bytes: int
    target: Path
    run_dir: Path
    control_dir: Path

    @property
    def run_shard(self) -> Path:
        return self.run_dir / self.shard_name

    @property
    def control_shard(self) -> Path:
        return self.control_dir / self.shard_name


def _write_safetensors_shard(path: Path, tensor_key: str, payload_bytes: int) -> None:
    """Write a valid one-tensor shard with deterministic payload bytes."""
    header = json.dumps(
        {
            tensor_key: {
                "dtype": "U8",
                "shape": [payload_bytes],
                "data_offsets": [0, payload_bytes],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    repeating = bytes(range(256)) * ((payload_bytes + 255) // 256)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(header).to_bytes(8, "little") + header + repeating[:payload_bytes])


def _write_index(directory: Path, tensor_key: str, shard_name: str, payload_bytes: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": payload_bytes},
                "weight_map": {tensor_key: shard_name},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture()
def checkpoint_arms(tmp_path: Path) -> _CheckpointArms:
    """Build the indistinguishable-to-``stat`` run and regular-file control."""
    arms = _CheckpointArms(
        tensor_key=_TENSOR_KEY,
        shard_name=_SHARD_NAME,
        payload_bytes=_PAYLOAD_BYTES,
        target=tmp_path / "base-model" / _SHARD_NAME,
        run_dir=tmp_path / "run-arm" / "checkpoint",
        control_dir=tmp_path / "control-arm" / "checkpoint",
    )

    _write_safetensors_shard(arms.target, arms.tensor_key, arms.payload_bytes)
    _write_index(arms.run_dir, arms.tensor_key, arms.shard_name, arms.payload_bytes)
    _write_index(arms.control_dir, arms.tensor_key, arms.shard_name, arms.payload_bytes)

    # Absolute targets also occur on the cluster, so the link is written with
    # one. The fixture deliberately verifies that a link-following size query
    # is blind to this link before inviting the module under test to make the
    # same mistake.
    arms.run_shard.symlink_to(arms.target)
    shutil.copyfile(arms.target, arms.control_shard)

    assert arms.run_shard.is_symlink(), (
        "fixture setup failed to create the run arm as a symlink; the test "
        "would then measure an ordinary copied shard and silently lose the "
        "#343 indirection it is supposed to detect"
    )
    assert not arms.control_shard.is_symlink(), (
        "fixture setup left the control arm as a link; a RED control would no "
        "longer distinguish symlink detection from shard-content rejection"
    )
    assert arms.run_shard.samefile(arms.target), (
        "the run-arm link does not resolve to the external base shard; the "
        "defect's identity aliasing is absent from the fixture"
    )
    assert arms.control_shard.read_bytes() == arms.target.read_bytes(), (
        "the control shard differs from the run link's target; any verdict "
        "difference could then be explained by content instead of inode type"
    )
    assert arms.run_shard.stat().st_size == arms.target.stat().st_size, (
        "the link-following size query defect #343 relied on did not report "
        "the external target bytes; the adversarial fixture is not reproducing "
        "the blind spot measured on the cluster"
    )
    assert os.lstat(arms.run_shard).st_size != arms.target.stat().st_size, (
        "the symlink inode and its target have the same size, so this fixture "
        "cannot prove which inode supplied the recorded byte count"
    )
    return arms


def _metadata_verdict(checkpoint_dir: Path) -> int:
    """Translate the checkpoint-format boundary used by the gate runner.

    A malformed or dishonest checkpoint artifact is RED. There is no broad
    exception mapping here: harness defects must fail the test rather than
    becoming UNMEASURED, and neither fixture is a CANNOT-MEASURE/REFUSE row.
    """
    try:
        dcp_meta.read_metadata(os.fspath(checkpoint_dir))
    except CheckpointFormatError:
        return _RED
    return _GREEN


def test_symlinked_tensor_shard_run_arm_is_red(checkpoint_arms: _CheckpointArms) -> None:
    """The checkpoint must not inherit bytes by naming a base-model shard."""
    verdict = _metadata_verdict(checkpoint_arms.run_dir)
    assert verdict == _RED, (
        f"expected RED ({_RED}) for the symlinked run arm but got {verdict}; "
        "a non-RED result still lets metadata describe the base shard as if "
        "this checkpoint wrote those bytes (#343)"
    )

    with pytest.raises(CheckpointFormatError, match=_SYLINK_RE) as caught:
        dcp_meta.read_metadata(os.fspath(checkpoint_arms.run_dir))

    assert checkpoint_arms.shard_name in str(caught.value), (
        "the RED diagnostic does not identify implicated shard "
        f"{checkpoint_arms.shard_name!r}; an operator would be left unable to "
        "distinguish the symlink defect from an unrelated malformed export"
    )


def test_byte_identical_regular_shard_control_is_green(
    checkpoint_arms: _CheckpointArms,
) -> None:
    """The detector must classify the link, not these otherwise valid bytes."""
    verdict = _metadata_verdict(checkpoint_arms.control_dir)
    assert verdict == _GREEN, (
        f"expected GREEN ({_GREEN}) for the byte-identical real-file control "
        f"but got {verdict}; rejecting this arm means the fix rejects valid "
        "local checkpoint bytes rather than the symlink behavior (#343)"
    )

    summary = dcp_meta.read_metadata(os.fspath(checkpoint_arms.control_dir))
    assert summary.format == "safetensors", (
        "the valid control artifact changed format classification; the symlink "
        "fix has widened behavior beyond the inode-type defect"
    )
    assert set(summary.tensors) == {checkpoint_arms.tensor_key}, (
        "the control's tensor surface changed while only symlink handling was "
        "being fixed; ordinary safetensors reads must remain byte-for-byte in "
        "behavior"
    )

    meta = summary.tensors.get(checkpoint_arms.tensor_key)
    assert meta is not None, (
        "the control tensor disappeared from metadata even though the real "
        "control shard is byte-identical to the run link's target"
    )
    assert meta.shape == (checkpoint_arms.payload_bytes,), (
        "the control tensor shape was not read from the real on-disk shard; "
        "the metadata path is no longer describing the control artifact"
    )
    assert meta.dtype == "uint8", (
        "the control tensor dtype changed under the symlink fix; valid real "
        "files must follow exactly the existing metadata mapping"
    )
    assert meta.storage_id == (f"{checkpoint_arms.shard_name}:0:{checkpoint_arms.payload_bytes}"), (
        "the control's storage identity does not name the local shard and its "
        "actual payload span; byte accounting is no longer anchored in this "
        "checkpoint's own regular file"
    )


def test_recorded_size_is_the_link_inode_not_its_target(
    checkpoint_arms: _CheckpointArms,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrument stat boundaries so target size cannot be silently recorded.

    ``Path.stat()`` and ``os.path.getsize()`` follow links; ``Path.lstat()``
    and ``os.lstat()`` do not. Instrumenting both groups distinguishes a real
    non-following inspection from a late string check added after the target
    was already observed or opened.
    """
    # absolute(), never resolve(): resolve() follows symlinks, so on this
    # fixture it would return the TARGET path and the interceptor below would
    # stop recognising the link it exists to watch. The comparison must be over
    # the link's own name.
    run_shard_text = os.fspath(Path(checkpoint_arms.run_shard).absolute())
    link_inode_size = os.lstat(checkpoint_arms.run_shard).st_size
    target_size = arms_target_size = checkpoint_arms.target.stat().st_size

    assert link_inode_size != arms_target_size, (
        "the accounting arm cannot distinguish source and target sizes; a "
        "pass would be vacuous and could not expose the #343 byte credit"
    )

    followed_sizes: list[int] = []
    nonfollow_sizes: list[int] = []

    def is_run_shard(candidate: object) -> bool:
        try:
            return os.fspath(Path(os.fspath(candidate)).absolute()) == run_shard_text
        except TypeError:
            return False

    real_path_stat = dcp_meta.Path.stat
    real_path_lstat = dcp_meta.Path.lstat
    real_os_stat = os.stat
    real_os_lstat = os.lstat

    def recording_path_stat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_path_stat(self, *args, **kwargs)
        if is_run_shard(self):
            bucket = followed_sizes if kwargs.get("follow_symlinks", True) else nonfollow_sizes
            bucket.append(int(result.st_size))
        return result

    def recording_path_lstat(self: Path) -> os.stat_result:
        result = real_path_lstat(self)
        if is_run_shard(self):
            nonfollow_sizes.append(int(result.st_size))
        return result

    def recording_os_stat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_os_stat(path, *args, **kwargs)
        if is_run_shard(path):
            bucket = followed_sizes if kwargs.get("follow_symlinks", True) else nonfollow_sizes
            bucket.append(int(result.st_size))
        return result

    def recording_os_lstat(path: object, *args: object, **kwargs: object) -> os.stat_result:
        result = real_os_lstat(path, *args, **kwargs)
        if is_run_shard(path):
            nonfollow_sizes.append(int(result.st_size))
        return result

    monkeypatch.setattr(dcp_meta.Path, "stat", recording_path_stat)
    monkeypatch.setattr(dcp_meta.Path, "lstat", recording_path_lstat)
    monkeypatch.setattr(os, "stat", recording_os_stat)
    monkeypatch.setattr(os, "lstat", recording_os_lstat)

    with pytest.raises(CheckpointFormatError, match=_SYLINK_RE):
        dcp_meta.read_metadata(os.fspath(checkpoint_arms.run_dir))

    assert not followed_sizes, (
        f"a link-following stat recorded shard sizes {followed_sizes}; that "
        f"observes the {target_size}-byte external target rather than the "
        "run checkpoint's own tiny link inode and reopens the #343 byte "
        "credit even if a later check happens to refuse"
    )
    assert nonfollow_sizes, (
        "the symlink refusal never inspected the shard through a "
        "non-following filesystem API; without an inode observation this is "
        "not a detector for checkpoint-plane symlink indirection"
    )
    assert all(size == link_inode_size for size in nonfollow_sizes), (
        f"non-following observations recorded sizes {nonfollow_sizes}, not "
        f"only the link inode's own size {link_inode_size}; 'on-disk bytes' "
        "has drifted back toward the link target"
    )
    assert target_size not in nonfollow_sizes, (
        f"the {target_size}-byte target size appeared in the non-following "
        "recorded sizes; the accounting arm can no longer show that the "
        "metadata plane distinguished the local link from the base shard"
    )


def _reader_for(
    checkpoint_dir: Path,
    arms: _CheckpointArms,
) -> tuple[SafetensorsReader, object]:
    """Reach the exact shard-handle seam without importing the ML stack."""
    shard_path = os.fspath(checkpoint_dir / arms.shard_name)
    sentinel = object()
    reader = object.__new__(SafetensorsReader)
    reader.path = os.fspath(checkpoint_dir)
    reader._weight_map = {arms.tensor_key: arms.shard_name}
    reader._handles = {shard_path: sentinel}
    return reader, sentinel


def test_weight_reader_does_not_hand_out_the_symlink_but_keeps_control(
    checkpoint_arms: _CheckpointArms,
) -> None:
    """Pin dcp.py's existence check: presence of a target is not ownership."""
    run_reader, _ = _reader_for(checkpoint_arms.run_dir, checkpoint_arms)
    with pytest.raises(CheckpointFormatError, match=_SYLINK_RE):
        run_reader._handle_and_key(checkpoint_arms.tensor_key)

    control_reader, control_sentinel = _reader_for(
        checkpoint_arms.control_dir,
        checkpoint_arms,
    )
    handle, shard_path = control_reader._handle_and_key(checkpoint_arms.tensor_key)
    assert handle is control_sentinel, (
        "the real-file control did not receive its prepared shard handle; "
        "symlink protection has altered successful regular-file resolution"
    )
    assert Path(shard_path) == checkpoint_arms.control_shard, (
        "the control reader resolved a different shard path; the fix must "
        "reject link indirection without changing where a valid local "
        "weight_map entry points"
    )
