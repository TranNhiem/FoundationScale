"""C2.3: every written launch spec carries a valid harness manifest; launch refuses a malformed one."""

import json

import pytest

from foundationskills.core.artifacts import Artifact, write_artifact
from foundationskills.core.harness import validate_manifest
from foundationskills.core.provenance import make_provenance
from foundationskills.interfaces.fs.launch import LaunchRefused, launch


def _spec(tmp_path):
    return {"stage_name": "probe", "entry": "fskills-merge", "argv": ["/bin/true"], "env": {}, "sbatch": None,
            "dry_run_argv": None, "expected_outputs": [], "executable": True, "missing": None,
            "cwd": str(tmp_path), "output_dir": str(tmp_path / "out")}


def test_write_artifact_attaches_a_valid_manifest(tmp_path):
    ref = write_artifact(Artifact("fs_launch_spec", "probe", _spec(tmp_path), make_provenance("merging", "0.1.0")),
                         tmp_path)
    payload = json.loads(open(ref.path).read())["payload"]
    assert validate_manifest(payload["harness_manifest"]) == []
    assert payload["harness_manifest"]["skills"][0]["name"] == "merging"


def test_launch_refuses_a_malformed_manifest(tmp_path):
    from foundationskills.core.orchestrator import plan_hash

    spec = {**_spec(tmp_path), "harness_manifest": {"schema": "fs.harness_manifest.v1"}}
    with pytest.raises(LaunchRefused, match="harness_manifest"):
        launch(spec, confirm=plan_hash(spec), submit=False)
