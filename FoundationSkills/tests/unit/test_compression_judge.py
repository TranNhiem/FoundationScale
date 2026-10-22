"""Unit tests for the compression judge (judge_results) and the RTN compress launch spec (fs.emit_compress)."""

import hashlib
import json
import sys

import pytest

from foundationskills.interfaces.fs import emit_compress as fsmod
from foundationskills.skills.compression.judge import judge_results

SCHEMA = "fs.compress.rtn_results.v1"


def good_results(**over):
    res = {"schema": SCHEMA, "status": "MEASURED", "model": "m", "bits": 4, "group_size": 128,
           "text_sha256": "t", "instrument_certified": True, "instrument_rel_diff": 0.004,
           "base_ppl_a": 9.8, "compressed_ppl": 10.2, "ppl_ratio": 10.2 / 9.8, "quantized_linears": 10,
           "changed_fraction": 0.9, "effective_bits_total": 5.1}
    res.update(over)
    return res


def compress_block(**over):
    plan = {"model": "m", "bits": 4, "group_size": 128, "text_sha256": "t", "budget_ppl_ratio": 1.15}
    plan.update(over)
    return plan


def _unmeasured(res, plan):
    res.clear()
    res.update(schema=SCHEMA, status="UNMEASURED", reason="no usable text corpus")


def _write(tmp_path, res, name="results.json"):
    path = tmp_path / name
    path.write_text(json.dumps(res), encoding="utf-8")
    return str(path)


CASES = [
    ("missing-file", "MISSING", "REFUSED", "CP-IN-005", None),
    ("bad-schema", lambda r, p: r.update(schema="fs.other.v1"), "REFUSED", "CP-IN-005", None),
    ("unmeasured", _unmeasured, "UNMEASURED", "CP-HO-003", "no usable text corpus"),
    ("bits-8", lambda r, p: r.update(bits=8), "REFUSED", "CP-IN-006", None),
    ("uncertified", lambda r, p: r.update(instrument_certified=False), "UNMEASURED", "CP-HO-001", "instrument_rel_diff"),
    ("base-nan", lambda r, p: r.update(base_ppl_a=float("nan")), "RED", "CP-HO-002", None),
    ("base-0p5", lambda r, p: r.update(base_ppl_a=0.5), "RED", "CP-HO-002", None),
    ("changed-0", lambda r, p: r.update(changed_fraction=0), "RED", "CP-HO-006", None),
    ("quant-0", lambda r, p: r.update(quantized_linears=0), "RED", "CP-HO-006", None),
    ("budget-none", lambda r, p: p.update(budget_ppl_ratio=None), "REFUSED", "CP-IN-004", None),
    ("budget-2p5", lambda r, p: p.update(budget_ppl_ratio=2.5), "REFUSED", "CP-IN-004", None),
    ("ratio-1p31", lambda r, p: r.update(ppl_ratio=1.31), "RED", "CP-HO-011", None),
    ("ratio-nan", lambda r, p: r.update(ppl_ratio=float("nan")), "RED", "CP-HO-011", None),
    ("good", lambda r, p: None, "PASS", None, None),
]


@pytest.mark.parametrize("case_id,mut,verdict,rule,note_sub", CASES, ids=[c[0] for c in CASES])
def test_judge_results_branches(tmp_path, case_id, mut, verdict, rule, note_sub):
    plan = compress_block()
    if mut == "MISSING":
        path = str(tmp_path / "absent.json")
    else:
        res = good_results()
        mut(res, plan)
        path = _write(tmp_path, res)
    out = judge_results(plan, path)
    assert (out["verdict"], out["rule"]) == (verdict, rule)
    if note_sub is not None:
        assert note_sub in (out["note"] or "")


def test_pass_report_and_refused_report(tmp_path):
    good = judge_results(compress_block(), _write(tmp_path, good_results()))
    report = good["report"]
    assert good["verdict"] == "PASS" and report["verdict"] == "PASS"
    assert report["efficiency"] == "UNMEASURED" and report["base_ppl"] == 9.8
    refused = judge_results(compress_block(budget_ppl_ratio=2.5),
                            _write(tmp_path, good_results(), "other.json"))
    assert refused["verdict"] == "REFUSED" and refused["report"] is None


def _patch(monkeypatch, **kw):
    for name, value in kw.items():
        for mod in {fsmod, sys.modules[fsmod.emit_compress.__module__]}:
            monkeypatch.setattr(mod, name, value, raising=False)


@pytest.fixture
def stage(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    (model / "a.safetensors").write_bytes(b"weights")
    text = tmp_path / "corpus.txt"
    text.write_bytes(b"rtn corpus bytes")
    worker, pre = tmp_path / "rtn_worker.py", tmp_path / "preflight.py"
    worker.write_text("")
    pre.write_text("")
    _patch(monkeypatch, WORKER=worker, PREFLIGHT=pre, import_gaps=lambda python=None: [])
    return model, text, tmp_path / "run" / "results.json"


def test_emit_compress_ready(stage):
    model, text, results = stage
    spec = fsmod.emit_compress(str(model), str(text), 4, 128, 1.15, str(results))
    assert spec["executable"] is True and spec["missing"] is None
    assert spec["entry"] == "fskills-compress"
    argv = spec["argv"]
    assert argv[argv.index("--bits") + 1] == "4" and str(fsmod.WORKER) in argv
    assert spec["compress"]["text_sha256"] == hashlib.sha256(text.read_bytes()).hexdigest()
    assert {"vision", "audio"} <= set(spec["compress"]["exempt"])


def test_emit_compress_never_overwrites(stage):
    model, text, results = stage
    results.parent.mkdir(parents=True)
    results.write_text("{}", encoding="utf-8")
    spec = fsmod.emit_compress(str(model), str(text), 4, 128, 1.15, str(results))
    assert spec["executable"] is False and "never overwritten" in spec["missing"]


@pytest.mark.parametrize("bits,group_size,frag", [(4, 100, "power of two"), (3, 128, "bits 3")])
def test_emit_compress_param_gaps(stage, bits, group_size, frag):
    model, text, results = stage
    spec = fsmod.emit_compress(str(model), str(text), bits, group_size, 1.15, str(results))
    assert spec["executable"] is False and frag in spec["missing"]


def test_preflight_argv():
    argv = fsmod.preflight_argv("/models/x", "/artifacts", 4)
    assert argv[argv.index("--planned-formats") + 1] == "rtnste_int4"
    assert argv[argv.index("--memory-profile") + 1] == "ptq"
    assert argv[argv.index("--out") + 1].endswith("cap.json")
    argv8 = fsmod.preflight_argv("/models/x", "/artifacts", 8)
    assert "--planned-formats" not in argv8
    assert argv8[argv8.index("--memory-profile") + 1] == "ptq"
    assert argv8[argv8.index("--out") + 1].endswith("cap.json")
