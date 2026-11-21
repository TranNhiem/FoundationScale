"""Unit tests for foundationskills.core.experience (C3 memory / P13)."""
import json
from datetime import datetime, timedelta, timezone

from foundationskills.core import experience

BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
LATER = BASE + timedelta(days=31)


def _row(idx=0, verdict="RED", rule_ids=("CP-HO-011",), action="a", skill="sk", now=BASE):
    row = experience.outcome_row(
        skill, "1.0", {"action": action, "workdir": f"/w{idx}"}, verdict,
        list(rule_ids), now=now,
    )
    row["run_id"] = f"run{idx:04d}"
    return row


def test_task_signature_ignores_volatile_and_paths():
    a = {"action": "cleanup", "workdir": "/tmp/eins", "created": 1, "out": "/o/x",
         "spec": {"output_dir": "/tmp/out", "gpu": 0}, "target": "~/files/x.py", "keep": 7}
    b = {"action": "cleanup", "workdir": "/tmp/zwei", "created": 9, "out": "/p/y",
         "spec": {"output_dir": "/opt/out", "gpu": 3}, "target": "/abs/y.py", "keep": 7}
    sa = experience.task_signature("sk", a)
    assert isinstance(sa, str) and len(sa) == 16
    assert sa == experience.task_signature("sk", b)


def test_task_signature_keeps_action_and_skill():
    a = {"action": "cleanup", "workdir": "/w1", "keep": 7}
    other_action = {"action": "rewrite", "workdir": "/w1", "keep": 7}
    other_keep = {"action": "cleanup", "workdir": "/w1", "keep": 8}
    assert experience.task_signature("sk", a) != experience.task_signature("sk", other_action)
    assert experience.task_signature("sk", a) != experience.task_signature("sk", other_keep)
    assert experience.task_signature("sk", a) != experience.task_signature("other", a)


def test_record_writes_to_workdir_bank(tmp_path, monkeypatch):
    monkeypatch.delenv("FSKILLS_EXPERIENCE_DIR", raising=False)
    wd = tmp_path / "wd"
    row = _row()
    assert experience.record(str(wd), False, row) is True
    bank = wd / ".fskills" / "experience.jsonl"
    assert bank.is_file()
    parsed = json.loads(bank.read_text().splitlines()[0])
    assert parsed["schema"] == experience.SCHEMA
    assert parsed["verdict"] == "RED" and parsed["measured"] is True
    assert parsed["expires_utc"] is None


def test_record_refused_writes_nothing_in_workdir(tmp_path, monkeypatch):
    monkeypatch.delenv("FSKILLS_EXPERIENCE_DIR", raising=False)
    wd = tmp_path / "wd"
    row = _row(verdict="REFUSED")
    assert experience.record(str(wd), True, row) in (True, False)
    assert not (wd / ".fskills" / "experience.jsonl").exists()


def test_record_refused_goes_to_shared_dir(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    wd = tmp_path / "wd"
    monkeypatch.setenv("FSKILLS_EXPERIENCE_DIR", str(shared))
    assert experience.record(str(wd), True, _row(verdict="REFUSED")) is True
    assert (shared / FILE_NAME).is_file() if False else (shared / experience.FILENAME).is_file()
    assert not (wd / ".fskills" / "experience.jsonl").exists()
    got = json.loads((shared / experience.FILENAME).read_text().splitlines()[0])
    assert got["verdict"] == "REFUSED" and got["measured"] is False
    assert got["expires_utc"] is None


def test_record_never_raises_on_unwritable_path(tmp_path, monkeypatch):
    blocker = tmp_path / "a_file_not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv("FSKILLS_EXPERIENCE_DIR", str(blocker / "sub"))
    row = _row(verdict="REFUSED")
    assert experience.record(str(blocker / "other"), True, row) in (True, False)
    monkeypatch.setenv("FSKILLS_EXPERIENCE_DIR", "")
    assert experience.record(None, False, row) is False


def test_read_drops_expired_unmeasured_and_torn_lines(tmp_path, monkeypatch):
    monkeypatch.delenv("FSKILLS_EXPERIENCE_DIR", raising=False)
    p = tmp_path / experience.FILENAME
    fresh = _row(1, verdict="UNMEASURED", rule_ids=(), now=BASE)
    expired = _row(2, verdict="UNMEASURED", rule_ids=(), now=BASE)
    red = _row(3, verdict="RED", now=BASE)
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(fresh) + "\n")
        fh.write(json.dumps(red) + "\nnot-json{{{\n")
        fh.write('{"schema": "fskills.experience/1", "to')
        fh.write("\n")
        fh.write('{"schema": "other/9", "verdict": "RED"}\n')
        fh.write(json.dumps({**expired, "expires_utc": (BASE).isoformat()}) + "\n")
    assert experience.read(p) == list(experience.read(p))  # smoke
    out = experience.read(p, now=LATER)
    assert len(out) == 1 and out[0]["verdict"] == "RED"
    both = experience.read(p, now=BASE)
    assert sorted(r["verdict"] for r in both) == ["RED", "UNMEASURED"]
    assert experience.read(tmp_path / "missing.jsonl") == []


def _lesson_rows(verdict="RED", n=3, skill="sk", action="a"):
    return [_row(i, verdict=verdict, skill=skill, action=action, now=BASE + timedelta(hours=i))
            for i in range(n)]


def test_lessons_needs_three_distinct_run_ids():
    two = _lesson_rows(n=2)
    assert experience.lessons(two) == []
    three = _lesson_rows(n=3)
    three[1]["run_id"] = three[0]["run_id"]  # only 2 distinct run ids
    assert experience.lessons(three) == []
    good = _lesson_rows(n=3)
    out = experience.lessons(good, min_support=3)
    assert len(out) == 1
    lesson = out[0]
    assert lesson["support"] == 3 and lesson["verdict"] == "RED"
    assert lesson["rule_ids"] == ["CP-HO-011"] and lesson["signature"] == good[0]["signature"]
    assert lesson["last_seen"] == max(r["ts_utc"] for r in good)


def test_lessons_ignores_unmeasured_and_refused():
    rows = _lesson_rows("UNMEASURED") + _lesson_rows("REFUSED")
    assert experience.lessons(rows) == []
    assert experience.lessons(_lesson_rows("PASS")) != []


def test_lessons_groups_by_signature_and_rules():
    rows = _lesson_rows(n=3) + [
        _row(10 + i, rule_ids=("CP-HO-011", "CP-HO-007"), now=BASE) for i in range(3)
    ]
    out = experience.lessons(rows)
    assert len(out) == 2
    keys = {(tuple(r["rule_ids"]), r["verdict"]) for r in out}
    assert keys == {(("CP-HO-007", "CP-HO-011"), "RED"), (("CP-HO-011",), "RED")}


def test_advise_wording_and_empty_for_other_signature():
    rows = _lesson_rows(n=3)
    msgs = experience.advise("sk", {"action": "a", "workdir": "/elsewhere"}, rows)
    assert len(msgs) == 1
    assert "advisory" in msgs[0] and "CP-HO-011" in msgs[0] and "RED" in msgs[0]
    assert "guarantee" not in msgs[0].lower()
    assert experience.advise("sk", {"action": "other", "workdir": "/w1"}, rows) == []
    assert experience.advise("nope", {"action": "a", "workdir": "/w1"}, rows) == []
    assert experience.advise("sk", {"action": "a"}, _lesson_rows(n=2)) == []


def test_execute_records_outcomes_without_changing_them(tmp_path, monkeypatch):
    monkeypatch.delenv("FSKILLS_EXPERIENCE_DIR", raising=False)
    from foundationskills.core.contract import SkillContext
    from foundationskills.skills.compression.skill import CompressionSkill

    work = tmp_path / "w"
    res = CompressionSkill().execute({"action": "quantize"}, SkillContext(workdir=work))
    assert res.status.value == "REFUSED" and not (work / ".fskills").exists()  # REFUSED writes nothing in the workdir
    shared = tmp_path / "shared"
    monkeypatch.setenv("FSKILLS_EXPERIENCE_DIR", str(shared))
    CompressionSkill().execute({"action": "quantize"}, SkillContext(workdir=work))
    rows = experience.read(shared / experience.FILENAME)
    assert len(rows) == 1 and rows[0]["verdict"] == "REFUSED" and rows[0]["measured"] is False
