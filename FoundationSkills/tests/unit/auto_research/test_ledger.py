"""Ledger tests: canonical hashing, the append-only chain, verification and the derived TSV view."""
from __future__ import annotations

from datetime import datetime

import pytest

import foundationskills.skills.auto_research.ledger as ledger_mod
from foundationskills.skills.auto_research.ledger import GENESIS, Ledger, canonical, ledger_files, sha256_hex


def _payload(value: float, trial: str = "t1") -> dict:
    return {"trial": trial, "seed": 1, "metrics": {"val_accuracy": {"value": value, "se": 0.001}}}


class TestCanonical:
    def test_key_order_independent(self):
        assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})
        assert canonical({"a": 1}) == b'{"a":1}'

    def test_refuses_nan_and_inf(self):
        with pytest.raises(ValueError):
            canonical({"x": float("nan")})
        with pytest.raises(ValueError):
            canonical({"x": float("inf")})


class TestAppend:
    def test_entries_chain_hashes(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        first = ledger.append("trial_result", "c1", "t1", _payload(0.5))
        second = ledger.append("trial_result", "c1", "t1", _payload(0.6))
        assert first["seq"] == 0 and second["seq"] == 1
        assert first["prev_hash"] == GENESIS and second["prev_hash"] == first["hash"]
        assert ledger.head() == {"count": 2, "head_hash": second["hash"]}
        assert ledger.verify() == []
        assert [e["op"] for e in ledger.entries()] == ["trial_result", "trial_result"]

    def test_payload_is_content_addressed(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        first = ledger.append("trial_result", "c1", "t1", _payload(0.5))
        second = ledger.append("trial_result", "c1", "t2", _payload(0.5))  # identical payload bytes
        assert second["payload_hash"] == first["payload_hash"]
        assert ledger.payload(first) == _payload(0.5)

    def test_results_and_launches_filter_by_op_and_campaign(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        ledger.append("campaign_approved", "c1", "-", {"spec_hash": "h"})
        ledger.append("launch_authorised", "c1", "t1", {"gpu_hours_est": 4.0})
        ledger.append("trial_result", "c1", "t1", _payload(0.5))
        ledger.append("trial_result", "c2", "t9", _payload(0.9, trial="t9"))
        assert len(ledger.results("c1")) == 1 and ledger.results("c1")[0]["trial"] == "t1"
        assert len(ledger.launches("c1")) == 1
        assert len(ledger.results("c2")) == 1 and ledger.results("c2")[0]["trial"] == "t9"


class TestLedgerFiles:
    TS = "2026-01-01T00:00:00Z"
    EVENTS = [
        ("campaign_approved", "c1", "-", {"spec_hash": "h", "approver": "a"}),
        ("trial_result", "c1", "t1", {"trial": "t1", "metrics": {"val_accuracy": {"value": 0.5, "se": 0.001}}}),
        ("trial_result", "c1", "t2", {"trial": "t2", "metrics": {"val_accuracy": {"value": 0.6, "se": 0.001}}}),
    ]

    def test_bytes_match_ledger_append(self, tmp_path, monkeypatch):
        class Clock:
            @staticmethod
            def now(tz=None):
                return datetime(2026, 1, 1, tzinfo=tz)

        monkeypatch.setattr(ledger_mod, "datetime", Clock)  # freeze append()'s clock on the fixture ts
        live = Ledger(tmp_path / "live")
        for event in self.EVENTS:
            live.append(*event)
        files = ledger_files(list(self.EVENTS), ts=self.TS)
        assert files["ledger/chain.jsonl"] == live.chain.read_text(encoding="utf-8")
        for rel, text in files.items():
            if rel != "ledger/chain.jsonl":
                assert (tmp_path / "live" / rel.split("/", 1)[1]).read_text(encoding="utf-8") == text

    def test_written_archive_is_intact_and_readable(self, tmp_path):
        for rel, text in ledger_files(list(self.EVENTS), ts=self.TS).items():
            path = tmp_path / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        ledger = Ledger(tmp_path / "ledger")
        assert ledger.verify() == []
        assert ledger.head()["count"] == 3
        assert [e["ts"] for e in ledger.entries()] == [self.TS] * 3
        assert ledger.results("c1")[0]["trial"] == "t1"
        assert ledger.payload(ledger.entries()[1]) == self.EVENTS[1][3]

    def test_tamper_flips_the_last_objects_first_byte(self):
        clean = ledger_files(list(self.EVENTS), ts=self.TS, tamper=False)
        dirty = ledger_files(list(self.EVENTS), ts=self.TS, tamper=True)
        key = f"ledger/objects/{sha256_hex(canonical(self.EVENTS[-1][3]))}.json"
        assert set(clean) == set(dirty)
        assert dirty[key][0] == "X" and dirty[key][1:] == clean[key][1:]
        assert dirty["ledger/chain.jsonl"] == clean["ledger/chain.jsonl"]


class TestVerify:
    def _ledger(self, tmp_path) -> Ledger:
        ledger = Ledger(tmp_path / "ledger")
        ledger.append("trial_result", "c1", "t1", _payload(0.5))
        ledger.append("trial_result", "c1", "t1", _payload(0.6))
        return ledger

    def test_detects_object_tamper(self, tmp_path):
        ledger = self._ledger(tmp_path)
        obj = ledger.objects / f"{ledger.entries()[0]['payload_hash']}.json"
        obj.write_bytes(obj.read_bytes().replace(b"0.5", b"9.5"))
        problems = ledger.verify()
        assert problems and "seq 0" in problems[0] and "payload hash mismatch" in problems[0]

    def test_detects_chain_field_tamper(self, tmp_path):
        ledger = self._ledger(tmp_path)
        text = ledger.chain.read_text(encoding="utf-8")
        ledger.chain.write_text(text.replace('"trial":"t1"', '"trial":"tX"'), encoding="utf-8")
        problems = ledger.verify()
        assert problems and "seq 0" in problems[0] and "hash mismatch" in problems[0]

    def test_detects_bad_json_line(self, tmp_path):
        ledger = self._ledger(tmp_path)
        with open(ledger.chain, "a", encoding="utf-8") as fh:
            fh.write("not json\n")
        problems = ledger.verify()
        assert problems and "bad JSON line" in problems[0] and "seq 2" in problems[0]

    def test_detects_seq_gap(self, tmp_path):
        ledger = self._ledger(tmp_path)
        lines = ledger.chain.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace('"seq":1', '"seq":5', 1)
        ledger.chain.write_text("\n".join(lines) + "\n", encoding="utf-8")
        problems = ledger.verify()
        assert problems and "seq 1" in problems[0] and "sequence gap" in problems[0]

    def test_detects_prev_hash_mismatch(self, tmp_path):
        ledger = self._ledger(tmp_path)
        lines = ledger.chain.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace(f'"prev_hash":"{ledger.entries()[0]["hash"]}"', f'"prev_hash":"{GENESIS}"', 1)
        ledger.chain.write_text("\n".join(lines) + "\n", encoding="utf-8")
        problems = ledger.verify()
        assert problems and "seq 1" in problems[0] and "prev_hash mismatch" in problems[0]

    def test_detects_missing_object(self, tmp_path):
        ledger = self._ledger(tmp_path)
        (ledger.objects / f"{ledger.entries()[1]['payload_hash']}.json").unlink()
        problems = ledger.verify()
        assert problems and "seq 1" in problems[0] and "missing object" in problems[0]


class TestDerivedViews:
    def test_crash_rows_show_empty_value_never_zero(self, tmp_path):
        ledger = Ledger(tmp_path / "ledger")
        ledger.append("trial_result", "c1", "ok-trial", _payload(0.5))
        ledger.append(
            "trial_result", "c1", "boom",
            {"trial": "boom", "role": "candidate", "seed": 7, "status": "crash", "metrics": {}},
        )
        rows = ledger.tsv_view("c1").splitlines()  # unstripped: the crash row's empty columns are the evidence
        assert rows[0] == "seq\ttrial\trole\tseed\tstatus\tmetric\tvalue\tse"
        assert len(rows) == 3
        crash = rows[2].split("\t")
        assert crash[1] == "boom" and crash[4] == "crash"
        assert crash[5] == "" and crash[6] == "" and crash[7] == ""
        assert "0.0" not in rows[2]
