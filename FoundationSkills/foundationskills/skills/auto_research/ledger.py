"""Append-only, hash-chained evidence ledger for auto_research campaigns.

``chain.jsonl`` carries one hash-linked entry per event and ``objects/<sha256>.json``
carries the content-addressed payload bytes. There is NO update/delete API: evidence
is immutable, every chain field is covered by the entry hash, and derived views
(``tsv_view``) are regenerated from the chain and never read back. ``ledger_files``
packages exactly the same bytes as pure text (fixture archives, no I/O).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
_HASH_FIELDS = ("seq", "ts", "op", "campaign", "trial", "payload_hash", "prev_hash")
_OPS = ("campaign_approved", "launch_authorised", "trial_result", "campaign_closed")


def canonical(obj: Any) -> bytes:
    """Deterministic JSON bytes for hashing/signing; NaN/inf are refused (ValueError)."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(
    seq: int, ts: str, op: str, campaign: str, trial: str, payload_hash: str, prev_hash: str
) -> dict[str, Any]:
    """One hash-linked chain record: ``hash`` covers exactly the canonical hash fields."""
    record: dict[str, Any] = {
        "seq": seq,
        "ts": ts,
        "op": op,
        "campaign": campaign,
        "trial": trial,
        "payload_hash": payload_hash,
        "prev_hash": prev_hash,
    }
    return {**record, "hash": sha256_hex(canonical({k: record[k] for k in _HASH_FIELDS}))}


def _line(entry: dict[str, Any]) -> str:
    """The exact ``chain.jsonl`` line for one entry: the shape every writer emits."""
    return json.dumps(entry, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def ledger_files(
    events: list[tuple[str, str, str, dict[str, Any]]],
    root: str = "ledger",
    ts: str = "2026-01-01T00:00:00Z",
    tamper: bool = False,
) -> dict[str, str]:
    """{relative path: text} for the ledger of ``(op, campaign, trial, payload)`` events (pure, no I/O).

    Byte-identical to what ``Ledger.append`` writes (same canonical, hash fields and line format)
    except the fixed ``ts``, so fixture hashes stay stable. ``tamper=True`` flips the first byte of
    the LAST event's object text (a documented fixture-only knob for the AR-HO-003 fixture).
    """
    objects: dict[str, str] = {}
    lines: list[str] = []
    prev_hash, last_hash = GENESIS, ""
    for seq, (op, campaign, trial, payload) in enumerate(events):
        body = canonical(payload)
        payload_hash = sha256_hex(body)
        objects.setdefault(payload_hash, body.decode("utf-8"))
        entry = _entry(seq, ts, op, campaign, trial, payload_hash, prev_hash)
        lines.append(_line(entry))
        prev_hash, last_hash = str(entry["hash"]), payload_hash
    if tamper and last_hash:
        text = objects[last_hash]
        objects[last_hash] = ("X" if text[:1] == "{" else "{") + text[1:]
    files = {f"{root}/objects/{digest}.json": text for digest, text in objects.items()}
    files[f"{root}/chain.jsonl"] = "".join(line + "\n" for line in lines)
    return files


class Ledger:
    """Append-only hash chain over one campaign directory (chain.jsonl + objects/)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.chain = self.root / "chain.jsonl"
        self.objects = self.root / "objects"

    # ---- write (append only) ---------------------------------------------

    def append(self, op: str, campaign: str, trial: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append one hash-linked entry and its content-addressed payload object."""
        body = canonical(payload)
        payload_hash = sha256_hex(body)
        self.objects.mkdir(parents=True, exist_ok=True)
        obj_path = self.objects / f"{payload_hash}.json"
        if not obj_path.exists():
            obj_path.write_bytes(body)
        entries = self.entries()
        entry = _entry(
            len(entries),
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            op,
            campaign,
            trial,
            payload_hash,
            entries[-1].get("hash", GENESIS) if entries else GENESIS,
        )
        with open(self.chain, "a", encoding="utf-8") as fh:
            fh.write(_line(entry) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return entry

    # ---- read -------------------------------------------------------------

    def _lines(self) -> list[str]:
        if not self.chain.exists():
            return []
        return self.chain.read_text(encoding="utf-8").splitlines()

    def entries(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for line in self._lines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # verify() reports malformed lines by their expected seq
        return out

    def payload(self, entry: dict[str, Any]) -> dict[str, Any]:
        """The content-addressed payload object behind one chain entry."""
        path = self.objects / f"{entry['payload_hash']}.json"
        return json.loads(path.read_bytes().decode("utf-8"))

    def head(self) -> dict[str, Any]:
        entries = self.entries()
        return {"count": len(entries), "head_hash": entries[-1].get("hash", GENESIS) if entries else GENESIS}

    def results(self, campaign: str) -> list[dict[str, Any]]:
        return [
            self.payload(e)
            for e in self.entries()
            if e.get("op") == "trial_result" and e.get("campaign") == campaign
        ]

    def launches(self, campaign: str) -> list[dict[str, Any]]:
        return [
            self.payload(e)
            for e in self.entries()
            if e.get("op") == "launch_authorised" and e.get("campaign") == campaign
        ]

    def tsv_view(self, campaign: str) -> str:
        """NeMo-RL styled derived table (seq/trial/role/seed/status/metric/value/se). Never read back."""
        rows = ["seq\ttrial\trole\tseed\tstatus\tmetric\tvalue\tse"]
        for entry in self.entries():
            if entry.get("op") != "trial_result" or entry.get("campaign") != campaign:
                continue
            payload = self.payload(entry)
            status = str(payload.get("status", ""))
            head = f"{entry.get('seq')}\t{payload.get('trial')}\t{payload.get('role')}\t{payload.get('seed')}\t{status}"
            metrics = dict(payload.get("metrics") or {})
            if status == "crash" or not metrics:
                rows.append(f"{head}\t\t\t")  # crash rows show an empty value, never 0.0
                continue
            for name in sorted(metrics):
                point = dict(metrics.get(name) or {})
                value = "" if point.get("value") is None else str(point["value"])
                se = "" if point.get("se") is None else str(point["se"])
                rows.append(f"{head}\t{name}\t{value}\t{se}")
        return "\n".join(rows) + "\n"

    # ---- integrity --------------------------------------------------------

    def verify(self) -> list[str]:
        """Problems in the chain; [] means intact. Every problem names the first broken seq."""
        problems: list[str] = []
        expected_seq = 0
        expected_prev = GENESIS
        for idx, line in enumerate(self._lines()):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"seq {expected_seq}: bad JSON line {idx + 1}")
                break
            seq = entry.get("seq")
            if seq != expected_seq:
                problems.append(f"seq {expected_seq}: sequence gap (found seq {seq!r})")
                break
            if entry.get("prev_hash") != expected_prev:
                problems.append(
                    f"seq {seq}: prev_hash mismatch (expected {expected_prev}, found {entry.get('prev_hash')})"
                )
                break
            body = {k: entry.get(k) for k in _HASH_FIELDS}
            if sha256_hex(canonical(body)) != entry.get("hash"):
                problems.append(f"seq {seq}: hash mismatch (entry fields no longer hash to {entry.get('hash')})")
                break
            payload_hash = str(entry.get("payload_hash", ""))
            obj_path = self.objects / f"{payload_hash}.json"
            if not obj_path.is_file():
                problems.append(f"seq {seq}: missing object {payload_hash}")
            else:
                actual = sha256_hex(obj_path.read_bytes())
                if actual != payload_hash:
                    problems.append(f"seq {seq}: payload hash mismatch ({payload_hash} != {actual})")
            expected_seq = seq + 1
            expected_prev = str(entry.get("hash"))
        return problems
