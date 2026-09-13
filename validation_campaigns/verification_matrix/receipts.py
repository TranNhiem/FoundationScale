# validation_campaigns/verification_matrix/receipts.py
"""The WRITER and the READER of verification-matrix receipts -- one shape, so
all five adjudicators and the gate never disagree about what a result is.

matrix.json is the DECLARATION and stays that way. A file that is both the
claim and its own result is self-certifying: an edit to the claim would
silently re-label an old measurement as evidence for a new claim. Results
therefore live in per-pass receipt files under

    validation_campaigns/verification_matrix/receipts/<row_id>/<pass_id>.json

A receipt records what was true WHEN IT WAS WRITTEN, including the things
that can later drift out from under it: the claim text and its sha256, and
the adjudicator path and the sha256 of its bytes. Reading re-derives every
verdict from the receipts on every call -- no count of greens is ever stored
anywhere, because a stored count is a claim about the receipts that can
drift away from them. A receipt whose recorded hashes no longer match the
row it claims to measure is STALE: it reads UNMEASURED, with a reason that
names the drift. Absent receipts read UNMEASURED too -- "we did not measure
this" and "this failed" are different states, and the plane already has
codes for both.

No signing, no crypto beyond hashlib.sha256, stdlib only: a public
repository, and a gate machine with no torch and no GPU.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["VERDICTS", "ReceiptError", "write_receipt", "read_receipts", "row_verdict"]

# Exit code -> verdict string. The four-state contract is the only vocabulary
# a receipt may speak: 0 the control upheld the claim, 5 the control refuted
# it, 95 the deciding quantity could not be measured, 96 the program refused
# to run at all. Anything else is not a verdict; it is a crash wearing one.
VERDICTS: dict[int, str] = {0: "green", 5: "red", 95: "unmeasured", 96: "refused"}

_VALID_VERDICTS = frozenset(VERDICTS.values())

# Every receipt MUST carry every one of these keys. A file missing any of
# them is not a partial receipt; it is a malformed one, and malformed
# evidence refuses -- it never silently drops out of the denominator.
_REQUIRED_KEYS: tuple[str, ...] = (
    "row_id",
    "pass_id",
    "verdict",
    "reason",
    "claim_text",
    "claim_sha256",
    "adjudicator",
    "adjudicator_sha256",
    "arms",
    "interpreter",
    "platform",
    "written_at_utc",
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ReceiptError(ValueError):
    """Raised when a receipt file exists but cannot be trusted as evidence.

    This is deliberately a hard failure, not a skip: a receipt that cannot
    be read is not a receipt, and pretending it was never there would shrink
    the measured denominator without telling anyone.
    """


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _claim_hash(claim: str) -> str:
    return _sha256_hex(claim.encode("utf-8"))


def _check_path_component(value: Any, what: str) -> str:
    """Receipts are addressed by <row_id>/<pass_id>.json; a component that
    can walk the tree (``..``, separators) would let a caller write evidence
    somewhere it does not belong, so it is rejected outright."""
    if (
        not isinstance(value, str)
        or not value.strip()
        or value in (".", "..")
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(
            f"{what} {value!r} is not a usable path component; receipts are addressed "
            "by a bare row id and a bare pass id with no separators"
        )
    return value


def _parse_iso8601(text: Any, what: str) -> datetime:
    """Timestamps are parsed, never generated, in this module: the caller
    passes written_at_utc in, so the writer stays testable and a receipt
    cannot secretly borrow 'whenever the test ran'. Naive stamps are read as
    UTC so ordering never raises on a mixed tree."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{what}: written_at_utc must be a non-empty ISO-8601 string")
    candidate = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ValueError(
            f"{what}: written_at_utc {text!r} is not parseable ISO-8601; a receipt's "
            "timestamp must order it against later passes"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _row_fields(row: Mapping[str, Any]) -> tuple[str, str]:
    if not isinstance(row, Mapping):
        raise ValueError(f"a matrix row is a mapping, got {type(row).__name__}")
    row_id = row.get("id")
    claim = row.get("claim")
    if not isinstance(row_id, str) or not row_id.strip():
        raise ValueError("the row must carry a non-empty string 'id'")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError(
            f"row {row_id!r} must carry non-empty claim text; a receipt hashes the "
            "claim, and there is nothing to hash here"
        )
    return row_id, claim


def _stale_fragment(
    receipt: Mapping[str, Any],
    *,
    claim: str,
    current_claim_sha: str,
    adjudicator_path: str | None,
    current_adj_sha: str | None,
) -> str | None:
    """None if this receipt still measures the row as it reads today; else a
    sentence naming the drift. Naming is the entire point of recording the
    hashes: 'stale' without WHICH input moved is a shrug, not a diagnosis."""
    drifts: list[str] = []
    if receipt.get("claim_sha256") != current_claim_sha:
        drifts.append(
            "claim drift: the receipt hashed the claim as "
            f"{str(receipt.get('claim_sha256'))[:12]}... ({str(receipt.get('claim_text'))[:70]!r}) "
            f"but the row's claim now reads {claim[:70]!r} ({current_claim_sha[:12]}...); the "
            "measurement belongs to a claim that no longer reads this way"
        )
    rec_adj = receipt.get("adjudicator")
    rec_sha = receipt.get("adjudicator_sha256")
    if rec_adj != adjudicator_path or rec_sha != current_adj_sha:
        if rec_sha != current_adj_sha and rec_sha is not None and current_adj_sha is not None:
            drifts.append(
                f"adjudicator drift: the receipt recorded {rec_adj} at "
                f"{str(rec_sha)[:12]}... but its current bytes hash {current_adj_sha[:12]}...; "
                "the program that decides this row changed after the run"
            )
        elif current_adj_sha is None and (rec_sha is not None or rec_adj is not None):
            drifts.append(
                f"adjudicator drift: the receipt names {rec_adj}, but the row now names "
                f"{adjudicator_path!r}; which program decides this row changed after the run"
            )
        else:
            drifts.append(
                f"adjudicator drift: the receipt recorded no adjudicator hash but the row "
                f"now names {adjudicator_path!r}"
            )
    if not drifts:
        return None
    return f"pass {receipt.get('pass_id')!r}: " + "; ".join(drifts)


def write_receipt(
    *,
    receipts_root: str | Path,
    row: Mapping[str, Any],
    pass_id: str,
    exit_code: int,
    reason: str | None,
    arms: Mapping[str, Any],
    adjudicator_path: str | None,
    adjudicator_bytes: bytes | None,
    written_at_utc: str,
) -> Path:
    """Write one pass's verdict as the ONLY shape the reader accepts.

    WHY this function is fussy: the receipt is the campaign's whole
    evidentiary claim. A writer that tolerates a missing reason, an exit code
    outside 0/5/95/96, or an adjudicator path without the bytes it hashed
    produces evidence that fails loudly only later, on the gate, after the
    run that produced it is gone. Failing here instead keeps the failure next
    to its cause. Existing files are never overwritten: a receipt is the
    record of an event, and rewriting one would quietly change what was true
    when it was written.
    """
    if exit_code not in VERDICTS:
        raise ValueError(
            f"exit_code {exit_code!r} is outside the four-state contract "
            f"{sorted(VERDICTS)}; a crash is not a verdict and must not be recorded as one"
        )
    verdict = VERDICTS[exit_code]
    if reason is not None and not isinstance(reason, str):
        raise ValueError("reason must be a sentence or null")
    if verdict != "green" and not (isinstance(reason, str) and reason.strip()):
        raise ValueError(
            f"a {verdict} verdict MUST carry a reason sentence: the operator learns WHY "
            "from this string, not by re-running"
        )
    row_id, claim = _row_fields(row)
    _check_path_component(row_id, "row id")
    _check_path_component(pass_id, "pass id")
    if not isinstance(arms, Mapping):
        raise ValueError("arms must be a mapping of arm-name -> measured scalars")
    for arm_name in arms:
        if not isinstance(arm_name, str) or not arm_name.strip():
            raise ValueError(f"arm names must be non-empty strings, got {arm_name!r}")
    try:
        json.dumps(arms, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"arms must be JSON-serializable measured scalars: {exc}") from exc
    if adjudicator_path is None:
        if adjudicator_bytes is not None:
            raise ValueError(
                "adjudicator bytes without a path records evidence against an unnamed "
                "program; name the program or pass neither"
            )
        adjudicator_sha256: str | None = None
    else:
        if not isinstance(adjudicator_path, str) or not adjudicator_path.strip():
            raise ValueError("adjudicator_path must be a repo-relative path string or None")
        if not isinstance(adjudicator_bytes, (bytes, bytearray)):
            raise ValueError(
                "the receipt MUST hash the adjudicator's bytes as they were at run time; "
                "a path without its bytes cannot later prove which program decided the row"
            )
        adjudicator_sha256 = _sha256_hex(bytes(adjudicator_bytes))
    _parse_iso8601(written_at_utc, "write_receipt")

    payload = {
        "row_id": row_id,
        "pass_id": pass_id,
        "verdict": verdict,
        "reason": reason,
        "claim_text": claim,
        "claim_sha256": _claim_hash(claim),
        "adjudicator": adjudicator_path,
        "adjudicator_sha256": adjudicator_sha256,
        "arms": dict(arms),
        "interpreter": sys.version,
        "platform": platform.machine(),
        "written_at_utc": written_at_utc,
    }
    row_dir = Path(receipts_root) / row_id
    path = row_dir / f"{pass_id}.json"
    if path.exists():
        raise FileExistsError(
            f"receipt {path} already exists; a receipt records what was true when it was "
            "written, and overwriting it would change the past. Give the new pass its own "
            "pass_id."
        )
    row_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_receipts(receipts_root: str | Path, row_id: str) -> list[dict]:
    """Read every receipt a row owns, oldest first, or raise naming the file.

    WHY sorted and strict: the reader is where the aggregate is re-derived,
    so its order must not depend on the filesystem, and any file it cannot
    trust -- bad JSON, a missing required key, a timestamp that will not
    parse, a pass_id that disagrees with its own filename -- is a
    ReceiptError naming the path. Silently skipping such a file would shrink
    the measured denominator without telling anyone, which is exactly the
    failure receipts exist to prevent. An absent directory is different: it
    means the row was never run, and that reads as an empty list so
    row_verdict can say UNMEASURED.
    """
    _check_path_component(row_id, "row id")
    directory = Path(receipts_root) / row_id
    if not directory.is_dir():
        return []
    receipts: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        receipts.append(_read_one(path, expected_row_id=row_id))
    receipts.sort(key=lambda r: (_parse_iso8601(r["written_at_utc"], str(r)), str(r["pass_id"])))
    return receipts


def _read_one(path: Path, *, expected_row_id: str) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReceiptError(f"{path}: unreadable: {exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReceiptError(f"{path}: not JSON ({exc}); this file is not a receipt") from exc
    if not isinstance(data, dict):
        raise ReceiptError(
            f"{path}: top level is {type(data).__name__}, not an object; this file is not a receipt"
        )
    missing = [key for key in _REQUIRED_KEYS if key not in data]
    if missing:
        raise ReceiptError(
            f"{path}: missing required key(s) {missing}; a partial receipt is not evidence"
        )
    if data["row_id"] != expected_row_id:
        raise ReceiptError(
            f"{path}: row_id {data['row_id']!r} does not match its directory "
            f"{expected_row_id!r}; a receipt must live under the row it measures"
        )
    if not isinstance(data["pass_id"], str) or data["pass_id"] != path.stem:
        raise ReceiptError(
            f"{path}: pass_id {data['pass_id']!r} does not match its filename; a receipt "
            "must live at the pass it names"
        )
    if data["verdict"] not in _VALID_VERDICTS:
        raise ReceiptError(
            f"{path}: verdict {data['verdict']!r} is outside {sorted(_VALID_VERDICTS)}; "
            "the contract has no such state"
        )
    if data["reason"] is not None and not isinstance(data["reason"], str):
        raise ReceiptError(f"{path}: reason must be a sentence or null")
    if data["verdict"] != "green" and not (
        isinstance(data["reason"], str) and data["reason"].strip()
    ):
        raise ReceiptError(
            f"{path}: a {data['verdict']} verdict without a reason says nothing"
        )
    if not isinstance(data["claim_text"], str):
        raise ReceiptError(f"{path}: claim_text must be the claim as it read at run time")
    if not isinstance(data["claim_sha256"], str) or not _SHA256_RE.fullmatch(
        data["claim_sha256"]
    ):
        raise ReceiptError(f"{path}: claim_sha256 is not a lowercase sha256 hex digest")
    if data["adjudicator"] is not None and not isinstance(data["adjudicator"], str):
        raise ReceiptError(f"{path}: adjudicator must be a repo-relative path string or null")
    if data["adjudicator_sha256"] is not None and (
        not isinstance(data["adjudicator_sha256"], str)
        or not _SHA256_RE.fullmatch(data["adjudicator_sha256"])
    ):
        raise ReceiptError(f"{path}: adjudicator_sha256 is not a lowercase sha256 hex digest")
    if not isinstance(data["arms"], dict):
        raise ReceiptError(f"{path}: arms must be a mapping of arm-name -> measured scalars")
    for key in ("interpreter", "platform", "written_at_utc"):
        if not isinstance(data[key], str) or not data[key].strip():
            raise ReceiptError(f"{path}: {key} must be a non-empty string")
    try:
        _parse_iso8601(data["written_at_utc"], str(path))
    except ValueError as exc:
        raise ReceiptError(str(exc)) from exc
    return data


def row_verdict(
    *,
    row: Mapping[str, Any],
    receipts: list[dict],
    adjudicator_bytes: bytes | None,
) -> tuple[str, str]:
    """Reduce one row to a single (verdict, reason), re-derived every call.

    WHY these rules: the aggregate is never stored, so this function IS the
    aggregate's source.

    * Rule 3 -- a row whose "adjudicator" names a path with no bytes behind
      it (the gate passes None for bytes it could not read off disk) REFUSES
      96; naming a decider that does not exist is a broken declaration, not
      a failed measurement.
    * Rule 5 -- no receipts reads UNMEASURED with a reason that says so;
      never green, never red. "We did not run this" must stay a different
      sentence from "this failed".
    * Rule 6 -- a receipt whose claim_sha256 no longer matches the row's
      current claim, or whose adjudicator_sha256 no longer matches the
      adjudicator's current bytes, is STALE: it is skipped over, and if no
      non-stale receipt stands, the row reads UNMEASURED with a reason
      naming the drift (claim drift vs adjudicator drift). The freshest
      non-stale receipt decides the row; older passes are history, not the
      verdict.
    """
    row_id, claim = _row_fields(row)
    adjudicator_path = row.get("adjudicator")
    if adjudicator_path is not None:
        if not isinstance(adjudicator_path, str) or not adjudicator_path.strip():
            raise ValueError(
                f"row {row_id!r}: 'adjudicator' must be a repo-relative path string or null"
            )
        if adjudicator_bytes is None:
            return (
                "refused",
                f"row {row_id} names adjudicator {adjudicator_path!r}, but no such file "
                "exists; a row whose decider is missing REFUSES 96 -- it is not a red, "
                "because nothing was measured and nothing failed",
            )
    current_claim_sha = _claim_hash(claim)
    current_adj_sha = (
        _sha256_hex(bytes(adjudicator_bytes)) if adjudicator_bytes is not None else None
    )
    stale_notes: list[str] = []
    ordered = sorted(
        receipts,
        key=lambda r: (_parse_iso8601(r["written_at_utc"], "row_verdict"), str(r["pass_id"])),
    )
    for receipt in reversed(ordered):
        fragment = _stale_fragment(
            receipt,
            claim=claim,
            current_claim_sha=current_claim_sha,
            adjudicator_path=adjudicator_path,
            current_adj_sha=current_adj_sha,
        )
        if fragment is not None:
            stale_notes.append(fragment)
            continue
        verdict = receipt["verdict"]
        reason = receipt.get("reason")
        if isinstance(reason, str) and reason.strip():
            return verdict, reason
        return verdict, f"pass {receipt['pass_id']!r} recorded {verdict}"
    if stale_notes:
        return (
            "unmeasured",
            f"row {row_id} reads UNMEASURED: every receipt is stale -- "
            + "; ".join(stale_notes)
            + "; re-run the adjudicator so a receipt measures the claim as it reads now",
        )
    return (
        "unmeasured",
        f"row {row_id} has no receipts: it has never been measured, and 'never "
        "measured' is neither green nor red",
    )
