"""Drop every row of the given calls from a training manifest (counted), so they stay held out.

Usage: filter_calls.py IN.jsonl OUT.jsonl --calls-from longform_earnings22_dev6.json
Rows without a source_id (LibriSpeech, AMI) pass through untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--calls-from", required=True)
    args = ap.parse_args()
    calls = {json.loads(line)["id"] for line in Path(args.calls_from).read_text().splitlines() if line.strip()}
    rows = [json.loads(line) for line in Path(args.src).read_text().splitlines() if line.strip()]
    kept = [r for r in rows if str(r.get("source_id")) not in calls]
    Path(args.dst).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(json.dumps({"rows_in": len(rows), "rows_dropped": len(rows) - len(kept), "held_out_calls": sorted(calls)}))


if __name__ == "__main__":
    main()
