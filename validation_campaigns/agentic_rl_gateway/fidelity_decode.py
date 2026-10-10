"""Decode each sampled span and each context span of saved gateway episodes.

Usage (inside a container with transformers): fidelity_decode.py <model_dir> <probe_out_dir>
Sampled spans must be the model's own output verbatim (reasoning, tool-call markup, end of
turn); context spans must be only harness-supplied text (tool results, next header).
"""

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1])
for path in sorted(Path(sys.argv[2]).glob("episode_*.json")):
    with path.open() as fh:
        seg = json.load(fh)["segments"][0]
    resp, mask = seg["response_ids"], seg["loss_mask"]
    print("==", path.name)
    for start, end, gid in seg["generation_spans"]:
        print("  sampled", gid, repr(tok.decode(resp[start:end], skip_special_tokens=False))[:160])
    ctx = [i for i, bit in enumerate(mask) if bit == 0]
    if ctx:
        print(
            "  context ",
            repr(tok.decode(resp[ctx[0] : ctx[-1] + 1], skip_special_tokens=False))[:160],
        )
