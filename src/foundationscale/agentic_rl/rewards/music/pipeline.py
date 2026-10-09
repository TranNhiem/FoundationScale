# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from XiaomiMiMo/verl recipes/design/music/scorer/pipeline.py (commit a2ad9f61).
# Modifications Copyright (c) 2026 TranNhiem, licensed under the MIT License (see LICENSE).
# See THIRD_PARTY_NOTICES.md.
"""Entry point: ABC extraction + two-gate scoring for verl RL.

Ported from the upstream ``scorer/pipeline.py`` named in the header above with
the scoring arithmetic, constants and feature names kept verbatim. Three
behavioural edits were made, all declared (none hidden in a diff):

1. No environment-variable reads. Upstream read ``ABC2MIDI_BIN`` at import
   time; here the ``abc2midi`` binary path and the subprocess timeout are
   explicit parameters of :func:`do` (``timeout_s`` defaults to 60.0, matching
   upstream's hard-coded ``timeout=60``).
2. :func:`compute_score` keeps upstream's ``traceback.print_exc()`` swallow
   (it is upstream's public API, preserved for parity) -- the FoundationScale
   public adapter is :class:`~foundationscale.agentic_rl.rewards.music.MusicReward`
   in this package's ``__init__.py``, which does NOT swallow unexpected
   exceptions via ``print_exc``; it lets them propagate.
3. The baseline is :data:`~foundationscale.agentic_rl.rewards.music.baseline.REF_FULL4K`,
   the embedded Python literal, rather than a JSON file read from disk.

One further, NEW exception type, :class:`Abc2MidiError`, is added alongside
the existing :class:`Abc2MidiMissing`. Upstream's ``do`` folded every
subprocess failure other than "binary missing" into a model-attributable
``skip`` (scored 0.0) -- including an ``abc2midi`` call that timed out or
failed with some other OS-level error (permissions, resource limits...).
FoundationScale's None-abstention policy does not accept that: an
infrastructure fault must never be reported as "the policy wrote bad music",
so :func:`do` now raises :class:`Abc2MidiError` for a timeout or any other
``OSError`` (``FileNotFoundError`` -- binary missing -- is still its own
case), and the ``MusicReward`` adapter maps it to ``value=None``. This is a
deliberate divergence from upstream, which scored those cases 0.0; it touches
only the shape of error propagation out of the one subprocess call site and
no feature arithmetic.
"""

from __future__ import annotations

import re
import struct
import subprocess as sp
import tempfile
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .baseline import REF_FULL4K
from .feats import analyze
from .score import score as _score_v1

_FENCE_RE = re.compile(r"```(?:abc|ABC)?\s*\n(.*?)```", re.DOTALL)
_XA_RE = re.compile(r"X\s*:\s*\d+")


class Abc2MidiMissing(RuntimeError):
    """The `abc2midi` binary is absent from this process's PATH.

    Kept distinct from every other scoring failure because it is the one that
    must not be reported as a reward of 0.0: it applies to every sample rather
    than to one score, so it is indistinguishable from "the policy writes bad
    music" once flattened.
    """


class Abc2MidiError(RuntimeError):
    """``abc2midi`` ran but failed for a reason attributable to the execution
    environment rather than to the policy's ABC text: an ``OSError`` other
    than "binary missing", or the subprocess exceeding its timeout.

    See the module docstring for why this is a deliberate addition over
    upstream, which folded these into a model-attributable ``skip``.
    """

    def __init__(self, cause: BaseException) -> None:
        self.cause_name = type(cause).__name__
        super().__init__(f"abc2midi failed: {self.cause_name}: {cause}")


def extract_abc(text: str | None) -> str | None:
    if not text:
        return None
    for m in _FENCE_RE.finditer(text):
        body = m.group(1).strip()
        if _XA_RE.search(body):
            return body
    matches = list(_XA_RE.finditer(text))
    if matches:
        return text[matches[-1].start() :].strip()
    return None


def _midi_notes_progs(
    data: bytes,
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int]]] | None:
    if data[:4] != b"MThd":
        return None
    _fmt, ntrk, div = struct.unpack(">HHH", data[8:14])
    if div == 0:
        return None
    notes: list[tuple[int, int, int, int]] = []
    progs: list[tuple[int, int]] = []
    i = 14
    for _ in range(ntrk):
        if data[i : i + 4] != b"MTrk":
            break
        ln = struct.unpack(">I", data[i + 4 : i + 8])[0]
        end = i + 8 + ln
        j = i + 8
        t = 0
        last: int | None = None
        on: dict[tuple[int, int], list[int]] = {}
        while j < end:
            dt = 0
            while j < end:
                b = data[j]
                j += 1
                dt = (dt << 7) | (b & 0x7F)
                if not b & 0x80:
                    break
            t += dt
            if j >= end:
                break
            st_ = data[j]
            if st_ == 0xFF:
                j += 1
                j += 1
                l2 = 0
                while j < end:
                    b = data[j]
                    j += 1
                    l2 = (l2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                j += l2
            elif st_ in (0xF0, 0xF7):
                j += 1
                l2 = 0
                while j < end:
                    b = data[j]
                    j += 1
                    l2 = (l2 << 7) | (b & 0x7F)
                    if not b & 0x80:
                        break
                j += l2
            else:
                if st_ & 0x80:
                    last = st_
                    j += 1
                s = last
                if s is None:
                    break
                k = s & 0xF0
                ch = s & 0x0F
                if k == 0xC0:
                    progs.append((ch, data[j]))
                    j += 1
                elif k == 0xD0:
                    j += 1
                elif k == 0x90:
                    n, v = data[j], data[j + 1]
                    j += 2
                    if v > 0:
                        on.setdefault((ch, n), []).append(t)
                    else:
                        q = on.get((ch, n))
                        if q:
                            t0 = q.pop(0)
                            notes.append((t0, t - t0, ch, n))
                elif k == 0x80:
                    n = data[j]
                    j += 2
                    q = on.get((ch, n))
                    if q:
                        t0 = q.pop(0)
                        notes.append((t0, t - t0, ch, n))
                else:
                    j += 2
        i = end
    return notes, progs


def do(rec: Mapping[str, Any], abc2midi_bin: str, timeout_s: float = 60.0) -> dict[str, Any]:
    key = rec["key"]
    abc = rec.get("abc") or ""
    base = dict(
        key=key,
        id=rec["id"],
        rep=rec["rep"],
        tag=rec.get("tag"),
        lang=rec.get("lang"),
        abc_len=rec.get("abc_len", 0),
        nvoice=rec.get("nvoice", 0),
        latency=rec.get("latency"),
    )
    if not abc.strip():
        return {**base, "skip": "empty_abc"}

    ref = REF_FULL4K
    with tempfile.TemporaryDirectory() as td:
        ap = Path(td) / "a.abc"
        mp = Path(td) / "a.mid"
        ap.write_text(abc, encoding="utf-8")
        try:
            p = sp.run(
                [abc2midi_bin, str(ap), "-o", str(mp)], capture_output=True, timeout=timeout_s
            )
        except FileNotFoundError as e:
            raise Abc2MidiMissing(
                "abc2midi is not on PATH in this process. The scorer renders MIDI to "
                "extract features, so no score can be computed without it. Install the "
                "`abcmidi` package, or pass an explicit abc2midi_bin pointing at the "
                "binary (FoundationScale never reads it from the environment)."
            ) from e
        except (OSError, sp.TimeoutExpired) as e:
            raise Abc2MidiError(e) from e
        except Exception as e:
            return {**base, "skip": f"abc2midi:{repr(e)[:60]}"}
        log = (p.stdout + p.stderr).decode("utf-8", "replace")
        err = len(re.findall(r"^Error", log, re.M))
        bar = len(re.findall(r"Bar \d+ has", log))
        if not mp.exists():
            return {**base, "skip": "no_midi", "err": err, "bar": bar}
        data = mp.read_bytes()
        try:
            feat = analyze(mp)
        except Exception as e:
            feat = {"skip": f"analyze:{repr(e)[:60]}"}
        try:
            got = _midi_notes_progs(data)
        except Exception:
            got = None

    r: dict[str, Any] = {
        **base,
        "err": err,
        "bar": bar,
        "blank": 1 if any(not line.strip() for line in abc.split("\n")[:-1]) else 0,
    }

    if feat.get("skip"):
        r["scorer_skip"] = feat["skip"]
    else:
        s = _score_v1(feat, ref)
        assert s is not None
        r["total"] = s["total"]
        r["groups"] = s["groups"]
        r["n_chan"] = feat.get("n_chan")
        r["is_piano"] = feat.get("is_piano")

    if got and got[0]:
        notes, progs = got
        pit = [n[3] for n in notes]
        m = sum(pit) / len(pit)
        v = sum((x - m) ** 2 for x in pit) / len(pit)
        r["pit_min"] = min(pit)
        r["pit_max"] = max(pit)
        r["pit_range"] = max(pit) - min(pit)
        r["pit_std"] = round(v**0.5, 2)
        r["low_c3"] = round(sum(1 for p in pit if p < 48) / len(pit), 4)
        chp: dict[int, set[int]] = {}
        for ch, pr in progs:
            chp.setdefault(ch, set()).add(pr)
        r["ch_conflict"] = sum(1 for ch, se in chp.items() if len(se) > 1)

    r["reject"] = (
        1
        if (
            r.get("bar", 0) >= 10
            or r.get("err", 0) > 0
            or r.get("blank", 0)
            or r.get("ch_conflict", 0) > 0
        )
        else 0
    )
    return r


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: Any = None,
    extra_info: Any = None,
    *,
    abc2midi_bin: str = "abc2midi",
    timeout_s: float = 60.0,
) -> float:
    """Upstream's verl reward-fn contract, kept for parity. NOT the FoundationScale public
    adapter -- see :class:`~foundationscale.agentic_rl.rewards.music.MusicReward` for that.
    """
    del data_source, ground_truth, extra_info
    try:
        abc = extract_abc(solution_str)
        if not abc:
            return 0.0

        rec = {
            "key": "rl_rollout",
            "id": 0,
            "rep": 0,
            "abc": abc,
            "tag": None,
            "lang": None,
            "abc_len": len(abc),
            "nvoice": 0,
            "latency": None,
        }
        r = do(rec, abc2midi_bin, timeout_s)

        if r.get("skip") or r.get("reject", 0):
            return 0.0
        if "total" not in r:
            return 0.0

        return max(0.0, min(1.0, float(r["total"]) / 100.0))
    except Abc2MidiMissing:
        raise
    except Exception:
        traceback.print_exc()
        return 0.0
