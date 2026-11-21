from __future__ import annotations

"""Rewrite-only SKILL.md candidate generation and Pareto pruning for C9 guided self-improvement.
Builds the prompts, pulls the rewritten file out of the marked model output and keeps the
non-dominated candidates; the LLM backend is injected and never constructed or contacted here."""

from typing import Any, Protocol

FENCE = '`' * 3    # fences are spelled by repetition; three backticks never appear in a row
START = "<<<SKILL_MD"
END = "SKILL_MD>>>"

SYSTEM = (
    "You rewrite ONLY the prose of a SKILL.md so that the listed behaviour misses stop happening. "
    "These must stay byte-identical: the YAML frontmatter keys name, license, compatibility and "
    "allowed-tools, metadata.invocation, the generated usage block between the FS-USAGE markers, "
    "every code block fenced with " + FENCE + ", every rule id, every exit code and every URL. "
    "Change nothing else. "
    "Emit the complete new SKILL.md and nothing else between a line that is exactly " + START
    + " and a line that is exactly " + END + "."
)


class Backend(Protocol):
    """Injected duck-typed client: complete(messages, *, temperature, max_tokens, seed, json_mode)."""

    def complete(self, messages: list[dict[str, str]], *, temperature: float,
                 max_tokens: int, seed: int, json_mode: bool) -> Any: ...


def _flat(text: object) -> str:
    """Collapse whitespace so a hypothesis item prints as one clean line."""
    return " ".join(str(text if text is not None else "").split())


def candidate_messages(skill_md: str, hyps: list[dict]) -> list[dict]:
    """System + user message: the behaviour misses to fix followed by the current SKILL.md."""
    rows = ["- %s  [misses: %s]" % (_flat(h.get("item")), h.get("misses", 0)) for h in (hyps or [])]
    user = (
        "Rewrite the prose of the SKILL.md below so that these behaviour misses disappear:\n"
        + ("\n".join(rows) if rows else "- (no misses supplied)")
        + "\n\nCurrent SKILL.md:\n" + START + "\n"
        + (skill_md or "").rstrip() + "\n" + END + "\n"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def extract_candidate(content: str | None) -> str | None:
    """Text between the START/END markers, stripped and newline terminated, else None."""
    if not content:
        return None
    s = content.find(START)
    if s < 0:
        return None
    s += len(START)
    e = content.find(END, s)
    if e < 0:
        return None
    body = content[s:e].strip()
    return body + "\n" if body else None


def propose(backend: Backend, skill_md: str, hyps: list[dict], n: int = 3,
            *, temperature: float = 0.7, max_tokens: int = 12000) -> list[str]:
    """Take n seeded samples (seeds 0..n-1); keep distinct valid rewrites that differ from skill_md."""
    msgs = candidate_messages(skill_md, hyps)
    seen = {(skill_md or "").strip()}
    keep: list[str] = []
    for seed in range(max(0, n)):
        try:
            resp = backend.complete(msgs, temperature=temperature, max_tokens=max_tokens,
                                    seed=seed, json_mode=False)
        except Exception:                       # a flaky backend must not abort the search
            continue
        if resp is None or getattr(resp, "error", None) or not getattr(resp, "content", None):
            continue
        cand = extract_candidate(getattr(resp, "content", None))
        if cand is None or cand.strip() in seen:
            continue
        seen.add(cand.strip())
        keep.append(cand)
    return keep


def pareto(cands: list[dict], objectives: tuple[str, ...]) -> list[dict]:
    """Non-dominated subset (higher is better everywhere); a candidate lacking an objective is
    dominated, ties keep both, and the surviving candidates keep their input order."""
    objs = tuple(objectives)
    scored = [c for c in (cands or []) if all(c.get(o) is not None for o in objs)]
    keep: list[dict] = []
    for i, cand in enumerate(scored):
        mine = [cand[o] for o in objs]
        dominated = False
        for j, other in enumerate(scored):
            if j == i:
                continue
            theirs = [other[o] for o in objs]
            if all(t >= m for t, m in zip(theirs, mine)) and any(t > m for t, m in zip(theirs, mine)):
                dominated = True
                break
        if not dominated:
            keep.append(cand)
    return keep
