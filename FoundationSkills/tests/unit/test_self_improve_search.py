"""tests/unit/test_self_improve_search.py - unit tests for C9's governed
SKILL.md self-improvement: MODULE D (diagnose.hypotheses), MODULE E
(edit_search proposals/pareto) and MODULE F (prune_verify.stage0).  Offline:
the LLM backend is injected and faked here, never built by the modules."""

from types import SimpleNamespace as NS
from foundationskills.agent import frozen_surface
from foundationskills.agent.diagnose import SCHEMA, hypotheses
from foundationskills.agent.edit_search import (SYSTEM, candidate_messages,
                                                extract_candidate, pareto, propose)
from foundationskills.agent.prune_verify import MAX_GROWTH, stage0

def reply(content=None, error=None): return NS(content=content, error=error)
def wrapped(md): return "noise\n<<<SKILL_MD\n" + md + "SKILL_MD>>>\nnoise"

class FakeBackend:
    """Injected backend: one canned reply per seed, every call recorded."""
    def __init__(self, seeds):
        self.seeds_seen, self.kwargs, self.replies = [], {}, dict(seeds)
    def complete(self, messages, *, temperature, max_tokens, seed, json_mode):
        self.seeds_seen.append(seed)
        self.kwargs.update(temperature=temperature, max_tokens=max_tokens)
        return self.replies.get(seed, reply(error="unknown seed"))


FM = ("---\nname: tables\nlicense: MIT\ncompatibility: core-1\n"    # frozen surface
      "allowed-tools: check\nmetadata: {invocation: fs tables}\n---\n")
BODY = "Run the table checks.\nReport every problem you find.\n"
SKILL_MD = FM + BODY        # minimal SKILL.md: prose only, no usage block
NEW_MD = FM + "Run the table checks carefully.\nMention every problem.\n"
HYP = {"schema": SCHEMA, "package": "pkg.tables", "item": "tables need headers",
       "misses": 7, "cases": ["c1"], "seen": 9}


# MODULE D: behaviour-miss hypotheses from an eval report --------------------
def call(case, items, met, package="pkg.tables", arm="with_skill", measured=True):
    return {"arm": arm, "measured": measured, "package": package, "case_id": case,
            "item_texts": items, "met": met}


def group(pre, package, item, misses, seen):
    out = [call(f"{pre}{i}", [item], [False], package=package) for i in range(misses)]
    out += [call(f"{pre}p{i}", [item], [True], package=package) for i in range(seen - misses)]
    return out


ITEMS = ["tables need headers", "empty table errors"]
REPORT = {"calls": [call("c1", ITEMS, [False, True]), call("c2", ITEMS, [False, True]),
                    call("c3", ITEMS, [True, True]),
                    call("c4", [ITEMS[0]], [False], arm="baseline"),
                    call("c5", [ITEMS[0]], [False], measured=False)]}
HEADERS = {"schema": SCHEMA, "package": "pkg.tables", "item": ITEMS[0]}


def test_d_recurrent_miss_becomes_one_hypothesis():
    assert hypotheses(REPORT) == [dict(HEADERS, misses=2, cases=["c1", "c2"], seen=3)]
    assert hypotheses(REPORT, min_count=3) == []
    assert hypotheses({}) == [] and hypotheses({"calls": []}) == []


def test_d_sorts_by_misses_desc_then_package_then_item():
    big = {"calls": (group("x", "pkg.images", "alt text required", 3, 4)
                     + group("y", "pkg.aaa", "zebra item", 2, 2)
                     + group("z", "pkg.aaa", "apple item", 2, 2))}
    got = [(h["misses"], h["package"], h["item"], h["cases"], h["seen"]) for h in hypotheses(big)]
    assert got == [(3, "pkg.images", "alt text required", ["x0", "x1", "x2"], 4),
                   (2, "pkg.aaa", "apple item", ["z0", "z1"], 2),
                   (2, "pkg.aaa", "zebra item", ["y0", "y1"], 2)]


def test_d_min_count_gating_case_dedupe_and_seen():
    rep = {"calls": [call("b1", ["dup", "lonely"], [False, False]),
                     call("b1", ["dup", "lonely"], [False, True]),
                     call("b2", ["dup"], [True])]}
    assert hypotheses(rep) == [dict(HEADERS, item="dup", misses=2, cases=["b1"], seen=3)]
    assert [h["item"] for h in hypotheses(rep, min_count=1)] == ["dup", "lonely"]


def test_d_never_raises_on_missing_keys():
    for rep in ({}, {"calls": None}, {"calls": [{"arm": "with_skill"}]},
                {"calls": [{"arm": "with_skill", "measured": True}]},
                {"calls": [call("c1", ["x"], None)]}):
        assert isinstance(hypotheses(rep), list)
    mixed = {"calls": [call("o1", ["x"], None, package="pkg.other")] + REPORT["calls"]}
    assert hypotheses(mixed) == hypotheses(REPORT)


# MODULE E: seeded proposals and the pareto front ----------------------------
def test_e_messages_keep_the_contract_and_list_the_hypotheses():
    sys_msg, user_msg = candidate_messages(SKILL_MD, [HYP])
    assert (sys_msg["role"], user_msg["role"]) == ("system", "user")
    assert sys_msg["content"] == SYSTEM
    for token in ("name", "license", "compatibility", "allowed-tools",
                  "metadata.invocation", "FS-USAGE", "SKILL_MD"):
        assert token in SYSTEM
    up = user_msg["content"]
    assert HYP["item"] in up and str(HYP["misses"]) in up and SKILL_MD in up


def test_e_extract_candidate_bounds():
    assert extract_candidate("<<<SKILL_MD\n\n  Alpha\nBeta  \n\nSKILL_MD>>> tail") == "Alpha\nBeta\n"
    assert extract_candidate("<<<SKILL_MD\nAlpha\n") is None; assert extract_candidate(None) is None
    assert extract_candidate("no markers at all") is None


def test_e_propose_walks_seeds_and_skips_garbage_errors_empties():
    back = FakeBackend({0: reply(wrapped(NEW_MD)), 1: reply("total garbage"),
                       2: reply(error="boom")})
    assert propose(back, SKILL_MD, [HYP], 3) == [NEW_MD]
    assert back.seeds_seen == [0, 1, 2]
    assert (back.kwargs["temperature"], back.kwargs["max_tokens"]) == (0.7, 12000)
    empty = FakeBackend({0: reply("<<<SKILL_MD\n \n SKILL_MD>>>"), 1: reply(), 2: reply(error="x")})
    assert all(c.strip() for c in propose(empty, SKILL_MD, [HYP], 3))


def test_e_propose_keeps_only_distinct_new_candidates():
    extra = FM + BODY + "A dull extra prose line.\n"
    back = FakeBackend({0: reply(wrapped(NEW_MD)), 1: reply(wrapped(NEW_MD)),
                       2: reply(wrapped(SKILL_MD)), 3: reply(wrapped(extra))})
    assert propose(back, SKILL_MD, [HYP], 4) == [NEW_MD, extra]


def test_e_pareto_on_two_objectives():
    a, b, c = {"id": "a", "fix": 4, "safe": 1}, {"id": "b", "fix": 1, "safe": 4}, {"id": "c", "fix": 2, "safe": 2}
    assert pareto([a, b, c], ("fix", "safe")) == [a, b, c]    # nobody dominates
    assert pareto([b, c, a], ("fix", "safe")) == [b, c, a]    # stable order
    king = {"id": "a", "fix": 4, "safe": 5}
    assert pareto([king, b, c], ("fix", "safe")) == [king]    # strictly better: wins
    assert [x["id"] for x in pareto([{"id": "x", "fix": 9}, {"id": "y", "fix": 1, "safe": 1}],
                                    ("fix", "safe"))] == ["y"]  # missing objective: out


# MODULE F: stage-0 admission gates ------------------------------------------
def test_f_stage0_accepts_prose_only_edits():
    assert MAX_GROWTH == 1.3 and stage0(SKILL_MD, SKILL_MD) == []
    assert stage0(SKILL_MD, FM + "Check the tables carefully.\nSay what you found.\n") == []


def test_f_stage0_rejects_frozen_growth_and_empty():
    bad = SKILL_MD.replace("name: tables", "name: grids")
    assert frozen_surface.assert_editable(SKILL_MD, bad)        # non-empty problems
    assert stage0(SKILL_MD, bad)                               # every problem shown
    grown = SKILL_MD + "z" * (2 * len(SKILL_MD))
    assert "candidate grew more than 30%" in stage0(SKILL_MD, grown)
    assert "candidate is empty" in stage0(SKILL_MD, "   ")
    cap = int(MAX_GROWTH * len(SKILL_MD))        # exactly at the cap is allowed
    assert not any("30%" in p for p in stage0(SKILL_MD, SKILL_MD + "w" * (cap - len(SKILL_MD))))
