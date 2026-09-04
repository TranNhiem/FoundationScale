#!/usr/bin/env python3
"""#155: the estate's identifier vocabulary, as an INPUT rather than a checked-in list.

WHY THIS MODULE EXISTS. Every stage and gate in this build redacts estate identifiers,
so every one of them has to NAME those identifiers. Seven files ended up carrying their
own copy of the alternation:

    r"r0[0-9]+dgx[0-9]+|hh[0-9]{5,6}|<org>|<corp>|<hpc>|<ip>|dgpn0[0-9]|..."

which means the repository that exists to redact the estate published it seven times.

#151b already reached this conclusion and acted on it -- but only for the literals that
could not be written as a safe pattern. A bare five-digit account id is inexpressible,
because `[0-9]{5}` matches line counts and byte sizes and half the corpus, so those went
out to FS_REDACT_EXTRA. The ones that COULD be written as a pattern stayed compiled in.

That criterion is wrong, and the wrongness is easy to miss because it sounds technical.
Whether a token is expressible as a regex has nothing whatever to do with whether
publishing it discloses an estate. A corporate name is trivially expressible and names
the owner outright. The question is never "can I write a pattern for this", it is "does
this name somebody's estate" -- and if it does, it belongs in the environment.

So the whole identifier tier is an input, and the empty case is DECLARED (NONE) rather
than assumed, because an unset redaction vocabulary is UNMEASURED, not clean.

WHAT STAYS COMPILED IN, AND WHY. The token tier -- `/work/`, `ghp_` -- is deliberately
still here. That is #144: a secret PREFIX is not a secret. `ghp_` is documented by
GitHub; what must never ship is `ghp_` followed by a body. Publishing the prefix
discloses nothing and every consumer needs it, so parameterising it would be ceremony.
The two tiers are different in kind, not merely in degree.
"""

from __future__ import annotations

import os
import re
import sys

# The token tier: public vocabulary, safe to publish, needed by every caller. See #144.
TOKEN_TIER = r"/work/|ghp_[A-Za-z0-9]{20,}"

# Same as TOKEN_TIER but matching a bare `ghp_`. Callers that scan GENERATED artifacts
# want this: a generated file has no legitimate reason to contain even the prefix,
# whereas a redactor obviously does.
TOKEN_TIER_STRICT = r"/work/|ghp_"

_REFUSAL = (
    "REFUSE 96: FS_ESTATE_IDENT_PAT is unset (required, no default by design).\n"
    "  It is the |-separated regex alternation of this estate's identifiers: node\n"
    "  names, account ids, org segments, private hostnames, management IPs.\n"
    "  It is not stored in this repository, because a checked-in redaction list is a\n"
    "  published estate -- the exact defect the list was written to prevent.\n"
    "  Set it to NONE to declare that this estate contributes no identifiers.\n"
    "  Unset is UNMEASURED, not clean."
)


def estate_ident_pat(*, trailing_pipe: bool = True) -> str:
    """The identifier alternation, from the environment. Refuses 96 when unset.

    Returns "" for a declared-empty (NONE) estate rather than a bare "|", because an
    empty alternation branch matches every line: the blocklist would go from redacting
    an estate to rejecting the universe, and it would do it silently.
    """
    v = os.environ.get("FS_ESTATE_IDENT_PAT", "").strip()
    if not v:
        print(_REFUSAL, file=sys.stderr)
        raise SystemExit(96)
    if v == "NONE":
        return ""
    return v + "|" if trailing_pipe else v


_PARTITION_REFUSAL = (
    "REFUSE 96: FS_PARTITION_LITERAL is unset (required, no default by design).\n"
    "  It is this estate's Slurm partition name -- the literal being REMOVED from the\n"
    "  launcher by patch_partition_knob.py, and the literal three patch stages must\n"
    "  reproduce verbatim in their anchors to find the before-text that carries it.\n"
    "  It is a VALUE, not a pattern, so it cannot come from FS_ESTATE_IDENT_PAT.\n"
    "  Set it to NONE to declare that this estate's before-text does not name it; the\n"
    "  affected anchors then match on their estate-free form.\n"
    "  (The runtime knob the launcher READS is FS_PARTITION. Deliberately two names:\n"
    "   LITERAL is the build input being deleted, FS_PARTITION is the operator input\n"
    "   that replaces it. Do not collapse them.)"
)


def estate_partition_literal(default_if_none: str = "") -> str:
    """The estate's partition name as a LITERAL, from the environment. Refuses 96 when unset.

    #157. A second, distinct reason an estate identifier ends up in source, and not the
    redaction reason #155 solved. A patch stage locates its edit site by matching before-text
    exactly, and when that before-text names the estate the anchor must name it too. Three
    stages did, so three estate literals sat in the published tree -- invisible, because the
    generator scan only counted an identifier that touched a `/`.

    Parameterising it is not only a redaction fix. An anchor carrying one estate's name can
    only match one estate's launcher, so the stage was silently single-site; supplying the
    name makes the same stage work anywhere. The disclosure and the non-generality were one
    defect seen from two directions.

    ONE ORACLE, ON PURPOSE. The first version of this shipped a separate FS_ESTATE_SHORTNAME
    holding the same string. That is #153's defect -- two oracles for one fact, kept equal by
    nobody -- introduced by the very ticket that exists to remove duplicated estate literals.
    patch_partition_knob.py already enumerates all 13 sites of this literal, two of which are
    these anchors, so FS_PARTITION_LITERAL is the established name and the shortname knob was
    retracted before it ever left the build host.

    NONE returns `default_if_none` (empty by default), so an estate whose before-text does not
    name the partition gets the estate-free anchor rather than a literal "NONE" spliced in.
    """
    v = os.environ.get("FS_PARTITION_LITERAL", "").strip()
    if not v:
        print(_PARTITION_REFUSAL, file=sys.stderr)
        raise SystemExit(96)
    return default_if_none if v == "NONE" else v


def estate_blocklist(*, strict_token: bool = False) -> re.Pattern[str]:
    """Both tiers, compiled. The pattern every stage and gate in this build should use."""
    return re.compile(
        estate_ident_pat() + (TOKEN_TIER_STRICT if strict_token else TOKEN_TIER),
        re.IGNORECASE,
    )
