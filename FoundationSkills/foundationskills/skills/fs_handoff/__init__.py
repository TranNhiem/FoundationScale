"""fs_handoff: the FoundationSkills session handoff writer (/fs-handoff).

Reads the artifact envelopes of one session BY PATH (type, id, provenance, sha256 — never a payload),
redacts secrets and PII from `notes`/`next_steps`, and writes one `handoff` artifact for the next session.
An A3.2 user-invoked workflow skill modelled on fs_status.
"""
from foundationskills.skills.fs_handoff.skill import FsHandoffSkill

__all__ = ["FsHandoffSkill"]
