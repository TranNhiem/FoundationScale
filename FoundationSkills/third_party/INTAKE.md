# Third-party skill intake ledger (A6.2)

Source `https://github.com/mattpocock/skills`. Only a skill that PASSes the gate (VG-01..VG-15) at its pin is vendored. Failing checks are red-cased here and never waived.

| skill | upstream path | commit | licence | verdict | vendored |
|---|---|---|---|---|---|
| codebase-design | `skills/engineering/codebase-design` | `49dd158d1076` | MIT | PASS | yes |
| diagnosing-bugs | `skills/engineering/diagnosing-bugs` | `49dd158d1076` | MIT | REFUSED | **no** |
| domain-modeling | `skills/engineering/domain-modeling` | `49dd158d1076` | MIT | PASS | yes |
| grilling | `skills/productivity/grilling` | `49dd158d1076` | MIT | PASS | yes |
| grill-with-docs | `skills/engineering/grill-with-docs` | `49dd158d1076` | MIT | PASS | yes |
| pr | `skills/engineering/pr` | `49dd158d1076` | MIT | PASS | yes |
| retro | `skills/engineering/retro` | `49dd158d1076` | MIT | PASS | yes |
| wait-what | `skills/productivity/wait-what` | `49dd158d1076` | MIT | PASS | yes |
| writing-for-agents | `skills/productivity/writing-for-agents` | `49dd158d1076` | MIT | PASS | yes |

## Red-cased (not vendored)

- **diagnosing-bugs** (REFUSED):
  - VG-04 RED: 1 network primitive hit(s)
  - VG-10 REFUSED: no scripts review matches the current 1 scripts/ file(s) (files_sha256 9b8e4bcffd88...); unsigned scripts are never run: scripts/hitl-loop.template.sh

Notes on the red case (context only; the verdict stands):
- The VG-04 hit is `http://localhost:3000` in `scripts/hitl-loop.template.sh` line 34. That is a template the user edits, so the hit is
  likely benign. The gate does not judge intent, and the skill stays out.
- VG-10: the script has no signed `scripts` review. Vendoring it needs a human review of that file recorded in the lock at its
  digest, followed by a fresh intake. Nobody has done that review.
- Licence evidence for every row: the upstream repository's root LICENSE heading ("MIT License"); no mattpocock SKILL.md
  declares a licence in its frontmatter.
- Re-vet: `tests/contract/test_third_party_skills.py` re-runs the gate offline over every vendored skill, replaying the
  advisory recorded for its exact SKILL.md bytes. A changed byte fails VG-02 and VG-11.
