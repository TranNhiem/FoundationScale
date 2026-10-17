# Writing A/B ledger

Append-only: one row per case set per A2.2 writing round (kept AND rejected). The decision follows the pre-registered rule in artifacts/research/a22_writing_protocol.md; rows are never edited or removed.

| date | change_family | package | cases | reps | agent | judge | bundle_before | bundle_after | with_before | with_after | mean_delta | ci | paired | ab_verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | a22-anti-rationalizations | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 27844c4aba64 | e04724b8c8ec | 0.919 | 0.951 | 0.023 | [-0.023, 0.072] | 17 | reject | keep |
| 2026-10-17 | a22-anti-rationalizations | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 27844c4aba64 | e04724b8c8ec | 0.859 | 0.892 | 0.034 | [0.007, 0.067] | 33 | keep | keep |
| 2026-10-17 | a22-red-flags | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | cd6d47d3f274 | 0.951 | 0.976 | 0.023 | [-0.007, 0.062] | 17 | reject | reject |
| 2026-10-17 | a22-red-flags | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | cd6d47d3f274 | 0.892 | 0.872 | -0.020 | [-0.074, 0.017] | 33 | reject | reject |
