# Skill-impact ledger

Append-only: one row per measured change family (A2.1 baselines first). Rows are never edited or removed; a re-measurement is a new row.

| date | change_family | package | cases | reps | agent | judge | skill_bundle | cases_sha | with | without | delta | ci | paired/abstained | verdict | report_sha256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-10 | baseline | fskills-auto-research | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 2921effff120 | bdad34150796 | 0.922 | 0.241 | 0.678 | [0.561, 0.792] | 20/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-confirm-before-launch | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 8b649728aa4a | 210d066106e6 | 0.860 | 0.193 | 0.667 | [0.532, 0.794] | 7/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-data-engine | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 60b644f5c0bc | 126f11147596 | 0.970 | 0.167 | 0.803 | [0.672, 0.908] | 20/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-evaluation | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 55de5ac8f93a | ad8bcab0a68d | 0.974 | 0.175 | 0.774 | [0.625, 0.906] | 16/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-measured-or-unmeasured | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | f54fd5443be0 | 0c62ae9b19c3 | 0.923 | 0.154 | 0.806 | [0.583, 1.000] | 6/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-probe-first | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 5a15bfaca65a | 4250963b2bba | 0.824 | 0.137 | 0.648 | [0.352, 0.889] | 6/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-refusal-surface | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6f15e568f487 | 1c5bfc550793 | 0.963 | 0.111 | 0.861 | [0.694, 1.000] | 6/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 12a4006652d9 | b27a8f6b39b5 | 0.919 | 0.447 | 0.464 | [0.304, 0.621] | 17/0 | uplift | 87930946179b |
| 2026-10-10 | baseline | fskills-auto-research | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 2921effff120 | 76544b5562fb | 0.879 | 0.282 | 0.597 | [0.517, 0.680] | 43/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-confirm-before-launch | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 8b649728aa4a | 90aa3697829c | 0.778 | 0.333 | 0.444 | [0.278, 0.611] | 4/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-data-engine | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 60b644f5c0bc | 2c4b242b7f1e | 1.000 | 0.293 | 0.707 | [0.556, 0.848] | 11/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-evaluation | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 55de5ac8f93a | 9d125c5709ee | 0.926 | 0.284 | 0.642 | [0.494, 0.778] | 9/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-measured-or-unmeasured | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | f54fd5443be0 | 0fe2950f46d1 | 0.852 | 0.444 | 0.407 | [0.222, 0.556] | 3/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-probe-first | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 5a15bfaca65a | 182a21afbc04 | 0.844 | 0.333 | 0.511 | [0.356, 0.644] | 5/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-refusal-surface | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6f15e568f487 | 24f72b0f1ec1 | 1.000 | 0.222 | 0.778 | [0.667, 1.000] | 3/0 | uplift | 7dfebfdccb90 |
| 2026-10-10 | baseline | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 12a4006652d9 | 38cbf83076ae | 0.985 | 0.207 | 0.778 | [0.681, 0.867] | 15/0 | uplift | 7dfebfdccb90 |
| 2026-10-11 | a41-vlm-lora | fskills-probe-first | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 5a15bfaca65a | 4250963b2bba | 0.863 | 0.137 | 0.685 | [0.389, 0.907] | 6/0 | uplift | b2fc3cad2640 |
| 2026-10-11 | a41-vlm-lora | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 892be2ee6071 | b27a8f6b39b5 | 0.992 | 0.431 | 0.552 | [0.386, 0.709] | 17/0 | uplift | b2fc3cad2640 |
| 2026-10-11 | a41-vlm-lora | fskills-probe-first | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 5a15bfaca65a | 2ab79adda3f2 | 0.867 | 0.244 | 0.622 | [0.467, 0.778] | 5/0 | uplift | 919e7a56758a |
| 2026-10-11 | a41-vlm-lora | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 892be2ee6071 | b87d01d17d79 | 0.972 | 0.185 | 0.787 | [0.699, 0.875] | 24/0 | uplift | 919e7a56758a |
| 2026-10-11 | a41-vlm-lora | fskills-evaluation | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 3e2136f6f6ef | ad8bcab0a68d | 0.974 | 0.175 | 0.781 | [0.635, 0.917] | 16/0 | uplift | daacf07e78b4 |
| 2026-10-11 | a41-vlm-lora | fskills-evaluation | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 3e2136f6f6ef | dfe2c4239f74 | 0.911 | 0.274 | 0.637 | [0.511, 0.763] | 15/0 | uplift | 7ea190ac6c1f |
