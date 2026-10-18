# Writing A/B ledger

Append-only: one row per case set per A2.2 writing round (kept AND rejected). The decision follows the pre-registered rule in artifacts/research/a22_writing_protocol.md; rows are never edited or removed.

| date | change_family | package | cases | reps | agent | judge | bundle_before | bundle_after | with_before | with_after | mean_delta | ci | paired | ab_verdict | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-17 | a22-anti-rationalizations | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 27844c4aba64 | e04724b8c8ec | 0.919 | 0.951 | 0.023 | [-0.023, 0.072] | 17 | reject | keep |
| 2026-10-17 | a22-anti-rationalizations | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 27844c4aba64 | e04724b8c8ec | 0.859 | 0.892 | 0.034 | [0.007, 0.067] | 33 | keep | keep |
| 2026-10-17 | a22-red-flags | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | cd6d47d3f274 | 0.951 | 0.976 | 0.023 | [-0.007, 0.062] | 17 | reject | reject |
| 2026-10-17 | a22-red-flags | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | cd6d47d3f274 | 0.892 | 0.872 | -0.020 | [-0.074, 0.017] | 33 | reject | reject |
| 2026-10-17 | a22-process | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | dd6803f3efb8 | 0.951 | 0.943 | -0.003 | [-0.039, 0.039] | 17 | reject | reject |
| 2026-10-17 | a22-process | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | dd6803f3efb8 | 0.892 | 0.882 | -0.010 | [-0.044, 0.020] | 33 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-probe-first | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | d77c74e9b750 | e81c2874b114 | 0.902 | 0.843 | -0.065 | [-0.139, 0.000] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-probe-first | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | d77c74e9b750 | e81c2874b114 | 0.844 | 0.889 | 0.044 | [0.000, 0.133] | 5 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-measured-or-unmeasured | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6070d39a416c | 02e87546739b | 0.974 | 0.872 | -0.083 | [-0.194, 0.056] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-measured-or-unmeasured | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6070d39a416c | 02e87546739b | 0.889 | 0.926 | 0.037 | [0.000, 0.111] | 3 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-data-engine | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 56326d7b1a39 | 2cc073b766bf | 0.952 | 0.935 | -0.017 | [-0.050, 0.022] | 20 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-data-engine | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 56326d7b1a39 | 2cc073b766bf | 0.970 | 0.990 | 0.020 | [0.000, 0.051] | 11 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-confirm-before-launch | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 1ccd011c9eb2 | 4a236911c615 | 0.860 | 0.860 | 0.008 | [-0.048, 0.071] | 7 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-confirm-before-launch | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 1ccd011c9eb2 | 4a236911c615 | 0.861 | 0.722 | -0.139 | [-0.417, 0.000] | 4 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-refusal-surface | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | be551f9b3cc1 | 1efe44bd28a3 | 0.926 | 0.926 | 0.000 | [0.000, 0.000] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-refusal-surface | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | be551f9b3cc1 | 1efe44bd28a3 | 0.889 | 0.963 | 0.074 | [0.000, 0.222] | 3 | reject | reject |
| 2026-10-17 | a22-verification-evidence | fskills-training | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | 9428d38d60ea | 0.951 | 0.935 | -0.010 | [-0.056, 0.029] | 17 | reject | reject |
| 2026-10-17 | a22-verification-evidence | fskills-training | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | e04724b8c8ec | 9428d38d60ea | 0.892 | 0.869 | -0.024 | [-0.051, 0.000] | 33 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations | fskills-evaluation | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | c790e29ec609 | 4d9df206631f | 0.974 | 0.965 | -0.007 | [-0.052, 0.052] | 16 | reject | unmeasured |
| 2026-10-17 | a22-anti-rationalizations | fskills-evaluation | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | c790e29ec609 | 4d9df206631f | 0.896 | 0.864 | -0.037 | [-0.096, 0.030] | 15 | unmeasured | unmeasured |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-confirm-before-launch | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 1ccd011c9eb2 | 0da3e33d32ee | 0.860 | 0.877 | 0.024 | [0.000, 0.071] | 7 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-confirm-before-launch | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 1ccd011c9eb2 | 0da3e33d32ee | 0.861 | 0.778 | -0.083 | [-0.250, 0.000] | 4 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-measured-or-unmeasured | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6070d39a416c | fd66605afa02 | 0.974 | 0.897 | -0.083 | [-0.139, -0.028] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-measured-or-unmeasured | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 6070d39a416c | fd66605afa02 | 0.889 | 0.815 | -0.074 | [-0.222, 0.000] | 3 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-data-engine | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 56326d7b1a39 | 2d59c4db2653 | 0.952 | 0.964 | 0.014 | [-0.011, 0.039] | 20 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-data-engine | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 56326d7b1a39 | 2d59c4db2653 | 0.970 | 0.990 | 0.020 | [0.000, 0.051] | 11 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-evaluation | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | c790e29ec609 | e8119429c328 | 0.974 | 0.982 | 0.014 | [-0.021, 0.062] | 16 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-evaluation | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | c790e29ec609 | e8119429c328 | 0.896 | 0.867 | -0.030 | [-0.089, 0.015] | 15 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-probe-first | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | d77c74e9b750 | ada819d1b2aa | 0.902 | 0.863 | -0.046 | [-0.102, 0.000] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-probe-first | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | d77c74e9b750 | ada819d1b2aa | 0.844 | 0.867 | 0.022 | [0.000, 0.067] | 5 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-refusal-surface | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | be551f9b3cc1 | 2aa368ad8fbb | 0.926 | 0.852 | -0.083 | [-0.194, 0.000] | 6 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-refusal-surface | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | be551f9b3cc1 | 2aa368ad8fbb | 0.889 | 1.000 | 0.111 | [0.000, 0.333] | 3 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-auto-research | routing | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 9b62f3d44299 | 7a6c0bd18f46 | 0.915 | 0.858 | -0.058 | [-0.114, -0.014] | 20 | reject | reject |
| 2026-10-17 | a22-anti-rationalizations-only | fskills-auto-research | oracle | 3 | mimo-v2.6-flash | mimo-v2.6-pro | 9b62f3d44299 | 7a6c0bd18f46 | 0.879 | 0.873 | -0.005 | [-0.039, 0.028] | 43 | reject | reject |
