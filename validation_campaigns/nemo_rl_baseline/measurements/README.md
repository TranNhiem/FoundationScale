# measurements/

Raw result records from the Phase 3 measurements, kept so a claim in
`PHASE3_DESIGN.md` can be checked against the run that produced it rather than
against a summary of it.

## Contents

| file | what it records |
|---|---|
| `weight_sync_gb200_4gpu.json` | full result record for the weight-sync transport sweep — every measured row, the withheld cell and its reason, the elision control's adjudication, the poison gate's per-row result, and run provenance |
| `weight_sync_gb200_4gpu.log` | the runner's stdout for the same run, including the DENOMINATOR and ABSTENTION RULE banners and the final verdict line |
| `weight_sync_gb200_16gib_probe.json` | the 16 GiB probe: the size the sweep above excluded, attempted on the same 4-GPU tray. Verdict `UNMEASURED` — 5 of 8 cells admitted, 3 withheld |
| `weight_sync_gb200_4gib_control.json` | a same-shape re-take of the 4 GiB sweep, run as the control for the probe. Verdict `CLEAR_WITH_ABSTENTIONS` — 7 of 8 cells admitted, 1 withheld |

Producer: `../bench_weight_sync.py`. Reading: `../PHASE3_DESIGN.md` section 7,
item 3.

## Declared redaction

These four files were produced on a cluster and are published in a public
repository. Absolute home-directory paths of the form `/home/<group>/<user>/`
were replaced before commit.

In `weight_sync_gb200_4gpu.json` and `.log` the placeholder is the literal
`<HOME>/`, and two sites were affected: `provenance.python_executable` in the
JSON, and one interpreter path inside a `UserWarning` line in the log. Byte
counts before and after (238551 -> 238536 and 7891 -> 7876) account for exactly
the substituted characters.

In the probe and control records the placeholder is `<estate-home>/`, and one
site was affected in each: `provenance.python_executable`. Each of those two
records also carries a sibling key, `provenance.python_executable_note`, saying
in the record itself that the prefix is a placeholder — the two later files
declare their own redaction inline rather than relying on a reader finding this
README.

The substitution is declared here rather than left silent because a reader
comparing a record against a re-run needs to know that one field is a
placeholder and not a path that exists. Nothing else was altered — no number,
verdict, row, abstention or reason was touched in any of the four files.

The placeholders were chosen to BREAK the shape of the thing they replace rather
than to preserve it. A placeholder that still looked like an absolute path
would still match the identifier scan these files are checked against, and
excepting one's own residue from that scan is the failure mode the scan exists
to catch.

## What these records do not establish

**The sweep record is `CLEAR_WITH_ABSTENTIONS`, not `CLEAR`.** It withholds
exactly one cell, and that cell is named in the record's own `unmeasured` list:
`full_copy` at 4294967296 B in the `sweep` label, withheld because warmup never
stabilised (spread 74.1% of the median against a 10% tolerance, at the
40-iteration cap).

**The abstention is a property of the RUN, not of the cell.** This is the one
thing the three records together establish that no single record could. The same
harness, the same rule, the same tray, three runs:

| record | cells admitted | cell(s) withheld (`transport` @ `bytes`, `label`) |
|---|---|---|
| `weight_sync_gb200_4gpu.json` | 19 of 20 | `full_copy` @ 4294967296, `sweep` |
| `weight_sync_gb200_4gib_control.json` | 7 of 8 | `full_copy_pageable` @ 1048576, `per_call_floor` |
| `weight_sync_gb200_16gib_probe.json` | 5 of 8 | `full_copy` @ 17179869184, `sweep`; `full_copy_pageable` @ 17179869184, `sweep`; `collective` @ 1048576, `per_call_floor` |

The three columns are the record's own `denominator.measured_cells` /
`denominator.total_cells` and its `unmeasured` list, quoted with the field names
the records use so a reader can grep for them rather than match prose.

`full_copy` at 4 GiB — the cell the first record withheld — MEASURED cleanly in
the control at 0.046692 s. So "this cell cannot be measured" is not what the
first record's abstention means, and reading it that way would be wrong. What it
means is "this cell did not stabilise in that run." The withheld set moves
between repeats; nothing here bounds how often, because N is 3 and the three runs
were not a designed repeatability study.

**16 GiB was attempted, and the prediction that it could not be was wrong.**
`PHASE3_DESIGN.md` excluded 16 GiB on the reasoning that allgather output across
4 ranks plus ring buffers would need roughly 192 GiB against ~186 GiB of HBM and
would OOM the process. The probe ran it on the same tray and did not OOM: the
`collective` row completed 17 timed samples at 0.033833 s (1523.4 GB/s over
51539607552 aggregate bytes) and `reshard` completed 21 at 0.023935 s
(2153.3 GB/s), both with cross-rank equality verified. The exclusion rationale
had assumed an allgather-shaped output; this harness's `collective` declares its
own byte model as a rank-0 broadcast to `world_size-1` consumers, so the resident
footprint the rationale predicted was never the footprint that row would
allocate. The occupancy gate recorded `contended: false` with 197169512448 B free
of 197897617408 B per device at probe time.

**`total_cells` is derived from what was SWEPT, not from what was declared.**
The sweep record's 20 is 5 sizes x 4 transports and the probe's 8 is 2 x 4,
each computed from the sizes that run actually attempted; no field in any of
these records states the size set the run INTENDED to cover. So a size that
silently drops out of a sweep shrinks the denominator along with the numerator
and the ratio stays flattering — the record cannot distinguish "4 of 4 sizes
measured" from "4 of 5 attempted and one never got as far as a row". Nothing
here is known to have been lost that way; the point is that the record could not
say so if it had. Tracked as the residual of the run-scoped-abstention finding.

**What 16 GiB still does not establish.** The probe's verdict is `UNMEASURED`,
not CLEAR: `full_copy` and `full_copy_pageable` at 16 GiB were both withheld for
warmup instability (spread 56.7% and 64.9%), so the host-staged transports are
uncharacterised at that size, and with them the pinned-versus-pageable elision
control. The probe answers "does it crash?" with no. It does not answer "how fast
is a full copy at 16 GiB?" — that cell is withheld, not slow.
