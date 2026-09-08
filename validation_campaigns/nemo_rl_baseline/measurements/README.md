# measurements/

Raw result records from the Phase 3 measurements, kept so a claim in
`PHASE3_DESIGN.md` can be checked against the run that produced it rather than
against a summary of it.

## Contents

| file | what it records |
|---|---|
| `weight_sync_gb200_4gpu.json` | full result record for the weight-sync transport sweep — every measured row, the withheld cell and its reason, the elision control's adjudication, the poison gate's per-row result, and run provenance |
| `weight_sync_gb200_4gpu.log` | the runner's stdout for the same run, including the DENOMINATOR and ABSTENTION RULE banners and the final verdict line |

Producer: `../bench_weight_sync.py`. Reading: `../PHASE3_DESIGN.md` section 7,
item 3.

## Declared redaction

These two files were produced on a cluster and are published in a public
repository. Absolute home-directory paths of the form `/home/<group>/<user>/`
were replaced with the literal placeholder `<HOME>/` before commit. Two sites
were affected: `provenance.python_executable` in the JSON, and one interpreter
path inside a `UserWarning` line in the log.

The substitution is declared here rather than left silent because a reader
comparing the record against a re-run needs to know that one field is a
placeholder and not a path that exists. Nothing else was altered — no number,
verdict, row, abstention or reason was touched, and byte counts before and
after (238551 -> 238536 and 7891 -> 7876) account for exactly the substituted
characters.

The placeholder was chosen to BREAK the shape of the thing it replaces rather
than to preserve it. A placeholder that still looked like an absolute path
would still match the identifier scan these files are checked against, and
excepting one's own residue from that scan is the failure mode the scan exists
to catch.

## What these records do not establish

The weight-sync record is `CLEAR_WITH_ABSTENTIONS`, not `CLEAR`. It certifies
nothing about `full_copy` at 4 GiB, which was withheld, and nothing about
16 GiB, which was never attempted. Both are named in the record itself and in
section 7 item 3.
