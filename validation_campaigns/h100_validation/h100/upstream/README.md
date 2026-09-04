# `h100/upstream/` — build INPUTS, supplied by the operator

Everything here is consumed by the build; nothing here is produced by it. `h100/gen/` is
the opposite: everything there is produced by a stage and is deleted and rebuilt on every
invocation. Keeping the two in one directory is what caused #136 and #137, so they are
separated and the separation is written down.

The distinction is not cosmetic. `build_h100_plane.sh` opens with

    rm -f "$LAUNCHER" "$BACKEND" "$ENTRY" "$SPLICED" "$MODELROOT" "$MRTEST"

and the files below have no other copy anywhere. Had that `rm` ever been widened to a glob
over `h100/gen/`, the build would have destroyed its own ability to rebuild.

**#157: these files are no longer published.** They are the estate's own artifacts — the
before-text this framework generalizes, not part of the framework doing the generalizing —
and one of them names an organisational filesystem root four times. See below.

| File | sha256 (short) | Read by | Published? |
|---|---|---|---|
| `launchers__launch_fs_h100.sh` | `5eeb315b81eb0cbf` | `apply_113.py`, `dispatch_113.py` | **no — operator-supplied** |
| `tools__fs_train.py` | `b99dd791eafcc91b` | no stage (provenance only) | **no — zero consumers** |
| `launchers__fs_container_backend.sh` | `b9ff3b225bee9c0b` | no stage (provenance only) | **no — zero consumers** |

## How to supply what the build needs

| Input | Consumed by | How to supply it |
|---|---|---|
| `launchers__launch_fs_h100.sh` | `apply_113.py` | `--upstream-launcher <path>`, `$FS_UPSTREAM_H100_LAUNCHER`, `$FS_UPSTREAM_DIR`, or place it in this directory |
| `launchers/fs_container_backend.sh` | `apply_splice.py` | `--repo <path>`, `$FS_UPSTREAM_REPO`, any ancestor of `--root`, or a sibling checkout (#156) |

Each resolver tries its candidates most-explicit-first, accepts one only if the file is really
there, and **names every candidate it tried** in the refusal. A resolver that fails without
saying where it looked sends the reader to guess in turn.

Neither stage falls back to a previously generated artifact. That is the #136 rule: a build
that cannot find its input must refuse, because continuing would leave the last successful
output in place and let the build certify a stale artifact as freshly produced.

## What each one is, and what is unresolved about it

**`launchers__launch_fs_h100.sh` — load-bearing, and its provenance is unrecorded.**
The whole shipped launcher is derived from this file by `apply_113.py`. It is not
reconstructable: `find` over `fs-repo` for `launch_fs_h100*` returns nothing, so there is no
upstream to re-splice from. MUST_FIRE, run 2026-08-31: moving it aside makes `apply_113.py`
die and the build go red at that stage; restoring it turns the build green. It almost
certainly came off the estate during the Phase 1 inventory, but "almost certainly" is not
provenance, and that gap is the open part of #137.

It is also the reason this directory is unpublished. The file guards four paths against a
hard-coded organisational root, and those four sites are exactly what `patch_estate_roots.py`
replaces using a root supplied through `FS_ESTATE_ROOT`. So the build's **output** is clean
while its **input** names somebody's filesystem, and three consecutive clean scans reported
zero because no scan category owned this directory at all.

Redacting the snapshot in place would be wrong twice over: it would misrepresent the
provenance of a file whose only job is to be the faithful before-text, and the downstream
anchors are built *from* the estate root, so a pre-redacted input would resolve zero sites and
the build would refuse anyway — just less legibly. Withholding it is the honest repair.

**`tools__fs_train.py` — provenance only, read by no stage.** The shipped entrypoint
`h100/gen/fs_train.fixed.py` is extracted from `h100/fs_train.json` by `extract_fs_train.py`,
not from this file. It is kept locally because it is the text that envelope was generated
from and there is no other copy; it is not kept because anything reads it.

**`launchers__fs_container_backend.sh` — a stale copy, read by no stage.** `apply_splice.py`
splices from the framework repo's `launchers/fs_container_backend.sh` (53 KB). This is a 27 KB
snapshot that has diverged by 1181 lines. It is retained only so the divergence is visible
rather than silently discarded; nothing should ever be repointed at it.

Neither of the last two is published, for the same reason stated as a general rule: an unread
file in a public tree is pure disclosure surface. It can never fail a build, so nothing ever
forces anyone to look at it again.

## Rule

A file in this directory must be either (a) re-derivable from a named upstream, or (b)
documented above as unreconstructable with the measurement that established it. A third
category — a file the build reads that nobody can account for — is the #136/#137 defect
class, and it is exactly the state this directory was created to make impossible to reach
by accident.

A fourth category was added by #157: a file that ships but that no scan category owns. That
one is enforced mechanically rather than by this document, because a rule written down is a
rule nobody re-derives. `h100/PUBLISH_SET.txt` declares what ships, and the build refuses
UNMEASURED if any entry in it falls outside every scan.
