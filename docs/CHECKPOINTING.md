# Checkpointing and recovery

FoundationScale treats a checkpoint as a claim that must carry its own coverage.
The founding incident of the audited estate was a MoE checkpoint in which 128
experts collapsed to 16 distinct storages, aliased 8 ways, and the artifact
passed `rc=0`, resume, loss-curve, tensor-count and dtype checks for two full
training runs. The tool built to catch it then reported `all_identity: true`
because the expert tensor set was empty and `all([])` is `True`. Every layer of
the checkpoint stack described here exists to make "a check that never touched
the data reported success" unrepresentable.

Two modules do the work:

| Module | Role |
|---|---|
| `src/foundationscale/checkpoint/dcp.py` | Partial, read-only weight access over DCP and safetensors artifacts; streaming comparison |
| `src/foundationscale/gates/checkpoint_gates.py` | The gates that *judge* a checkpoint at `FIRST_SAVE` / `SAVE` |

`src/foundationscale/verify/parity.py` (referenced from the README) is where
checkpoints are judged against each other using the same verdict vocabulary;
this document covers the reader layer and the save-path gate layer.

## Reading checkpoints: `dcp.py`

### The `WeightSource` protocol

`DcpReader` (a `torch.distributed.checkpoint` directory) and
`SafetensorsReader` (an HF-style safetensors export or a single
`.safetensors` file) both implement `WeightSource`:

| Member | Contract |
|---|---|
| `tensor_keys()` | Sorted tensor keys. Guaranteed non-empty for any live instance |
| `nontensor_keys()` | Non-tensor entries (DCP `_extra_state` blobs); empty for safetensors |
| `shape(key)` / `dtype(key)` | Declared geometry and on-disk dtype |
| `chunks(key)` | Stored `Chunk` records (`offsets` + `sizes`) |
| `read_chunk(key, offsets)` | Exactly one stored chunk |
| `read_box(key, lo, hi)` | Assembled sub-box `[lo, hi)`, coverage verified |
| `read_full(key)` | Whole tensor; raises unless stored coverage is exact |
| `close()` | Idempotent release of cached handles |

The load-bearing invariant: **a source with an empty tensor key set cannot
exist.** Both constructors refuse a zero-tensor artifact with
`CheckpointFormatError`, because downstream an empty key set becomes
`all([])`. In the audited 26B checkpoint, `state_dict_metadata` carried ~8,970
keys of which only ~928 were tensors — the rest were `BytesStorageMetadata`
`_extra_state` blobs — so the readers filter by type and keep the non-tensor
keys visible via `nontensor_keys()` instead of silently dropping them.

### Why per-chunk reads are safe

A DCP checkpoint's `.metadata` file is a plain pickle, and every stored chunk
range is a complete, self-describing `torch.save` ZIP archive. Slicing bytes
`[offset, offset+length)` out of a `.distcp` shard and passing them to
`torch.load(io.BytesIO(...), weights_only=True)` self-validates: an off-by-one
offset raises `UnpicklingError`, a truncated slice raises the stream-reader
central-directory error, and a range that does not begin with the `PK\x03\x04`
magic is rejected before that. Corruption raises `ChunkReadError` — carrying
`path`, `key`, the chunk `offsets`, and the original exception — instead of
returning plausible garbage. This is what makes per-chunk reads safe enough to
run as a gate at `FIRST_SAVE`/`SAVE`/`EXPORT` without a process group, CUDA,
or a rank.

### Coverage is a returned fact

Every assembled read returns a `ReadResult`:

```python
ReadResult(
    key=...,
    tensor=...,            # on-disk dtype; no upcasting
    chunks_read=...,
    elements_covered=...,
    elements_expected=...,
    bytes_read=...,
)
```

`ReadResult.complete` is true iff stored chunks covered every declared element
exactly once. `DcpReader.read_full` and `read_box` raise
`IncompleteCoverageError` — carrying `elements_covered` and
`elements_expected` — unless covered elements exactly equal the declared
extent, and also if stored chunks overlap (overlap would double-count volume
and hide holes). This is the direct fix for the reference probe's `read_full`,
which silently zero-filled uncovered regions: the vacuous-truth bug wearing a
different hat. A caller that ignores the coverage fields and treats `tensor`
as ground truth is re-implementing that bug.

Box validation is equally strict: `_validate_box` rejects a zero-extent
dimension (`lo == hi`) on any dim where the tensor itself has extent, because
such a read assembles nothing yet would report complete coverage. Dims where
the tensor itself is empty (`shape[i] == 0`) are exempt — full coverage of
zero declared elements is a true statement.

### Memory behaviour, stated with its measured scope

From the module's own measurements on the audited 51.7 GB sharded checkpoint:

| Operation | Cost |
|---|---|
| Parsing `.metadata` alone | ~180 MB RSS |
| Reading 128 expert chunks (~4 MB each) | +18 MB RSS, ~1.9 s (peak 198 MB) |
| Materializing one `262144 x 2816` bf16 embedding + `.float()` | 11.5 GB peak RSS |

Memory is a function of what you request. Chunk-level reads are cheap;
whole-tensor materialization of a large embedding is a different operation
with a different bill. `compare_keys` exists for the second case.

## Comparing checkpoints: `compare_keys`

```python
comparison = compare_keys(source_a, source_b, key, block_rows=4096)
```

`compare_keys` streams row blocks through `read_box` and accumulates every
sum, sum of squares, and the dot product in **float64** — the fix for the
probe defect in which float32 reductions underflowed and a tensor's cosine
with itself printed 1.80. Rough memory is
`block_rows * row_size * (2 dtypes + float64 temporaries)` — about 0.5 GB for
`block_rows=4096` on a 2816-wide tensor, versus 11.5 GB peak for the
read-full-and-upcast path.

As a guard against that regression ever returning, when the two tensors are
bitwise identical, finite, and non-empty, the function *asserts* the computed
cosine is 1.0 and `max_abs_diff` is 0.0. A broken accumulator raises
`AssertionError` instead of laundering a bad number into a report.

### The verdict vocabulary

`TensorComparison.verdict` is machine-consumed by downstream gates. Adding a
value is a breaking change, and that is deliberate: a consumer written before
a new verdict must see a token it does not recognize and fail closed, never
silently treat a poisoned artifact as ordinary divergence.

| Verdict | Meaning |
|---|---|
| `EXACT` | Bitwise identical: same declared encoding AND same decoded values, elementwise |
| `CLOSE` | Within both tolerances (`close_max_abs_diff`, `close_min_cosine`) |
| `DIFFER` | Finite content outside both tolerances |
| `NON_FINITE` | NaN/Inf present without bitwise identity |
| `SHAPE_MISMATCH` | Sources disagree on geometry; zero elements compared, visibly so |
| `DTYPE_MISMATCH` | Declared encodings differ; adjudicated at metadata level before streaming |
| `NO_ELEMENTS` | Shapes agree but declare zero elements; a stated abstention, not a grade |

Three design rules to internalize:

1. **`NO_ELEMENTS` abstains, never passes.** An empty tensor is a legitimate
   artifact (zero-row padding embeddings, unallocated buffers), but an unread
   nothing must not earn a pass grade. The record mirrors the mismatch
   branch's "compared nothing, visibly so" expression: `bitwise_equal=False`,
   unbounded diffs, `elements=0` naming the denominator, `chunks_read=0`.
2. **`DTYPE_MISMATCH` is the abstention's mirror.** Unlike `NO_ELEMENTS`, the
   difference *was* observed — in the declared encodings, before any read — so
   abstaining would un-state a fact the function is holding. It cannot reuse
   `SHAPE_MISMATCH` (the geometry agrees), cannot stream into `DIFFER` (no
   `EXACT` claim is possible across encodings; a bf16 export whose content is
   exactly representable in f32 would otherwise compare equal on decoded
   values and launder a requantization into an identity claim), and is a
   finding, not a corruption, so it returns rather than raising.
3. **A negative extent raises `CheckpointFormatError`.** It is not a
   disagreement and not an emptiness — no tensor has a negative dimension, so
   both "identical" and "different" would be fabrications. `DcpReader.shape`
   reads an unvalidated pickle, so a negative dim means corrupt metadata and
   the artifact, not the comparison, failed.

Adjudication precedence — dtype mismatch, then shape mismatch, then
zero-element abstention — matches the parity layer's order, because two
layers of one framework must never disagree about which finding owns a key.

### Non-finite content and JSON

`nonfinite_elements` counts NaN/Inf on **every** verdict, including `EXACT`:
bitwise identity is a statement about bytes, not about the finiteness of the
weights those bytes encode. Identical ±inf pairs are parity over bytes with
the poison surfaced, not laundered into a tolerance verdict. `cosine` is
`None` whenever no finite, non-zero direction exists to compare — an all-zero
tensor on either side, or NaN/Inf in the content.

`TensorComparison.to_dict()` survives `json.dumps(..., allow_nan=False)`:
non-finite floats become their names (`"inf"`, `"-inf"`, `"nan"`) via
`_json_float`. The alternative is a strict encoder crashing on the report of
the very defect the comparison exists to surface. IEEE comparisons swallow
NaN (`max(0.0, nan) == 0.0`), which is exactly why the tally exists as its
own fact rather than being derivable from the statistics.

### `open_weights` and refused formats

`open_weights(path)` sniffs by layout, not by hope:

- A directory containing `.metadata` (`DCP_METADATA_FILENAME`) → `DcpReader`.
- `model.safetensors.index.json` (`SAFETENSORS_INDEX_FILENAME`) and/or
  `*.safetensors` shards, or a single `.safetensors` file →
  `SafetensorsReader`.
- Both formats present → `CheckpointFormatError` naming what was found.
- Only plain `.bin` torch pickles present → **refused**. A whole-file
  `torch.load` cannot be partially read or verified chunk-wise, and silently
  loading one in a gate is how verification quietly stopped happening in the
  audited estate.
- Nothing recognizable → `CheckpointFormatError` listing what was there.

`SafetensorsReader` holds shards in a bounded LRU `_HandleCache`
(`handle_cache_size=16` by default): the reference probe kept one handle per
shard forever, which is a file-descriptor leak across a long verification
run. Each safetensors tensor lives contiguously in exactly one shard, so
`chunks` always reports a single whole-tensor chunk and incomplete coverage
cannot occur silently.

## Judging checkpoints: `checkpoint_gates.py`

The gates consume `CheckpointGateContext` — everything they need with no
torch dependency. `CheckpointGateContext.from_path` builds it from an on-disk
checkpoint, lazily importing `foundationscale.checkpoint` for a torch-free
metadata summary plus the run manifest that declares what the checkpoint
*should* contain. The manifest-side fields are the denominators:
`declared_fqns`, `num_experts`, `num_moe_layers`, `expected_expert_bytes`, and
optionally a reader-measured `expert_storage_bytes`. Comparing what *is*
there against only what *is* there is the vacuity trap; comparing it against
what the run *declared* is the check. If the real module's API drifts, the
adapter in `from_path` is the single place to fix.

One typing discipline matters above all: `num_experts` distinguishes **None
(UNKNOWN)** from **0 (DECLARED DENSE)**. Python's equalities launder
look-alikes (`False == 0`, `0.0 == 0`, `0j == 0` are all True), so
`_checked_num_experts` rejects a bool, float, complex, or negative count
before any gate logic runs — such a value blocks VACUOUS as a malformed
denominator rather than buying the dense-model SKIP.

### `TensorMeta` and the price table

Each entry carries `fqn`, `shape`, `dtype`, optional `storage_id`, and
`kind` (`"tensor"` vs `"extra_state"`). `storage_id` is what makes aliasing
visible without reading a tensor: two FQNs naming one storage are one tensor
wearing two names — the "aliased 8 ways" signature. `implied_nbytes` prices a
tensor from shape and dtype alone and returns `None` for a dtype outside
`_DTYPE_BYTES`: the old behaviour defaulted unknown dtypes to 4 bytes
silently, pricing a 1-byte `float8_e4m3fn` expert set at 4x its true volume
and then "matching" an honest manifest. A guessed element width is a
fabricated denominator, and the `int | None` return type makes mypy enforce
that consumers handle it.

### Recognized expert layouts

The selector sees three per-expert namings — Megatron local-name shards
(`...linear_fc[12].weight<i>`, the incident), Megatron global names
(`...experts.42.linear_fc1.weight`), and Mixtral/Qwen
(`...experts.<i>.<proj>.weight`) — plus the stacked layout that dominates HF
MoE (`...experts.gate_up_proj` / `...experts.down_proj`; Megatron's fused,
suffix-less `...linear_fc1.weight` is the same thing in older spelling).
Expert-ish names matching none of these families **block as an unrecognized
layout — never read as a dense model.** Mixtures of per-expert shards and
stacked tensors block as their own named MIXED verdict, because neither
family's denominator is defined over the mixture.

The vocabulary is exported as data so the producer of a run's declaration
sorts names by the *same* rules the gates adjudicate:
`mentions_expert` (broad — any expert-related segment; used to refuse a dense
declaration) and `matches_expert_family` (narrow — an expert weight in a
verifiable layout; used to price denominators).

### The four gates

| Gate | ID | Events | What it establishes |
|---|---|---|---|
| `ExpertDistinctnessGate` | `checkpoint.expert_distinctness` | `FIRST_SAVE`, `SAVE` | Experts exist at the declared count and occupy distinct storage |
| `ExpertByteVolumeGate` | `checkpoint.expert_bytes` | `FIRST_SAVE`, `SAVE` | Expert byte volume matches the manifest-declared volume |
| `SaveCompletenessGate` | `checkpoint.save_complete` | `FIRST_SAVE`, `SAVE` | Every declared tensor FQN is present (counting real tensors only) |
| `FirstSaveGate` | `checkpoint.first_save` | `FIRST_SAVE` | Composite of the three above at the cheapest catch point |

**`ExpertDistinctnessGate`** checks, per shard-stem, the declared count (16
on disk against 128 declared is 87.5% of the experts missing, not a format
variation) and within-stem storage aliasing. On stacked layouts it verifies
everything metadata still permits — leading dims against the declared expert
count, sibling byte ratios across layers, cross-layer storage-span sharing —
and then **abstains NOT_ESTABLISHED**: N duplicated slices occupy exactly the
one storage span N distinct slices occupy, so per-expert identity inside a
stacked tensor is metadata-invisible *by construction*. A FAIL would be false
(nothing wrong was found); a PASS would pretend. On a declared-MoE model with
zero expert tensors present, the outcome is VACUOUS — the exact difference
between this gate and the audit tool it replaces.

**`ExpertByteVolumeGate`** prices bytes over *distinct storage* (128
shape-correct FQNs aliased to one tensor imply the full declared volume while
one physical tensor exists). The incident ratio — 5.71 GB where 45.70 GB was
correct — is an exact 1/8, catchable in milliseconds from metadata alone. A
deficit of strictly more than 1% fails (`_DEFICIT_PER_MILLE = 10`; exactly 99%
passes). An implied-versus-physical disagreement in either direction fails
under its own description: aliasing hides missing weights behind shared
spans, while a physical surplus means bytes no expert FQN names. Without a
manifest-supplied `expected_expert_bytes`, the gate abstains NOT_ESTABLISHED
and names the missing denominator — pricing a checkpoint against itself is
the vacuity trap.

**`SaveCompletenessGate`** intersects declared FQNs with present FQNs after
filtering both sides to real tensors. A missing manifest yields VACUOUS via
zero coverage ("'what is there matches what is there' is not a check"); an
empty declared set takes the same enforced-VACUOUS path.

**`FirstSaveGate`** runs the three sub-gates and prices abstentions off the
machine-readable `AbstentionKind`, never off prose:

- `NOT_APPLICABLE` — reachable only via a positive dense declaration
  (`num_experts == 0`) — is removed from the denominator, and the PASS names
  the inapplicable sub-gates: "verified 1/1 applicable …; 2 inapplicable
  (named)", never "verified 3/3".
- Any other SKIP (`NOT_ESTABLISHED`, or an undeclared None) **stays in the
  denominator**: "I could not check" is not "there was nothing to check". A
  clean stacked checkpoint therefore reports 2/3 verified and downgrades to
  UNDERCOVERED — blocked for a true reason instead of silently green.
- A blocking sub-gate fails the composite with the offenders named.

### Controls

Every gate ships fixtures in `fixtures.py` proving it fires, run without
torch or large I/O via `verify_controls`. The `empty-expert-set` control
exists solely to prevent the detector-itself-silently-passing bug from
recurring in this codebase; `manifestless-expert-set` pins that None
(UNKNOWN) blocks VACUOUS rather than taking the dense SKIP;
`malformed-dense-count-bool` / `-float` pin the type-laundering door;
`stacked-cross-layer-alias`, `stacked-a-fraction-of-experts`,
`unknown-expert-layout`, and `mixed-expert-layout` pin the stacked-layout
failures; `right-count-but-aliased` pins the variant where only
physical-byte pricing fires.

## Recovery knobs

The launch-plane environment surface includes `FS_RESUME_CKPT` and
`FS_RESUME_STEP`, wired to the launchers rather than the package.

**What is not here:** beyond the presence of those two variables in the
launch-plane environment surface, no resume implementation, validation of
resume behaviour, or CLI command for checkpoint recovery exists in the
package source material. The current behaviour is that checkpoint *judgment*
lives in the gates above; recovery itself is a harness concern.

## The stated limit

From README §4, repeated where it bites: **the suite writes real checkpoints
to disk and reads them back single-process.** Multi-rank save/reload shapes
are reproduced from the audit record, not re-observed — so multi-rank
recovery is **specified, not verified here**. The readers above run without a
process group, CUDA, or a rank by design; that is what makes them cheap
enough to run at every save, and it is also the boundary of what this
document may claim.
