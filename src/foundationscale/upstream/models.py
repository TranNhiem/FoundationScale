"""The speech model registry: every model this plane will load, as typed data.

PHASE 2a of docs/research/upstream_integration.md. The upstream design's first rule is
"declare, don't discover": a model is registered here or it is refused. Nothing searches
Hugging Face, nothing globs a checkpoint directory, and nothing infers an architecture
from a name -- a person transcribes the model card's facts into one :class:`ModelEntry`
once, and :func:`registry_problems` checks the transcription for self-contradiction on
every CI run. A discovered model is a model nobody has looked at; a registered one has a
license, a locator, an architecture declaration and, when the card publishes one, a
number to reproduce.

What one entry pins down, and why each piece is data rather than behaviour:

* ``upstream_ref`` -- the HF repo id or NeMo pretrained name exactly as upstream spells
  it (case included: ``google/gemma-4-E4B-it``). "The gemma 4 e4b instruct model" is not
  a locator, and a hub search for it returns two repos with different licenses.
* ``family`` / ``nemo_class`` -- where the architecture's declaration lives. For HF
  models, a FamilySpec name (a registration in ``foundationscale.families.registry``,
  which holds the measured tower and scope layout); for NeMo models, the dotted class
  path ``restore_from`` dispatches to. Exactly one is meaningful per backend, and the
  registry refuses the mismatch rather than silently ignoring the field that does not
  apply (``hf_with_nemo_class``, ``nemo_without_class``).
* ``license_weights`` + ``attribution`` -- what the weights may be used under and who
  must be credited. This is transcription of the card, not policy: the adjudication
  plane acts on these and must never have to guess (``empty_field``).
* ``trainable_prefixes`` / ``frozen_prefixes`` -- what fine-tuning trains and what it
  freezes, as module-name prefixes adjudication reads directly. ``lora_marker`` is the
  one designed carve-out: inside a frozen scope, PEFT tensors matching the marker (e.g.
  ``.lora_``) stay trainable while the rest of the scope does not. With no frozen scope
  the marker is meaningless (``lora_marker_without_frozen``); the same prefix declared on
  both sides is a contradiction (``overlapping_scopes``); and a NeMo entry with no
  trainable prefixes would make the run's trainable set depend on whatever default the
  loader happens to carry (``nemo_without_scopes``).
* ``reference`` -- one task and metric from the model card: the value it publishes, the
  URL to read it off, and the ABSOLUTE tolerance that decides ``reproduced``, fixed
  BEFORE the run (rule 4 of the upstream design, verification-first). A tolerance picked
  after seeing our number is not a tolerance, it is a description. Both numbers must be
  read over the SAME test set: when we can only measure another split (this estate holds
  LibriSpeech dev-clean; the cards report test-clean), the reading goes in
  ``measured_note`` with its provenance and ``measured`` stays None, because an
  across-split comparison exported as ``reproduced`` is a lie every later stage will
  happily read.

``status`` is the honest stamp over all of it. EXPERIMENTAL: loads and runs here, card
not reproduced. SUPPORTED: ``reference.reproduced`` holds on the card's own test set --
and an entry claiming SUPPORTED without a reproduction is a registry error
(``supported_without_reproduction``), not a hope.

Stdlib only, Python >= 3.10: this module must import without transformers, nemo_toolkit
or peft installed, so catalogue inspection and every refusal path work in a bare venv.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "MODELS",
    "Backend",
    "ModelEntry",
    "ModelKind",
    "ReferenceResult",
    "SupportStatus",
    "entries_for_backend",
    "get_model",
    "registry_problems",
]


class Backend(str, Enum):
    """Which upstream framework serves a model."""

    HF = "hf"  # Hugging Face transformers (plus its processors / tokenizers)
    NEMO = "nemo"  # NeMo: restore_from on a .nemo checkpoint


class ModelKind(str, Enum):
    """The decoding shape of a model: what the evaluation lane must drive it as.

    A zero-shot lane that sends a transcription instruction to an AUDIO_LLM gets an
    answer ("Sure! The audio says ...") where a SEQ2SEQ model would return the transcript
    -- a real measured failure mode (see the qwen2-audio entry's ``measured_note``). Kind
    is declared so the correct prompt/decode path is chosen from data, not discovered
    from a wrong number afterwards.
    """

    AUDIO_LLM = "audio_llm"  # chat-style audio+text LLM: prompt in, generated text out
    SEQ2SEQ = "seq2seq"  # encoder-decoder transcription (Whisper)
    CTC = "ctc"  # encoder + CTC head: frames to tokens, no decoder
    AED = "aed"  # attention encoder-decoder (Canary)
    SPEECH_LLM = "speech_llm"  # speech language model (SALM: perception + a pretrained LLM)


class SupportStatus(str, Enum):
    """How honest we can be about a model's agreement with its own card."""

    EXPERIMENTAL = "experimental"  # loads and runs here; the card's number is NOT reproduced
    SUPPORTED = "supported"  # reference.reproduced holds on the card's own test set


@dataclass(frozen=True)
class ReferenceResult:
    """One model-card claim, and what we measured against it.

    Acceptance is ``abs(measured - upstream_value) <= tolerance`` -- absolute, in the
    metric's own units, fixed BEFORE the run. A relative tolerance rewards rounding in
    the card (our 2.28 vs the card's 2.2 is 3.6% relative and 0.08 absolute); the
    disagreement we actually care about lives in the metric's units, so that is where the
    boundary is drawn.

    ``upstream_value`` is None while the number has not been looked up yet: the claim is
    registered with its task, source and tolerance first, so the lookup cannot be
    skipped, and the value is filled in when it is read off the card.

    ``measured`` is None until OUR number exists for the SAME split. When we can only
    measure another split -- we run the full LibriSpeech dev-clean where the cards report
    test-clean -- the reading goes in ``measured_note`` with the campaign that produced
    it and is never copied into ``measured``. A cross-split comparison exported as a
    boolean is indistinguishable from a reproduction to every consumer downstream, which
    is exactly the failure this registry exists to prevent.
    """

    task: str
    metric: str
    upstream_value: float | None
    source: str
    tolerance: float
    measured: float | None = None
    measured_note: str = ""

    @property
    def reproduced(self) -> bool:
        """True only when BOTH numbers exist and agree within the registered tolerance.

        The boundary is inclusive (exactly at tolerance reproduces): the tolerance is the
        width of "same number, different rounding", and asking for strict inequality
        would make a card that rounds to its own precision unreproducible by
        construction. Missing data is never a pass -- measured absent is not 0.
        """
        return (
            self.measured is not None
            and self.upstream_value is not None
            # the epsilon keeps the inclusive boundary inclusive under float addition
            and abs(self.measured - self.upstream_value) <= self.tolerance + 1e-9
        )


@dataclass(frozen=True)
class ModelEntry:
    """Everything the training and evaluation planes need to know about one speech model.

    Field-order note: the always-present facts come first (locator, kind, license); the
    optional declarations follow with defaults, so an entry that omits one states
    "not declared" in the data rather than inventing a value.

    ``trainable_prefixes`` and ``frozen_prefixes`` are module-name prefixes exactly as
    the framework spells them (``encoder``, ``perception.proj``), not regexes: adjudication
    matches them against qualified module names and refuses on overlap, so a declaration
    must be exact enough to copy into a scope list. ``lora_marker`` (e.g. ``.lora_``) is
    how a frozen scope keeps its PEFT tensors trainable while everything else in the scope
    stays frozen -- the marker can only mean something when there is a frozen scope to
    carve it out of.

    ``requires`` is (package, version spec) as data: the worker's environment check reads
    it to REFUSE rather than ImportError at model build. ``unpinned`` is a stated version
    spec too -- it means "installed the day of the run, recorded in the run manifest", not
    "whatever happens to resolve".
    """

    id: str
    backend: Backend
    upstream_ref: str
    kind: ModelKind
    license_weights: str
    attribution: str = ""
    family: str = ""  # FamilySpec name; HF backend only
    nemo_class: str = ""  # dotted class path; NeMo backend only
    trainable_prefixes: tuple[str, ...] = ()
    frozen_prefixes: tuple[str, ...] = ()
    lora_marker: str = ""  # e.g. ".lora_" when a frozen scope keeps PEFT tensors trainable
    requires: tuple[tuple[str, str], ...] = ()
    reference: ReferenceResult | None = None
    status: SupportStatus = SupportStatus.EXPERIMENTAL
    notes: str = ""
    profile: str = ""  # upstream profile it runs in; "" = its backend's default profile
    eval_script: str = ""  # own decode script (repo-relative); decode-only HF entries need one


# Status rule (rule 4, "integrated" means "reproduced"): an entry is SUPPORTED only when its
# card's own LibriSpeech TEST number was reproduced here on the full split under the card's
# normalizer (validation_campaigns/speech_repro). Cards that publish no such number stay
# EXPERIMENTAL; their ``measured_note`` records what we did measure, on dev-clean.
MODELS: tuple[ModelEntry, ...] = (
    ModelEntry(
        id="gemma-4-e4b-it",
        backend=Backend.HF,
        upstream_ref="google/gemma-4-E4B-it",
        kind=ModelKind.AUDIO_LLM,
        license_weights="apache-2.0",
        family="gemma4",
        requires=(("transformers", ">=5.5"),),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=None,
            source="",
            tolerance=0.3,
            measured=None,
            measured_note="dev-clean full 4.07 zero-shot (validation_campaigns/speech_p7)",
        ),
    ),
    ModelEntry(
        id="whisper-large-v3",
        backend=Backend.HF,
        upstream_ref="openai/whisper-large-v3",
        kind=ModelKind.SEQ2SEQ,
        license_weights="apache-2.0",
        family="whisper",
        requires=(("transformers", ">=4.40"),),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=None,
            source="",
            tolerance=0.3,
            measured=None,
            measured_note="dev-clean full 2.25 zero-shot",
        ),
    ),
    ModelEntry(
        id="qwen2-audio-7b-instruct",
        backend=Backend.HF,
        upstream_ref="Qwen/Qwen2-Audio-7B-Instruct",
        kind=ModelKind.AUDIO_LLM,
        license_weights="apache-2.0",
        family="qwen2_audio",
        requires=(("transformers", ">=4.45"),),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=None,
            source="",
            tolerance=0.3,
            measured=None,
            measured_note=(
                "dev-clean full 35.6 zero-shot with our prompt (answers instead of "
                "transcribing); 1.79 after fine-tuning"
            ),
        ),
    ),
    ModelEntry(
        id="parakeet-ctc-1.1b",
        backend=Backend.HF,
        upstream_ref="nvidia/parakeet-ctc-1.1b",
        kind=ModelKind.CTC,
        license_weights="cc-by-4.0",
        attribution="NVIDIA, Parakeet-CTC-1.1B",
        family="parakeet_ctc",
        requires=(("transformers", ">=5.5"),),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=1.83,
            source="https://huggingface.co/nvidia/parakeet-ctc-1.1b",
            tolerance=0.3,
            measured=1.846,
            measured_note=(
                "full split, Whisper EnglishTextNormalizer (the card's); FS normalizer reads "
                "2.054 -- validation_campaigns/speech_repro, 2026-10-10"
            ),
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="canary-1b-flash",
        backend=Backend.NEMO,
        upstream_ref="nvidia/canary-1b-flash",
        kind=ModelKind.AED,
        license_weights="cc-by-4.0",
        attribution="NVIDIA, Canary-1B-Flash",
        nemo_class="nemo.collections.asr.models.EncDecMultiTaskModel",
        # AED fine-tuning on NV-TRML-EN translates both towers: the encoder hears the
        # new domain and the transf_decoder has to emit the new target. Both train;
        # neither scope listed here is frozen.
        trainable_prefixes=("encoder", "transf_decoder"),
        requires=(("nemo_toolkit", ">=3.1"),),
        reference=ReferenceResult(
            task="librispeech_test_other",
            metric="wer",
            upstream_value=2.87,
            source="https://huggingface.co/nvidia/canary-1b-flash",
            tolerance=0.3,
            measured=2.874,
            measured_note=(
                "full split, Whisper EnglishTextNormalizer (the card's); FS normalizer reads "
                "3.084 -- validation_campaigns/speech_repro, 2026-10-10"
            ),
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="canary-qwen-2.5b",
        backend=Backend.NEMO,
        upstream_ref="nvidia/canary-qwen-2.5b",
        kind=ModelKind.SPEECH_LLM,
        license_weights="cc-by-4.0",
        attribution="NVIDIA, Canary-Qwen-2.5B (LLM: Qwen3-1.7B, apache-2.0)",
        nemo_class="nemo.collections.speechlm2.models.SALM",
        # The SALM split is asymmetric by design: the perception side (audio encoder and
        # its projection) adapts to the domain in full, the Qwen3 LLM with its embedding
        # stays as upstream tuned it -- except the LoRA tensors the marker names inside
        # the frozen scope. The carve-out is data so adjudication can print the trainable
        # set and see what a run will actually update.
        trainable_prefixes=("perception.encoder", "perception.proj"),
        frozen_prefixes=("llm", "embed_tokens"),
        lora_marker=".lora_",
        requires=(("nemo_toolkit", ">=3.1"), ("peft", "unpinned")),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=1.61,
            source="https://huggingface.co/nvidia/canary-qwen-2.5b",
            tolerance=0.3,
            measured=1.624,
            measured_note=(
                "full split, Whisper EnglishTextNormalizer (the card's); FS normalizer reads "
                "1.801 -- validation_campaigns/speech_repro, 2026-10-10"
            ),
        ),
        status=SupportStatus.SUPPORTED,
    ),
    ModelEntry(
        id="qwen3-asr-1.7b",
        backend=Backend.HF,
        upstream_ref="Qwen/Qwen3-ASR-1.7B-hf",
        kind=ModelKind.AUDIO_LLM,
        license_weights="apache-2.0",
        attribution="Qwen team, Alibaba Cloud, Qwen3-ASR-1.7B",
        family="qwen3_asr",
        # The class needs transformers >= 5.13, so it trains and decodes in the candidate
        # profile. Its own eval script drives apply_transcription_request (the forced-language
        # prompt); the generic audio-LLM eval would build a different prompt.
        requires=(("transformers", ">=5.13"),),
        reference=ReferenceResult(
            task="librispeech_test_clean",
            metric="wer",
            upstream_value=1.63,
            source="https://huggingface.co/Qwen/Qwen3-ASR-1.7B (README results table)",
            tolerance=0.3,
            measured=1.643,
            measured_note=(
                "full split, Whisper EnglishTextNormalizer; FS normalizer reads 1.883. "
                "test_other 3.368 vs README 3.38 -- validation_campaigns/speech_repro, 2026-10-10. "
                "FS fine-tune on Earnings-22: held-out 14.54 -> 9.66 (validation_campaigns/"
                "speech_q3asr)"
            ),
        ),
        status=SupportStatus.SUPPORTED,
        profile="hf-cand-519",
        eval_script="validation_campaigns/speech_repro/eval_qwen3_asr.py",
    ),
)


def get_model(model_id: str) -> ModelEntry:
    """The registry entry for ``model_id``.

    A KeyError-style refusal raised as a ``ValueError`` naming every known id: the
    missing model is a data gap, not an internal invariant break, and a refusal that
    does not say what IS registered is just a stop. The message repeats what the module
    docstring promises -- that models are declared here, never discovered -- because
    whoever hit it was almost certainly hoping for the opposite.
    """
    for entry in MODELS:
        if entry.id == model_id:
            return entry
    known = ", ".join(entry.id for entry in MODELS)
    raise ValueError(
        f"unknown model id {model_id!r}; the model registry holds: {known}. Model ids "
        "are declared in foundationscale.upstream.models.MODELS and never discovered "
        "from the hub, a checkpoint directory or a prompt: add a ModelEntry with its "
        "upstream_ref, license_weights and reference, or use one of the ids above"
    )


def registry_problems(
    models: tuple[ModelEntry, ...] = MODELS,
    family_names: Collection[str] | None = None,
) -> list[str]:
    """Every way the registry disagrees with itself; ``[]`` when it is consistent.

    Codes: ``duplicate_id:<id>``, ``empty_field:<id>:<field>`` (for the id,
    upstream_ref and license_weights fields that identification and licensing read),
    ``hf_without_family:<id>`` (unless it declares its own ``eval_script``),
    ``unknown_family:<id>:<name>``, ``unknown_profile:<id>:<name>``,
    ``hf_with_nemo_class:<id>``, ``nemo_without_class:<id>``,
    ``nemo_without_scopes:<id>``, ``overlapping_scopes:<id>``,
    ``lora_marker_without_frozen:<id>``, ``supported_without_reproduction:<id>``.

    ``family_names`` defaults to the names of every registered FamilySpec, imported
    LAZILY inside this function: callers checking a crafted table pass their own set and
    pay no import cost, while the real check still names the real registrations rather
    than a copy of their names that could drift here unnoticed. ``unknown_family`` is
    what catches that drift for the real table.
    """
    if family_names is None:
        from foundationscale.families.registry import REGISTRY as _FAMILY_REGISTRY

        family_names = tuple(spec.name for spec in _FAMILY_REGISTRY)

    from foundationscale.upstream.profiles import PROFILES

    profile_names = {p.name for p in PROFILES}
    problems: list[str] = []
    seen: set[str] = set()
    for entry in models:
        if entry.id in seen:
            problems.append(f"duplicate_id:{entry.id}")
        seen.add(entry.id)
        for field in ("id", "upstream_ref", "license_weights"):
            if not str(getattr(entry, field)).strip():
                problems.append(f"empty_field:{entry.id}:{field}")
        if entry.backend == Backend.HF:
            # no FamilySpec means FS cannot train it; it may still be decoded by its own script
            if not entry.family.strip():
                if not entry.eval_script.strip():
                    problems.append(f"hf_without_family:{entry.id}")
            elif entry.family not in family_names:
                problems.append(f"unknown_family:{entry.id}:{entry.family}")
            if entry.nemo_class.strip():
                problems.append(f"hf_with_nemo_class:{entry.id}")
        elif entry.backend == Backend.NEMO:
            if not entry.nemo_class.strip():
                problems.append(f"nemo_without_class:{entry.id}")
            # NeMo adjudication prints and freezes exactly these prefixes; an entry that
            # declares none would leave the trainable set to the loader's default.
            if not entry.trainable_prefixes:
                problems.append(f"nemo_without_scopes:{entry.id}")
        # The same prefix on both sides is a contradiction no ordering rule can resolve.
        if set(entry.trainable_prefixes) & set(entry.frozen_prefixes):
            problems.append(f"overlapping_scopes:{entry.id}")
        # The marker only carves something out of a frozen scope; without one it names
        # an exception to nothing.
        if entry.lora_marker.strip() and not entry.frozen_prefixes:
            problems.append(f"lora_marker_without_frozen:{entry.id}")
        if entry.profile and entry.profile not in profile_names:
            problems.append(f"unknown_profile:{entry.id}:{entry.profile}")
        if entry.status == SupportStatus.SUPPORTED and not (
            entry.reference is not None and entry.reference.reproduced
        ):
            problems.append(f"supported_without_reproduction:{entry.id}")
    return problems


def entries_for_backend(
    backend: Backend | str,
    models: tuple[ModelEntry, ...] = MODELS,
) -> tuple[ModelEntry, ...]:
    """The registry entries served by one backend, in registry order.

    Accepts the string spelling too (``entries_for_backend("hf")``): the worker config
    carries ``Backend`` values as strings, and coercing here means an unknown spelling
    raises from ``Backend(...)`` instead of silently returning an empty tuple and making
    a run look like it knew two fewer models than the tree has.
    """
    wanted = Backend(backend)
    return tuple(entry for entry in models if entry.backend == wanted)
