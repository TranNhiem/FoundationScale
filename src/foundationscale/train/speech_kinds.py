"""The P2 SPEECH model kinds: which geometry a config declares, and one verified batch per geometry.

audio.py answers "may sound enter this run at all" and builds the ONE batch shape it
measured (a causal LM with an audio tower, through the chat template). FoundationScale
trains that same sound through THREE geometries whose label contracts differ -- where
the loss lives, what pads a label row, how long one row may be -- and this module is
the dispatch: which geometry a config declares, which Auto class loads it, what the
processor must expose per geometry, and how one verified batch per non-chat geometry
is shaped.

  audio_llm   causal LM with an audio tower (Gemma-4, Qwen2-Audio), prompt via
              ``processor.apply_chat_template``. Already handled by audio.py's
              ``train_audio_collator_or_refuse`` -- this module detects the kind and
              DELEGATES. A second implementation of that batch would be a second
              contract for one measured shape (P0 measured input_features [B, T, 128]
              arriving in ONE apply_chat_template call, 2026-10-06).
  seq2seq     encoder-decoder ASR (Whisper), ``AutoModelForSpeechSeq2Seq``. Labels
              carry the tokenizer's own prefix tokens and pad with -100; one row is
              bounded by Whisper's 30 s window.
  ctc         encoder + CTC head (Parakeet, wav2vec2, hubert), ``AutoModelForCTC``.
              Labels pad with ``pad_token_id`` -- the CTC BLANK -- and never -100
              (see ``train_ctc_collator_or_refuse``).

Three rules hold here:

  never guess            ``speech_model_kind`` returns None for a config whose own
                          declarations name no geometry, and the CALLER refuses. Each
                          kind decides the loss module and the label padding; a guess
                          here trains labels through the wrong one and still reports
                          success.
  refuse at 96, never    every collator builder refuses through the SAME mechanism as
  fix silently           audio.py -- its ``_audio_refuse_exit_96``, imported here
                          rather than reimplemented. No resampling, no truncation, no
                          dropped rows: a row the loader rejects is a REFUSED run with
                          its row id and its reason, never a shorter batch.
  abstain where there    the placeholder-vs-tower gate is a GEMMA-4 measurement
  is no measurement      (audio.py's per-row ``_compute_audio_num_tokens`` check);
                          seq2seq and ctc have no placeholder tokens to count, so that
                          gate calls NOTHING here -- an abstention, not a pass.

torch, numpy, soundfile and transformers are unreachable from here at import time
(stdlib and this package's own audio module are not): the speech gates run on boxes
without the training stack, so every heavy import enters inside the function that
needs it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from foundationscale.train.audio import (
    AUDIO_COLUMN_ENV,
    AudioLoadError,
    SharedAudioCoverage,
    _as_int,
    _audio_refusal,
    _audio_refuse_exit_96,
    load_audio,
    max_audio_seconds,
    refuse_if_audio_features_dropped,
)

__all__ = [
    "SPEECH_KINDS",
    "WHISPER_WINDOW_SECONDS",
    "SpeechCollator",
    "auto_class_name",
    "load_speech_model",
    "refuse_if_features_dropped",
    "speech_model_kind",
    "speech_support_refusal",
    "train_ctc_collator_or_refuse",
    "train_seq2seq_collator_or_refuse",
]

# The private imports above are audio.py's own pieces, imported rather than
# rewritten (its docstring sanctions duplication to keep a bare interpreter's
# import contract simple; that contract is stdlib-only for audio too, so nothing is
# gained by copying):
#   _audio_refuse_exit_96  the ONE refusal mechanism -- a "REFUSAL (exit 96):" line
#                          on stderr then SystemExit(96). Duplicating it would let
#                          a future edit leave two mechanisms that disagree.
#   _audio_refusal         the ONE refusal wording ("... declared via
#                          AUDIO_COLUMN_ENV, but <missing piece>: ... Unset
#                          ... to train text-only"), so the loop, the audio plane
#                          and this module tell an operator one story about one
#                          declared column.
#   _as_int                ints as ints and bools NOT as ints -- True read as token 1
#                          would place every audio block in somebody else's
#                          vocabulary.


SPEECH_KINDS: tuple[str, ...] = ("audio_llm", "seq2seq", "ctc")
"""The speech geometries, closed and in dispatch order. A kind outside this tuple has
no declared loss module, no declared label padding and no declared row window, so it
has no collator and no Auto class -- the caller refuses."""

WHISPER_WINDOW_SECONDS: float = 30.0
"""One row's maximum waveform for the seq2seq kind. MEASURED, not configured: Whisper's
encoder consumes exactly 30 s of 16 kHz audio (3000 mel frames -- its feature extractor
pads every clip UP to that width), so a longer row cannot be fed whole and must not be
truncated (a truncated row is the silent-drop defect one layer down: measured sound
quietly missing from the training input)."""

# The config's own declaration of family -> geometry. Everything is one table so a
# reader can see every mapping this package makes in one place -- and see that no
# model_type maps to a kind by fuzzy matching ("anything at all" -> audio_llm would
# be a guess wearing a table's clothes).
_SPEECH_KIND_BY_MODEL_TYPE: Mapping[str, str] = {
    "whisper": "seq2seq",
    "parakeet_ctc": "ctc",
    "parakeet": "ctc",
    "wav2vec2": "ctc",
    "hubert": "ctc",
    "gemma4": "audio_llm",
    "qwen2_audio": "audio_llm",
    "qwen3_asr": "audio_llm",
}

# Geometry -> the transformers Auto class whose from_pretrained carries it. The
# class NAME is data (and ``auto_class_name``'s contract) rather than a repeated if:
# one table decides both the string a caller logs and the attribute loaded.
_AUTO_CLASS_BY_KIND: Mapping[str, str] = {
    "audio_llm": "AutoModelForCausalLM",
    "seq2seq": "AutoModelForSpeechSeq2Seq",
    "ctc": "AutoModelForCTC",
}


def speech_model_kind(config: Any) -> str | None:
    """Which of :data:`SPEECH_KINDS` the HF ``config`` declares, or ``None`` -- never a guess.

    ``None`` is an OUTCOME, not an error: the caller refuses the run when the config's
    own declarations name no geometry, because each kind decides the loss module, the
    label padding and the row window, and a guess here trains labels through the wrong
    one while reporting success.

    Two axes, in THIS order:

      1. ``model_type`` against :data:`_SPEECH_KIND_BY_MODEL_TYPE` -- the config's own
         family declaration, read with ``getattr`` because a config saved without one
         is a real artefact and its absence is another ``None``.
      2. the ``architectures`` class names -- the auto-map's own choice of head.
         ``*ForCTC`` says the CTC head outright and settles it. ``*ForConditionalGeneration``
         does NOT settle it (Gemma4ForConditionalGeneration is an ``audio_llm``), so
         the seq2seq reading additionally needs the class name to say Whisper;
         ``model_type == "whisper"`` never reaches this branch (the table answers it
         first). Any other name is UNKNOWN and stays ``None``.

    When the two axes disagree (``model_type="gemma4"`` with an ``*ForCTC`` class
    name) the TABLE wins: ``model_type`` is the config's declaration of family and the
    class name is one saved-away string -- the declaration outranks the artefact, and
    in list order the first recognised architecture outranks the ones after it.
    """
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str):
        kind = _SPEECH_KIND_BY_MODEL_TYPE.get(model_type)
        if kind is not None:
            return kind
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        for name in architectures:
            if not isinstance(name, str):
                continue
            if name.endswith("ForCTC"):
                return "ctc"
            if name.endswith("ForConditionalGeneration") and "Whisper" in name:
                return "seq2seq"
    return None


def auto_class_name(kind: str) -> str:
    """The transformers Auto class NAME that loads ``kind``; ValueError outside SPEECH_KINDS.

    Raises rather than refuses, and :func:`load_speech_model` is the surface that turns
    it into the run's exit-96 refusal: a pure name lookup that killed the interpreter
    for a caller's typo would make the typo untestable, and the judgement of what ends
    a run belongs to the surface that runs it.
    """
    name = _AUTO_CLASS_BY_KIND.get(kind)
    if name is None:
        raise ValueError(
            f"speech kind {kind!r} is not one of {list(SPEECH_KINDS)}; there is no Auto class, "
            "no label padding and no row window for a kind nobody declared"
        )
    return name


def load_speech_model(kind: str, model_path: str, **kwargs: Any) -> Any:
    """Load the model for ``kind``; refuse (96) rather than guess.

    ``audio_llm`` loads through the class the checkpoint NAMES in ``config.architectures``
    when transformers has it: measured on transformers 5.5, AutoModelForCausalLM raises
    ValueError for qwen2_audio, while for Gemma-4 the named class is exactly the
    Gemma4ForConditionalGeneration AutoModelForCausalLM builds. The other kinds use
    ``auto_class_name(kind)`` (AutoModelForCausalLM on a Whisper config builds a
    decoder-only WhisperForCausalLM, which is why the kind decides the class).

    Two refusals, both exit-96 like every other train-loop stop here: a kind outside
    :data:`SPEECH_KINDS` (loading one would train against an objective nobody declared),
    and an absent transformers (mirrors ``resolve_audio_surface`` -- a missing training
    stack is a RED run, never a silent downgrade to some class that happens to exist).
    """
    import importlib  # deferred: this module must import without transformers

    class_name = _AUTO_CLASS_BY_KIND.get(kind)
    if class_name is None:
        _audio_refuse_exit_96(
            f"speech kind {kind!r} is not one of {list(SPEECH_KINDS)}: its geometry is "
            "undeclared (which loss module, which label padding, which row window), so "
            "loading one would train against an objective nobody measured. Refusing "
            "rather than guessing."
        )
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        _audio_refuse_exit_96(
            f"speech kind {kind!r} requires the {class_name} load surface but transformers "
            "is absent; the model cannot be constructed and no text-only fallback is "
            "permitted (it would train on the transcripts and drop the audio)"
        )
        raise  # pragma: no cover -- _audio_refuse_exit_96 never returns
    assert class_name is not None  # refused above when unknown
    if kind == "audio_llm":
        config = transformers.AutoConfig.from_pretrained(model_path)
        archs = getattr(config, "architectures", None) or []
        named = getattr(transformers, archs[0], None) if archs else None
        if named is not None:
            return named.from_pretrained(model_path, **kwargs)
    auto_class: Any = getattr(transformers, class_name)
    return auto_class.from_pretrained(model_path, **kwargs)


def speech_support_refusal(kind: str, processor: Any) -> str | None:
    """None when ``kind`` may train this run, else the one reason it may not.

    Checks in THIS order and names the FIRST piece missing -- a fixed order, so the
    text is a function of the state and not of argument order. Every refusal is the
    audio plane's one wording (:func:`foundationscale.train.audio._audio_refusal`), so
    it names the missing piece FIRST, then the cost, then the declared column and its
    env var (:data:`AUDIO_COLUMN_ENV`) as the way out. a refusal naming no way out is
    a stop, not an explanation.

    Per kind, what the processor must expose:

      audio_llm   non-``None`` ``audio_token`` and an int ``audio_token_id`` (the
                  placeholder expansion counts and masks by it) and
                  ``feature_extractor.sampling_rate`` a positive int. ``boa``/``eoa``
                  are a Gemma-4 surround convention and are NOT required: Qwen2-Audio
                  spells them ``audio_bos``/``audio_eos`` and inserts them itself --
                  requiring them would refuse the second family in this kind for
                  obeying its own template.
      seq2seq /
      ctc         ``feature_extractor`` with a positive int ``sampling_rate`` and a
                  ``tokenizer`` (the transcript path is the LABEL path for these
                  kinds; audio_llm's text rides the chat template instead and needs
                  no tokenizer of its own).

    ``processor`` is ``Any`` deliberately: every attribute is read with ``getattr``
    because a missing attribute is an OUTCOME this function returns, and an
    annotation that said the attribute was present would hide exactly the gap the
    refusal exists to name.
    """
    if kind not in SPEECH_KINDS:
        return _audio_refusal(
            f"the speech kind {kind!r} is not one of {list(SPEECH_KINDS)}, so what its "
            "processor must expose is undeclared and nothing here can verify the rows"
        )
    if processor is None:
        return _audio_refusal(
            f"the {kind!r} trainer has no audio processor, so its sampling rate cannot be "
            "read and no waveform can be required to match anything"
        )
    if kind == "audio_llm":
        if getattr(processor, "audio_token", None) is None:
            return _audio_refusal(
                f"the {kind!r} processor exposes no audio_token, so the audio placeholders "
                "a sequence carries can neither be counted nor masked out of the labels"
            )
        if _as_int(getattr(processor, "audio_token_id", None)) is None:
            return _audio_refusal(
                f"the {kind!r} processor exposes audio_token_id "
                f"{getattr(processor, 'audio_token_id', None)!r}, which is not an int "
                "(a bool reads as token 1 and every audio block lands in somebody "
                "else's vocabulary)"
            )
        # boa/eoa deliberately unchecked here -- see the docstring.
    extractor = getattr(processor, "feature_extractor", None)
    if extractor is None:
        return _audio_refusal(
            f"the {kind!r} processor exposes no feature_extractor, so its sampling_rate "
            "cannot be read and no waveform can be verified against it"
        )
    sampling_rate = _as_int(getattr(extractor, "sampling_rate", None))
    if sampling_rate is None or sampling_rate <= 0:
        return _audio_refusal(
            f"the {kind!r} processor's feature_extractor reports no positive int "
            f"sampling_rate (got {getattr(extractor, 'sampling_rate', None)!r}); every "
            "carrier must match it exactly, so no row could be accepted"
        )
    if kind != "audio_llm" and getattr(processor, "tokenizer", None) is None:
        return _audio_refusal(
            f"the {kind!r} processor exposes no tokenizer, so the {kind!r} labels "
            "(the transcripts the loss actually supervises) have no encoder"
        )
    return None


class SpeechCollator(Protocol):
    """The TRAIN loop's speech collator surface: rows -> batch dict, with coverage.

    A Protocol (not a class) for the same reason audio.py's ``AudioCollator`` is one:
    the train loop builds these through closures that satisfy the shape at runtime,
    and tests pass their own doubles. ``coverage`` is a :class:`SharedAudioCoverage`
    (same surface as ``AudioCoverage``, counters in shared memory) because the
    collate_fn runs in DataLoader WORKER processes -- see that class for the measured
    2026-10-06 defect and why a per-process count is not a count.
    """

    coverage: SharedAudioCoverage

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]: ...  # pragma: no cover


def refuse_if_features_dropped(batch_keys: Any, audio_column: str) -> None:
    """REFUSE (96) when output destined for the forward lost ``input_features``.

    The declared modality's carrier for BOTH non-chat kinds is the semantic key
    ``input_features`` -- the same key the chat path emits (audio.py's
    ``AUDIO_FEATURES_KEY``) and the same drop probe applies. Delegating rather than
    re-spelling it keeps ONE message and ONE mechanism for "a declared axis that is
    not executed is a refusal, not a pass" (#371/#410/#422's class).
    """
    refuse_if_audio_features_dropped(batch_keys, audio_column)


def _as_output_dict(output: Any) -> dict[str, Any]:
    """The extractor/tokenizer output as a plain dict of the keys it actually carries.

    ``BatchFeature`` and ``BatchEncoding`` are Mappings (item access, attribute access
    and ``.keys()`` all answer), while a stub or a hand-rolled encoder may answer only
    one spelling -- reading both here means a collator sees the same keys either way.
    A plain dict also lets ``refuse_if_features_dropped`` run on an output BEFORE the
    batch is assembled, which is precisely where a silently missing key is detectable.
    """
    if isinstance(output, Mapping):
        return {str(key): value for key, value in output.items()}
    return {str(key): value for key, value in vars(output).items()}


def _sampling_rate_or_refuse(processor: Any, audio_column: str) -> int:
    """The positive int ``sampling_rate`` every carrier must match exactly, or a refusal (96).

    ``speech_support_refusal`` measured this before the first row was read; this
    re-reads it rather than trusting ``int(...)`` because the number is also what the
    manifest records and what ``load_audio`` compares with ``!=`` -- a bool or a string
    in the slot would then refuse the CORPUS as ``sample_rate_mismatch``, blaming the
    rows for a processor defect.
    """
    extractor = getattr(processor, "feature_extractor", None)
    rate = _as_int(getattr(extractor, "sampling_rate", None))
    if rate is None or rate <= 0:
        _audio_refuse_exit_96(
            f"audio column {audio_column!r}: the processor's feature_extractor reports no "
            f"positive int sampling_rate (got {getattr(extractor, 'sampling_rate', None)!r}); "
            "every waveform is required to match it exactly, so no row can be verified"
        )
        raise AssertionError  # pragma: no cover -- _audio_refuse_exit_96 never returns
    return rate


def _refuse_row_load(*, kind: str, audio_column: str, row_id: str, reason: str, value: Any) -> None:
    """One row the loader rejected is a REFUSED RUN, never a silently shorter batch.

    audio.py's ``train_audio_collator_or_refuse`` wording, with the kind named: the
    row id, the counted reason (one of ``AUDIO_LOAD_REASONS``) and the DECLARED column
    so an operator can fix the corpus without guessing which axis was in play. Strict
    mode only -- tolerating the row is the defect this plane exists to fix.
    """
    _audio_refuse_exit_96(
        f"audio column {audio_column!r} row {row_id!r} refused by the {kind} loader: "
        f"reason={reason!r}, source={value!r}. Strict mode does not drop rows silently "
        "(that is the defect being fixed); fix the corpus or the loader, or narrow the "
        f"audio column. Unset {AUDIO_COLUMN_ENV} to train text-only on the same corpus"
    )
    raise AssertionError  # pragma: no cover -- _audio_refuse_exit_96 never returns


def _pad_label_rows(rows: Sequence[Sequence[int]], fill: int) -> Any:
    """One rectangular ``[B, L]`` long tensor: every row padded to the batch max with ``fill``.

    ``fill`` is a PARAMETER and not a convention: -100 is the seq2seq ignore index while
    a CTC label row pads with the CTC blank (see ``train_ctc_collator_or_refuse``), and
    a batch that mixed the two would train one of the kinds against its own contract.
    Rows are padded in python first so no row is truncated to reach a common width; torch
    enters HERE and never at module load.
    """
    import torch  # noqa: PLC0415 - the import contract forbids this at module load

    width = max((len(row) for row in rows), default=0)
    return torch.tensor(
        [list(row) + [fill] * (width - len(row)) for row in rows],
        dtype=torch.long,
    )


def train_seq2seq_collator_or_refuse(
    processor: Any,
    *,
    audio_column: str,
    answer_field: str = "answer",
    max_label_length: int,
    decoder_start_token_id: int,
    language: str | None = None,
    task: str = "transcribe",
) -> SpeechCollator:
    """Rows -> one verified encoder-decoder ASR batch; refuse (96) rather than fix anything.

    Whisper's prompt is part of the label: ``<|startoftranscript|><|lang|><|task|>
    <|notimestamps|>`` then the text. Its tokenizer emits only the prefix it is CONFIGURED
    with, and by default that has no language or task token. Measured on GB200 with
    whisper-large-v3: labels built that way start ``<|notimestamps|>``, the two prompt
    positions carry NLL 20 and 24 (the model predicts ``<|en|>`` / ``<|translate|>``),
    and fine-tuning on them would teach the model to drop its own prompt. So the
    language is DECLARED (never guessed from the data), set with ``set_prefix_tokens``
    at construction, and every batch is checked to start with that prefix.

    The shape is Whisper's measured fine-tune recipe (transformers 5.5):

      * one row -> one (waveform, ``duration``) via ``load_audio`` against
        ``processor.feature_extractor.sampling_rate``, bounded by
        :data:`WHISPER_WINDOW_SECONDS` -- Whisper's encoder window. An
        ``AudioLoadError`` is a refused run naming the row, the reason and the column.
        No resampling, no truncation (audio.py's rules, unchanged).
      * features come out of the extractor in ONE call
        (``processor.feature_extractor(waves, sampling_rate=sr, return_tensors="pt")``).
        Whisper pads every clip to 30 s, and whatever the extractor returns is kept
        verbatim -- including its ``attention_mask`` when it emits one (whisper only
        does when configured to). An invented mask of ones over padded frames would
        re-describe the input, so none is ever constructed here.
      * labels are ``processor.tokenizer(texts).input_ids`` -- Whisper's tokenizer adds
        its own prefix tokens (language / task / no-timestamps and the trailing token)
        exactly as inference does, so nothing is invented around the transcript.
      * the decoder-start strip: when EVERY row's first label id is
        ``decoder_start_token_id``, it is dropped from every row -- the model shifts
        labels right and PREPENDS ``decoder_start_token_id`` itself (its
        ``_shift_right``/decoder path), so keeping it would double the symbol the loss
        is taught to emit. When even ONE row lacks it, NOTHING is stripped: a tokenizer
        that disagrees with itself across rows gets no half-repair here (fixing only
        the rows that have it would leave one batch holding two label framings), and
        leaving every row as tokenised is the one choice that changes nothing.
      * labels pad to the batch max with **-100** (the ignore index) and a row longer
        than ``max_label_length`` REFUSES -- labels are NEVER truncated: a truncated
        target teaches a transcript the corpus does not contain, which is the
        silent-drop defect in the text.
      * the placeholder-count gate ABSTAINS here. seq2seq has no audio placeholder
        tokens (there is no tower slot in ``input_ids`` to count), so this collator
        calls neither ``add_placeholder_verified`` nor ``add_placeholder_unmeasured``:
        an abstention, not a pass, and the manifest's two placeholder numbers stay 0
        rather than claiming measurement where there was none.

    ``coverage`` (``.coverage`` on the returned callable) is a
    :class:`SharedAudioCoverage`: ``add_expected`` per row the run declares,
    ``record_ok(duration)`` per row the loader accepted, and nothing else -- the
    counters survive the DataLoader workers this closure runs in.
    """
    refusal = speech_support_refusal("seq2seq", processor)
    if refusal is not None:
        # Verify before decode: the surface must carry the kind's parts before a
        # single row is opened (audio.py's rule, at construction rather than at the
        # first batch, so the run dies before the corpus is touched).
        _audio_refuse_exit_96(refusal)
        raise AssertionError  # pragma: no cover -- _audio_refuse_exit_96 never returns
    target_sr = _sampling_rate_or_refuse(processor, audio_column)
    extractor: Any = processor.feature_extractor
    tokenizer: Any = processor.tokenizer
    expected_prefix: list[int] = []
    set_prefix = getattr(tokenizer, "set_prefix_tokens", None)
    if callable(set_prefix):
        multilingual = tokenizer.convert_tokens_to_ids("<|en|>") != getattr(
            tokenizer, "unk_token_id", None
        )
        if multilingual and language is None:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r}: this seq2seq tokenizer is multilingual and "
                "its prompt carries a language token, but no language was declared "
                "(FOUNDATIONSCALE_TRAIN_AUDIO_LANGUAGE, e.g. 'en'). Refusing rather than "
                "guessing it, or training labels with no language token"
            )
        set_prefix(language=language, task=task, predict_timestamps=False)
        expected_prefix = [int(t) for t in tokenizer.prefix_tokens[1:]]
    coverage = SharedAudioCoverage(rows_expected=0)

    def collate(rows: Sequence[Any]) -> dict[str, Any]:
        waves: list[Any] = []
        texts: list[str] = []
        for index, raw_row in enumerate(rows):
            row = raw_row if isinstance(raw_row, dict) else vars(raw_row)
            coverage.add_expected()
            row_id = f"train-row[{index}]"
            value = row.get(audio_column)
            try:
                wave, duration = load_audio(
                    value,
                    target_sr=target_sr,
                    row_id=row_id,
                    max_seconds=WHISPER_WINDOW_SECONDS,
                )
            except AudioLoadError as exc:
                _refuse_row_load(
                    kind="seq2seq",
                    audio_column=audio_column,
                    row_id=row_id,
                    reason=exc.reason,
                    value=value,
                )
                raise  # pragma: no cover -- _refuse_row_load never returns
            waves.append(wave)
            texts.append(str(row.get(answer_field, "")))
            # Per LOADED row (audio.py records at the end of a batch, when the batch
            # may NOT have succeeded): both refusals this collator can still raise are
            # exit-96 -- a whole run, never a partial batch -- so there is no window
            # in which a counted row could still be dropped from the training set.
            coverage.record_ok(duration)
        if not waves:
            return {}

        features = _as_output_dict(extractor(waves, sampling_rate=target_sr, return_tensors="pt"))
        # The tower's carrier must come OUT of the extractor: a text-only encode
        # sneaking in here is the silent-drop defect one layer down, and this is the
        # last point where its absence is detectable.
        refuse_if_features_dropped(features.keys(), audio_column)

        tokenized = _as_output_dict(tokenizer(texts))
        id_rows = tokenized.get("input_ids")
        if id_rows is None:
            _audio_refuse_exit_96(
                f"answer column {answer_field!r} (audio column {audio_column!r}): the "
                f"tokenizer produced no input_ids (keys {sorted(tokenized)}), so the labels "
                "the loss supervises do not exist. Refusing rather than training against "
                "an objective that is not the task"
            )
            raise AssertionError  # pragma: no cover
        label_rows: list[list[int]] = [[int(token) for token in row] for row in id_rows]
        if len(label_rows) != len(texts):
            # Same rule as audio.placeholder_mismatches: a denominator mismatch is not
            # a pass -- zip() would label only the shared prefix and the rest of the
            # batch would carry transcripts against the wrong waveforms.
            _audio_refuse_exit_96(
                f"audio column {audio_column!r} answer column {answer_field!r}: the tokenizer "
                f"returned {len(label_rows)} label rows for {len(texts)} rows; labels would "
                "land against the wrong waveforms. Refusing rather than mislabelling the batch"
            )
            raise AssertionError  # pragma: no cover

        if label_rows and all(row and row[0] == decoder_start_token_id for row in label_rows):
            # The model PREPENDS decoder_start_token_id when it shifts labels right, so
            # keeping the tokenizer's copy would double the symbol being predicted.
            label_rows = [row[1:] for row in label_rows]
        if expected_prefix:
            bad = [
                i for i, r in enumerate(label_rows) if r[: len(expected_prefix)] != expected_prefix
            ]
            if bad:
                _audio_refuse_exit_96(
                    f"audio column {audio_column!r}: label row(s) {bad[:5]} do not start with "
                    f"the declared prompt {expected_prefix} (language={language!r}, "
                    f"task={task!r}); training on them would teach the model a different "
                    "prompt. Refusing rather than mislabelling the batch"
                )

        # Measured AFTER the strip above: this is the label length the loss will see.
        oversized = [
            (index, len(row)) for index, row in enumerate(label_rows) if len(row) > max_label_length
        ]
        if oversized:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r} answer column {answer_field!r}: "
                + ", ".join(f"train-row[{index}] is {length} tokens" for index, length in oversized)
                + f" wider than the declared max_label_length={max_label_length}. Labels are "
                "never truncated -- a truncated target teaches a transcript the corpus does "
                "not contain, which is the silent-drop defect in the text. Raise the budget, "
                "shorten the answers, or narrow the corpus; this refuses rather than corrupts"
            )
            raise AssertionError  # pragma: no cover

        batch: dict[str, Any] = {
            "input_features": features["input_features"],
            "labels": _pad_label_rows(label_rows, -100),
        }
        attention = features.get("attention_mask")
        if attention is not None:
            # Pass the extractor's own mask through when it arrived, and never invent
            # one (see the docstring).
            batch["attention_mask"] = attention
        # Last gate (audio.py's, on the batch the model would actually receive): a
        # declared audio column that is not executed is a refusal, not a pass.
        refuse_if_features_dropped(batch.keys(), audio_column)
        return dict(batch)

    # Functions cannot nominally carry an attribute (PEP 544): assigning .coverage on
    # the callable is what satisfies SpeechCollator at runtime.
    collate.coverage = coverage  # type: ignore[attr-defined]
    return collate  # type: ignore[return-value]


def train_ctc_collator_or_refuse(
    processor: Any,
    *,
    audio_column: str,
    answer_field: str = "answer",
    pad_token_id: int,
    max_seconds: float | None = None,
) -> SpeechCollator:
    """Rows -> one verified CTC batch; refuse (96) rather than fix anything.

    The label contract here is the one MEASURED in transformers 5.5's
    ``ParakeetForCTC.forward`` (2026-10): the loss masks targets with
    ``labels != config.pad_token_id`` -- the CTC BLANK -- under a comment that says
    ``-100``. So -100 padding is NOT an ignore index here: it reaches ``ctc_loss`` as a
    real target, inflates ``target_lengths``, and trains the head to emit a token that
    is nowhere in the vocabulary. Labels therefore pad with ``pad_token_id`` and a
    ``-100`` anywhere in a CTC label row is a defect, never a convention.

      * ``features = processor.feature_extractor(waves, sampling_rate=sr,
        return_tensors="pt", return_attention_mask=True)`` -> ``input_features`` and
        **``attention_mask``**. The mask is REQUIRED for this kind: Parakeet's loss
        derives ``input_lengths`` from ``attention_mask.sum(-1)`` (and falls back to
        treating every padded frame as valid when it is absent -- input lengths that no
        measurement produced). An extractor that returns none refuses.

      * ``labels = processor.tokenizer(texts, add_special_tokens=False).input_ids`` --
        no boundary or separator tokens: CTC collapses FRAME logits and needs the bare
        characters/subwords of the transcript as targets to collapse into. An EMPTY
        label row refuses: with zero non-blank targets there is nothing to collapse and
        the objective silently becomes "emit frames for nothing".

    ``max_seconds`` bounds ONE waveform -- a longer row is refused and counted, never
    truncated. ``None`` is UNMEASURED and defers to
    :func:`max_audio_seconds` (audio.py): the cap the PROCESSOR itself measured
    (``audio_seq_length * audio_ms_per_token``), which Parakeet's processor declares
    none of and so leaves this collator unbounded exactly as ``None`` spells it. The
    placeholder-count gate ABSTAINS for this kind (like seq2seq -- there is no audio
    placeholder in a CTC ``input_ids`` sequence at all), so neither placeholder counter
    is updated.
    """
    refusal = speech_support_refusal("ctc", processor)
    if refusal is not None:
        _audio_refuse_exit_96(refusal)
        raise AssertionError  # pragma: no cover -- _audio_refuse_exit_96 never returns
    blank = _as_int(pad_token_id)
    if blank is None or blank < 0:
        _audio_refuse_exit_96(
            f"audio column {audio_column!r}: pad_token_id {pad_token_id!r} is not a "
            "non-negative int, and for this kind it IS the CTC blank the loss masks with "
            "(a bool reads as blank 1; a negative token has no row to be). Refusing rather "
            "than padding labels into a blank nobody chose"
        )
        raise AssertionError  # pragma: no cover
    target_sr = _sampling_rate_or_refuse(processor, audio_column)
    extractor: Any = processor.feature_extractor
    tokenizer: Any = processor.tokenizer
    # None is UNMEASURED (load_audio applies exactly the bounds given): defer to the
    # cap the processor measured, if it measured one at all.
    cap = max_seconds if max_seconds is not None else max_audio_seconds(processor)
    coverage = SharedAudioCoverage(rows_expected=0)

    def collate(rows: Sequence[Any]) -> dict[str, Any]:
        waves: list[Any] = []
        texts: list[str] = []
        for index, raw_row in enumerate(rows):
            row = raw_row if isinstance(raw_row, dict) else vars(raw_row)
            coverage.add_expected()
            row_id = f"train-row[{index}]"
            value = row.get(audio_column)
            try:
                wave, duration = load_audio(
                    value,
                    target_sr=target_sr,
                    row_id=row_id,
                    max_seconds=cap,
                )
            except AudioLoadError as exc:
                _refuse_row_load(
                    kind="ctc",
                    audio_column=audio_column,
                    row_id=row_id,
                    reason=exc.reason,
                    value=value,
                )
                raise  # pragma: no cover -- _refuse_row_load never returns
            waves.append(wave)
            texts.append(str(row.get(answer_field, "")))
            # Per LOADED row -- see train_seq2seq_collator_or_refuse for why.
            coverage.record_ok(duration)
        if not waves:
            return {}

        features = _as_output_dict(
            extractor(
                waves,
                sampling_rate=target_sr,
                return_tensors="pt",
                return_attention_mask=True,
            )
        )
        refuse_if_features_dropped(features.keys(), audio_column)
        attention = features.get("attention_mask")
        if attention is None:
            _audio_refuse_exit_96(
                f"audio column {audio_column!r}: the feature extractor returned no "
                "attention_mask although it was asked for one. For this kind the CTC loss "
                "derives input lengths FROM that mask; without it, every padded frame "
                "would count as valid and the input lengths would be numbers no "
                "measurement produced. Refusing rather than training them"
            )
            raise AssertionError  # pragma: no cover

        tokenized = _as_output_dict(tokenizer(texts, add_special_tokens=False))
        id_rows = tokenized.get("input_ids")
        if id_rows is None:
            _audio_refuse_exit_96(
                f"answer column {answer_field!r} (audio column {audio_column!r}): the "
                f"tokenizer produced no input_ids (keys {sorted(tokenized)}), so the labels "
                "the loss supervises do not exist. Refusing rather than training against "
                "an objective that is not the task"
            )
            raise AssertionError  # pragma: no cover
        label_rows: list[list[int]] = [[int(token) for token in row] for row in id_rows]
        if len(label_rows) != len(texts):
            _audio_refuse_exit_96(
                f"audio column {audio_column!r} answer column {answer_field!r}: the tokenizer "
                f"returned {len(label_rows)} label rows for {len(texts)} rows; labels would "
                "land against the wrong waveforms. Refusing rather than mislabelling the batch"
            )
            raise AssertionError  # pragma: no cover
        for index, label_row in enumerate(label_rows):
            if not label_row:
                _audio_refuse_exit_96(
                    f"audio column {audio_column!r} row {row_id_or(index)!r}: answer column "
                    f"{answer_field!r} tokenised to ZERO target tokens. CTC needs at least one "
                    "non-blank target to collapse into -- an empty row would supervise the "
                    "head to emit frames for nothing, which is not the task. Fix the corpus "
                    "rather than letting a blank transcript pass as training data"
                )
                raise AssertionError  # pragma: no cover

        batch: dict[str, Any] = {
            "input_features": features["input_features"],
            "attention_mask": attention,
            # The CTC blank, NOT -100 (see the docstring: transformers 5.5's
            # ParakeetForCTC masks with pad_token_id despite its own comment).
            "labels": _pad_label_rows(label_rows, blank),
        }
        refuse_if_features_dropped(batch.keys(), audio_column)
        return dict(batch)

    collate.coverage = coverage  # type: ignore[attr-defined]
    return collate  # type: ignore[return-value]


def row_id_or(index: int) -> str:
    """``f"train-row[{index}]"`` -- the one row-id spelling every refusal here uses.

    Extracted (rather than inlined a third time) so a row that is named in a refusal is
    named exactly as ``load_audio`` names it: the manifest merges per-row accounts by
    this string, and two spellings would split one row into two accounts.
    """
    return f"train-row[{index}]"


def speech_plane_refusal(kind: str | None, family: Any, processor: Any) -> str | None:
    """The one reason this run may not train speech, or None; checked in a fixed order.

    1. The model kind must be known (``speech_model_kind``); an unknown kind has no
       declared loss, label padding or row window, so nothing about it is guessed.
    2. The family must declare an audio tower: the tower-movement gate proves the audio
       path trained, and without a declared tower it would have nothing to prove.
    3. The processor must carry what this kind's collator needs (``speech_support_refusal``).
    """
    from foundationscale.train.audio import AUDIO_COLUMN_ENV  # noqa: PLC0415

    if kind is None:
        return (
            f"{AUDIO_COLUMN_ENV} is declared but this checkpoint's model type is not a "
            f"known speech kind ({list(SPEECH_KINDS)}); its loss, label padding and row "
            "window are undeclared, so it is refused rather than guessed (96)"
        )
    towers = getattr(family, "towers", None) or ()
    if not any(modality == "audio" for _, modality in towers):
        name = getattr(family, "name", None)
        label = f"family {name!r}" if name else "this checkpoint (no registered family)"
        return (
            f"{AUDIO_COLUMN_ENV} is declared but {label} declares no audio tower, so the "
            "tower-movement gate would have nothing to prove the audio path trained; "
            "register the family's audio towers (FamilySpec.towers). Refusing (96)"
        )
    return speech_support_refusal(kind, processor)


def speech_tower_resolution_refusal(family: Any, model: Any) -> str | None:
    """Refuse (96) before a step when a declared audio tower is absent from the loaded model.

    FamilySpec paths are exact module paths, and transformers moves them between releases:
    Qwen2-Audio's ``audio_tower`` (5.5) is ``model.audio_tower`` in 5.18 and 5.19. An
    unresolved tower used to surface only after training, as a VACUOUS tower-movement gate;
    naming it here costs nothing and spends no GPU time on a run that cannot pass.
    """
    from foundationscale.families.towers import resolve_module_path  # noqa: PLC0415

    towers = getattr(family, "towers", None) or ()
    missing = [
        path
        for path, modality in towers
        if modality == "audio" and resolve_module_path(model, path) is None
    ]
    if not missing:
        return None
    children = getattr(model, "named_children", None)
    top = sorted(name for name, _ in children()) if callable(children) else []
    return (
        f"family {getattr(family, 'name', None)!r} declares audio tower(s) {missing} that do "
        f"not resolve on the loaded model (its top-level modules: {top}). The module layout "
        "differs from the one the FamilySpec was measured on -- usually a transformers "
        "version that nests the model differently. Re-measure the module paths for this "
        "transformers version and update FamilySpec.towers. Refusing (96) before training"
    )


def build_speech_collator(
    kind: str,
    surface: Any,
    *,
    audio_column: str,
    max_length: int,
    model_config: Any,
    language: str | None,
) -> Any:
    """The train collator for ``kind``, built from the loaded model's own config.

    audio_llm reuses the collators in train/audio.py: the request-API transcription
    collator for processors exposing ``apply_transcription_request`` (Qwen3-ASR), the
    chat-template collator otherwise. seq2seq caps label length at the decoder's
    ``max_target_positions`` when the config has
    one (Whisper: 448), because a longer label would overrun the decoder's position
    table; ctc pads labels with the config's ``pad_token_id`` (its blank, see
    :func:`train_ctc_collator_or_refuse`).
    """
    from foundationscale.train.audio import (  # noqa: PLC0415
        train_audio_collator_or_refuse,
        train_transcription_collator_or_refuse,
    )

    processor = surface.surface
    if kind == "audio_llm":
        if callable(getattr(processor, "apply_transcription_request", None)):
            return train_transcription_collator_or_refuse(
                surface,
                audio_column=audio_column,
                max_length=max_length,
                language=language,
            )
        return train_audio_collator_or_refuse(
            surface, audio_column=audio_column, max_length=max_length
        )
    if kind == "seq2seq":
        decoder_cap = getattr(model_config, "max_target_positions", None)
        cap = min(max_length, decoder_cap) if isinstance(decoder_cap, int) else max_length
        return train_seq2seq_collator_or_refuse(
            processor,
            audio_column=audio_column,
            max_label_length=cap,
            decoder_start_token_id=int(model_config.decoder_start_token_id),
            language=language,
        )
    if kind == "ctc":
        return train_ctc_collator_or_refuse(
            processor, audio_column=audio_column, pad_token_id=int(model_config.pad_token_id)
        )
    _audio_refuse_exit_96(f"speech kind {kind!r} has no collator (known: {list(SPEECH_KINDS)})")
    raise AssertionError  # pragma: no cover -- _audio_refuse_exit_96 never returns


def freeze_batchnorm_statistics(model: Any) -> int:
    """Keep every BatchNorm layer on its stored running statistics; return how many.

    Measured on parakeet-ctc-1.1b (GB200, 500 steps, batch 8): fine-tuning moved held-out
    WER from 1.66% to 2.39%, and restoring only the base model's BatchNorm buffers into the
    fine-tuned weights brought it back to exactly 1.66%. The damage was the running
    statistics, updated in train mode from small batches zero-padded to the longest clip.
    A forward pre-hook puts each BatchNorm in inference behaviour on every call (the
    Trainer calls ``model.train()`` each step, so setting ``eval()`` once would not last):
    it normalises with -- and never updates -- the stored statistics, while gradients
    still reach its affine weight and bias.
    """
    import torch  # noqa: PLC0415 -- this module must import without torch

    def _keep_inference_stats(module: Any, _inputs: Any) -> None:
        module.training = False

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.register_forward_pre_hook(_keep_inference_stats)
            count += 1
    return count


GROUP_BY_DURATION_ENV = "FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION"


def group_by_duration_settings(
    declared: str | None, columns: Sequence[str]
) -> dict[str, str] | str | None:
    """TrainingArguments for duration-grouped batches, a refusal string, or None (undeclared).

    Declared, never implicit: grouping changes which rows share a batch, so it is opt-in via
    ``FOUNDATIONSCALE_TRAIN_AUDIO_GROUP_BY_DURATION`` and uses the manifest's own measured
    ``duration`` column (transformers 5.5 ``train_sampling_strategy="group_by_length"``). Audio
    batches pad to their longest clip, so grouping similar durations cuts padded frames. A
    declaration on a dataset without the column refuses rather than grouping by nothing.
    """
    if not declared:
        return None
    if "duration" not in columns:
        return (
            f"{GROUP_BY_DURATION_ENV} is declared but the dataset has no 'duration' column "
            f"(columns {list(columns)}); grouping needs each row's measured length. Refusing (96)"
        )
    return {"train_sampling_strategy": "group_by_length", "length_column_name": "duration"}


def apply_group_by_duration(
    training_kwargs: dict[str, Any], environ: Mapping[str, str], columns: Sequence[str]
) -> str | None:
    """Apply the declared duration grouping to ``training_kwargs`` in place.

    Returns a refusal string (the caller refuses 96), an announcement line when grouping was
    applied, or None when it was not declared. Kept here so the train loop holds one call.
    """
    settings = group_by_duration_settings(environ.get(GROUP_BY_DURATION_ENV), columns)
    if settings is None or isinstance(settings, str):
        return settings
    training_kwargs.update(settings)
    return (
        "[   ok] speech.group_by_duration: batches group rows of similar measured "
        "duration (declared), so each pads to a closer longest clip"
    )


def full_train_announcement(audio_column: str, modules: Sequence[str], family: Any) -> str:
    """The line announcing what trains in full and where LoRA attaches, for an adapter run."""
    scope = list(getattr(family, "adapter_scope_prefixes", ()) or ())
    return (
        f"adapter='lora' with audio_column={audio_column!r}: {', '.join(modules)} train in "
        f"full (peft modules_to_save); LoRA covers the declared adapter scope {scope}"
    )


def lora_audio_plan(family: Any, audio_column: str) -> tuple[list[str], str]:
    """For a LoRA run that declares audio: ``(full_train_modules, line)``.

    Non-empty modules: ``line`` is the announcement and the caller sets peft
    ``modules_to_save``. Empty modules: ``line`` is the refusal (96) -- the family declares no
    measured wrap points, so peft would freeze its audio towers and the run would train text.
    """
    from foundationscale.train.audio import audio_full_train_modules  # noqa: PLC0415

    modules = audio_full_train_modules(family)
    if not modules:
        return [], (
            f"adapter='lora' with audio_column={audio_column!r}: this family declares no "
            "adapter_full_train modules for its audio towers, so an adapter run cannot train "
            "them (peft would freeze them). Run a full fine-tune, or measure and register the "
            "family's wrap points"
        )
    return modules, full_train_announcement(audio_column, modules, family)


def build_speech_collator_checked(
    kind: str,
    surface: Any,
    probe_rows: Sequence[Any],
    *,
    audio_column: str,
    max_length: int,
    model_config: Any,
    language: str | None,
) -> Any:
    """Build the collator, prove input_features reaches a real batch, then reset coverage.

    The survival probe collates real rows before a step is paid for and refuses (96) --
    naming the column -- if ``input_features`` is not in the batch; the probe rows proved the
    path and did not train, so the coverage record is reset to count training rows only.
    """
    from foundationscale.train.audio import refuse_if_audio_features_dropped  # noqa: PLC0415

    collator = build_speech_collator(
        kind,
        surface,
        audio_column=audio_column,
        max_length=max_length,
        model_config=model_config,
        language=language,
    )
    refuse_if_audio_features_dropped(collator(list(probe_rows)).keys(), audio_column)
    collator.coverage.reset()
    return collator
