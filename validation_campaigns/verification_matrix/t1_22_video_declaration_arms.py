"""T1-22 -- "declaring video REFUSES cleanly today". Adjudicated over TWO planes.

WHAT THIS ROW ASSERTED, AND WHY IT WAS NEVER TRUE. matrix.json carried
`status: "refuses by design"` with an empty findings list and
`adjudicator: None` -- an assertion, never a measurement. Its own control_arm
field spelled out the failure mode to guard against: "must be 96, not a silent
text-only train". Before fix #490 that is exactly what happened: the deciding
corpus trained and the word "video" appeared nowhere.

WHAT FIX #490 CHANGED, AND WHAT THIS ADJUDICATOR NOW MEASURES. `train()` gained
a declaration boundary. Four module-level names -- UNTRAINABLE_MODALITIES,
_declared_untrainable_modality, _untrainable_modality_refusal,
_dropped_column_notice -- make the boundary concrete: an audio or video column
DECLARED through its env var (FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN here) is
refused 96 immediately after the image column is read, before the model,
tokenizer or dataset are touched, with a [fs:train:refuse] line naming the
modality, the env var, the declared column, and the word "text" for what the
plane CAN do. A corpus that merely CONTAINS such a column with nothing declared
still trains -- it must -- but the drop is now NAMED on the console via
[fs:train:data] instead of being silent. This adjudicator therefore scores the
boundary itself: refusal keyed on the DECLARATION, never on the DATA. A run
that refuses a corpus for having a column NAMED "video" is name-sniffing and
breaks legitimate text corpora; that over-correction is RED here, not a pass.

WHY THIS ADJUDICATOR IS NOT A COPY OF THE T1-23 AUDIO ONE. Audio has zero
occurrences in shipped source, so there is one plane to measure and the finding
is a flat absence. Video has fifteen, all in `src/foundationscale/rl/`, and
`rl/trainer.py` refuses it deliberately, by name, with a written rationale about
frame budgets. So video is not an unsupported modality handled badly -- it is a
modality this repository has already decided to refuse, where the refusal lives
in a plane most operators never reach. Measuring only the training plane would
miss the actionable part; measuring only the RL plane would report a refusal on
a plane nobody trains with. The claim is adjudicated over BOTH planes.

THE DECIDING ARM. The old deciding arm (a bare-`video` corpus) refused for a
missing `text` column, which a corpus of bare timestamps would earn just as
well; a refusal about a missing text column is not a refusal about video. Its
successor, text_video_declared, carries a usable `text` column AND declares the
modality through FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN, so the only term that
differs from an ordinary accepted corpus is the DECLARATION under test. The
corpus that merely carries an undeclared `video` key alongside `text`
(text_video) is kept as the second leg: it must train -- refusal would be
name-sniffing -- and the dropped-column notice must name "video" on the
console. Right code with the wrong reason (96 about 'text') is the original
defect in a new outfit and scores RED, as does training through a declaration,
as does the silent drop returning, as does refusing on the column name.

WHY THE SFT PLANE'S REFUSALS ARE READABLE AT ALL. A plane that never refuses
any declaration would be silent on video for reasons having nothing to do with
video. The image_control arm declares an image column the dataset does not
contain, and the trainer refuses 96 naming that column. So the SFT plane
demonstrably CAN refuse a declared-but-unsatisfiable modality, and its
behaviour on video is a choice about video rather than a general incapacity.

WHY EXIT CODES ALONE DECIDE NOTHING ON THE RL PLANE. The video check sits before
the tokenizer load, and that later load ALSO refuses 96 when the model path is
bogus. Both the positive leg and its control exit 96. Scoring exit codes would
compare 96 to 96 and conclude nothing while appearing to conclude something, so
every RL leg is classified by the SUBJECT of its refusal and the control must
demonstrably have reached the model load -- i.e. got PAST the video check.

VERDICT ORDER (load-bearing; pinned by --self-test controls, not by comments):
  1. any arm not measured                                  -> UNMEASURED 95
  2. SFT refusal-capability control silent                  -> REFUSE 96
  3. RL attribution control claims video                    -> REFUSE 96
  4. RL attribution control never passed 331                -> REFUSE 96
  5. RL key-is-read control silent                          -> REFUSE 96
  6. deciding arm refused 96 on a MISSING 'text' column     -> RED 5
  7. deciding arm TRAINED despite the declaration           -> RED 5
  8. undeclared-corpus arm REFUSED (name-sniffing)          -> RED 5
  9. undeclared-corpus arm trained and stayed quiet         -> RED 5
 10. declaration refused by name; undeclared corpus trained -> GREEN 0
     with the drop named on the console

The verdict function is PURE -- no I/O, no torch, no datasets -- and --self-test
drives it over synthetic payloads under `python3 -S`.

MEASURED 2026-09-16, pre-#490: the undeclared arm exited 0 having tokenized 2
examples with the word absent from its output and from 30,671 characters of run
manifest, and scored RED. The baseline payload in --self-test encodes the shape
#490 is designed to produce; re-running the emitted SFT plan regenerates it.
"""

from __future__ import annotations

import argparse
import json
import os  # stdlib only: the self-test runs under `python3 -S`
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

GREEN = 0
RED = 5
UNMEASURED = 95
REFUSE = 96

ROW_ID = "T1-22"
PROG = "t1_22_video_declaration_arms"
CLAIM = "declaring video REFUSES cleanly today"
CONTROL = (
    "declared video: 96 naming the modality; undeclared video: trains, "
    "with the drop named on the console -- never a silent text-only train"
)

VERDICT_NAMES = {GREEN: "GREEN", RED: "RED", UNMEASURED: "UNMEASURED", REFUSE: "REFUSE"}

NEEDLE = "video"
CORPUS_FILENAME = "corpus.jsonl"
MANIFEST_GLOB = "*manifest*.json"

IMAGE_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_IMAGE_COLUMN"
DECLARED_IMAGE_COLUMN = "t22_img_probe"
# The env var fix #490 added. The refusal is keyed on THIS declaration, never
# on the data: the adversarial arm carries both so the declaration is the only
# term that varies.
VIDEO_COLUMN_ENV = "FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN"
DECLARED_VIDEO_COLUMN = "video"

SFT_ARMS = ("video_only", "text_video", "text_video_declared", "image_control")
# The deciding arm DECLARES the modality through its env var on an otherwise
# trainable corpus; refusal must be keyed on that declaration and name it.
SFT_DECIDING_ARM = "text_video_declared"
# The pre-#490 deciding arm, kept as the boundary's other leg: nothing is
# declared, so it must TRAIN -- and the drop must be NAMED, not silent.
SFT_UNDECLARED_ARM = "text_video"
SFT_REFUSAL_CONTROL = "image_control"

SFT_TEXT_EXAMPLE = "A short, ordinary training example with nothing to refuse."
# The run plan for the SFT plane: this adjudicator cannot drive a tray, so it
# EMITS the plan (corpus columns + env per arm) and a tray-side measurer
# executes it, handing the arms back through --sft-payload. Each "why" is the
# reason that arm exists; removing an arm without answering its "why" breaks
# the measurement it guards.
SFT_ARM_SPECS: dict[str, dict[str, Any]] = {
    "video_only": {
        "columns": {"video": "clip0.mp4"},
        "env": {},
        "why": (
            "no text column at all: any refusal here is about 'text' and never about "
            "video, so a missing-'text' refusal can never be mistaken for a modality "
            "refusal; under #490 the [fs:train:data] notice names the dropped video "
            "column first, then the textless corpus earns its own refusal"
        ),
    },
    "text_video": {
        "columns": {"text": SFT_TEXT_EXAMPLE, "video": "clip0.mp4"},
        "env": {},
        "why": (
            "nothing declared: the corpus merely CONTAINS a video key. It must TRAIN "
            "-- refusing on the column name is name-sniffing and breaks legitimate "
            "text corpora -- and the [fs:train:data] notice must name 'video' among "
            "the dropped columns instead of dropping it silently"
        ),
    },
    "text_video_declared": {
        "columns": {"text": SFT_TEXT_EXAMPLE, "video": "clip0.mp4"},
        "env": {VIDEO_COLUMN_ENV: DECLARED_VIDEO_COLUMN},
        "why": (
            "the deciding arm: identical corpus to text_video except the modality is "
            "DECLARED through its env var. It must exit 96 via [fs:train:refuse] "
            "naming the modality, the env var and the declared column -- the refusal "
            "is keyed on the declaration, which is the whole design"
        ),
    },
    "image_control": {
        "columns": {"text": SFT_TEXT_EXAMPLE},
        "env": {IMAGE_COLUMN_ENV: DECLARED_IMAGE_COLUMN},
        "why": (
            "the refusal-capability control: a declared-but-unsatisfiable image "
            "column earns a named 96, proving the SFT plane CAN refuse a declaration "
            "at all; without it, silence about video is an absence of evidence"
        ),
    },
}

RL_ARMS = ("video_declared", "no_video_control", "non_string_video")
RL_POSITIVE = "video_declared"
RL_ATTRIBUTION_CONTROL = "no_video_control"
RL_KEY_READ_CONTROL = "non_string_video"

# The stable fragment of the refusal at rl/trainer.py, not the whole sentence: a
# reworded refusal must still score as one. Specificity is checked separately,
# by requiring the offending sample id, so a generic message cannot pass as a
# named one.
VIDEO_REFUSAL_FRAGMENT = "carry video"
RL_SAMPLE_ID = "t22-rl-vid-0"
# Never loaded. Every RL leg must decide before the model matters; a leg that
# names this path got PAST the video check, which is information, not noise.
RL_MODEL_SENTINEL = "/nonexistent/t1-22-model-must-never-be-loaded"
# This probe's own sentinel for "propagated an exception instead of exiting",
# which is a distinct outcome from any code the program can choose for itself.
RAISED_SENTINEL = 70

MAX_STEPS = 2
PER_DEVICE_BATCH_SIZE = 1
NODES = 1
GPUS_PER_NODE = 1
SEED = 42
ARM_TIMEOUT_S = 3600
RL_TIMEOUT_S = 300

_TOKENIZED_RE = re.compile(r"(\d+)\s+examples tokenized")

RL_DRIVER = """
import json, sys, traceback
sys.path.insert(0, {src!r})
out = {{"exit": None, "exc": None, "reached_run": False}}
try:
    from foundationscale.rl.trainer import RLTrainConfig, RLTrainer
    cfg = RLTrainConfig(model={model!r}, dataset={dataset!r})
    t = RLTrainer(cfg)
    out["reached_run"] = True
    t.run()
except SystemExit as exc:
    out["exit"] = exc.code
except BaseException as exc:
    out["exc"] = "%s: %s" % (type(exc).__name__, exc)
    traceback.print_exc()
print("T1_22_RL_JSON " + json.dumps(out))
sys.stdout.flush()
sys.exit(out["exit"] if out["exit"] is not None else ({raised} if out["exc"] else 0))
"""

RL_RECORD: dict[str, Any] = {
    "id": RL_SAMPLE_ID,
    "conversations": [
        {"from": "human", "value": "What happens in the clip?"},
        {"from": "gpt", "value": "A refusal should happen before this is read."},
    ],
}
RL_LEG_VIDEO_VALUE: dict[str, Any] = {
    "video_declared": "clip0.mp4",
    "no_video_control": None,  # key deleted entirely
    "non_string_video": 123,  # wrong type: the parser must guard it
}


# --------------------------------------------------------------------------
# the pure verdict
# --------------------------------------------------------------------------
def video_declaration_verdict(payload: Mapping[str, Any]) -> tuple[int, str]:
    """Decide T1-22 from a two-plane payload. Pure: no I/O, no imports beyond stdlib.

    Decision order is load-bearing and pinned by --self-test controls. Controls
    are judged BEFORE the claim, always: a claim read through a detector that was
    never shown to fire is not a measurement, however green or red it looks.
    """
    sft = payload.get("sft") or {}
    rl = payload.get("rl") or {}

    missing = [f"sft/{a}" for a in SFT_ARMS if a not in sft]
    missing += [f"rl/{a}" for a in RL_ARMS if a not in rl]
    if missing:
        return UNMEASURED, (
            f"{len(missing)} of {len(SFT_ARMS) + len(RL_ARMS)} arm(s) absent: "
            f"{', '.join(missing)}. An arm that was never run is not evidence "
            "either way; 95, NOT a refutation"
        )
    unmeasured = [
        f"{plane}/{name}"
        for plane, arms in (("sft", sft), ("rl", rl))
        for name, arm in arms.items()
        if arm.get("status") != "measured"
    ]
    if unmeasured:
        return UNMEASURED, (
            f"{len(unmeasured)} arm(s) did not complete: {', '.join(sorted(unmeasured))}; "
            "the measurement never happened, so nothing is refuted; 95"
        )

    # -- control 1: can the SFT plane refuse ANY declaration at all? ---------
    cap = sft[SFT_REFUSAL_CONTROL]
    if cap.get("launcher_exit_code") != REFUSE or not cap.get("names_own_declaration"):
        return REFUSE, (
            f"the SFT refusal-capability control ({SFT_REFUSAL_CONTROL}) exited "
            f"{cap.get('launcher_exit_code')} and "
            f"{'named' if cap.get('names_own_declaration') else 'did NOT name'} the column it "
            "declared. Until that plane is shown to refuse a declared-but-unsatisfiable "
            "modality, its behaviour about video measures nothing; 96"
        )

    # -- control 2: is the RL refusal attributable to the KEY? --------------
    attr = rl[RL_ATTRIBUTION_CONTROL]
    if attr.get("refusal_subject") == "video":
        return REFUSE, (
            f"the RL attribution control ({RL_ATTRIBUTION_CONTROL}) carries no video key and "
            "still earned the video refusal, so that refusal is not attributable to the key "
            "and the positive leg proves nothing; 96"
        )
    if not attr.get("reached_model_load"):
        return REFUSE, (
            f"the RL attribution control ({RL_ATTRIBUTION_CONTROL}) never reached the model "
            "load, so it is not shown to have passed the video check at all; its silence "
            "about video may be an earlier stop rather than a pass; 96"
        )

    # -- control 3: is the key READ, or merely present? ---------------------
    keyread = rl[RL_KEY_READ_CONTROL]
    if keyread.get("refusal_subject") != "video_key_type":
        return REFUSE, (
            f"the RL key-is-read control ({RL_KEY_READ_CONTROL}) declared a non-string video "
            f"key and refused on {keyread.get('refusal_subject')!r} rather than on the key's "
            "type, so there is no evidence the key is read rather than incidentally present; 96"
        )

    # -- the claim, plane by plane -------------------------------------------
    pos = rl[RL_POSITIVE]
    rl_refuses = (
        pos.get("code_chose_exit") == REFUSE
        and pos.get("refusal_subject") == "video"
        and pos.get("names_offending_sample")
    )

    # Leg 1: the DECLARATION must be refused, for the right reason. Exit 96
    # alone proves nothing: refusing on a missing 'text' column is the original
    # defect -- right code, wrong reason -- wearing the fix's clothes.
    declared = sft[SFT_DECIDING_ARM]
    dec_exit = declared.get("launcher_exit_code")
    if dec_exit == 0:
        return RED, (
            f"{CLAIM!r} is FALSE: the deciding arm ({SFT_DECIDING_ARM}) declared {NEEDLE!r} "
            f"through {VIDEO_COLUMN_ENV} on an otherwise trainable corpus and the run trained "
            f"anyway -- exit 0, {declared.get('examples_tokenized')} example(s) tokenized. The "
            "declaration, the only signal a user has that the modality is wanted, is still "
            "ignored; 5"
        )
    if dec_exit != REFUSE:
        return REFUSE, (
            f"the deciding arm ({SFT_DECIDING_ARM}) exited {dec_exit}, a shape this "
            "adjudicator does not classify; refusing rather than forcing it into GREEN "
            "or RED; 96"
        )
    if declared.get("refusal_names_missing_text") and not declared.get("output_names_needle"):
        return RED, (
            f"the RIGHT code for the WRONG reason is the original defect in a new outfit: the "
            f"deciding arm ({SFT_DECIDING_ARM}) declared {NEEDLE!r} via {VIDEO_COLUMN_ENV} and "
            "the run refused 96 on a missing 'text' column -- a refusal any textless corpus "
            "would earn just as well -- instead of naming the modality it cannot train. The "
            "declaration is still not the thing refused on; 5"
        )
    if not declared.get("output_names_needle"):
        return REFUSE, (
            f"the deciding arm ({SFT_DECIDING_ARM}) refused {REFUSE} but named neither the "
            f"modality {NEEDLE!r} nor a missing 'text' column -- a refusal this adjudicator "
            "cannot attribute; refusing rather than forcing it into GREEN or RED; 96"
        )

    # Leg 2: with NOTHING declared, the corpus must TRAIN -- the boundary is
    # the declaration, never the data -- and the drop must be NAMED on the
    # console. Refusing here is name-sniffing; silence here is the old drop.
    undeclared = sft[SFT_UNDECLARED_ARM]
    und_exit = undeclared.get("launcher_exit_code")
    if und_exit == REFUSE:
        return RED, (
            f"over-correction, not a fix: the undeclared arm ({SFT_UNDECLARED_ARM}) carries a "
            f"usable text column AND a {NEEDLE!r} key with NOTHING declared, and the run "
            f"refused {und_exit}. A refusal keyed on the column NAME breaks legitimate text "
            f"corpora that happen to carry such a column -- the boundary #490 drew is the "
            "DECLARATION, never the data; 5"
        )
    if und_exit == 0:
        if not undeclared.get("output_names_needle"):
            contrast = (
                "Meanwhile the RL plane REFUSES the same modality 96, naming both the modality "
                "and the offending sample, before any model is consulted."
                if rl_refuses
                else "The RL plane did not refuse it either."
            )
            return RED, (
                f"the silent drop is back: the undeclared arm ({SFT_UNDECLARED_ARM}) trained "
                f"(exit 0, {undeclared.get('examples_tokenized')} example(s) tokenized) and "
                f"the word {NEEDLE!r} never reached the console. Training an undeclared column "
                f"is intended; saying nothing about dropping it is the defect this row exists "
                f"to catch. {contrast}; 5"
            )
    else:
        return REFUSE, (
            f"the undeclared arm ({SFT_UNDECLARED_ARM}) exited {und_exit}, a shape this "
            "adjudicator does not classify; refusing rather than forcing it into GREEN "
            "or RED; 96"
        )

    return GREEN, (
        f"{CLAIM!r} holds on the plane operators train with, for the right reason: the "
        f"deciding arm ({SFT_DECIDING_ARM}) declared {NEEDLE!r} through {VIDEO_COLUMN_ENV} "
        f"and refused {REFUSE} naming the modality, while the undeclared arm "
        f"({SFT_UNDECLARED_ARM}) carried a mere {NEEDLE!r} key, trained "
        f"({undeclared.get('examples_tokenized')} example(s) tokenized), and the "
        "[fs:train:data] notice named the dropped column on the console. The refusal is "
        "keyed on the DECLARATION and the data-only case trains with its drop NAMED -- "
        "exactly the boundary #490 drew; 0"
    )


# --------------------------------------------------------------------------
# measurement -- the RL plane (needs no model, runs in seconds)
# --------------------------------------------------------------------------
def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    return stream if isinstance(stream, str) else stream.decode("utf-8", "replace")


def _write_rl_corpus(path: Path, leg: str) -> None:
    rec = json.loads(json.dumps(RL_RECORD))  # deep copy; legs must not share state
    if leg != "no_video_control":
        rec["video"] = RL_LEG_VIDEO_VALUE[leg]
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_rl_leg(leg: str, src_root: Path, out_dir: Path) -> dict[str, Any]:
    leg_dir = out_dir / f"rl_{leg}"
    leg_dir.mkdir(parents=True, exist_ok=True)
    corpus = leg_dir / CORPUS_FILENAME
    _write_rl_corpus(corpus, leg)

    driver = RL_DRIVER.format(
        src=os.fspath(src_root),
        model=RL_MODEL_SENTINEL,
        dataset=os.fspath(corpus),
        raised=RAISED_SENTINEL,
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.fspath(src_root)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", driver],
            capture_output=True,
            text=True,
            env=env,
            timeout=RL_TIMEOUT_S,
        )
        rc: int | None = proc.returncode
        out = _as_text(proc.stdout) + _as_text(proc.stderr)
    except subprocess.TimeoutExpired as exc:
        rc = None
        out = _as_text(exc.stdout) + _as_text(exc.stderr)

    parsed = None
    for line in out.splitlines():
        if line.startswith("T1_22_RL_JSON "):
            parsed = json.loads(line[len("T1_22_RL_JSON ") :])
    driver_exit = parsed.get("exit") if parsed else None
    raised = parsed.get("exc") if parsed else None
    low = out.lower()
    return {
        "status": "measured" if rc is not None else "timeout",
        "arm": f"rl_{leg}",
        "launcher_exit_code": rc,
        "code_chose_exit": driver_exit,
        # The driver re-raises, so these must agree. A disagreement means the
        # harness is overwriting the measurement it exists to take, and is
        # recorded rather than quietly reconciled.
        "exit_agrees": (rc == driver_exit) if driver_exit is not None else None,
        "raised": raised,
        "names_offending_sample": RL_SAMPLE_ID in out,
        "reached_model_load": RL_MODEL_SENTINEL in out,
        "refusal_subject": (
            "video"
            if VIDEO_REFUSAL_FRAGMENT in low
            else "model_load"
            if RL_MODEL_SENTINEL in out
            else "video_key_type"
            if raised and "video key" in raised
            else "other"
        ),
        "excerpts": [out[-1200:]],
    }


def _sft_plan() -> dict[str, Any]:
    """The run plan for the SFT plane, emitted for the tray-side measurer."""
    return {
        "row": ROW_ID,
        "arms": SFT_ARM_SPECS,
        "hyperparameters": {
            "max_steps": MAX_STEPS,
            "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
            "nodes": NODES,
            "gpus_per_node": GPUS_PER_NODE,
            "seed": SEED,
            "timeout_s": ARM_TIMEOUT_S,
        },
    }


# --------------------------------------------------------------------------
# --self-test: drive the PURE verdict over synthetic payloads
# --------------------------------------------------------------------------
def _sft_arm(**over: Any) -> dict[str, Any]:
    base = {
        "status": "measured",
        "launcher_exit_code": 0,
        "output_names_needle": False,
        "refusal_names_missing_text": False,
        "manifests_naming_needle": [],
        "manifest_chars": 30671,
        "examples_tokenized": 2,
        "names_own_declaration": False,
    }
    base.update(over)
    return base


def _rl_arm(**over: Any) -> dict[str, Any]:
    base = {
        "status": "measured",
        "launcher_exit_code": REFUSE,
        "code_chose_exit": REFUSE,
        "exit_agrees": True,
        "raised": None,
        "names_offending_sample": True,
        "reached_model_load": False,
        "refusal_subject": "video",
    }
    base.update(over)
    return base


def _measured_payload() -> dict[str, Any]:
    """The shape fix #490 is designed to produce, used as the self-test baseline.

    The deciding arm refuses 96 naming the modality; the undeclared arm trains
    and its [fs:train:data] notice names the dropped column; the bare-video
    arm's notice names the drop before its textless corpus earns the ordinary
    missing-'text' refusal; the RL plane is unchanged.
    """
    return {
        "sft": {
            "video_only": _sft_arm(
                launcher_exit_code=REFUSE,
                output_names_needle=True,
                refusal_names_missing_text=True,
            ),
            "text_video": _sft_arm(output_names_needle=True),
            "text_video_declared": _sft_arm(
                launcher_exit_code=REFUSE, output_names_needle=True
            ),
            "image_control": _sft_arm(launcher_exit_code=REFUSE, names_own_declaration=True),
        },
        "rl": {
            "video_declared": _rl_arm(),
            "no_video_control": _rl_arm(
                refusal_subject="model_load",
                reached_model_load=True,
                names_offending_sample=False,
            ),
            "non_string_video": _rl_arm(
                refusal_subject="video_key_type",
                code_chose_exit=None,
                launcher_exit_code=RAISED_SENTINEL,
                exit_agrees=None,
                raised="BatchRefusal: record 0 has a non-string video key (int)",
            ),
        },
    }


def _mutate(payload: dict[str, Any], plane: str, arm: str, **over: Any) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    out[plane][arm].update(over)
    return out


def _self_test() -> int:
    base = _measured_payload()
    controls: list[tuple[str, str, dict[str, Any], int]] = [
        # -- MUST_PASS: the post-fix shape, leg by leg -------------------------
        (
            "MUST_PASS",
            "the deciding arm refuses 96 AND names the modality -- the GREEN leg #490 added",
            base,
            GREEN,
        ),
        (
            "MUST_PASS",
            "the undeclared-corpus arm trains AND names the dropped column on the console -- "
            "the second GREEN leg (training is intended; naming the drop is the fix)",
            base,
            GREEN,
        ),
        (
            "MUST_PASS",
            "controls are judged BEFORE the claim: a broken control outranks a red-looking arm",
            _mutate(
                _mutate(base, "sft", "image_control", names_own_declaration=False),
                "sft",
                "text_video",
                output_names_needle=False,
            ),
            REFUSE,
        ),
        # -- MUST_FIRE: each way the boundary can be broken --------------------
        (
            "MUST_FIRE",
            "the RIGHT code for the WRONG reason is the original defect: 96 about a "
            "missing 'text' column instead of about the modality",
            _mutate(
                base,
                "sft",
                "text_video_declared",
                output_names_needle=False,
                refusal_names_missing_text=True,
            ),
            RED,
        ),
        (
            "MUST_FIRE",
            "a declaration that TRAINS anyway is a declaration ignored, which is the "
            "failure this row measures",
            _mutate(
                base,
                "sft",
                "text_video_declared",
                launcher_exit_code=0,
                output_names_needle=False,
            ),
            RED,
        ),
        (
            "MUST_FIRE",
            "an undeclared corpus that TRAINS without the drop named on the console is "
            "the silent drop returning",
            _mutate(base, "sft", "text_video", output_names_needle=False),
            RED,
        ),
        (
            "MUST_FIRE",
            "a manifest mention is not the console notice: naming the drop only in files "
            "operators never read changes nothing",
            _mutate(
                base,
                "sft",
                "text_video",
                output_names_needle=False,
                manifests_naming_needle=["run_manifest.json"],
            ),
            RED,
        ),
        (
            "MUST_FIRE",
            "refusing an UNDECLARED corpus is name-sniffing: refusal keyed on the column "
            "name breaks legitimate text corpora -- the opposite over-correction",
            _mutate(
                base, "sft", "text_video", launcher_exit_code=REFUSE, output_names_needle=True
            ),
            RED,
        ),
        (
            "MUST_FIRE",
            "a silent console drop with NO RL refusal to contrast it is still RED -- the "
            "contrast is context, not the defect",
            _mutate(
                _mutate(base, "rl", "video_declared", refusal_subject="other", code_chose_exit=0),
                "sft",
                "text_video",
                output_names_needle=False,
            ),
            RED,
        ),
        # -- MUST_FIRE: controls that never fired make the claim unreadable ----
        (
            "MUST_FIRE",
            "an SFT plane that cannot refuse ANY declaration makes video unreadable",
            _mutate(base, "sft", "image_control", names_own_declaration=False),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "a capability control that exits 0 is not a control",
            _mutate(base, "sft", "image_control", launcher_exit_code=0),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "an attribution control that ALSO claims video makes the positive leg meaningless",
            _mutate(base, "rl", "no_video_control", refusal_subject="video"),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "an attribution control that never reached the model load never passed the check",
            _mutate(
                base, "rl", "no_video_control", reached_model_load=False, refusal_subject="other"
            ),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "a key-is-read control that refuses on something else proves the key is unread",
            _mutate(base, "rl", "non_string_video", refusal_subject="other"),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "an absent arm is 95, never RED",
            {"sft": base["sft"], "rl": {k: v for k, v in base["rl"].items() if k != RL_POSITIVE}},
            UNMEASURED,
        ),
        (
            "MUST_FIRE",
            "a timed-out arm is 95, never RED",
            _mutate(base, "sft", "text_video", status="timeout"),
            UNMEASURED,
        ),
        (
            "MUST_FIRE",
            "a deciding arm that refuses 96 naming NEITHER the modality nor 'text' is "
            "unattributable -- refused rather than forced to a verdict",
            _mutate(
                base,
                "sft",
                "text_video_declared",
                output_names_needle=False,
                refusal_names_missing_text=False,
            ),
            REFUSE,
        ),
        (
            "MUST_FIRE",
            "an undeclared arm exiting outside the 0/96 boundary is unclassifiable, not "
            "GREEN or RED",
            _mutate(base, "sft", "text_video", launcher_exit_code=1),
            REFUSE,
        ),
    ]

    passed = 0
    for kind, name, payload, want in controls:
        got, reason = video_declaration_verdict(payload)
        ok = got == want
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(
            f"  [{mark}] {kind} {name}: want {VERDICT_NAMES[want]}, "
            f"got {VERDICT_NAMES.get(got, got)}"
        )
        if not ok:
            print(f"         reason: {reason}")

    # A suite whose controls all agree because the verdict never varies is not a
    # suite. Require every declared outcome to be exercised by something.
    exercised = {video_declaration_verdict(p)[0] for _, _, p, _ in controls}
    if exercised != {GREEN, RED, UNMEASURED, REFUSE}:
        print(f"  [FAIL] the suite never exercises {sorted(set(VERDICT_NAMES) - exercised)}")
        return 1
    print(f"{ROW_ID} self-test: {passed}/{len(controls)} controls PASS, all 4 verdicts exercised")
    return 0 if passed == len(controls) else 1


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=PROG, description=f"{ROW_ID}: {CLAIM}")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out-dir")
    ap.add_argument("--src-root", help="path to the checkout's src/ (RL plane)")
    ap.add_argument(
        "--sft-payload",
        help="JSON file holding the already-measured SFT arms; the SFT plane needs a model "
        "and a tray, so it is emitted as a plan, measured separately and joined here",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.out_dir or not args.src_root:
        print("REFUSE: --out-dir and --src-root are required without --self-test", file=sys.stderr)
        return REFUSE

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # The SFT plane needs a model and a tray this process does not have. Emit
    # the plan -- corpus columns and env per arm, including the new
    # FOUNDATIONSCALE_TRAIN_VIDEO_COLUMN declaration on the deciding arm -- so
    # the tray-side measurer runs exactly the boundary this verdict scores.
    (out_dir / "t1_22_sft_plan.json").write_text(
        json.dumps(_sft_plan(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rl = {leg: _run_rl_leg(leg, Path(args.src_root).resolve(), out_dir) for leg in RL_ARMS}
    sft: dict[str, Any] = {}
    if args.sft_payload:
        # An unreadable payload is an unmet PRECONDITION, not a failed claim, and
        # certainly not an uncaught traceback: rc 1 is outside this repository's
        # four-state contract entirely. Measured the hard way -- /tmp is
        # node-local on this estate, so a payload written on the login node is
        # simply absent from the compute node, and the first version of this
        # tool answered that with a stack trace and rc 1.
        try:
            sft = json.loads(Path(args.sft_payload).read_text(encoding="utf-8"))
        except OSError as exc:
            print(
                f"UNMEASURED: --sft-payload {args.sft_payload!r} is unreadable "
                f"({exc.__class__.__name__}: "
                f"{exc}); the SFT arms were never joined, so nothing was adjudicated; 95",
                file=sys.stderr,
            )
            return UNMEASURED
        except ValueError as exc:
            print(
                f"REFUSE: --sft-payload {args.sft_payload!r} is not valid JSON ({exc}); 96",
                file=sys.stderr,
            )
            return REFUSE

    payload = {"sft": sft, "rl": rl}
    verdict, reason = video_declaration_verdict(payload)
    for plane, arms in payload.items():
        for name, arm in arms.items():
            (out_dir / f"t1_22_{plane}_{name}.json").write_text(
                json.dumps(arm, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    print(f"{ROW_ID} CLAIM   : {CLAIM}")
    print(f"{ROW_ID} CONTROL : {CONTROL}")
    print(f"{ROW_ID} VERDICT {VERDICT_NAMES.get(verdict, verdict)}: {reason}")
    (out_dir / "t1_22_verdict.txt").write_text(
        f"{VERDICT_NAMES.get(verdict, verdict)}: {reason}\n", encoding="utf-8"
    )
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
