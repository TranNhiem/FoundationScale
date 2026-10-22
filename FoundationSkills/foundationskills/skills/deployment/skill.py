"""DeploymentSkill (A4.9): serve a checkpoint with the measured vLLM container, inside the measured envelope or UNMEASURED.

Actions:
- ``plan`` (default): probe the deployment (interfaces/fs/emit_serve.py) and write an ``fs_launch_spec`` that starts the
  ``vllm031`` server confined to the leased GPU UUID. The user launches it with ``fskills hash --spec S`` and then
  ``fskills launch --spec S --confirm <hash> --no-submit`` (under nohup or an srun step: the server runs in the
  foreground). A target outside the measured serving envelope plans as UNMEASURED.
- ``verify``: against the running endpoint, check that `/v1/models` lists every served name, that every server GPU
  process sits on the leased UUID, and that a smoke completion per served name returns text. It writes a
  ``serving_spec`` record.

Exit contract:
- 96 REFUSED: DP-IN-001..004.
- 5 RED: health or smoke failed (DP-HO-003), or the server is on the wrong GPU (DP-HO-004).
- 95 UNMEASURED: the target is outside the measured envelope (DP-HO-001), or a LoRA deployment in a cell whose served
  fidelity is not measured (DP-HO-002).
- 0 PASS only for a healthy, correctly placed deployment in a measured cell. A LoRA cell also needs its served fidelity
  measured; gemma4 LoRA keeps 99.5% of the adapter's held-out gain.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Callable

from foundationskills.core.artifacts import Artifact, write_artifact
from foundationskills.core.contract import (
    BaseSkill,
    Diagnosis,
    Finding,
    FSInterface,
    RuleSpec,
    Scope,
    Severity,
    SkillContext,
    SkillResult,
)
from foundationskills.core.provenance import make_provenance
from foundationskills.core.status import Status
from foundationskills.interfaces.fs.emit_serve import emit_serve, model_family, serve_probes
from foundationskills.interfaces.fs.serving_envelope import ENGINE, lookup

_GAP_RULES = (("not a model directory", "DP-IN-001"), ("no adapter_config", "DP-IN-001"),
              ("was trained on", "DP-IN-002"), ("not a GPU UUID", "DP-IN-003"), ("already in use", "DP-IN-003"))
_LORA_FIDELITY = ("served-LoRA fidelity is not measured for this cell: it needs a task-level held-out NLL check through "
                  "the server (per-token logprobs are dominated by bf16 noise; a49/FINDINGS.md)")


def _http_json(url: str, body: dict[str, Any] | None = None, timeout: int = 120) -> dict[str, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - localhost endpoint from the spec
        return json.load(resp)


def _gpu_placement(port: int) -> dict[str, Any]:
    """The GPUs holding processes of the vLLM server on ``port`` (this host's nvidia-smi; None when unmeasurable)."""
    try:
        pids = subprocess.run(["pgrep", "-u", str(Path.home().name), "-f",
                               f"^/usr/bin/python3 /usr/local/bin/vllm serve /models/served .*--port {port}"],
                              capture_output=True, text=True, check=False).stdout.split()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=False).stdout.splitlines()
    except OSError as exc:
        return {"measured": False, "reason": str(exc)}
    if not pids:
        return {"measured": False, "reason": f"no vllm serve process for port {port} on this host"}
    roots, gpus = set(pids), set()
    for line in apps:
        uuid, _, pid = (x.strip() for x in line.partition(","))
        current = pid
        while current and current not in ("0", "1"):
            if current in roots:
                gpus.add(uuid)
                break
            current = subprocess.run(["ps", "-o", "ppid=", "-p", current], capture_output=True, text=True,
                                     check=False).stdout.strip()
    return {"measured": True, "gpus": sorted(gpus), "server_pids": sorted(roots)}


class DeploymentSkill(BaseSkill):
    name = "deployment"
    version = "0.1.0"
    description = (
        "Serves a checkpoint (full or merged weights, an already-quantized checkpoint such as fp8, or base + LoRA "
        "adapter unmerged) with the measured vLLM container: plans a GPU-confined serving fs_launch_spec probe-first "
        "(launched only with the plan hash the user types back), then verifies the running endpoint (served names, GPU "
        "placement, smoke completion) and records a serving_spec. Whether a target fits the measured serving envelope "
        "is answered here; targets outside it are UNMEASURED, never PASS. Do NOT use to quantize or compress weights "
        "(fskills-compression), merge adapters (fskills-merging), train (fskills-training) or benchmark checkpoints "
        "(fskills-evaluation)."
    )
    scope = Scope(model_types=("llm", "vlm"), families="any", stages=("sft", "preference", "rl"), algorithms=(),
                  methods=("full", "lora", "qlora"), hardware=("gb200",))
    consumes = ("checkpoint",)
    produces = ("fs_launch_spec", "serving_spec")
    input_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "action": {"type": "string", "enum": ["plan", "verify"]},
            "model": {"type": "string", "minLength": 1},
            "adapter": {"type": "string", "minLength": 1},
            "gpu": {"type": "string"},
            "port": {"type": "integer", "minimum": 1024, "maximum": 65535},
            "served_name": {"type": "string", "minLength": 1},
            "modality": {"type": "string", "enum": ["text", "vision"]},
            "max_model_len": {"type": "integer", "minimum": 256},
            "spec": {"type": "string", "minLength": 1},
        },
    }
    output_schema: dict[str, Any] = {"type": "object", "additionalProperties": True}
    rules = (
        RuleSpec("DP-IN-001", "plan: the model is not a model directory, or the adapter has no adapter_config.json",
                 Severity.BLOCK, "input"),
        RuleSpec("DP-IN-002", "plan: the adapter was trained on a different base than the served model", Severity.BLOCK,
                 "input"),
        RuleSpec("DP-IN-003", "plan: no leased GPU UUID given, or the port is already in use (the server must be "
                 "confined to one leased GPU)", Severity.BLOCK, "input"),
        RuleSpec("DP-IN-004", "verify: the spec is unreadable or the endpoint does not answer /v1/models", Severity.BLOCK,
                 "input"),
        RuleSpec("DP-HO-001", "the deployment target (engine x family x mode x modality) is outside the measured serving "
                 "envelope - UNMEASURED, never PASS", Severity.WARN, "handoff"),
        RuleSpec("DP-HO-002", "LoRA deployment in a cell whose served-adapter fidelity is not measured - UNMEASURED even when "
                 "healthy",
                 Severity.WARN, "handoff"),
        RuleSpec("DP-HO-003", "verify: a served name is missing from /v1/models or a smoke completion returned no text - "
                 "RED", Severity.BLOCK, "handoff"),
        RuleSpec("DP-HO-004", "verify: a server GPU process is on a GPU other than the leased UUID - RED", Severity.BLOCK,
                 "handoff"),
    )
    fs_interface = FSInterface(entries=("fskills serve",), emits=("fs_launch_spec", "serving_spec"),
                               apis=("emit_serve", "serving_envelope.lookup"),
                               notes="Launching goes through `fskills launch --confirm` like every other spec.")

    def __init__(self, http: Callable[..., dict[str, Any]] = _http_json,
                 placement: Callable[[int], dict[str, Any]] = _gpu_placement) -> None:
        super().__init__()
        self.http, self.placement = http, placement  # injectable: unit tests never start a server

    @staticmethod
    def _spec(request: dict[str, Any]) -> dict[str, Any]:
        try:
            data = json.loads(Path(str(request.get("spec") or "")).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data.get("payload", data) if isinstance(data, dict) else {}

    def check_inputs(self, request: dict[str, Any], ctx: SkillContext) -> list[Finding]:
        if str(request.get("action") or "plan") == "verify":
            serve = self._spec(request).get("serve") or {}
            try:
                self.http(f"{serve['endpoint']}/v1/models", None, 10)
            except Exception as exc:  # noqa: BLE001 - unreadable spec or a dead endpoint both refuse
                return [self.finding("DP-IN-004", f"missing input: endpoint not answering ({type(exc).__name__}: {exc})",
                                     {"spec": request.get("spec")}, "launch the serving spec, then verify")]
            return []
        if not request.get("model"):
            return [self.finding("DP-IN-001", "missing input: model", {}, "pass the model (or merged checkpoint) path")]
        gaps = serve_probes(Path(request["model"]).expanduser(),
                            Path(request["adapter"]).expanduser() if request.get("adapter") else None,
                            str(request.get("gpu") or ""), int(request.get("port") or 18841))
        return [self.finding(next((r for t, r in _GAP_RULES if t in g), "DP-IN-003"), g, {"gap": g},
                             "fix the named input and re-plan") for g in gaps]

    def run(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        if str(request.get("action") or "plan") == "verify":
            return self._verify(request, ctx)
        spec = emit_serve(request["model"], str(request["gpu"]), int(request.get("port") or 18841),
                          request.get("adapter"), str(request.get("served_name") or "model"),
                          str(request.get("modality") or "text"), int(request.get("max_model_len") or 16384))
        serve = spec["serve"]
        cell = lookup(str(serve["family"]), serve["mode"], serve["modality"])
        stamp = make_provenance(self.name, self.version, {"model": serve["model"], "adapter": str(serve["adapter"])})
        ref = write_artifact(Artifact("fs_launch_spec", "serve", spec, stamp), ctx.artifacts_dir)
        payload = {"spec": spec, "spec_path": ref.path, "envelope": cell,
                   "next": f"fskills hash --spec {ref.path} (read it yourself), then nohup fskills launch --spec "
                           f"{ref.path} --confirm <hash> --no-submit &, then fskills serve verify --spec {ref.path}"}
        if cell is None:
            return SkillResult(Status.UNMEASURED, payload, artifacts=(ref,), provenance=stamp, findings=(self.finding(
                "DP-HO-001", f"{ENGINE} x {serve['family']} x {serve['mode']} x {serve['modality']} was never served "
                "here: the deployment is UNMEASURED", {"target": serve}, "measure it and add a row to "
                "interfaces/fs/serving_envelope.py, or deploy a measured cell"),))
        return SkillResult(Status.PASS, payload, artifacts=(ref,), provenance=stamp)

    def _verify(self, request: dict[str, Any], ctx: SkillContext) -> SkillResult:
        serve = self._spec(request)["serve"]
        listed = [m.get("id") for m in self.http(f"{serve['endpoint']}/v1/models", None, 30).get("data", [])]
        missing = [n for n in serve["served_names"] if n not in listed]
        smoke: dict[str, Any] = {}
        for name in serve["served_names"]:
            if name in missing:
                continue
            try:
                out = self.http(f"{serve['endpoint']}/v1/chat/completions",
                                {"model": name, "messages": [{"role": "user", "content": "Reply with the word OK."}],
                                 "max_tokens": 16, "temperature": 0}, 300)
                smoke[name] = str(((out.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
            except Exception as exc:  # noqa: BLE001 - a failed completion is a measured failure
                smoke[name] = ""
                smoke[f"{name}.error"] = f"{type(exc).__name__}: {exc}"
        place = self.placement(int(serve["port"]))
        cell = lookup(str(serve["family"]), serve["mode"], serve["modality"])
        checks = {"served_names_listed": not missing, "smoke_text": all(smoke.get(n) for n in serve["served_names"]),
                  "placement": place}
        findings: list[Finding] = []
        if missing or not checks["smoke_text"]:
            findings.append(self.finding("DP-HO-003", f"served names missing {missing} or empty smoke {smoke}",
                                         {"listed": listed, "smoke": smoke}, "read the server log; re-launch"))
        if place.get("measured") and place.get("gpus") and set(place["gpus"]) != {serve["gpu"]}:
            findings.append(self.finding("DP-HO-004", f"server on {place['gpus']}, leased {serve['gpu']}",
                                         {"placement": place}, "stop the server now; re-plan with the leased UUID"))
        red = any(f.severity is Severity.BLOCK for f in findings)
        if not red and cell is None:
            findings.append(self.finding("DP-HO-001", "target outside the measured serving envelope", {"target": serve},
                                         "add the measured row before claiming this deployment"))
        if not red and serve["mode"] == "lora" and (cell or {}).get("fidelity") != "measured":
            findings.append(self.finding("DP-HO-002", _LORA_FIDELITY, {"envelope": cell},
                                         "serve a float32-stored merged checkpoint (A4.7) for a measured full-weights "
                                         "cell, or wait for the task-level fidelity check"))
        verdict = "RED" if red else ("UNMEASURED" if findings else "PASS")
        if not place.get("measured") and verdict == "PASS":
            verdict = "UNMEASURED"  # placement unmeasured (verify ran off the server host) is never a pass
            findings.append(self.finding("DP-HO-001", f"GPU placement unmeasured: {place.get('reason')}",
                                         {"placement": place}, "run verify on the server's host"))
        record = {"endpoint": serve["endpoint"], "served_names": serve["served_names"], "family": serve["family"],
                  "mode": serve["mode"], "modality": serve["modality"], "engine": ENGINE, "envelope": cell,
                  "checks": checks, "smoke": smoke, "gpu": serve["gpu"], "verdict": verdict}
        stamp = make_provenance(self.name, self.version, {"endpoint": serve["endpoint"]})
        ref = write_artifact(Artifact("serving_spec", "serve", record, stamp), ctx.artifacts_dir)
        status = {"PASS": Status.PASS, "RED": Status.RED, "UNMEASURED": Status.UNMEASURED}[verdict]
        return SkillResult(status, record, artifacts=(ref,), findings=tuple(findings), provenance=stamp)

    def check_handoff(self, result: SkillResult, ctx: SkillContext) -> list[Finding]:
        return []  # every DP-HO verdict is decided in run(), next to its measured evidence

    def diagnose(self, failure: BaseException | SkillResult) -> Diagnosis:
        symptom = (f"deployment {type(failure).__name__}: {failure}" if isinstance(failure, BaseException)
                   else f"deployment {failure.status.value}: {failure.refusal or 'see findings'}")
        return Diagnosis(symptom=symptom,
                         likely_causes=("a path, GPU UUID or port is wrong (DP-IN-001..003)", "the server is not up "
                                        "(DP-IN-004)", "the target was never measured (DP-HO-001)", "LoRA fidelity "
                                        "unmeasured (DP-HO-002)", "wrong GPU (DP-HO-004)"),
                         checks=("read the finding: it names the rule", "curl <endpoint>/v1/models",
                                 "nvidia-smi --query-compute-apps=gpu_uuid,pid"),
                         recovery=("fix the input and re-plan", "launch with the hash you read", "stop a misplaced "
                                   "server at once and re-plan with the leased UUID"))

    def must_fire_fixtures(self) -> dict[str, dict[str, Any]]:
        plan = {"action": "plan", "model": "merged/gemma-4-12B-sft", "gpu": "GPU-447bb30a", "port": 18841}
        verify = {"action": "verify", "spec": "artifacts/fs_launch_spec.serve.json"}
        return {
            "DP-IN-001": {"request": {**plan, "model": "merged/missing"}, "observed": "no config.json in model",
                          "expected": "REFUSED naming DP-IN-001"},
            "DP-IN-002": {"request": {**plan, "adapter": "runs/sft/final"}, "observed": "adapter trained on gemma-4-31B-it",
                          "expected": "REFUSED naming DP-IN-002"},
            "DP-IN-003": {"request": {**plan, "gpu": ""}, "expected": "REFUSED naming DP-IN-003 (no leased GPU UUID)"},
            "DP-IN-004": {"request": verify, "observed": "endpoint connection refused", "expected": "REFUSED naming DP-IN-004"},
            "DP-HO-001": {"request": {**plan, "model": "models/nemotron-nano"}, "observed": "family nemotron never served",
                          "expected": "UNMEASURED naming DP-HO-001"},
            "DP-HO-002": {"request": verify, "observed": "LoRA cell fidelity marked unmeasured",
                          "expected": "UNMEASURED naming DP-HO-002 (healthy, but served-LoRA fidelity unmeasured)"},
            "DP-HO-003": {"request": verify, "observed": "smoke completion returned empty text",
                          "expected": "RED naming DP-HO-003"},
            "DP-HO-004": {"request": verify, "observed": "leased GPU-447bb30a, server on GPU-29098e5e",
                          "expected": "RED naming DP-HO-004 (not the leased GPU)"},
        }
