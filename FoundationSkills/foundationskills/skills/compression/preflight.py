#!/usr/bin/env python3
"""preflight.py — capability probe. Runs BEFORE any GPU-hours are spent.

Exit 0 = PASS / PASS_WITH_WARNINGS (may submit)
Exit 1 = FAIL (>=1 failed check; do not submit)
Exit 2 = PROBE_ERROR (probe itself crashed; treat as FAIL, read the FULL log)

Statuses: PASS / WARN / FAIL / UNKN. UNKN means the check could not be
evaluated (e.g. a config-derived check when the config never loaded). It is
counted as non-passing and never as PASS — reporting "all clear" from an
input that was never read is the exact failure mode this probe exists to stop.

Stdlib + torch only. Optional libraries (transformers, flashinfer) are
imported defensively: absence degrades to a check result, never a crash.
"""
import argparse, importlib.util, importlib.metadata as md
import json, os, platform, re, shutil, site, socket, subprocess, sys, tempfile, traceback
from pathlib import Path

SCHEMA = "forge.preflight.v1"

# Planning heuristics (NOT measurements): estimated memory multiplier on source
# weight bytes per compression profile. Tune with --memory-multiplier.
MEM_FACTOR = {"ptq": 1.25, "rtnste": 2.0, "qat": 5.0, "cad": 2.5, "pad": 3.0, "serve": 1.35}
SM100_REAL = {"nvfp4_real", "w4a4_real", "mxfp4_real"}
SM100_FAKE = {"nvfp4_fake", "w4a4_fake", "mxfp4_fake", "nvfp4", "w4a4"}
# Formats that need no hardware gate but ARE valid vocabulary. Anything outside
# KNOWN_FORMATS is rejected rather than ignored: a typo'd format token would
# otherwise silently skip its gate and report PASS.
FORMAT_UNGATED = {"rtnste_int4", "int4_modelopt", "kv_fp8", "kv_int8",
                  "fp8_w8a8", "int8_w8a8", "gptq_int4", "awq_int4", "bf16", "fp16"}
KNOWN_FORMATS = SM100_REAL | SM100_FAKE | FORMAT_UNGATED
KV_PAT = re.compile(r"sliding_window|layer_types|gated.?delta|delta_net|linear_attention|"
                    r"linear_num_key_heads|mamba|ssm|state_size|conv_kernel|hybrid|recurrent", re.I)


def run(cmd, timeout=30):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 127, "", str(e)


def ver_tuple(s):
    if not s:
        return None
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", str(s))
    return tuple(int(x) for x in m.groups(default="0")) if m else None


def add(checks, name, status, detail, fatal=False, metrics=None, remedy="", wall=None):
    checks.append({"name": name, "status": status, "fatal": fatal, "detail": detail,
                   "metrics": metrics or {}, "remedy": remedy, "wall": wall})
    print(f"[{status:4s}] {name}: {detail}", flush=True)


# ---------------------------------------------------------------- wall #8/#9
def check_env_hygiene(a, checks, env):
    bad = []
    env["PYTHONNOUSERSITE"] = os.environ.get("PYTHONNOUSERSITE")
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        bad.append("PYTHONNOUSERSITE is not 1")
    try:
        if site.ENABLE_USER_SITE:
            bad.append("site.ENABLE_USER_SITE is True")
        usp = site.getusersitepackages()
        env["user_site_in_sys_path"] = any(
            p and Path(p).resolve() == Path(usp).resolve() for p in sys.path)
        if env["user_site_in_sys_path"]:
            bad.append(f"user site on sys.path: {usp}")
    except Exception as e:
        bad.append(f"could not inspect user site: {e!r}")
    if a.repo:
        src = str((Path(a.repo) / "src").resolve())
        if src not in os.environ.get("PYTHONPATH", "").split(os.pathsep):
            bad.append(f"PYTHONPATH missing {src}")
    for mod in a.required_imports:
        if importlib.util.find_spec(mod) is None:
            bad.append(f"required import not found: {mod}")
    if bad:
        add(checks, "env_hygiene", "FAIL", "; ".join(bad), fatal=True, wall=8,
            remedy="export PYTHONNOUSERSITE=1; set PYTHONPATH=$REPO/src; use a clean venv/conda env")
    else:
        add(checks, "env_hygiene", "PASS", "no user-site shadowing; required imports resolve")


def nvidia_smi():
    rc, out, err = run(["nvidia-smi", "--query-gpu=name,driver_version",
                        "--format=csv,noheader"])
    gpus = []
    if rc == 0:
        for line in out.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 2:
                gpus.append({"name": parts[0], "driver_version": parts[1]})
    rc2, out2, err2 = run(["nvidia-smi"])
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", out2 or "")
    return gpus, (m.group(1) if m else None), (err or err2 or "")


# ---------------------------------------------------------------- wall #1
def check_torch_gpu(a, checks, hw):
    try:
        import torch
    except Exception as e:
        add(checks, "torch_import", "FAIL", repr(e), fatal=True,
            remedy="install the correct CUDA torch wheel for this image")
        return None
    hw["torch"] = torch.__version__
    hw["torch_cuda"] = torch.version.cuda
    if torch.version.cuda is None or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        add(checks, "gpu_present", "FAIL",
            f"torch_cuda={torch.version.cuda} is_available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}", fatal=True,
            remedy="run on a GPU node with a CUDA-enabled torch build, not a login/CPU node")
        return None
    hw["gpus"] = [{"name": torch.cuda.get_device_name(i),
                   "compute_capability": list(torch.cuda.get_device_capability(i)),
                   "mem_total_bytes": torch.cuda.get_device_properties(i).total_memory}
                  for i in range(torch.cuda.device_count())]
    add(checks, "gpu_present", "PASS",
        f"{torch.cuda.device_count()} GPU(s); dev0={hw['gpus'][0]['name']} "
        f"cc={tuple(hw['gpus'][0]['compute_capability'])}")

    # Wall #1: torch.cuda.is_available() is NOT proof. cu130-built torch on a
    # 12.2 driver SILENTLY falls back to CPU and returns plausible-but-slow
    # numbers. Proof = exact known-answer integer kernel + real CUDA allocation
    # + device placement + finite fp16 matmul + a throughput floor no CPU reaches.
    try:
        dev = 0
        torch.cuda.set_device(dev)
        n = 1_000_000
        x = torch.arange(n, device="cuda", dtype=torch.int64)
        alloc = torch.cuda.memory_allocated(dev)
        y = x * x
        torch.cuda.synchronize(dev)
        exact = int(y[-1].item()) == (n - 1) * (n - 1)
        del x, y
        m = 4096
        am = torch.randn(m, m, device="cuda", dtype=torch.float16)
        bm = torch.randn(m, m, device="cuda", dtype=torch.float16)
        for _ in range(3):
            cm = am @ bm                       # warm-up: cold start poisons timing
        torch.cuda.synchronize(dev)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        reps = 10
        for _ in range(reps):
            cm = am @ bm
        end.record()
        torch.cuda.synchronize(dev)
        ms = start.elapsed_time(end)
        finite = bool(torch.isfinite(cm.float()).all().item())
        on_gpu = cm.device.type == "cuda"
        tflops = reps * 2.0 * m**3 / (ms / 1e3) / 1e12 if ms > 0 else 0.0
        del am, bm, cm
        torch.cuda.empty_cache()
        floor = float(os.environ.get("PREFLIGHT_TFLOPS_FLOOR", "5.0"))
        metrics = {"fp16_4096_matmul_ms_total": ms, "tflops_fp16": tflops,
                   "cuda_alloc_bytes": alloc, "tflops_floor": floor}
        hw["tflops_fp16"] = tflops
        if not (on_gpu and alloc > 0 and exact and finite):
            add(checks, "real_gpu_execution", "FAIL",
                f"on_gpu={on_gpu} alloc={alloc} exact={exact} finite={finite}",
                fatal=True, metrics=metrics, wall=1,
                remedy="torch is not really executing on GPU; check driver/toolkit major versions and the container image")
        elif tflops < floor:
            add(checks, "real_gpu_execution", "FAIL",
                f"{tflops:.1f} TFLOP/s < {floor} floor — SUSPECT SILENT CPU FALLBACK "
                f"(torch {torch.__version__}, cu{torch.version.cuda}). Do NOT trust any benchmark from this env.",
                fatal=True, metrics=metrics, wall=1,
                remedy="install a torch wheel whose CUDA major matches the driver; override the floor via "
                       "PREFLIGHT_TFLOPS_FLOOR only after validating on known-good hardware")
        else:
            add(checks, "real_gpu_execution", "PASS",
                f"exact int kernel, finite fp16 matmul on-device, {tflops:.1f} TFLOP/s",
                metrics=metrics)
    except Exception as e:
        add(checks, "real_gpu_execution", "FAIL", repr(e), fatal=True, wall=1,
            remedy="CUDA kernel launch failed; do not benchmark until this passes")
    return torch


def check_versions(a, checks, hw):
    gpus_smi, driver_cuda, smi_err = nvidia_smi()
    if gpus_smi:
        hw["driver_version"] = gpus_smi[0]["driver_version"]
    hw["driver_cuda"] = driver_cuda
    rc, out, _ = run(["nvcc", "--version"])
    m = re.search(r"release\s+(\d+\.\d+)", out or "")
    hw["nvcc_cuda"] = m.group(1) if (rc == 0 and m) else None
    tc, dc = ver_tuple(hw.get("torch_cuda")), ver_tuple(driver_cuda)
    if not tc:
        add(checks, "cuda_version_matrix", "FAIL", "torch.version.cuda is None (CPU-only torch build)",
            fatal=True, wall=1, remedy="install a CUDA-enabled torch wheel")
    elif not dc:
        add(checks, "cuda_version_matrix", "WARN",
            f"could not parse driver CUDA from nvidia-smi ({smi_err.strip() or 'no output'}); "
            f"torch_cuda={hw.get('torch_cuda')}", wall=1,
            remedy="run nvidia-smi manually; confirm driver CUDA major >= torch CUDA major")
    elif tc[0] > dc[0]:
        add(checks, "cuda_version_matrix", "FAIL",
            f"torch built for CUDA {hw.get('torch_cuda')} but driver supports only {driver_cuda}: "
            f"major-version mismatch — this is the silent-CPU-fallback wall", fatal=True, wall=1,
            remedy=f"install torch built for CUDA {dc[0]}.x or upgrade the driver")
    elif tc > dc:
        add(checks, "cuda_version_matrix", "WARN",
            f"torch CUDA {hw.get('torch_cuda')} newer than driver CUDA {driver_cuda} but same major: "
            f"forward-compat only (cu128-on-12.x-driver is the known-good case); "
            f"real_gpu_execution must pass on every node type", wall=1,
            remedy="prefer matching minor versions; re-run preflight per node type")
    else:
        add(checks, "cuda_version_matrix", "PASS",
            f"driver={hw.get('driver_version')} driver_cuda={driver_cuda} "
            f"torch_cuda={hw.get('torch_cuda')} nvcc={hw.get('nvcc_cuda')}")


# ---------------------------------------------------------------- walls #3/#4
def check_format_vocabulary(a, checks):
    """Pure string validation — must run even when torch/GPU probing is unavailable,
    otherwise a typo'd format token silently skips its hardware gate."""
    unknown = set(a.planned_formats) - KNOWN_FORMATS
    if unknown:
        add(checks, "format_vocabulary", "FAIL",
            f"unrecognized --planned-formats token(s): {sorted(unknown)}. An unknown token silently "
            f"skips its hardware gate and would report PASS", fatal=True,
            remedy=f"use documented tokens: {', '.join(sorted(KNOWN_FORMATS))}")


def check_format_gates(a, checks, hw):
    planned = set(a.planned_formats)
    if not hw.get("gpus"):
        return
    cc = tuple(hw["gpus"][0]["compute_capability"])
    real = planned & SM100_REAL
    fake = planned & SM100_FAKE
    if real and cc < (10, 0):
        add(checks, "format_hardware_gate", "FAIL",
            f"planned {sorted(real)} requires sm100 (Blackwell); visible cc={cc}",
            fatal=True, wall=3,
            remedy="move those cells to GB200/B200 (sm100) or convert them to fake-quant measurement cells")
    elif (real | fake) and cc < (10, 0):
        add(checks, "format_hardware_gate", "WARN",
            f"{sorted(real | fake)} on cc={cc}: fake-quant only — distributional metrics "
            f"(ppl, accept-length) are valid; NO tok/s speedup may be claimed", wall=3,
            remedy="label all efficiency numbers from these cells as fake-quant/Hopper")
    else:
        add(checks, "format_hardware_gate", "PASS", f"cc={cc} planned={sorted(planned) or 'none'}")
    if "int4_modelopt" in planned:
        add(checks, "modelopt_int4_scope", "WARN",
            "NVIDIA modelopt INT4 configs observed in the field are symmetric-only "
            "(INT4_BLOCKWISE_WEIGHT_ONLY_CFG, INT4_AWQ_CFG — no zero-point)", wall=4,
            remedy="verify available configs in the installed modelopt before promising asymmetric "
                   "INT4; otherwise plan RTN+STE")


# ---------------------------------------------------------------- wall #2
def check_toolchain_and_arch(a, checks, model_info, env):
    def pkgv(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None
    tf = pkgv("transformers")
    env["transformers"] = tf
    env["gptqmodel"] = pkgv("gptqmodel")
    env["llmcompressor"] = pkgv("llmcompressor")
    tv = ver_tuple(tf)
    if tf is None:
        add(checks, "config_load", "FAIL", "transformers is not installed", fatal=True, wall=2,
            remedy="install transformers (>=5.5 for Gemma-4/Qwen3.6-class archs; 4.x for "
                   "gptqmodel/llmcompressor toolchains — never both in one env)")
        return
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(a.model, revision=a.revision,
                                         trust_remote_code=a.trust_remote_code)
        model_info.update({"architectures": getattr(cfg, "architectures", None),
                           "model_type": getattr(cfg, "model_type", None),
                           "native_quantization": getattr(cfg, "quantization_config", None),
                           "config": cfg, "config_inspected": True})
    except Exception as e:
        add(checks, "config_load", "FAIL", f"AutoConfig failed: {e!r}", fatal=True, wall=2,
            remedy="fix model path/revision/token, pass --trust-remote-code if the model needs it, "
                   "or change transformers version")
        return
    add(checks, "config_load", "PASS",
        f"model_type={model_info['model_type']} archs={model_info['architectures']}")

    wants_gptq = bool({"gptq", "gptqmodel", "llmcompressor"} & set(a.planned_tools))
    has_gptq = env["gptqmodel"] is not None or env["llmcompressor"] is not None
    if wants_gptq and tv and tv[0] >= 5:
        add(checks, "toolchain_conflict", "FAIL",
            f"plan uses {a.planned_tools} but transformers={tf}: gptqmodel/llmcompressor require "
            f"transformers 4.x; Gemma-4/Qwen3.6-class archs require >=5.5 — mutually exclusive in ONE env",
            fatal=True, wall=2,
            remedy="separate venv per toolchain (tf4.x+GPTQ | tf>=5.5+new arch), or drop GPTQ and use "
                   "RTN+STE which has no such pin")
    elif has_gptq and tv and tv[0] >= 5:
        add(checks, "toolchain_conflict", "WARN",
            f"gptqmodel/llmcompressor co-installed with transformers {tf}; not in the current plan. "
            f"Do not add GPTQ cells to this env later.", wall=2,
            remedy="keep GPTQ work in a separate tf-4.x env with its own preflight run")
    else:
        add(checks, "toolchain_conflict", "PASS",
            f"transformers={tf} gptqmodel={env['gptqmodel']} llmcompressor={env['llmcompressor']} "
            f"planned_tools={a.planned_tools or 'none'}")

    # Prove the arch class instantiates WITHOUT weights (meta device, zero memory).
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoModel
        with torch.device("meta"):
            try:
                mdl = AutoModelForCausalLM.from_config(cfg, trust_remote_code=a.trust_remote_code)
            except Exception:
                mdl = AutoModel.from_config(cfg, trust_remote_code=a.trust_remote_code)
        n_meta = sum(1 for p in mdl.parameters() if p.device.type == "meta")
        del mdl
        if n_meta == 0:
            raise RuntimeError("from_config produced no meta-device parameters")
        model_info["arch_load_verified"] = True
        add(checks, "arch_class_load", "PASS",
            f"arch instantiated on meta device ({n_meta} meta params); no weights loaded")
    except Exception as e:
        model_info["arch_load_verified"] = False
        st = "WARN" if a.allow_unverified_arch else "FAIL"
        add(checks, "arch_class_load", st, repr(e), fatal=not a.allow_unverified_arch, wall=2,
            remedy="newest archs need transformers>=5.5 (Gemma4ForConditionalGeneration, "
                   "Qwen3_5MoeForConditionalGeneration); fix the env or pass --allow-unverified-arch — "
                   "but do not sweep an uninstantiable arch")


# ---------------------------------------------------------------- wall #6
def scan_obj(o, pat, path="$", hits=None, depth=0):
    hits = hits if hits is not None else []
    if depth > 8 or len(hits) >= 50:
        return hits
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{path}.{k}"
            if pat.search(str(k)):
                hits.append(p)
            scan_obj(v, pat, p, hits, depth + 1)
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o[:64]):
            scan_obj(v, pat, f"{path}[{i}]", hits, depth + 1)
    elif pat.search(str(o)):
        hits.append(f"{path}={o}")
    return hits


def check_partial_kv(a, checks, model_info):
    cfg = model_info.get("config")
    raw = None
    if cfg is not None and hasattr(cfg, "to_dict"):
        raw = cfg.to_dict()
    else:
        # transformers may be absent (wall #2) but a local config.json is still readable.
        try:
            raw = json.loads((Path(a.model) / "config.json").read_text())
        except Exception:
            raw = None
    # A config-derived check must NEVER report PASS on a config it never read:
    # "no hybrid signals found" and "config unavailable" are different answers.
    if raw is None:
        add(checks, "partial_kv_cache", "UNKN",
            "config was not inspected (see config_load) — hybrid/partial-KV status is UNKNOWN, not clear",
            wall=6, remedy="resolve config_load, or stage the checkpoint locally so config.json is readable")
        return
    hits = scan_obj(raw, KV_PAT)
    archs = " ".join(model_info.get("architectures") or [])
    if KV_PAT.search(archs):
        hits.append(f"architectures={archs}")
    model_info["partial_kv_suspected"] = bool(hits)
    if not hits:
        add(checks, "partial_kv_cache", "PASS", "no sliding-window/hybrid/linear-attention signals found")
        return
    detail = ("partial/hybrid KV signals: " + "; ".join(hits[:12]) +
              ". Only a SUBSET of layers may keep a KV cache: DynamicCache subclasses crash; "
              "vLLM refuses draft-model speculative decoding ('All drafting layers should belong "
              "to the same kv cache group').")
    remedy = ("KV quantization: use forward hooks on k_proj/v_proj Linear modules, not cache subclasses. "
              "Speculative decoding: plan n-gram, or run a serving canary that captures the exact vLLM "
              "error before claiming any speedup. KV memory math must count only cached layers.")
    if a.plan_spec_decode:
        add(checks, "partial_kv_cache", "FAIL",
            detail + " Draft-model spec-decode is in the plan → replan.",
            fatal=True, wall=6, remedy=remedy)
    else:
        add(checks, "partial_kv_cache", "WARN", detail, wall=6, remedy=remedy)


# ---------------------------------------------------------------- wall #5
def check_native_quant(a, checks, model_info):
    q = model_info.get("native_quantization")
    if q is None and Path(a.model).is_dir():
        try:
            raw = json.loads((Path(a.model) / "config.json").read_text())
            q = raw.get("quantization_config")
            model_info["native_quantization"] = q
        except Exception:
            pass
    # Absence of evidence is not evidence of absence: only claim "not quantized"
    # if we actually managed to read a config (via transformers or local config.json).
    inspected = bool(model_info.get("config_inspected")) or (Path(a.model) / "config.json").is_file()
    if not inspected:
        add(checks, "native_quantized_checkpoint", "UNKN",
            "config was not inspected (see config_load) — cannot rule out a native low-precision "
            "checkpoint; base_ppl_guard requirement is UNDETERMINED",
            wall=5, remedy="resolve config_load first, then re-run preflight")
        model_info["require_base_ppl_guard"] = None
        return
    native_low = bool(q) and any(t in json.dumps(q, default=str).lower()
                                 for t in ("mxfp4", "fp4", "int4", "awq", "gptq", "compressed-tensors"))
    model_info["require_base_ppl_guard"] = native_low
    if native_low and "dequant_requant" in a.planned_tools and not a.allow_native_quant_requant:
        add(checks, "native_quantized_checkpoint", "FAIL",
            f"quantization_config={q}; dequantize→requantize of a native low-precision checkpoint is a "
            f"SUSPECT path (measured: ppl ~183 where ~4 expected, via BOTH plain dtype cast and "
            f"Mxfp4Config(dequantize=True))", fatal=True, wall=5,
            remedy="start from the bf16/fp16 sibling, or pass --allow-native-quant-requant AND set a "
                   "base_ppl_guard band from the published bf16 sibling")
    elif native_low:
        add(checks, "native_quantized_checkpoint", "WARN",
            f"native quantized checkpoint: {q}. require_base_ppl_guard armed.", wall=5,
            remedy="first training stage MUST be a base_ppl_guard on the exact eval protocol; the "
                   "expected-ppl band comes from the published bf16/fp16 sibling, never from this checkpoint")
    else:
        add(checks, "native_quantized_checkpoint", "PASS", "no low-precision quantization_config detected")


# ---------------------------------------------------------------- ledger §6/§7
def local_weight_bytes(model):
    p = Path(model)
    if not p.is_dir():
        return None
    idxs = list(p.glob("*.safetensors.index.json"))
    if idxs:
        try:
            return sum(json.loads(x.read_text()).get("metadata", {}).get("total_size", 0) for x in idxs)
        except Exception:
            return None
    st = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
    return sum(x.stat().st_size for x in st) if st else None


def check_feasibility(a, checks, hw, plan):
    wb = local_weight_bytes(a.model)
    if wb is None and a.total_params:
        wb = int(a.total_params * a.source_bytes_per_param)
    if wb is None:
        add(checks, "memory_feasibility", "WARN",
            "cannot size weights (model not local and --total-params not given)",
            remedy="stage the checkpoint locally, or pass --total-params; for hub models verify file "
                   "sizes before queueing")
        return
    plan["weight_bytes"] = wb
    gib = lambda b: b / 2**30
    tib = lambda b: b / 2**40
    if hw.get("gpus"):
        per_cell = int(wb * MEM_FACTOR[a.memory_profile] * a.memory_multiplier)
        visible = sum(g["mem_total_bytes"] for g in hw["gpus"][: max(a.gpus_per_cell, 1)])
        budget = int(visible * a.mem_util)
        plan.update(estimated_gpu_mem_bytes_per_cell=per_cell, gpu_budget_bytes_per_cell=budget)
        if per_cell > budget:
            add(checks, "memory_feasibility", "FAIL",
                f"~{gib(per_cell):.0f} GiB ({a.memory_profile} profile) > budget {gib(budget):.0f} GiB "
                f"({a.gpus_per_cell} GPU(s) x mem_util={a.mem_util})", fatal=True,
                remedy="more GPUs per cell, tensor/expert parallelism, sharded optimizer, or split the "
                       "cell — decide BEFORE submitting")
        elif per_cell > 0.85 * budget:
            add(checks, "memory_feasibility", "WARN",
                f"~{gib(per_cell):.0f} GiB within 15% of budget {gib(budget):.0f} GiB",
                remedy="run exactly one canary cell before launching the grid")
        else:
            add(checks, "memory_feasibility", "PASS",
                f"~{gib(per_cell):.0f} GiB est. <= {gib(budget):.0f} GiB budget")
    ckpt = a.checkpoint_bytes_per_cell or int(wb * a.checkpoint_multiplier)
    required = int(a.sweep_cells * ckpt * 1.10)          # logs/index/temp overhead
    free = shutil.disk_usage(a.artifact_root).free
    plan.update(checkpoint_bytes_per_cell=ckpt, estimated_disk_bytes=required, disk_free_bytes=free)
    if required > 0.85 * free:
        add(checks, "disk_feasibility", "FAIL",
            f"sweep needs ~{tib(required):.2f} TiB, only {tib(free):.2f} TiB free at {a.artifact_root}",
            fatal=True,
            remedy="reduce cells/checkpoint multiplier, and enable results-index + has_result GC "
                   "(KEEP_PREFIX allowlist, dry-run first) BEFORE the sweep, not after")
    elif required > 0.65 * free:
        add(checks, "disk_feasibility", "WARN",
            f"disk tight: ~{tib(required):.2f} TiB of {tib(free):.2f} TiB free",
            remedy="install the KEEP_PREFIX allowlist and dry-run GC now; 28 TB accumulated silently once")
    else:
        add(checks, "disk_feasibility", "PASS",
            f"~{tib(required):.2f} TiB planned vs {tib(free):.2f} TiB free")


# ---------------------------------------------------------------- wall #7
def check_flashinfer(a, checks):
    if importlib.util.find_spec("flashinfer") is None:
        return
    try:
        import flashinfer
        d = Path(flashinfer.__file__).parent
        with tempfile.TemporaryFile(dir=d):
            pass
        add(checks, "flashinfer_writable", "PASS", f"{d} writable")
    except OSError:
        add(checks, "flashinfer_writable", "WARN",
            "flashinfer install dir is READ-ONLY → cubin JIT crashes with [Errno 30] in containers",
            wall=7, remedy="submit with --container-writable (pyxis/enroot) or pre-populate the cubin cache")
    except Exception as e:
        add(checks, "flashinfer_writable", "WARN", f"could not probe flashinfer writability: {e!r}",
            wall=7, remedy="verify the cubin cache path for your flashinfer version is writable "
                           "inside the container")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF id or local checkpoint path")
    ap.add_argument("--revision")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--planned-formats", default="",
                    help="comma list, e.g. rtnste_int4,nvfp4_fake,nvfp4_real,kv_fp8,int4_modelopt")
    ap.add_argument("--planned-tools", default="",
                    help="comma list, e.g. gptq,llmcompressor,dequant_requant")
    ap.add_argument("--plan-spec-decode", action="store_true",
                    help="draft-model speculative decoding is in the serving plan")
    ap.add_argument("--sweep-cells", type=int, default=1)
    ap.add_argument("--artifact-root", required=True)
    ap.add_argument("--repo", help="repo root whose src/ must be on PYTHONPATH")
    ap.add_argument("--required-imports", default="")
    ap.add_argument("--total-params", type=int)
    ap.add_argument("--source-bytes-per-param", type=float, default=2.0)
    ap.add_argument("--gpus-per-cell", type=int, default=1)
    ap.add_argument("--mem-util", type=float, default=0.90)
    ap.add_argument("--memory-profile", choices=sorted(MEM_FACTOR), default="qat")
    ap.add_argument("--memory-multiplier", type=float, default=1.0)
    ap.add_argument("--checkpoint-bytes-per-cell", type=int)
    ap.add_argument("--checkpoint-multiplier", type=float, default=2.0)
    ap.add_argument("--allow-unverified-arch", action="store_true")
    ap.add_argument("--allow-native-quant-requant", action="store_true")
    ap.add_argument("--strict", action="store_true", help="treat WARN as FAIL")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.planned_formats = [x.strip() for x in a.planned_formats.split(",") if x.strip()]
    a.planned_tools = [x.strip() for x in a.planned_tools.split(",") if x.strip()]
    a.required_imports = [x.strip() for x in a.required_imports.split(",") if x.strip()]

    checks = []
    hw = {"node": socket.gethostname()}
    env = {"python": platform.python_version()}
    model_info = {"id_or_path": a.model}
    plan = {"formats": a.planned_formats, "tools": a.planned_tools, "sweep_cells": a.sweep_cells}
    probe_error = None
    try:
        check_env_hygiene(a, checks, env)
        torch = check_torch_gpu(a, checks, hw)
        check_versions(a, checks, hw)
        check_format_vocabulary(a, checks)
        if torch is not None:
            check_format_gates(a, checks, hw)
        check_toolchain_and_arch(a, checks, model_info, env)
        check_partial_kv(a, checks, model_info)
        check_native_quant(a, checks, model_info)
        check_feasibility(a, checks, hw, plan)
        check_flashinfer(a, checks)
    except Exception:
        probe_error = traceback.format_exc()
        print(probe_error, file=sys.stderr)
        add(checks, "preflight_internal_error", "FAIL",
            "probe crashed; inspect the FULL unfiltered log — never a grepped tail", fatal=True)

    fails = sum(c["status"] == "FAIL" for c in checks)
    # UNKN = the check could not be evaluated. It is deliberately NOT a pass:
    # an unevaluated check is the failure mode this whole probe exists to prevent.
    unkns = sum(c["status"] == "UNKN" for c in checks)
    warns = sum(c["status"] == "WARN" for c in checks) + unkns
    if probe_error is not None:
        overall, code = "PROBE_ERROR", 2
    elif fails or (a.strict and warns):
        overall, code = "FAIL", 1
    elif warns:
        overall, code = "PASS_WITH_WARNINGS", 0
    else:
        overall, code = "PASS", 0

    report = {"schema": SCHEMA, "overall": overall, "strict": a.strict,
              "checks": checks, "hardware": hw, "env": env,
              "model": {k: v for k, v in model_info.items() if k != "config"},
              "plan": plan}
    try:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2, default=str))
    except Exception as e:
        print(f"FATAL: cannot write report to {a.out}: {e!r}", file=sys.stderr)
        return 2

    print(f"\nOVERALL: {overall} ({fails} FAIL, {warns} WARN) -> {a.out}")
    if overall not in ("PASS",):
        print("Non-passing checks:")
        for c in checks:
            if c["status"] != "PASS":
                print(f"  - {c['name']} [{c['status']}]: {c['remedy'] or c['detail'][:200]}")
    return code


if __name__ == "__main__":
    sys.exit(main())
