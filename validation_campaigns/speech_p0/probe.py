#!/usr/bin/env python3
"""speech_p0_probe.py — Gemma-4-E4B audio-tower trainability de-risk probe.

Honesty rule: every check records MEASURED values. Where API is uncertain, try the
documented route and record which route worked and any exception text in the JSON.

Usage:
  python speech_p0_probe.py --model HF_DIR --librispeech-root LS_ROOT --out probe.json
Exit code 0 = overall PASS, 1 = anything else.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import platform
import random
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import transformers
from transformers import AutoConfig, AutoProcessor

MODEL_KEYS = {"input_ids", "attention_mask", "input_features", "input_features_mask", "mm_token_type_ids"}
GROUPS = ("audio_tower", "embed_audio", "vision_tower", "embed_vision", "language_model", "other")


# ---------------------------------------------------------------- helpers
def to_jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def safe(results, name, fn, *a, **kw):
    """Run a step; record exception text as evidence instead of crashing."""
    try:
        out = fn(*a, **kw)
        results.setdefault("step_status", {})[name] = "ok"
        return out
    except Exception as ex:
        results.setdefault("step_status", {})[name] = f"EXCEPTION: {ex!r}"
        results.setdefault("errors", {})[name] = traceback.format_exc()
        return None


def mkcheck(verdict, **detail):
    return {"verdict": verdict, **detail}


def _to_id_list(x):
    """Flatten a apply_chat_template(tokenize=True) return value to list[int]."""
    if isinstance(x, torch.Tensor):
        return x.view(-1).tolist()
    if hasattr(x, "get") and "input_ids" in x:
        x = x["input_ids"]
        return x.view(-1).tolist() if isinstance(x, torch.Tensor) else _to_id_list(x)
    if isinstance(x, (list, tuple)):
        if x and isinstance(x[0], (list, tuple)):
            return [int(t) for t in x[0]]
        return [int(t) for t in x]
    return []


def _norm_words(s):
    return [w for w in re.sub(r"[^a-z0-9' ]+", " ", (s or "").lower()).split() if w]


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r_, h_ = _norm_words(ref), _norm_words(hyp)
    if not r_:
        return 0.0 if not h_ else 1.0
    return _lev(r_, h_) / len(r_)


# ---------------------------------------------------------------- step 1: environment
def step_env(r):
    e = {"python": sys.version.split()[0], "platform": platform.platform(),
         "torch": torch.__version__, "transformers": transformers.__version__,
         "numpy": np.__version__, "soundfile": getattr(sf, "__version__", "unknown"),
         "cuda_available": bool(torch.cuda.is_available()),
         "cuda_compiled_version": getattr(torch.version, "cuda", None),
         "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}
    if torch.cuda.is_available():
        e["cuda_device_count"] = torch.cuda.device_count()
        e["cuda_get_device_name_0"] = torch.cuda.get_device_name(0)
        e["bf16_supported_by_gpu"] = bool(torch.cuda.is_bf16_supported())
    else:
        e["cuda_device_count"] = 0
        e["bf16_supported_by_gpu"] = False
    r["1_env"] = e


# ---------------------------------------------------------------- step 2: data
def step_data(r, root, n, processor):
    sr_expected = int(processor.feature_extractor.sampling_rate)
    dev = Path(root) / "dev-clean"
    transcripts = {}
    for tp in sorted(dev.rglob("*.trans.txt")):
        for line in tp.read_text(encoding="utf-8").splitlines():
            uid, _, txt = line.strip().partition(" ")
            if uid:
                transcripts[uid] = txt
    flacs = sorted(str(p) for p in dev.rglob("*.flac"))
    samples, skipped = [], {"duration_out_of_range": 0, "empty": 0}
    for path in flacs:
        if len(samples) >= n:
            break
        wave2d, sr = sf.read(path, dtype="float32", always_2d=True)
        mono = wave2d.mean(axis=1).astype(np.float32) if wave2d.shape[1] > 1 else wave2d[:, 0].astype(np.float32)
        if mono.shape[0] == 0:
            skipped["empty"] += 1
            continue
        dur = mono.shape[0] / float(sr)
        if not (2.0 <= dur <= 15.0):
            skipped["duration_out_of_range"] += 1
            continue
        uid = Path(path).stem
        samples.append({"id": uid, "wave": mono, "duration_s": dur,
                        "sample_rate": int(sr), "ref": transcripts.get(uid),
                        "ref_found": uid in transcripts,
                        "sr_match": int(sr) == sr_expected})
    sr_ok = bool(samples) and all(s["sr_match"] for s in samples)
    ref_ok = bool(samples) and all(s["ref_found"] for s in samples)
    enough = len(samples) == n
    reasons = []
    if not sr_ok:
        reasons.append("SR != processor SR (resampling NOT allowed)")
    if not ref_ok:
        reasons.append("missing ref transcript")
    if not enough:
        reasons.append(f"only {len(samples)}/{n} with 2s<=dur<=15s")
    r["2_data"] = {
        "n_selected": len(samples), "n_requested": n,
        "processor_sampling_rate": sr_expected,
        "measured_sample_rates": sorted({s["sample_rate"] for s in samples}) if samples else [],
        "resample_allowed_by_contract": False,
        "samples": [{k: v for k, v in s.items() if k != "wave"} for s in samples],
        "check": mkcheck("PASS" if (sr_ok and ref_ok and enough) else "FAIL",
                         sampled_rate_match=sr_ok, refs_found=ref_ok,
                         count_ok=enough, reasons=reasons)}
    return samples


# ---------------------------------------------------------------- step 3: class probe (meta device)
def step_class_probe(r, config):
    out = {"method": "with torch.device('meta'): AutoX.from_config(config)"}
    for label in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        ent = {}
        try:
            cls = getattr(importlib.import_module("transformers"), label)
            with torch.device("meta"):
                m = cls.from_config(config)
            ent["resolved_class"] = type(m).__module__ + "." + type(m).__name__
            ent["has_audio_tower"] = any("audio_tower" in nm for nm, _ in m.named_modules())
            ent["audio_tower_param_tensors"] = sum(1 for nm, _ in m.named_parameters() if "audio_tower" in nm)
            del m
            gc.collect()
        except Exception as ex:
            ent["exception"] = repr(ex)
        out[label] = ent
    r["3_class_probe"] = out


# ---------------------------------------------------------------- step 4: load real model
def step_load(r, path, device):
    info = {"model_path": path, "device": device}
    try:
        from transformers import Gemma4ForConditionalGeneration
    except Exception:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
    config = AutoConfig.from_pretrained(path)
    try:
        model = Gemma4ForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        info["from_pretrained_dtype_route"] = "dtype=torch.bfloat16"
    except TypeError:
        model = Gemma4ForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16)
        info["from_pretrained_dtype_route"] = "torch_dtype=torch.bfloat16 (dtype= raised TypeError)"
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(path)
    info["model_class"] = type(model).__module__ + "." + type(model).__name__
    info["processor_class"] = type(processor).__module__ + "." + type(processor).__name__
    info["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})
    info["total_params_unique"] = int(sum(p.numel() for p in model.parameters()))
    info["audio_token_id"] = getattr(processor, "audio_token_id", None)
    info["audio_seq_length"] = getattr(processor, "audio_seq_length", None)
    info["audio_ms_per_token"] = getattr(processor, "audio_ms_per_token", None)
    info["padding_side"] = getattr(processor.tokenizer, "padding_side", "<no attr>")
    info["top_level_submodules"] = {
        cn: {"class": type(c).__name__, "params": int(sum(p.numel() for p in c.parameters()))}
        for cn, c in model.model.named_children()}
    at = [(nm, p) for nm, p in model.named_parameters() if "audio_tower" in nm]
    info["audio_tower_param_tensors"] = len(at)
    info["audio_tower_numel"] = int(sum(p.numel() for _, p in at))
    r["4_load"] = info
    return model, processor, config


def to_dev(batch, model):
    """Move every tensor in a processor batch to the model's device (floats to the model dtype)."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(model.dtype)
        out[k] = v
    return out


def param_groups(model):
    """Bucket named parameters into GROUPS using submodule-name matching."""
    g = {k: {} for k in GROUPS}
    for nm, p in model.named_parameters():
        key = next((c for c in GROUPS[:-1] if f".{c}." in f".{nm}." or nm.startswith(c + ".")), "other")
        g[key][nm] = p
    return g


# ---------------------------------------------------------------- step 5: prompt build
def step_prompt(r, processor, samples):
    res = {"messages_schema": (
        "user:[{type:audio,audio:<np.float32 mono>},{type:text,text:'Transcribe this audio exactly.'}]"
        " + assistant:[{type:text,text:<ref>}]")}

    def msgs(i, with_answer):
        content = [{"type": "audio", "audio": samples[i]["wave"]},
                   {"type": "text", "text": "Transcribe this audio exactly."}]
        m = [{"role": "user", "content": content}]
        if with_answer:
            m.append({"role": "assistant", "content": [{"type": "text", "text": samples[i]["ref"]}]})
        return m

    # Route A: apply_chat_template with tokenize+return_dict on the batch
    batch = None
    try:
        batch = processor.apply_chat_template(
            [msgs(i, True) for i in range(len(samples))],
            tokenize=True, return_dict=True, return_tensors="pt", padding=True)
        res["route_worked"] = "A: processor.apply_chat_template(messages, tokenize=True, return_dict=True)"
    except Exception as ex:
        res["route_a_exception"] = repr(ex)
        texts = [processor.apply_chat_template(msgs(i, True), tokenize=False) for i in range(len(samples))]
        batch = processor(text=texts, audio=[s["wave"] for s in samples], return_tensors="pt", padding=True)
        res["route_worked"] = "B: apply_chat_template(tokenize=False) -> processor(text=, audio=[], ...)"

    raw_keys = list(batch.keys())
    batch = {k: v for k, v in batch.items() if k in MODEL_KEYS}
    res["batch_keys_returned"] = raw_keys
    res["batch_keys_dropped_before_forward"] = [k for k in raw_keys if k not in MODEL_KEYS]
    res["batch_shapes"] = {k: list(v.shape) for k, v in batch.items()}
    res["batch_has_mm_token_type_ids"] = "mm_token_type_ids" in batch
    res["batch_has_input_features_mask"] = "input_features_mask" in batch

    # Build prompt-only token ids (user turn + add_generation_prompt=True) for label spans
    prompt_ids, prefix_ok = [], []
    side = getattr(processor.tokenizer, "padding_side", "right")
    for i in range(len(samples)):
        x = processor.apply_chat_template(msgs(i, False), tokenize=True, add_generation_prompt=True)
        ids = _to_id_list(x)
        prompt_ids.append(ids)
        row, attn = batch["input_ids"][i], batch["attention_mask"][i]
        off = 0 if side == "right" else int(row.shape[0] - int(attn.sum()))
        got = [int(t) for t in row[off:off + len(ids)].tolist()]
        prefix_ok.append(got == ids)
    res["prompt_only_token_len"] = [len(p) for p in prompt_ids]
    res["prompt_token_ids_are_prefix_of_batch_row"] = prefix_ok
    res["padding_side"] = side
    r["5_prompt"] = res
    return batch, prompt_ids, msgs


# ---------------------------------------------------------------- step 6: placeholder coverage + hook
def step_coverage(r, processor, model, samples, batch):
    out = {}
    audio_tok = getattr(processor, "audio_token_id", None)
    out["audio_token_id"] = audio_tok
    ids = batch["input_ids"]
    out["placeholder_count_per_sample"] = [int((ids[i] == audio_tok).sum()) for i in range(ids.shape[0])]
    if hasattr(processor, "_compute_audio_num_tokens"):
        out["prompt_placeholder_expected_per_sample"] = [
            int(processor._compute_audio_num_tokens(s["wave"], int(processor.feature_extractor.sampling_rate)))
            for s in samples]
    ifm = batch.get("input_features_mask")
    out["input_features_mask_sum_per_sample"] = (
        [int(x) for x in ifm.sum(-1).tolist()] if ifm is not None else "absent")

    # Find the audio_tower module (modeling_gemma4.py: Gemma4Model.audio_tower)
    tower, tower_name = None, None
    for nm, mod in model.named_modules():
        if nm.split(".")[-1] == "audio_tower":
            tower, tower_name = mod, nm
    out["audio_tower_module_name"] = tower_name
    out["audio_tower_module_class"] = type(tower).__module__ + "." + type(tower).__name__ if tower else None
    out["hook_registered"] = tower is not None

    cap: dict = {}
    if tower is not None:
        def hook(module, inp, output):
            mask = getattr(output, "attention_mask", None)
            hs = getattr(output, "last_hidden_state", None)
            if hs is None and isinstance(output, (tuple, list)):
                hs = output[0]
            cap["output_seq_len_per_sample"] = [int(hs.shape[1])] * int(hs.shape[0]) if hs is not None else None
            if torch.is_tensor(mask):
                cap["mask_returned"] = True
                cap["valid_len_per_sample"] = [int(v) for v in mask.reshape(mask.shape[0], -1).sum(-1).tolist()]
            elif hs is not None:
                cap["mask_returned"] = False
                cap["valid_len_per_sample"] = [int(hs.shape[1])] * int(hs.shape[0])
        cap["handle"] = tower.register_forward_hook(hook)

    r["6_coverage"] = out
    r["_coverage_capture"] = cap


def finalize_coverage(r):
    out = r.get("6_coverage", {})
    cap = r.pop("_coverage_capture", {})
    ph = out.get("placeholder_count_per_sample")
    vl = cap.get("valid_len_per_sample")
    out["hook_mask_returned"] = cap.get("mask_returned", "hook never fired")
    out["audio_tower_valid_len_per_sample_MEASURED"] = vl if vl else "UNMEASURED (hook did not fire)"
    crit = "placeholder_count == audio_tower output valid length (per sample)"
    if isinstance(ph, list) and isinstance(vl, list) and len(ph) == len(vl):
        out["check"] = mkcheck("PASS" if ph == vl else "FAIL",
                               criterion=crit,
                               placeholder_count_per_sample=ph, valid_len_per_sample=vl)
    else:
        out["check"] = mkcheck("UNMEASURED", criterion=crit,
                               reason="one of the two arrays was not measured",
                               placeholder_count_per_sample=ph, valid_len_per_sample=vl)
    h = cap.get("handle")
    if h is not None:
        h.remove()


# ---------------------------------------------------------------- step 7: labels
def step_labels(r, processor, batch, prompt_ids):
    ids, attn = batch["input_ids"], batch["attention_mask"]
    side = getattr(processor.tokenizer, "padding_side", "right")
    labels = torch.full_like(ids, -100)
    sup, spans, prefix_ok = [], [], []
    for i in range(ids.shape[0]):
        full_len = int(attn[i].sum())
        off = 0 if side == "right" else int(ids.shape[1] - full_len)
        p_len = len(prompt_ids[i])
        start, end = off + p_len, off + full_len
        got = ids[i][off:off + p_len].tolist()
        prefix_ok.append(got == prompt_ids[i])
        if start < end:
            labels[i, start:end] = ids[i, start:end]
        spans.append([start, end, full_len, p_len, off])
        sup.append(int((labels[i] != -100).sum()))
    ok = all(s > 0 for s in sup) and all(prefix_ok)
    r["7_labels"] = {
        "method": ("assistant span = [prompt_len, valid_len] inside the padded row; "
                   "pad tokens and prompt tokens set to -100; padding side accounted for"),
        "spans=[start,end,valid_len,prompt_len,pad_offset]": spans,
        "supervised_tokens_per_sample": sup,
        "prompt_token_ids_prefix_verified": prefix_ok,
        "check": mkcheck("PASS" if ok else "FAIL",
                         criterion="supervised token count > 0 for every sample and prompt prefix verified",
                         supervised_tokens_per_sample=sup, prompt_prefix_verified=prefix_ok)}
    return labels


# ---------------------------------------------------------------- step 8: zero-shot ASR sanity
def step_zeroshot(r, args, model, processor, samples, msgs_fn, route_a_ok):
    model.eval()
    gen_msgs = [msgs_fn(i, False) for i in range(len(samples))]
    try:
        if route_a_ok:
            gb = processor.apply_chat_template(
                gen_msgs, tokenize=True, return_dict=True,
                return_tensors="pt", padding=True, add_generation_prompt=True)
        else:
            texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in gen_msgs]
            gb = processor(text=texts, audio=[s["wave"] for s in samples], return_tensors="pt", padding=True)
    except Exception as ex:
        r["8_zeroshot"] = mkcheck("UNMEASURED", reason="prompt-only batch construction failed",
                                  exception=repr(ex), traceback=traceback.format_exc())
        return
    gb = {k: v for k, v in gb.items() if k in MODEL_KEYS}
    with torch.no_grad():
        out_ids = model.generate(**to_dev(gb, model), max_new_tokens=args.max_new_tokens, do_sample=False)
    new_tokens = out_ids[:, gb["input_ids"].shape[1]:]
    hyps = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    wers = [wer(samples[i]["ref"], hyps[i]) for i in range(len(samples))]
    r["8_zeroshot"] = {
        "mode": "greedy (do_sample=False)", "max_new_tokens": args.max_new_tokens,
        "hypotheses": [h.strip() for h in hyps], "references": [s["ref"] for s in samples],
        "wer_per_sample": wers, "wer_mean": float(sum(wers) / max(len(wers), 1)),
        "red_flag_all_wer_near_1": bool(all(w > 0.9 for w in wers)),
        "note": "WER near 1.0 everywhere is a red flag to record, not a hard assert"}


# ---------------------------------------------------------------- step 9: train step + grad census
def step_train(r, model, batch, labels, groups):
    for p in model.parameters():
        p.requires_grad_(True)
    for meth in ("gradient_checkpointing_disable", "disable_input_require_grads"):
        try:
            getattr(model, meth, lambda: None)()
        except Exception:
            pass
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model(**to_dev(batch, model), labels=labels.to(model.device), use_cache=False).loss
    finite = bool(torch.isfinite(loss.detach()).item())
    loss_val = float(loss.detach().float().cpu())
    loss.backward()
    census = {}
    for g in GROUPS:
        wg, nz, sq = 0, 0, 0.0
        for p in groups[g].values():
            if p.grad is None:
                continue
            wg += 1
            n = float(p.grad.detach().float().norm().item())
            sq += n * n
            if int(torch.count_nonzero(p.grad).item()) > 0:
                nz += 1
        census[g] = {"tensors_total": len(groups[g]), "tensors_with_grad": wg,
                     "tensors_with_nonzero_grad": nz, "grad_l2_norm": float(sq ** 0.5)}
    at_ok = census["audio_tower"]["tensors_with_nonzero_grad"] > 0
    ok = at_ok and finite
    r["9_train"] = {
        "loss": loss_val, "loss_is_finite": finite,
        "grad_census_per_group": census,
        "check": mkcheck("PASS" if ok else "FAIL",
                         criterion="loss finite AND >=1 audio_tower tensor has nonzero grad",
                         loss_finite=finite,
                         audio_tower_nonzero_grad_tensors=census["audio_tower"]["tensors_with_nonzero_grad"])}
    r["loss_finite"] = mkcheck("PASS" if finite else "FAIL", loss=loss_val)


# ---------------------------------------------------------------- step 10: SGD movement
def step_movement(r, model, groups, lr):
    snap: dict = {}
    method: dict = {}
    for g in ("audio_tower", "language_model"):
        try:
            snap[g] = {n: p.detach().float().cpu().clone() for n, p in groups[g].items()}
            method[g] = "cpu fp32 copy + torch.equal (exact)"
        except Exception as ex:
            snap[g] = {n: hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest()
                       for n, p in groups[g].items()}
            method[g] = f"sha256 fallback: {ex!r}"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
    opt.step()
    res: dict = {"optimizer": "torch.optim.SGD", "lr": lr, "snapshot_method": method}
    for g in ("audio_tower", "language_model"):
        changed = total = 0
        for n, p in groups[g].items():
            total += 1
            before = snap[g][n]
            after = (p.detach().float().cpu() if not isinstance(before, str)
                     else hashlib.sha256(p.detach().float().cpu().numpy().tobytes()).hexdigest())
            same = torch.equal(before, after) if not isinstance(before, str) else (before == after)
            changed += 0 if same else 1
        res[g] = {"tensors_total": total, "tensors_changed": changed, "tensors_unchanged": total - changed}
    at_ok = res["audio_tower"]["tensors_changed"] > 0
    res["check"] = mkcheck("PASS" if at_ok else "FAIL",
                           criterion="audio_tower params moved after SGD step",
                           audio_tower_changed=res["audio_tower"]["tensors_changed"],
                           audio_tower_total=res["audio_tower"]["tensors_total"],
                           language_model_changed=res["language_model"]["tensors_changed"])
    del opt, snap
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    r["10_movement"] = res


# ---------------------------------------------------------------- step 11: verdict + summary
def compute_verdict(r):
    checks = {}
    for k in ("2_data", "6_coverage", "7_labels", "9_train", "10_movement"):
        checks[k] = r.get(k, {}).get("check", {}).get("verdict", "UNMEASURED")
    loss_ok = r.get("loss_finite", {}).get("verdict") == "PASS"
    all_pass = all(v == "PASS" for v in checks.values()) and loss_ok
    return {"overall": "PASS" if all_pass else "FAIL",
            "required_checks": checks,
            "loss_finite": bool(loss_ok),
            "exit_code": 0 if all_pass else 1}


def summary(r):
    e = r.get("1_env", {})
    d = r.get("2_data", {})
    lo = r.get("4_load", {})
    cp = r.get("3_class_probe", {})
    pr = r.get("5_prompt", {})
    cov = r.get("6_coverage", {})
    lab = r.get("7_labels", {})
    zs = r.get("8_zeroshot", {})
    tr = r.get("9_train", {})
    mv = r.get("10_movement", {})
    vd = r.get("verdict", {})
    z = zs if isinstance(zs, dict) else {}
    return [
        f"1 env       python={e.get('python')} torch={e.get('torch')} "
        f"transformers={e.get('transformers')} cuda={e.get('cuda_compiled_version')} "
        f"dev0={e.get('cuda_get_device_name_0', 'N/A')} bf16={e.get('bf16_supported_by_gpu')}",
        f"2 data      n={d.get('n_selected')} sr={d.get('measured_sample_rates')} "
        f"(processor={d.get('processor_sampling_rate')}) -> {d.get('check', {}).get('verdict')}",
        f"3 class     CausalLM={cp.get('AutoModelForCausalLM', {}).get('resolved_class', '?')} "
        f"audio_tower_tensors={cp.get('AutoModelForCausalLM', {}).get('audio_tower_param_tensors', '?')}",
        f"4 load      {lo.get('model_class', '?')} params={lo.get('total_params_unique')} "
        f"audio_tower={lo.get('audio_tower_numel')}",
        f"5 prompt    route={str(pr.get('route_worked', ''))[:50]} "
        f"shapes={pr.get('batch_shapes')}",
        f"6 coverage  placeholders={cov.get('placeholder_count_per_sample')} "
        f"valid={cov.get('audio_tower_valid_len_per_sample_MEASURED')} "
        f"-> {cov.get('check', {}).get('verdict')}",
        f"7 labels    supervised={lab.get('supervised_tokens_per_sample')} "
        f"prefix_ok={lab.get('prompt_token_ids_prefix_verified')} "
        f"-> {lab.get('check', {}).get('verdict')}",
        f"8 zero-shot wer={z.get('wer_per_sample')} mean={z.get('wer_mean')} "
        f"red_flag={z.get('red_flag_all_wer_near_1')}",
        f"9 train     loss={tr.get('loss')} finite={tr.get('loss_is_finite')} "
        f"at_nonzero={tr.get('grad_census_per_group', {}).get('audio_tower', {}).get('tensors_with_nonzero_grad')} "
        f"-> {tr.get('check', {}).get('verdict')}",
        f"10 movement audio_tower {mv.get('audio_tower', {}).get('tensors_changed', '?')}/"
        f"{mv.get('audio_tower', {}).get('tensors_total', '?')} moved "
        f"(LM {mv.get('language_model', {}).get('tensors_changed', '?')}/"
        f"{mv.get('language_model', {}).get('tensors_total', '?')}) "
        f"OVERALL={vd.get('overall')} exit={vd.get('exit_code')}",
    ]


# ---------------------------------------------------------------- orchestration
def run(r, args):
    safe(r, "1_env", step_env, r)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    loaded = safe(r, "4_load", step_load, r, args.model, device)
    if loaded is None:
        r.setdefault("fatal_note", "model load failed: nothing downstream can be measured")
        return
    model, processor, config = loaded

    samples = safe(r, "2_data", step_data, r, args.librispeech_root, args.n, processor)
    safe(r, "3_class_probe", step_class_probe, r, config)

    if not samples:
        r.setdefault("fatal_note", "no usable utterances: prompt/train steps cannot run")
        return

    built = safe(r, "5_prompt", step_prompt, r, processor, samples)
    if built is None:
        return
    batch, prompt_ids, msgs_fn = built
    route_a_ok = str(r.get("5_prompt", {}).get("route_worked", "")).startswith("A:")

    safe(r, "6_coverage", step_coverage, r, processor, model, samples, batch)
    labels = safe(r, "7_labels", step_labels, r, processor, batch, prompt_ids)
    safe(r, "8_zeroshot", step_zeroshot, r, args, model, processor, samples, msgs_fn, route_a_ok)
    safe(r, "6_coverage_final", finalize_coverage, r)

    groups = param_groups(model)
    r["param_groups"] = {g: {"tensors": len(groups[g]),
                              "params": int(sum(p.numel() for p in groups[g].values()))}
                          for g in GROUPS}

    if labels is not None:
        safe(r, "9_train", step_train, r, model, batch, labels, groups)
    safe(r, "10_movement", step_movement, r, model, groups, args.lr)


def main():
    ap = argparse.ArgumentParser(description="Gemma-4 audio-tower trainability probe")
    ap.add_argument("--model", required=True)
    ap.add_argument("--librispeech-root", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    results = {"probe": "speech_p0_probe", "args": vars(args)}
    out_path = Path(args.out)
    try:
        run(results, args)
    except Exception:
        results["fatal"] = traceback.format_exc()
    finally:
        results.setdefault("verdict", compute_verdict(results))
        try:
            results["summary"] = summary(results)
        except Exception:
            results["summary"] = ["summary generation failed"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=to_jsonable), encoding="utf-8")

    print("\n".join(results.get("summary", ["no summary available"])))
    return results.get("verdict", {}).get("exit_code", 1)


if __name__ == "__main__":
    sys.exit(main())

