"""RTN (quantize-dequantize) quality worker; single GPU, standalone.
Measures only base-vs-compressed perplexity after an integer round-trip on the
weights and records raw numbers; it never decides a verdict. Runtime and memory
are not measured here and no speed claim may be made from this script's output
(W3). Harness B (the model's own labels= loss) cross-checks harness A's shift, reduction and
log-softmax; both share one forward pass, so it certifies the loss computation, not the forward."""
import argparse, hashlib, json, math, os, socket, sys, time, traceback

import torch
import transformers

SCHEMA = "fs.compress.rtn_results.v1"


def write_results(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="RTN quality worker (measure-only)")
    p.add_argument("--model", required=True, help="local HF dir")
    p.add_argument("--text-file", required=True, help=".txt or .jsonl with 'text' rows")
    p.add_argument("--bits", type=int, required=True, choices=(4, 8))
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-eval-tokens", type=int, default=65536)
    p.add_argument("--exempt", default="embed,lm_head,vision,audio")
    p.add_argument("--results", required=True)
    return p.parse_args()


def rtn_quantize(model, bits, g, exempt):
    st, qmax = {"linears": 0, "params": 0, "groups": 0, "changed": 0}, 2 ** bits - 1
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or any(s in name for s in exempt):
            continue
        old, (out_dim, in_dim) = mod.weight.data.clone(), mod.weight.shape
        W = old.float()
        if in_dim % g:
            W = torch.nn.functional.pad(W, (0, g - in_dim % g))
        Wg = W.view(out_dim, -1, g)
        lo, hi = Wg.amin(-1, keepdim=True), Wg.amax(-1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zp = torch.round(-lo / scale).clamp(0, qmax)
        q = (torch.round(Wg / scale) + zp).clamp(0, qmax)
        deq = ((q - zp) * scale).view(out_dim, -1)[:, :in_dim].to(old.dtype)
        st["changed"] += int((deq != old).sum())
        mod.weight.data.copy_(deq)
        st["linears"] += 1
        st["params"] += old.numel()
        st["groups"] += Wg.shape[0] * Wg.shape[1]
    return st


def main():
    t0, a = time.time(), parse_args()
    exempt = [s.strip() for s in a.exempt.split(",") if s.strip()]

    def unmet(reason):
        write_results(a.results, {"schema": SCHEMA, "status": "UNMEASURED", "reason": reason,
                                  "model": a.model, "bits": a.bits, "group_size": a.group_size,
                                  "ctx": a.ctx, "text_file": a.text_file, "exempt": exempt})
        sys.exit(95)

    if not torch.cuda.is_available():
        unmet("no CUDA device: CPU fallback is never a measurement (W1)")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        except (ValueError, KeyError):
            model = transformers.AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
        model.to("cuda")
        model.eval()
        devs = sorted({str(p.device) for p in model.parameters()})
        if any(not d.startswith("cuda") for d in devs):
            unmet("parameter device(s) %s: CPU fallback is never a measurement (W1)" % ",".join(devs))
        raw = open(a.text_file, "rb").read()
        text_sha = hashlib.sha256(raw).hexdigest()
        if a.text_file.endswith(".jsonl"):
            text = "\n\n".join(json.loads(x)["text"] for x in raw.decode().splitlines() if x.strip())
        else:
            text = raw.decode()
        ids = tok(text, add_special_tokens=False).input_ids[: a.max_eval_tokens]
        step, bos, wins = a.ctx - 1, tok.bos_token_id, []
        for i in range(0, len(ids), step):
            chunk = ids[i: i + step]
            w = [bos] + chunk if bos is not None else list(chunk)
            if len(w) >= 64:
                wins.append(torch.tensor([w], dtype=torch.long, device="cuda"))
        if not wins:
            unmet("no eval window with at least 64 tokens: nothing to measure")

        def harness_a(m):
            s, n = 0.0, 0
            for w in wins:
                lg = m(input_ids=w).logits.float()
                s += float(torch.nn.functional.cross_entropy(lg[0, :-1], w[0, 1:], reduction="sum"))  # (T, V) vs (T,)
                n += w.shape[1] - 1
            return math.exp(s / n)

        def harness_b(m):
            s, n = 0.0, 0
            for w in wins:
                out = m(input_ids=w, labels=w)
                s += float(out.loss.float()) * (w.shape[1] - 1)
                n += w.shape[1] - 1
            return math.exp(s / n)

        with torch.no_grad():
            base_a, base_b = harness_a(model), harness_b(model)
        rel = abs(base_a - base_b) / base_b if base_b else 0.0
        stats = rtn_quantize(model, a.bits, a.group_size, exempt)
        with torch.no_grad():
            comp = harness_a(model)
        eval_tokens = sum(w.shape[1] - 1 for w in wins)
        qk, groups = stats["params"], stats["groups"]
        total = sum(p.numel() for p in model.parameters())
        eff_q = a.bits + groups * (16 + a.bits) / qk if qk else float(a.bits)
        eff_t = (qk * a.bits + groups * (16 + a.bits) + (total - qk) * 16) / total
        cfg = os.path.join(a.model, "config.json")
        cfg_sha = hashlib.sha256(open(cfg, "rb").read()).hexdigest() if os.path.isfile(cfg) else None
        props = torch.cuda.get_device_properties(0)
        try:
            uuid = str(props.uuid)
        except Exception:
            uuid = None
        write_results(a.results, {
            "schema": SCHEMA, "status": "MEASURED", "model": a.model,
            "model_config_sha256": cfg_sha, "text_file": a.text_file, "text_sha256": text_sha,
            "eval_tokens": eval_tokens, "windows": len(wins), "ctx": a.ctx,
            "bits": a.bits, "group_size": a.group_size, "exempt": exempt,
            "quantized_linears": stats["linears"], "quantized_params": qk,
            "total_params": total, "changed_fraction": (stats["changed"] / qk if qk else 0.0),
            "effective_bits_quantized": eff_q, "effective_bits_total": eff_t,
            "base_ppl_a": base_a, "base_ppl_b": base_b,
            "instrument_rel_diff": rel, "instrument_certified": bool(rel <= 0.02),
            "compressed_ppl": comp, "ppl_ratio": (comp / base_a if base_a else None),
            "gpu_name": props.name, "gpu_uuid": uuid, "hostname": socket.gethostname(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "seconds": time.time() - t0})
    except Exception:
        unmet(traceback.format_exc()[-800:])


if __name__ == "__main__":
    sys.exit(main())

