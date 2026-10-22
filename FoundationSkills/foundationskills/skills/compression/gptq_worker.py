"""GPTQ one-shot (W4A16) quality and size worker; single GPU, standalone. Quality is measured
on the in-memory quantized model and size is measured from the bytes on disk of the saved
artifact; only raw numbers are recorded and this script never decides a verdict. Speed is
never claimed here (it needs the served artifact and is measured by the deployment skill);
harness B (the model's own labels= loss) cross-checks harness A on one shared forward pass."""
import argparse, hashlib, importlib.metadata as _im, json, math, os, socket, sys, time, traceback
import datasets, torch, transformers

SCHEMA = "fs.compress.gptq_results.v1"

def write_results(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh: json.dump(payload, fh, indent=2)

def parse_args():
    p = argparse.ArgumentParser(description="GPTQ W4A16 worker (measure-only)")
    p.add_argument("--model", required=True, help="local HF dir")
    p.add_argument("--text-file", required=True, help=".txt/.jsonl")
    p.add_argument("--calib-file", required=True, help=".txt calibration corpus")
    p.add_argument("--calib-samples", type=int, default=128); p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--group-size", type=int, default=128); p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-eval-tokens", type=int, default=65536)
    p.add_argument("--ignore",
                   default="lm_head,re:.*embed.*,re:.*per_layer_model_projection.*,re:.*vision.*,re:.*audio.*")
    # basic = full-model calibration forwards: models that share KV across layers (Gemma-4 E-models, wall W6) break
    # the layer-isolated sequential pipeline; sequential keeps GPTQ's layer-to-layer error propagation
    p.add_argument("--pipeline", default="sequential", choices=("sequential", "basic"))
    p.add_argument("--save-dir", required=True, help="must not exist"); p.add_argument("--results", required=True)
    return p.parse_args()

def ngrams(rows, n):
    return {tuple(r[i: i + n]) for r in rows for i in range(max(0, len(r) - n + 1))}

def sbytes(root):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(root)
               for f in fs if f.endswith(".safetensors"))

def pver(mod, name):
    try: return getattr(mod, "__version__", None) or _im.version(name)
    except Exception: return None

def main():
    t0, a = time.time(), parse_args()
    ignore = [s.strip() for s in a.ignore.split(",") if s.strip()]

    def unmet(reason):
        write_results(a.results, {"schema": SCHEMA, "status": "UNMEASURED", "reason": reason,
                                  "model": a.model, "method": "gptq_w4a16", "bits": 4, "group_size": a.group_size,
                                  "ctx": a.ctx, "text_file": a.text_file, "calib_file": a.calib_file,
                                  "save_dir": a.save_dir, "ignore": ignore})
        sys.exit(95)
    if not torch.cuda.is_available():
        unmet("no CUDA device: CPU fallback is never a measurement (W1)")
    if a.group_size != 128:
        unmet("only group 128 is wired in v2.0")
    if os.path.exists(a.save_dir):
        unmet("save-dir %s already exists: refusing to overwrite" % a.save_dir)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(a.model)
        try: proc = transformers.AutoProcessor.from_pretrained(a.model)
        except Exception: proc = None  # saved with the artifact only if it loads
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
        except (ValueError, KeyError):
            model = transformers.AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16)
        model.to("cuda"); model.eval()
        devs = sorted({str(p.device) for p in model.parameters()})
        if any(not d.startswith("cuda") for d in devs):
            unmet("parameter device(s) %s: CPU fallback is never a measurement (W1)" % ",".join(devs))
        raw = open(a.text_file, "rb").read(); text_sha = hashlib.sha256(raw).hexdigest()
        if a.text_file.endswith(".jsonl"):
            text = "\n\n".join(json.loads(x)["text"] for x in raw.decode().splitlines() if x.strip())
        else:
            text = raw.decode()
        ids = tok(text, add_special_tokens=False).input_ids[: a.max_eval_tokens]
        step, bos, wins, rows = a.ctx - 1, tok.bos_token_id, [], []
        for i in range(0, len(ids), step):
            w = [bos] + ids[i: i + step] if bos is not None else list(ids[i: i + step])
            if len(w) >= 64:
                wins.append(torch.tensor([w], dtype=torch.long, device="cuda"))
                rows.append(w[1:] if bos is not None else w)  # eval tokens with BOS excluded
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
                s += float(out.loss.float()) * (w.shape[1] - 1); n += w.shape[1] - 1
            return math.exp(s / n)

        with torch.no_grad():
            base_a, base_b = harness_a(model), harness_b(model)
        rel = abs(base_a - base_b) / base_b if base_b else 0.0
        craw = open(a.calib_file, "rb").read()
        calib_sha = hashlib.sha256(craw).hexdigest()
        cids = tok(craw.decode(), add_special_tokens=False).input_ids
        eg = ngrams(rows, 13)  # eval 13-grams (BOS excluded)
        # decontaminate: drop every calibration chunk sharing any 13-gram with the eval tokens, take the next ones
        chunks, dropped = [], 0
        for i in range(0, len(cids) - a.calib_seqlen + 1, a.calib_seqlen):
            c = cids[i: i + a.calib_seqlen]
            if ngrams([c], 13) & eg:
                dropped += 1
                continue
            chunks.append(c)
            if len(chunks) == a.calib_samples:
                break
        if len(chunks) < a.calib_samples:
            unmet("only %d clean calibration chunks of %d tokens (wanted %d)" % (len(chunks), a.calib_seqlen,
                                                                                 a.calib_samples))
        cg = ngrams(chunks, 13)  # re-measured on the kept chunks: must be zero
        overlap = (len(eg & cg) / len(eg)) if eg else 0.0  # fraction of distinct eval 13-grams in calibration
        if overlap > 0.0:
            unmet("calibration and eval overlap: %d/%d eval 13-grams in %s" % (len(eg & cg), len(eg), a.calib_file))
        ds = datasets.Dataset.from_dict({"input_ids": chunks,
                                        "attention_mask": [[1] * len(c) for c in chunks]})
        from llmcompressor import oneshot
        try:
            from llmcompressor.modifiers.quantization import GPTQModifier
        except Exception:
            from llmcompressor.modifiers.gptq import GPTQModifier
        import llmcompressor as _lc, compressed_tensors as _ct
        recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=ignore)  # W4A16 preset: groups of 128
        tq = time.time()
        oneshot(model=model, dataset=ds, recipe=recipe, max_seq_length=a.calib_seqlen,
                num_calibration_samples=len(chunks), pipeline=a.pipeline)
        quant_seconds = time.time() - tq
        model.to("cuda")  # the sequential pipeline offloads layers to CPU while calibrating
        devs = sorted({str(p.device) for p in model.parameters()})
        if any(not d.startswith("cuda") for d in devs):
            unmet("after oneshot, parameter device(s) %s: the compressed ppl would not be a GPU measurement"
                  % ",".join(devs))
        # the quantized weights held in memory are the ones llmcompressor will serialize below
        with torch.no_grad():
            comp = harness_a(model)
        model.save_pretrained(a.save_dir, save_compressed=True)  # size is measured from these bytes
        tok.save_pretrained(a.save_dir)
        if proc is not None: proc.save_pretrained(a.save_dir)
        art, src = sbytes(a.save_dir), sbytes(a.model)
        pj = os.path.join(a.save_dir, "config.json")
        qcfg = json.load(open(pj)).get("quantization_config") if os.path.isfile(pj) else None
        cfg = os.path.join(a.model, "config.json")
        cfg_sha = hashlib.sha256(open(cfg, "rb").read()).hexdigest() if os.path.isfile(cfg) else None
        props = torch.cuda.get_device_properties(0)
        uuid = getattr(props, "uuid", None)
        write_results(a.results, {
            "schema": SCHEMA, "status": "MEASURED", "method": "gptq_w4a16", "bits": 4, "group_size": 128,
            "model": a.model, "model_config_sha256": cfg_sha, "text_file": a.text_file, "text_sha256": text_sha,
            "eval_tokens": sum(w.shape[1] - 1 for w in wins), "windows": len(wins), "ctx": a.ctx, "ignore": ignore,
            "base_ppl_a": base_a, "base_ppl_b": base_b, "instrument_rel_diff": rel,
            "instrument_certified": bool(rel <= 0.02), "compressed_ppl": comp,
            "ppl_ratio": (comp / base_a if base_a else None), "calib_file": a.calib_file, "calib_sha256": calib_sha,
            "calib_samples": len(chunks), "calib_seqlen": a.calib_seqlen, "calib_eval_overlap": overlap,
            "calib_chunks_dropped_for_overlap": dropped, "pipeline": a.pipeline,
            "quant_seconds": quant_seconds, "llmcompressor_version": pver(_lc, "llmcompressor"),
            "compressed_tensors_version": pver(_ct, "compressed_tensors"), "save_dir": a.save_dir,
            "disk_bytes_source": src, "disk_bytes_artifact": art, "disk_ratio": (src / art if art else None),
            "quantization_format": (qcfg.get("format") if isinstance(qcfg, dict) else None), "gpu_name": props.name,
            "gpu_uuid": (str(uuid) if uuid is not None else None), "hostname": socket.gethostname(),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "seconds": time.time() - t0})
    except Exception:
        unmet(traceback.format_exc()[-800:])


if __name__ == "__main__":
    sys.exit(main())