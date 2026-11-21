"""Measure a RUNNING vLLM OpenAI-compatible server: served perplexity (fidelity) and serving throughput.
This tool only measures and always writes a results JSON; it never decides a verdict.
A speed number is valid only for a real low-bit artifact served on labeled hardware, benchmarked against a
same-hardware bf16 baseline measured the same way on the same stack (estate wall W3)."""

import argparse, hashlib, json, math, os, socket, statistics, sys, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer


def post(url, body):
    req = urllib.request.Request(url, body, {"Content-Type": "application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))


def build_windows(ids, ctx, bos):
    step, out = ctx - 1, []
    for i in range(0, len(ids), step):
        chunk = ids[i:i + step]
        w = ([bos] + chunk) if bos is not None else list(chunk)
        if len(w) >= 64:
            out.append(w)
    return out


def lp(e):  # entry -> logprob: {"logprob": ...} or (rank, logprob, text)
    return float(e["logprob"]) if isinstance(e, dict) else float(e[1])


def measure_ppl(ep, name, windows, acc):
    for w in windows:
        body = json.dumps({"model": name, "prompt": w, "max_tokens": 1,
                           "prompt_logprobs": 0, "temperature": 0}).encode()
        pl = (post(ep + "/v1/completions", body)["choices"][0].get("prompt_logprobs") or [])
        for t in range(1, len(w)):
            d = pl[t] if t < len(pl) else None
            e = d.get(str(w[t])) if isinstance(d, dict) else None
            if e is None:
                acc["missing"] += 1
                continue
            acc["nll"] -= lp(e)
            acc["count"] += 1


def one_chat(ep, name, line, max_tokens):
    body = json.dumps({"model": name, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user",
                                     "content": "Continue this passage:\n\n" + line[:1500]}]}).encode()
    return int((post(ep + "/v1/chat/completions", body).get("usage") or {}).get("completion_tokens", 0))


def round_run(ep, name, prompts, a):
    cts, errs = 0, 0
    with ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        futs = [ex.submit(one_chat, ep, name, p, a.bench_max_tokens) for p in prompts]
        for f in futs:
            try:
                cts += f.result()
            except Exception:
                errs += 1
    return cts, errs


def empty_bench(prompts, a):
    return {"prompts": len(prompts), "max_tokens": a.bench_max_tokens, "concurrency": a.concurrency,
            "reps": a.reps, "rounds": [], "median_tok_per_s": 0.0, "spread_pct": 0.0,
            "mean_completion_tokens": 0.0}


def bench_run(ep, name, prompts, a):
    out, rounds = empty_bench(prompts, a), []
    if not prompts:
        return out
    round_run(ep, name, prompts, a)  # warm-up round: all prompts once, results discarded
    for _ in range(a.reps):
        t0 = time.time()
        cts, errs = round_run(ep, name, prompts, a)  # natural stopping, never ignore_eos
        secs = max(time.time() - t0, 1e-9)
        rounds.append({"seconds": round(secs, 3), "completion_tokens": cts,
                       "tok_per_s": round(cts / secs, 3), "errors": errs, "failed": errs > 0})
    vals = [r["tok_per_s"] for r in rounds if not r["failed"]] or [r["tok_per_s"] for r in rounds]
    med = statistics.median(vals)
    out.update({"rounds": rounds, "median_tok_per_s": round(med, 3),
                "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 3) if med > 0 else 0.0,
                "mean_completion_tokens": round(sum(r["completion_tokens"] for r in rounds)
                                                / (len(prompts) * len(rounds)), 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    for n, dv, ty in [("endpoint", None, str), ("served-name", None, str), ("tokenizer", None, str),
                      ("text-file", None, str), ("ctx", 2048, int), ("max-eval-tokens", 65536, int),
                      ("bench-prompts", 16, int), ("bench-max-tokens", 256, int), ("concurrency", 8, int),
                      ("reps", 3, int), ("label", "", str), ("results", None, str)]:
        ap.add_argument("--" + n, dest=n.replace("-", "_"), type=ty, default=dv, required=(dv is None))
    a = ap.parse_args()
    t0 = time.time()
    raw = open(a.text_file, "rb").read()
    text = raw.decode("utf-8", "replace")
    ep = a.endpoint.rstrip("/")
    tk = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = tk(text, add_special_tokens=False)["input_ids"][: a.max_eval_tokens]
    windows = build_windows(ids, a.ctx, tk.bos_token_id)
    prompts = []
    for line in text.splitlines():
        s = line.strip()
        if s and len(s) >= 200:
            prompts.append(s)
            if len(prompts) >= a.bench_prompts:
                break
    res = {"schema": "fs.compress.serve_bench.v1", "label": a.label, "endpoint": ep,
           "served_name": a.served_name,
           "text_sha256": hashlib.sha256(raw).hexdigest(), "ctx": a.ctx, "windows": len(windows),
           "served_ppl": None, "served_tokens": 0, "missing_tokens": 0,
           "bench": empty_bench(prompts, a), "hostname": socket.gethostname(), "seconds": 0.0}
    acc, status, reason = {"nll": 0.0, "count": 0, "missing": 0}, "MEASURED", None
    try:
        measure_ppl(ep, a.served_name, windows, acc)
        if acc["count"] == 0:
            raise RuntimeError("served_tokens == 0 (no scored positions)")
    except Exception as e:  # ppl phase cannot complete (HTTP error) -> UNMEASURED
        status, reason = "UNMEASURED", "%s: %s" % (type(e).__name__, e)
    res.update({"served_tokens": acc["count"], "missing_tokens": acc["missing"]})
    if acc["count"]:
        res["served_ppl"] = math.exp(acc["nll"] / acc["count"])
    if status == "MEASURED":
        res["bench"] = bench_run(ep, a.served_name, prompts, a)
    res.update({"status": status, "seconds": round(time.time() - t0, 3)})
    if reason:
        res["reason"] = reason
    os.makedirs(os.path.dirname(os.path.abspath(a.results)), exist_ok=True)
    with open(a.results, "w") as f:
        json.dump(res, f, indent=2)
    sys.exit(0 if status == "MEASURED" else 95)


if __name__ == "__main__":
    main()
