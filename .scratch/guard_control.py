"""Positive control: does the #371 guard actually fire, and only when it should?"""
import json, tempfile, pathlib, subprocess, sys, textwrap
def run(records):
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "c.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    code = textwrap.dedent(f"""
        from foundationscale.rl.trainer import RLTrainConfig, RLTrainer
        cfg = RLTrainConfig(model="never-loaded", dataset={str(d / 'c.jsonl')!r},
                            group_size=4, temperature=1.0)
        RLTrainer(config=cfg).run()
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return r.returncode, (r.stderr or "")

text_only = {"id": "t1", "conversations": [
    {"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}]}
with_image = {"id": "v1", "image": "/some/pic.png", "conversations": [
    {"from": "human", "value": "What is this?"}, {"from": "gpt", "value": "a cat"}]}
with_video = {"id": "v2", "video": "/some/clip.mp4", "conversations": [
    {"from": "human", "value": "What happens?"}, {"from": "gpt", "value": "it moves"}]}

for label, recs, expect_guard in (
    ("text-only (guard must NOT fire)", [text_only], False),
    ("one image record   (MUST fire)",  [text_only, with_image], True),
    ("one video record   (MUST fire)",  [text_only, with_video], True),
):
    rc, err = run(recs)
    fired = "cannot yet route to the model" in err
    ok = "OK " if fired == expect_guard else "WRONG"
    print(f"  {ok} {label}: rc={rc} guard_fired={fired}")
    if fired:
        line = [l for l in err.splitlines() if "cannot yet route" in l][0]
        print(f"        {line[:150]}")
