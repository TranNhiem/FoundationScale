"""Real-hardware check of foundationscale.agentic_rl.sandbox on an enroot node."""

import json
import os
import sys
import time
from pathlib import Path

from foundationscale.agentic_rl.sandbox.build import build_image
from foundationscale.agentic_rl.sandbox.enroot import EnrootSandbox, SandboxSpec

H = Path(os.environ["HOME"])
W = H / "fsar-data" / "sbxreal"
W.mkdir(parents=True, exist_ok=True)
roots = dict(
    data_root=str(H / ".enroot-sbx"), image_store=str(W / "images"), scratch_root=str(W / "scratch")
)
for d in roots.values():
    Path(d).mkdir(parents=True, exist_ok=True)
rep = {}
t = time.time()
r = build_image(Path(sys.argv[1]) / "Dockerfile", sys.argv[1], W / "toy.sqsh", **roots)
rep["build"] = {
    "image": r.image,
    "steps": r.steps_applied,
    "env": dict(r.env),
    "workdir": r.workdir,
    "secs": round(time.time() - t, 1),
}
sb = EnrootSandbox(
    SandboxSpec(name="toy-trial-1", image=str(W / "toy.sqsh"), network="no-network", **roots)
)
t = time.time()
sb.start()
rep["start_secs"] = round(time.time() - t, 1)
res = sb.exec("python -m pytest -q 2>&1 | tail -1")
rep["pytest"] = [res.return_code, res.stdout.strip()]
rep["env_FOO"] = sb.exec("echo $FOO").stdout.strip()
rep["pwd"] = sb.exec("pwd").stdout.strip()
net = sb.exec(
    "python -c \"import urllib.request;urllib.request.urlopen('https://pypi.org',timeout=5)\""
)
rep["no_network_blocks"] = net.return_code != 0
up = W / "up.txt"
up.write_text("uploaded")
sb.upload_file(up, "/testbed/up.txt")
rep["upload_seen_inside"] = sb.exec("cat /testbed/up.txt").stdout.strip()
sb.exec("echo produced > /testbed/out.txt")
sb.download_file("/testbed/out.txt", W / "down.txt")
rep["download"] = (W / "down.txt").read_text().strip()
sb.exec("echo private > /tmp/p.txt")
rep["tmp_private"] = (Path(roots["scratch_root"]) / "toy-trial-1" / "tmp" / "p.txt").exists()
sb.stop()
rep["removed"] = not (Path(roots["data_root"]) / "toy-trial-1").exists()
ok = (
    rep["pytest"][0] == 0
    and "1 passed" in rep["pytest"][1]
    and rep["env_FOO"] == "bar"
    and rep["pwd"] == "/testbed"
    and rep["no_network_blocks"]
    and rep["upload_seen_inside"] == "uploaded"
    and rep["download"] == "produced"
    and rep["tmp_private"]
    and rep["removed"]
)
rep["ok"] = ok
print(json.dumps(rep))
sys.exit(0 if ok else 5)
