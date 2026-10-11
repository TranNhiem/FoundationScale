# P0.7 sandbox integrity on real enroot (2026-10-11)

`real_integrity.py`, run on a GB200 tray (aarch64, enroot), reusing the toy image built by
`../agentic_rl_sandbox/real_sandbox.py`. The full output is in `report.json`.

| check | expected | measured |
|---|---|---|
| seal `/tests` + `/testbed/.git` | both moved to the host vault | `sealed: ["/tests", "/testbed/.git"]` |
| agent phase view | hidden paths absent | `ls` / `cat` fail with "No such file or directory" |
| unseal for the verifier | content restored byte-for-byte | `verifier_view: ["gold", "ref"]` |
| dangling symlink planted at `/tests` | MUST_FIRE in assert_sealed and unseal | both FIRED |
| file planted at `/tests/t.py` | MUST_FIRE in assert_sealed and unseal | both FIRED |
| `/testbed` replaced by a symlink to `/elsewhere` | MUST_FIRE in assert_sealed and unseal | both FIRED |
| egress probe, `network="no-network"` | False | False |
| egress probe, `network="public"` | True | True |

A bug was found and fixed on this hardware. In the first run, `assert_sealed` did NOT fire on the
symlinked-parent attack (`unseal` did). The symlink target `/elsewhere` is a container path, and the
host cannot resolve it, so the hidden path looked absent from the host even though the agent could see
it. `assert_sealed` now refuses any symlinked ancestor, with a unit test covering it.
