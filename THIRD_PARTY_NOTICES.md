# Third-Party Notices

FoundationScale includes material adapted from the third-party project below.
Apart from that material, FoundationScale is licensed under the MIT License
(see LICENSE).

## verl

- Component: `src/foundationscale/agentic_rl/token_trace.py`, adapted from the
  file named in its attribution header.
- Upstream repository: https://github.com/XiaomiMiMo/verl
- Upstream path: `recipes/arvo/token_trace.py`
- Upstream commit: a2ad9f61
- Copyright: Copyright the original authors of the adapted file.
- License: Apache License, Version 2.0.
  Full text: https://www.apache.org/licenses/LICENSE-2.0.txt
- Upstream NOTICE (verbatim from `Notice.txt`):
  "Copyright 2023-2024 Bytedance Ltd. and/or its affiliates"
- Modifications: Copyright (c) 2026 TranNhiem, licensed under the MIT License
  (see LICENSE). See the modified file's header and module docstring for the
  nature of the modifications.

## verl (music scorer port)

- Component: `src/foundationscale/agentic_rl/rewards/music/core.py`,
  `src/foundationscale/agentic_rl/rewards/music/feats.py`,
  `src/foundationscale/agentic_rl/rewards/music/score.py`,
  `src/foundationscale/agentic_rl/rewards/music/pipeline.py`, and
  `src/foundationscale/agentic_rl/rewards/music/baseline.py`, adapted from the
  files named in each file's attribution header.
- Upstream repository: https://github.com/XiaomiMiMo/verl
- Upstream path: `recipes/design/music/scorer/core.py`,
  `recipes/design/music/scorer/feats.py`,
  `recipes/design/music/scorer/score.py`,
  `recipes/design/music/scorer/pipeline.py`, and
  `recipes/design/music/scorer/baselines/ref_full4k.json` (embedded as the
  `REF_FULL4K` Python literal in `baseline.py`, not shipped as package data).
- Upstream commit: a2ad9f61
- Copyright: Copyright the original authors of the adapted files.
- License: Apache License, Version 2.0.
  Full text: https://www.apache.org/licenses/LICENSE-2.0.txt
- Upstream file header (verbatim, each adapted file):
  "Copyright 2026 Bytedance Ltd. and/or its affiliates"
- Modifications: Copyright (c) 2026 TranNhiem, licensed under the MIT License
  (see LICENSE). See each modified file's header and module docstring for the
  nature of the modifications -- in short: added type hints; the `abc2midi`
  binary path and subprocess timeout are explicit `MusicReward` fields rather
  than environment-variable reads; the public adapter
  (`foundationscale.agentic_rl.rewards.music.MusicReward`) does not swallow
  unexpected exceptions behind `traceback.print_exc()`; the baseline is the
  embedded `REF_FULL4K` literal rather than a JSON file read from disk; and a
  new `Abc2MidiError` exception lets an `abc2midi` timeout or other OS-level
  failure abstain (`value=None`) instead of upstream's measured `0.0`.
