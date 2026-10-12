# ruff: noqa: E501, E401
"""verl loads the FS music reward through its own custom_reward_function hook (real abc2midi)."""
import sys, inspect
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import get_custom_reward_fn
import verl.trainer.ppo.reward as R
from foundationscale.agentic_rl import verl_rewards as vr
cfg = OmegaConf.create({"reward": {"custom_reward_function": {"path": vr.__file__, "name": "music_compute_score", "reward_kwargs": {}}}})
fn = get_custom_reward_fn(cfg)
info = {"abc2midi_bin": sys.argv[1], "timeout_s": 60.0}
tune = ("X:1\nT:Scale\nM:4/4\nL:1/4\nQ:1/4=100\nK:C\n" + "C D E F | G A B c | c B A G | F E D C |\n" * 2 + "[CEG]4 | [FAc]4 | [GBd]4 | [CEGc]4 |]\n")
print("direct:", vr.music_compute_score("fs_music", tune, "", info))
print("hook positional:", fn("fs_music", tune, "", info))
print("hook keyword:", fn(data_source="fs_music", solution_str=tune, ground_truth="", extra_info=info))
src = inspect.getsource(R._call_with_kwargs); print(src[:600])
