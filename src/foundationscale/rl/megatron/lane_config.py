"""Configuration and named refusal rules for the Megatron-Core RL lane.

``MegatronLaneConfig.validate(world_size)`` derives DP as
``world / (TP * PP * CP)`` and refuses the design G section 2/config classes
that are measured or structural failures: non-divisible world/batch, CP on an
unsupported TE head_dim (head_dim 512 is measured-unsupported under CP),
EP/ETP inconsistency, sequence parallelism without TP, VPP, and refit/save
staleness. There are no silent fallbacks and no estate-specific constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TE_CP_SUPPORTED: frozenset[int] = frozenset((64, 128, 256))

RefMode = Literal["swap", "instance"]


@dataclass(slots=True)
class MegatronLaneConfig:
    """Parallel/batch layout; ``dp`` is derived by ``validate(world_size)``."""

    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    etp: int = 0
    sp: bool = False
    dp: int | None = None
    mbs: int = 1
    gbs: int = 1
    seq_len: int = 4096
    softcap: float | None = None
    head_dim: int = 128
    num_experts: int = 1
    ref_mode: RefMode = "swap"
    inner_epochs: int = 1
    refit_every: int = 0
    save_interval: int = 0
    vpp: int | None = None

    def validate(self, world_size: int) -> int:
        """Refuse invalid layouts by name; returns and stores derived ``dp``."""
        for name in ("tp", "pp", "cp", "ep", "mbs", "gbs", "seq_len", "inner_epochs"):
            value = int(getattr(self, name))
            if value < 1:
                raise ValueError(
                    f"lane_nonpositive_{name}: {name}={value}; parallel and "
                    f"batch extents must be positive integers"
                )
        if self.ref_mode not in ("swap", "instance"):
            raise ValueError(
                f"lane_ref_mode_unknown: ref_mode={self.ref_mode!r}; only "
                f"'swap' (default CPU weight swap) and 'instance' exist"
            )
        if self.vpp not in (None, 1):
            raise ValueError(
                "lane_vpp_not_supported: virtual pipeline parallelism changes "
                "the forward_backward schedule and metrics plumbing; rungs 0-1 "
                "use non-interleaved PP only"
            )
        model_parallel = int(self.tp) * int(self.pp) * int(self.cp)
        if int(world_size) % model_parallel != 0:
            raise ValueError(
                "lane_world_not_divisible: "
                f"world_size={world_size} is not divisible by tp*pp*cp="
                f"{model_parallel} (tp={self.tp}, pp={self.pp}, cp={self.cp}); "
                f"DP cannot be derived from a partial model-parallel replica"
            )
        dp = int(world_size) // model_parallel
        if int(self.gbs) % (dp * int(self.mbs)) != 0:
            raise ValueError(
                "lane_gbs_not_divisible: "
                f"gbs={self.gbs} is not divisible by dp*mbs={dp * int(self.mbs)} "
                f"(dp={dp}, mbs={self.mbs}); microbatch counts must be exact "
                f"so skipped microbatches are refused at collate time"
            )
        if int(self.cp) > 1 and int(self.head_dim) not in TE_CP_SUPPORTED:
            raise ValueError(
                "lane_cp_head_dim_unsupported: "
                f"cp={self.cp} with head_dim={self.head_dim}; design section 2 "
                f"marks head_dim 512 measured-unsupported under TE CP kernels, "
                f"supported set={sorted(TE_CP_SUPPORTED)}"
            )
        if int(self.ep) > 1 and int(self.etp) < 1:
            raise ValueError(
                "lane_ep_requires_etp: "
                f"ep={self.ep} but etp={self.etp}; EP>1 requires expert tensor "
                f"parallelism to bound alltoall extent and expert memory"
            )
        if int(self.ep) > 1 and int(self.num_experts) % int(self.ep) != 0:
            raise ValueError(
                "lane_ep_experts_not_divisible: "
                f"num_experts={self.num_experts} is not divisible by ep={self.ep}; "
                f"an expert shard cannot own a fractional expert"
            )
        if bool(self.sp) and int(self.tp) == 1:
            raise ValueError(
                "lane_sp_without_tp: sp=True with tp=1 has no tensor-parallel "
                "sequence region to shard; refusing a no-op that changes batch "
                "accounting"
            )
        if int(self.refit_every) > 0 and int(self.save_interval) > int(self.refit_every):
            raise ValueError(
                "lane_refit_save_staleness: "
                f"save_interval={self.save_interval} exceeds refit_every={self.refit_every}; "
                f"served weights would silently go stale between refreshes"
            )
        self.dp = dp
        return dp
