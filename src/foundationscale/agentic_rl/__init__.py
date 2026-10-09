"""FoundationScale agentic RL: the torch-free contract plane.

WHAT IS CLAIMED HERE: the trajectory/turn/tool-call contracts and their named
refusals (``contracts``), the ``DECLARED_COLUMNS`` batch schema
:func:`~foundationscale.agentic_rl.contracts.flatten` emits, and the incremental
:class:`~foundationscale.agentic_rl.token_trace.TokenTrace` bookkeeping a
rollout bridge needs to keep one flat token sequence without ever re-rendering
a sampled turn.

The rollout HOST, the harness adapter, the environment and the engine adapters
are later-slice modules and are deliberately neither imported nor re-exported
here -- the same rule ``foundationscale.rl.__init__`` applies to its
torch-bearing modules. Import them by submodule
(``foundationscale.agentic_rl.<module>``) when they land: a contract plane that
re-exported its engines would decide what an engine is, and this one only
decides what a rollout REPORTS. ``ExperienceBatch`` is likewise not claimed: it
is ``foundationscale.rl.interfaces``' contract and is reached for through that
module, never restated here.

WHAT IS NOT CLAIMED: any behaviour behind ``metadata`` (harness-private strings
that never reach a batch), any verdict on model tool-call text (the harness's
verdict is recorded verbatim and never re-derived), or any engine semantics at
all -- nothing in this namespace imports torch and nothing in it may.
"""

from __future__ import annotations

from foundationscale.agentic_rl.contracts import (
    DECLARED_COLUMNS,
    SegmentKind,
    Termination,
    ToolCall,
    Trajectory,
    TrajectoryRefusal,
    Turn,
    flatten,
    group_by_uid,
)
from foundationscale.agentic_rl.token_trace import (
    ResponseBudgetExhausted,
    TokenTrace,
    TokenTraceRefusal,
    select_delta_messages,
)

__all__ = (
    "DECLARED_COLUMNS",
    "ResponseBudgetExhausted",
    "SegmentKind",
    "Termination",
    "TokenTrace",
    "TokenTraceRefusal",
    "ToolCall",
    "Trajectory",
    "TrajectoryRefusal",
    "Turn",
    "flatten",
    "group_by_uid",
    "select_delta_messages",
)
