"""Engine-adapter package: the shared infra-fault type every engine client,
engine fleet and the harness itself import.

``EngineInfraError`` lives HERE -- not in :mod:`.sglang` -- so a caller that
only needs to catch an engine's infra faults (e.g.
``harness.native_tool_loop``) never pulls in ``urllib``, ``subprocess`` or any
other transport machinery to do it. This module imports nothing from
``.sglang`` or ``.fleet`` on purpose: importing ``foundationscale.agentic_rl.
engines`` alone must never start a subprocess or open a socket.

WHAT IS NOT CLAIMED: any transport, process-lifecycle or HTTP behaviour --
those are :mod:`.sglang` (the HTTP client) and :mod:`.fleet` (the subprocess
fleet), both leaf modules this package does not import eagerly.
"""

from __future__ import annotations

__all__ = (
    "EngineInfraError",
    "EngineRefusal",
)


def _described(value: object) -> str:
    """Short ``type-name repr`` rendering of a rejected value for refusal messages."""
    return f"{type(value).__name__} {value!r}"


class EngineRefusal(ValueError):
    """A programming/config error in an engine declaration or call -- not a runtime
    infra fault. Raised by construction-time validation across this package (client,
    server spec, fleet); never raised for a transport or protocol failure, which is
    :class:`EngineInfraError`'s job.
    """


class EngineInfraError(RuntimeError):
    """A RUNTIME fault from a generation engine (transport broke, bad response shape,
    startup failed, HTTP error) -- never the model's fault.

    ``kind`` is the short machine-readable tag ("transport", "timeout",
    "bad_response", "http_<code>", "startup_timeout", "startup_crashed", ...) that
    ``harness.native_tool_loop`` maps to ``abstention_reason=f"infra:engine_{kind}"``,
    mirroring how ``envs.base.EnvInfraError.kind`` is mapped to
    ``abstention_reason=f"infra:{kind}"``.
    """

    def __init__(self, kind: str, message: str) -> None:
        """Store the machine-readable ``kind`` tag; refuse a non-str/empty ``kind`` or a
        non-str ``message`` because an untagged or undescribed infra error cannot be
        routed or reported.
        """
        if not isinstance(kind, str):
            raise EngineRefusal(
                f"EngineInfraError.__init__: parameter 'kind' is {_described(kind)}: an "
                f"infra error kind is a short machine-readable str tag -- a non-str kind "
                f"cannot be routed into abstention_reason"
            )
        if not kind:
            raise EngineRefusal(
                f"EngineInfraError.__init__: parameter 'kind' is {_described(kind)}: an "
                f"infra error kind is a non-empty str tag -- an empty kind cannot be "
                f"routed into abstention_reason"
            )
        if not isinstance(message, str):
            raise EngineRefusal(
                f"EngineInfraError.__init__: parameter 'message' is {_described(message)}: "
                f"an infra error message is a str -- a non-str message cannot be reported "
                f"to a harness"
            )
        super().__init__(message)
        self.kind = kind
