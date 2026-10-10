"""Backend adapters: the only place FoundationScale's VLA plane meets an upstream project.

One subpackage per backend (``gr00t`` now; ``openpi`` and ``verl`` follow). An adapter reads
or drives its upstream through that project's public surface only -- a checkpoint's own config
files, a documented entry point, a policy server -- and converts to and from FoundationScale's
owned contracts (robot data schema, checkpoint sidecar, run config, evaluation harness).
Upstream packages are never imported at module scope here: each backend runs in its own
environment, so the core install must import cleanly without any of them.
"""
