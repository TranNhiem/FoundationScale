"""Alpamayo adapter: NVIDIA's driving VLAs, served from their own environment."""

from __future__ import annotations

from foundationscale.vla.adapters.alpamayo.client import AlpamayoError, AlpamayoTrajectoryClient

__all__ = ["AlpamayoError", "AlpamayoTrajectoryClient"]
