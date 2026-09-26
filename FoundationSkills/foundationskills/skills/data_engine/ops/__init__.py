
"""Data Engine operator registration.

Importing this package registers every operator in ``OPS`` via module import
side effects. Import ``base`` first so the registry exists before the ops
modules call ``register_op``.

Note: this module also makes the stdlib-looking name ``format`` importable as
``foundationskills.skills.data_engine.ops.format``; that is intentional.
"""
from __future__ import annotations

from foundationskills.skills.data_engine.ops import base
from foundationskills.skills.data_engine.ops import clean
from foundationskills.skills.data_engine.ops import decontam
from foundationskills.skills.data_engine.ops import dedup
from foundationskills.skills.data_engine.ops import format
from foundationskills.skills.data_engine.ops import ingest
from foundationskills.skills.data_engine.ops import mix
from foundationskills.skills.data_engine.ops import quality
from foundationskills.skills.data_engine.ops import tokenize

OPS = base.OPS

__all__ = [
    "OPS",
    "base",
    "ingest",
    "clean",
    "dedup",
    "quality",
    "decontam",
    "format",
    "tokenize",
    "mix",
]
