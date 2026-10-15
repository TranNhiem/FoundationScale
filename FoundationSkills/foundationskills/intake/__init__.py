"""Intake: importing third-party skill bundles into the FoundationSkills measurement pipeline.

Intake is the boundary between the outside world and FoundationSkills' own measurement
machinery. Nothing enters the corpus raw: a bundle that was authored elsewhere -- by a
different vendor, under a different licence, pinned to a different base image -- is
re-expressed here as something FoundationSkills can name, gate and record.

The importer (:mod:`foundationskills.intake.skill2env`) reads a skill-2-environment corpus,
joins it against the bundle's licence manifest and emits ``fskills.skill2env_import/1``:
one row per task with its licence verdict, its Dockerfile preflight and its acceptance
gates (oracle and no-op), reduced to a single worst-of-anything verdict with rule ids.
Absence of evidence never becomes evidence: a missing licence row is UNMEASURED, an
unavailable builder is UNMEASURED, a crash is REFUSED and a weak-copyleft licence is RED.

Intake modules are core, not registered skills: they share the repo doctrine that a check
which did not run is never a result.
"""
