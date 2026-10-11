"""The boundary between FoundationScale and every upstream framework.

This package is WHERE upstreamity is contained. Nothing outside it may import
NeMo, transformers or any other framework's internals, and nothing inside it
caries a framework import except the per-backend worker subpackages
(``upstream/nemo/`` and friends), which are the only code allowed to import the
framework they adapt -- and they run in their own pinned container anyway, so
the import never reaches the control plane.

What lives here, and why exactly two modules and no more:

* ``contracts.py`` -- the data contracts FoundationScale OWNS: the speech
  manifest row schema, its reader and problem codes, and the speech run config
  together with its CLI projection. A contract is ours because campaigns are
  reproduced, diffed and regression-tested over months, while an upstream
  manifest format is rewritten whenever a vendored example script is tidied up
  (measured: the NeMo example scripts write the transcript field as ``text``
  while the checkpoint's inherited ``train_ds`` names it ``answer``; leave the
  inheritance alone and targets come back empty and nothing says so). An owned
  contract makes that disagreement a REFUSAL with a code, not a quiet WER.

* ``ledger.py`` -- the copy/workaround ledger. Every formula copied out of an
  upstream repo, every private symbol touched, and every example script run by
  file path is recorded there with why, where, from which upstream version, and
  an expiry probe. The ledger exists because unrecorded copies do not announce
  themselves when upstream changes underneath them: the number is simply wrong
  one campaign later and nobody knows which copy went stale. With the ledger,
  a version bump becomes a checklist of entries to re-verify.

Every refusal here is NAMED and countable (problem codes in the manifest
contract, ``ValueError`` messages that name the offending key in the run-config
contract). Nothing silently defaults: an unrecognised manifest key is a
problem, an unrecognised run-config key is a refusal, and a value a schema does
not describe is never "probably fine".

This module deliberately imports nothing but its own submodules: it is the
first thing a worker and the control plane have in common, and it must stay
loadable without any ML stack on the path.
"""

from __future__ import annotations

from foundationscale.upstream import contracts, ledger

__all__ = ["contracts", "ledger"]
