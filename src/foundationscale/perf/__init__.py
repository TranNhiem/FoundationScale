"""Performance measurement for the training plane.

What this package is FOR: turning "the run finished" into "the run finished at
this throughput, at this MFU, with the GPUs starved for this fraction of the
time". The training loop already records four outcome numbers -- runtime,
samples/s, steps/s and peak memory -- and those are the trainer's own summary
statistics over the whole run. They cannot answer the questions an operator
scaling a job actually asks: is the step time stable or is one rank straggling,
are the GPUs waiting on the dataloader, how much of the silicon's arithmetic
is being used, and does any of that change when the job is spread over more
devices.

What this package deliberately does NOT do: it does not tune anything, it does
not choose a kernel, and it never turns a run RED. It is an instrument. A
measurement that can fail a run is a gate, and gates live in
``foundationscale.gates`` where their verdicts are adjudicated.

The one rule every entry here follows: **an unmeasured number is None paired
with a reason that names the missing input, never 0.0 and never absent.** A
zero throughput and an unread counter are different facts, and a benchmark
that cannot tell them apart will eventually publish the wrong one. Two
consequences are worth stating because they look like omissions:

* ``perf_mfu`` is unmeasured unless the operator DECLARES the device peak
  through ``FS_DEVICE_PEAK_TFLOPS`` / ``FS_DEVICE_PEAK_SOURCE``. The peak is a
  vendor number for a specific chip and a specific precision; there is no way
  to discover it from inside the process, and a built-in table would silently
  go stale one silicon generation later. A malformed declaration is REFUSED
  rather than demoted to unmeasured, so an operator's typo is visible instead
  of being hidden inside an honest-looking gap.
* ``perf_model_tflops_per_second`` is an ESTIMATE from the standard
  ``6N + 12*L*h*s`` formula, not a hardware counter. It is labelled "derived"
  for that reason. Where the model's architecture cannot be read from its
  config, there is no estimate at all rather than one built on an assumed
  layer count.

``telemetry`` imports no torch at module scope and guards its transformers
import, so importing this package costs nothing on a login node and does not
end the torch-free property of the modules that import it.
"""

from __future__ import annotations

from foundationscale.perf.telemetry import (
    PERF_TELEMETRY_UNITS,
    DevicePeak,
    FlopsModel,
    StepTelemetry,
)

__all__ = (
    "PERF_TELEMETRY_UNITS",
    "DevicePeak",
    "FlopsModel",
    "StepTelemetry",
)
