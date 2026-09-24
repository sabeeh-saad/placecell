# Saturation and backpressure

Day 17 bounds waiting work, exposes rejected work, and prevents ingestion retries from
turning a provider outage into a request burst. The [validation record](validation/day-17.json)
separates unit regressions from the controlled ROS workload. No paid providers were used;
the five deferred live product missions remain paused.

## Admission and shutdown

`max_queue` includes active, waiting and exhausted ingestion jobs. The reference profile
allows four. Normal camera callbacks check capacity before JPEG encoding or image writes.
If admission loses a race after encoding, cleanup removes unowned evidence while preserving
files referenced by accepted jobs or memories. Arrival observations may bypass ordinary
ingestion admission; they retain their existing bounded command workers and deadlines.

`camera_max_message_bytes` defaults to 8 MiB per raw RGB, compressed RGB or depth payload.
Larger payloads are rejected before encoding or synchronization storage. RGB, depth and
calibration caches each retain at most eight messages; eviction and invalid/oversized
input counters are visible. This limits retained application buffers. DDS can allocate a
message before delivering it to Python, so this is not a middleware memory limit.

Questions must contain 1–2,000 characters and non-whitespace content. Validation occurs
before worker admission; invalid replies include at most 128 input characters. Defaults
are two active question workers and eight waiting requests. Overflow receives an explicit
busy response. Command admission has its own bounded pool.

Task admission and shutdown are atomic. Shutdown rejects new submissions and discards
waiting work, releasing its payloads. Already running external calls are allowed to finish;
Python cannot forcibly stop them. Idle workers also release completed request payloads.
The [maintenance lease](backup-restore.md) remains held if a worker cannot stop, preventing
a backup from treating the process as quiescent.

Repeated refinement, curation and consolidation ticks coalesce while that same maintenance
task is queued or running. Each task type has its own key. Explicit user requests do not
coalesce, so a deliberate repeated command is not silently removed.

## Provider retries

The ROS ingestion captioner, detector and cloud embedder make one HTTP attempt per
individual provider operation. The durable worker owns retries. A job may need several
operations or batches; this is not a one-request-per-job promise.

On failure, the journal persists a queue-wide deadline alongside per-job attempt counts.
The delay is the larger of the provider's `Retry-After` and exponential backoff, whose
local component is capped at 300 seconds. The deadline still applies if the failed head
job exhausts its attempts, or if memory work committed before a later provider failure.
Restarting or manually retrying failed jobs cannot bypass this deadline. The single writer
resumes eligible observations in timestamp order.

Defaults remain five durable attempts and a one-second base delay. Exhausted jobs stay
visible and pin their evidence and queue capacity until explicitly retried or discarded.
Zero base delay is allowed for deterministic fixtures. Production configurations should
use a positive delay. This cooldown applies to one ingestion queue; it is not an
account-wide limit covering navigation, questions and maintenance.

The shared HTTP adapter preserves numeric and HTTP-date `Retry-After` values. If the
requested wait exceeds its inline sleep budget, it returns a typed provider error with
the cooldown instead of retrying early. Other callers receive that error without a new
automatic retry loop.

## Diagnostics

Periodic node diagnostics include ingestion capacity, queued/failed counts, oldest job
age, remaining cooldown and drops. Task pools report active/waiting work, capacity,
accepted/completed/failed/discarded work, full/stopped rejections, coalescing and queue
high-water counts. Camera diagnostics include buffer depths and discard counts. Ingestion
drop warnings are emitted at most once per five seconds; every drop is still counted.
In-memory counters reset on restart. Durable jobs, attempts and cooldowns survive it.

## Reproducing the checks

```bash
.venv/bin/pytest tests/test_overload.py tests/test_jobs.py tests/test_image_synchronization.py
simulation/sim build
simulation/sim check-overload --samples 100
```

The normal Python test matrix includes the unit regressions. The simulation CI workflow
also runs the 100-trial overload harness and retains its artifacts on failure. Hosted CI
results require a pushed revision; the Day 17 record reports local checks only.

Unit tests cover 8,000 concurrent submissions, atomic shutdown, 2,000 repeated maintenance
ticks, payload release, image ownership after overflow, RGB/depth buffer eviction,
exhaustion/restart/manual-retry cooldowns and HTTP retry limits. Memory and LanceDB stores
both exercise the durable journal regressions.

The ROS harness runs production camera ingestion, question admission and maintenance
alongside the real controller/Nav2 adapter in a shared four-thread executor. A separate
four-thread executor hosts a controlled DDS action server. It publishes 640 × 480 RGB
at up to 20 Hz, holds a scripted caption provider with four accepted jobs, blocks one
question worker with two waiting slots, and repeats a maintenance tick 1,000 times.
It then measures 100 cancellation trials and completes a two-goal positive control.
Releasing the provider must drain exactly four memories with no duplicate sightings.

Capture poses are authored, bypassing TF/localization. There is no depth, object detection,
Gazebo motion or physical stopping in this overload harness. Unit buffer checks do not
substitute for a complete RGB-D workload. Published frames and delivered callbacks have
separate denominators because DDS may supersede samples under load.

Cancellation latency runs from command callback receipt to return from the asynchronous
cancel-request API. Its p99 target remains 500 ms; per-trial publish-to-request and
request-to-acknowledgement timings are retained separately. Loaded trials use the reference
deployment's 10-second goal acknowledgement deadline. Deliberate fault phases use a shorter
deadline. The record also preserves an initial run using the old harness's 300 ms loaded
deadline that blocked safely on uncertain ownership after a late goal acknowledgement.

These are short, controlled stress checks. They do not satisfy the 24-hour endurance
gate, qualify live-model resource use, enforce a filesystem quota or establish an accepted
ingestion-age budget for the supported provider workload.
