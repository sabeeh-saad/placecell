# Sensor and clock fault handling

Day 11 requires valid capture provenance before dispatch and explicit outcomes when that
provenance is lost. The [validation record](validation/day-11.json) identifies the exact
sources, reports and ROS image. These are software and simulated-message checks; physical
sensor behavior and live-model mission quality remain unqualified.

## Which inputs a goal needs

All navigation requires recent, valid localization in the configured map and an unchanged
source clock. Named places use their configured poses. Remembered scene destinations also
require a live camera with validated capture-time TF/localization. Object destinations
additionally require aligned depth and calibration. Losing depth permits scene-only
observations; it never fabricates an object position or proves that an object is absent.

`sensor_max_age_s` defaults to **5 seconds** and is explicit in the reference profile.
Both source timestamp age and monotonic receipt age must pass. This is a bounded freshness
policy, not a measured hardware latency budget. Localization has its separate existing
`localization_max_age_s` policy. A paused simulation clock cannot keep either input trusted.
Camera health is evaluated before ingestion sampling and worker-capacity checks, so a
stationary robot's sampling interval does not look like camera loss.

RGB must have a nonempty frame, a normalized positive timestamp, supported encoding and a
consistent bounded buffer (at most 16 MB and four million pixels). Compressed input must
decode as bounded JPEG; the ROS reference image includes Pillow for validation. Zero RGB
time is rejected because `Time(0)` asks TF for its latest transform, not capture-time TF.
Malformed base-frame quaternions, absent TF, or a pose inconsistent with localization
cannot authorize a capture. TF is requested at the RGB timestamp with the configured
timeout; a later transform does not retroactively validate an already discarded capture.

RGB, depth and calibration must share the optical frame and permitted skew (default
80 ms). Static CameraInfo with timestamp zero remains supported. Rectification, dimensions,
depth units/stride, camera rotation and localization uncertainty are still validated.
Out-of-order or repeated RGB never refreshes receipt age. Zero, stale and future RGB are
refused before they can advance the synchronizer's timestamp watermark. Invalid depth
timestamps are refused before entering the bounded cache. The RGB-D wait timer uses a
steady clock, including when `/clock` pauses.

The subsequent [Gazebo checkpoint](gazebo-checkpoint.md) refined packet-loss handling:
after an unpaired RGB capture's wait expires, a newer complete queued capture takes
precedence. If none exists, scene-only fallback remains available. An unpaired capture
does not refresh depth age or revoke a still-fresh valid depth sample. Sustained depth
loss expires at the same source/monotonic `sensor_max_age_s` bound and increments the
trust generation. Invalid camera/TF and clock inputs still revoke trust immediately.
This intentionally replaces Day 11's immediate revocation on every missing depth pair;
the historical Day 11 validation record describes the earlier source hashes.

## Mission behavior

Localization and camera/depth trust carry generation counters. A mission cannot use a
newly recovered generation to conceal a failure that occurred while it was active.
Checks apply during planning, before dispatch, during motion, at terminal action success,
during local search and before accepting a visual verdict. The check after publishing
`submitting` also covers a loss of trust during that publication/persistence interval.

- Missing localization/clock provenance blocks instruction admission with `unavailable`.
  A resolved destination missing its required live camera/depth is also `unavailable`.
- Interrupted planning/lookup is discarded without dispatch; its final status explains
  that it expired or became unavailable.
- During motion, the controller requests cancellation and remains busy until a known
  terminal action result. Missing cancellation acknowledgment does not prove a stop.
- A named-goal success arriving after trust was lost is `canceled`, with no next mission
  step. A memory-goal success without trusted arrival verification is
  `destination_unverified`. The interruption reason is retained in the final status.
- After ordinary sensor/localization recovery and confirmed completion of cancellation,
  a **new instruction** may start. The interrupted mission never resumes automatically.

The ROS node wires these guards into the controller. Direct Python users supplying their
own sensors should pass `localization_ready`, `localization_generation`, `sensor_ready`
and `sensor_generation` to `NavigationCommands`; its compatibility defaults cannot
inspect an external sensor stack.

## Clock resets and recovery

Any backward ROS-time jump, ROS clock activation/deactivation after node construction,
or an observed nonfinite source clock latches a fault. The jump callback only sets an
event; it never waits for provider work or a controller lock. Steady controller timers
request cancellation, and the sensor timer clears pending RGB/depth/calibration. New
captures and navigation remain blocked even if localization starts publishing again.
Forward jumps age existing samples; sufficiently large jumps invalidate trust. New
samples can recover ordinary freshness, but cannot resume an interrupted mission.

Memory identities and keyframe paths contain timestamps. Reusing them after a simulated
reset could mix observations from different runs, so automatic reset recovery is not
supported. Confirm Nav2 is quiescent, archive the old run, then restart with a fresh
collection and keyframe directory. Use a new map ID if the map itself changed. Preserve
the command journal for the same robot/map/conversation scope so retries remain suppressed.
Do not interpret controller restart as evidence that a previous Nav2 goal stopped;
post-crash action reconciliation remains Day 15 work.

Already queued observations from before the reset may finish ingestion in the old run;
they retain their original timestamps. No post-reset capture is admitted into that run.
The fault latch is process-local, and choosing fresh storage on restart remains an
operator responsibility.

## Reproduce

```bash
pytest tests/test_sensor_faults.py tests/test_localization.py tests/test_image_synchronization.py
python -m placecell.fault_injection --repeat 3 --output simulation/artifacts/day-11-faults.json
simulation/sim build
simulation/sim check-sensors
simulation/sim check-operator
simulation/sim check-cancel
```

The new ROS check uses actual DDS RGB-D, covariance, TF and `/clock` messages through the
production node, with scripted destinations/navigation and provider calls blocked. It
checks valid input, missing camera/depth, skew, duplicate/zero/future RGB, malformed RGB,
missing/delayed TF, localization loss/recovery and a real clock reset. It runs in an
isolated network namespace with eight CPUs and 16 GiB RAM. Reports are retained under
`simulation/artifacts/check-sensors-*` and the command is part of simulation CI.

The unit suite additionally forces terminal-result races, invalid quaternions, nonfinite
times, malformed covariance and recovery between polls. Thirteen new fault-runner
scenarios bring the suite to 73 cases (71 faults and two successful controls). The original
pre-repair run reproduced six failing cases; its report is retained separately. These
checks do not qualify physical stopping, full RGB-D throughput or endurance.
