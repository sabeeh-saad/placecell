# Gazebo checkpoint through Day 12

The 22 September checkpoint exercised the accumulated implementation in the actual
Gazebo office, with RGB-D, TF, AMCL and Nav2. **It is not an all-green qualification.**
The live-provider mission reached the printer but refused arrival confirmation because
continuous ingestion had created a competing printer identity. It did not start the
second destination or publish a false success.

The [validation manifest](validation/gazebo-checkpoint-2026-09-22.json) records source
hashes, image provenance, reports and outcomes. Detailed local artifacts are under
`simulation/artifacts/gazebo-checkpoint-20260922/`. Earlier Day 10–12 records remain
historical evidence for their original source hashes.

## Three distinct kinds of evidence

- **Gazebo baseline:** real camera/depth/lidar streams, clock advancement, RGB-D capture
  admission, forward motion, rotation and command-silence stopping. Actual Nav2 reached
  two configured poses. The baseline video proves navigation only.
- **Live providers:** the production node used the configured hosted embedding,
  captioning, detection, planning and independent review adapters, persistent memory,
  command journal and real Nav2. It learned a localized printer, rejected a duplicate
  instruction and traveled 2.034 m on the semantic trip. The terminal result was
  `destination_ambiguous` / `identity`; the requested printer-then-home mission failed.
- **Deterministic providers:** the same production controller, storage, ROS interfaces,
  geometry and Nav2 used scripted planning/review and an office-specific blue-display
  pixel detector. HTTP provider calls were blocked. These checks isolate software
  behavior; they do not measure recognition quality or generalize to other objects.

The continuous-ingestion deterministic matrix passed 6 of 11 contracts: ordered named
destinations with duplicate suppression, single-object arrival, stop during planning,
stop during motion, malformed-plan rejection and depth-loss cancellation. Five trials
were blocked by ambiguous/excessive object candidates before their intended injection:
camera loss, removal, movement, occlusion and lookalikes. They remain failures in that
matrix's denominator.

An additional `--isolate-arrival` profile defers **background object refresh** for 3,600
seconds after learning the initial reference. Scene ingestion, sensor-health checks,
fresh arrival detection/geometry, paired comparison and actual navigation remain active.
This permits independent testing of the intended faults without claiming that continuous
identity association was repaired. Its results are recorded separately in the manifest.
It passed 3 of 5 strict expectations: camera loss canceled and the simulated robot stopped,
occlusion returned `unobserved` / `geometry`, and a second printer returned `ambiguous` /
`identity`. Removal returned `unobserved` rather than the expected `missing`; movement
returned `unavailable` / `geometry` because reliable depth or an equivalent recorded
viewpoint was unavailable. Both remained unverified, with no false success. These two
strict expectation failures remain failures rather than being relabeled as passes.

World changes occur only after PlaceCell has dispatched a real Nav2 goal. Removal moves
the authored printer below the world; movement changes its lateral position; occlusion
adds an opaque box; the lookalike duplicates the printer mesh. Simulator entity positions
are test stimuli and never inputs to the perception fixtures.

## Repairs supported by reproduced failures

1. **Dropped depth packet blocked a later complete capture.** The synchronizer held the
   oldest RGB message until its deadline, then returned it without depth even when a
   newer complete RGB-D capture was waiting. It now skips that expired incomplete
   capture in favor of the newer complete one. Capture order, queue bounds, timestamp
   checks and scene-only fallback when no complete pair exists remain enforced.
2. **One unpaired frame canceled a healthy object trip.** Depth health used to revoke
   its generation immediately on any RGB capture without usable depth. It now preserves
   a still-fresh valid sample, without refreshing its source or monotonic age. Sustained
   loss expires at the existing five-second bound. Camera/TF invalidity and clock faults
   still revoke trust. Unit, fault-runner and DDS expectations now express this deadline.

Both repairs have failing-before and passing-after regression artifacts. A stationary
25-second Gazebo diagnostic changed from 58 complete / 20 incomplete captures to 71
complete / zero incomplete captures. This is one diagnostic, not a throughput guarantee:
intermittent incomplete captures still occurred during full ingestion/navigation. The
depth-health repair allowed the subsequent live semantic trip to reach its stopping pose.

## Remaining blockers and limits

- **Identity fragmentation:** an RGB-only sighting from a different viewpoint can become
  another object record instead of associating with the existing printer. That record
  remains an appearance rival and correctly prevents a unique arrival verdict. Fixing
  association/reconciliation requires preserving true lookalike ambiguity; discarding
  rivals or weakening arrival checks would hide the problem.
- **Trace loss under Nav2 feedback load:** the default bounded trace queue dropped
  thousands of events during the successful ordered trial, including one dispatch
  event. Goal counts in the harness therefore come from an independent wrapper around
  the actual action client's `send_goal_async`, and trace losses are reported separately.
  Critical-event retention and feedback volume need repair before claiming complete
  mission observability. An early harness attempt also read traces before the writer
  had drained; the revised harness flushes and preserves read errors explicitly.
- **Arrival availability and geometric evidence:** a moved target was not confirmed
  when the arrival capture lacked reliable depth. Removing the target did not establish
  a visibly empty old region. Further tests must separate packet delivery, capture
  selection and geometric clearance before changing thresholds or claiming movement/
  absence recognition. The narrower deterministic detector is also a limitation.
- These are individual trials in one authored office. They do not qualify held-out
  model accuracy, endurance, real sensor loss rates, physical braking, crash recovery
  or every release criterion in the [readiness contract](production-readiness.md).

The full software regression suite passed **1,187 tests**, with **95.68% branch-inclusive
coverage**, plus **246/246** fault executions across 82 scenarios. Ruff and mypy passed.
The rebuilt image also passed 13 real-DDS sensor checks and 23 operator checks. Those
controlled ROS transports are separate from the actual Gazebo mission outcomes above.
The separate 100-trial cancellation benchmark passed at **2.86 ms p99** (5.42 ms maximum),
measuring callback receipt to async cancel API return under its synthetic load. It does
not measure physical stopping or the full Gazebo workload.

## Reproduce

```bash
simulation/sim build
simulation/sim start-nav
simulation/sim check-missions
simulation/sim check-missions --isolate-arrival \
  --case camera_loss --case removed --case moved --case occluded --case lookalike

# Opt-in paid run: configure OPENROUTER_API_KEY privately in the environment first.
simulation/sim check-live-mission
simulation/sim stop
```

Run only one motion harness at a time. Each attempt writes to a fresh output directory
and preserves its failure report. The live harness also accepts `--key-file` for a
private mounted credential. It records provider metadata, never authorization headers
or request bodies. Each run stops before additional requests after 80 calls, 900 seconds,
or $1 in provider-reported usage; in-flight/unknown costs mean this is not a billing cap.
The checkpoint's four live attempts made 128 requests reporting **$0.03409245** total;
all returned usage costs. The first attempt ended on an incorrect harness receipt
expectation, two exposed false depth-loss cancellation, and the final one exposed
identity ambiguity. These attempts are not independent model-quality measurements.

The test container had eight CPUs, 16 GiB RAM and a private ROS/Gazebo network namespace.
It used bridge networking for live providers; deterministic mission checks blocked model
HTTP calls in the harness. Separate DDS/contract containers used `--network none`.
The temporary credential file and task-owned Gazebo container were removed after testing.
No physical robot or unrelated running deployment was changed. Work remains local and unpushed.
