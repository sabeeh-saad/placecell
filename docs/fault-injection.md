# Exercise failure handling

The offline fault runner connects the production mission planner, independent plan reviewer,
destination resolver, mission controller, localization gate and Nav2 adapter. Scripted
providers and a controlled action client inject failures at their interfaces. Context uses
a real temporary SQLite database. No API key, ROS installation or robot is needed.

The suite has **73 scenarios: 71 fault cases and two successful controls**. Its checks
specify expected status, dispatch count and mission ownership at intermediate checkpoints,
as well as the final outcome. A successful fault check means the software handled the
specified failure; it does not mean a navigation mission succeeded.

## Run and reproduce

Use the active development environment:

```bash
python -m placecell.fault_injection \
  --output simulation/artifacts/faults/run-01.json
```

A fresh installation also exposes `placecell-check-faults` with the same arguments.
Select one or more cases by repeating `--case`, and repeat the selected suite with
`--repeat` (1–1000). For example, reproduce the late-success regression:

```bash
placecell-check-faults \
  --case nav_timeout_late_success \
  --case nav_timeout_visual_success \
  --case nav_late_acceptance_success \
  --repeat 10 \
  --output simulation/artifacts/faults/late-success-01.json
```

`--help` lists the available case IDs. Reports must use a new path; existing files are
never overwritten. Exit code 0 means all selected contracts passed, 1 means one or more
failed (the report is still written), and 2 means invalid arguments. Unexpected scenario
exceptions are recorded as failures, not dropped from the denominator. A scenario without
any checks cannot pass. Each repetition starts with fresh state and a new temporary database.

## Failure contracts

- **Planning and review:** provider timeouts, malformed replies, replies arriving after the
  controller deadline, reviewer rejection, and stop during review must dispatch no goal.
  The late-reply cases explicitly poll the controller while the scripted provider is active.
- **Provider contracts:** duplicate decision/verdict fields, unknown tools/actions, extra
  fields, oversized explanations, missing completion evidence, refusal-plus-positive output
  and HTTP errors must report a reason with zero dispatches. These cases exercise the actual
  chat/vision adapters against injected HTTP bodies.
- **Nav2:** unavailable server and rejected goals cannot advance a mission. A send error,
  late acceptance, lost terminal result, delayed cancel acknowledgement or rejected cancel
  retains ownership until a terminal result resolves the trip. New instructions receive
  `busy`. A cancel acknowledgement alone is not confirmation that the robot stopped.
- **Late success after cancellation:** transport deadlines are cancellation intent too.
  Named-place missions end `canceled`; remembered-pose missions end
  `destination_unverified`. Neither may publish `step_succeeded`, start another destination,
  or begin fresh arrival verification. Late feedback cannot reopen a completed trip.
- **Localization:** stale estimates prevent admission or request cancellation during motion.
  Receipt age still expires when the source/simulation clock is paused.
- **Visual evidence:** failed candidate verification prevents dispatch. Missing or stale
  arrival images, inconsistent capture poses, and arrival-verifier timeouts end
  `destination_unverified`, without advancing the mission.
- **Context storage:** failed instruction writes, history reads, and pre-dispatch status
  writes block movement. A failed motion-status write requests cancellation on the next
  controller poll. A failed step-status write prevents the next destination.
- **Depth synchronization:** missing or out-of-skew depth must wait for the bounded window,
  then deliver RGB once for scene-only handling. These are synchronizer checks; they do
  not exercise object geometry or claim that depth loss blocks named-place navigation.
- **Interrupted persistence:** a child process commits instruction/history events, begins
  another SQLite transaction, then exits without cleanup. Reopening must preserve committed
  history, discard the incomplete success record, and never automatically replay motion.

The controls complete two named destinations in order and one remembered destination with
scripted candidate and fresh-arrival verification. They prevent “always refuse” behavior
from appearing as a healthy suite.

Day 8 adds ten cancellation cases: stop during planning/candidate/arrival work, stop before
acceptance, deadlines enforced without a timely poll, and terminal results deliberately
delivered before cancellation events. Named and memory goals keep their distinct outcomes.
These forced callback interleavings are deterministic; threaded regressions and the
[real ROS cancellation check](cancellation-ownership.md) supply separate scheduling evidence.

Day 10 adds 14 [provider-contract scenarios](model-input-contracts.md). New malformed
planner and visual replies were reproduced before repair; positive controls remain required.

Day 11 adds 13 [sensor/clock provenance scenarios](sensor-clock-contracts.md), including
recovery before polling, late success after trust loss, camera/depth-dependent refusal,
paused source time and reset clocks. The separate `simulation/sim check-sensors` command
exercises real DDS and production-node capture callbacks; deterministic fixtures do not
stand in for that transport evidence.

## Read the report

Day 12 adds nine [target-freshness scenarios](target-freshness.md) and checks failure-stage
preservation in traces. Each result records `failure_stage`; `summary.final_failure_stages`
counts terminal attribution independently of the contract pass/fail totals. An expected
refusal is a passing software check, not a successful robot mission. The empty-string
bucket includes successful controls and cases without terminal failure attribution.
Target scenarios use a 30-second arrival phase to isolate the 5-second capture-age bound.

The JSON report includes:

- Suite version, creation time, Python version, selected cases and repeat count.
- SHA-256 hashes of the installed PlaceCell Python sources, including the scenario fixtures.
- Every repetition, injected fault, expected/actual checks and unexpected exceptions.
- Status events serialized through the same `navigation_payload` function used for the
  ROS topic, retaining request/mission/step IDs and goal descriptions. Events are collected
  in-process; DDS delivery is not tested.
- Goal submission attempts, cancellation calls, final ownership, queued tasks, scripted
  Python model/verifier fixture calls, virtual elapsed time and measured wall duration.
  For the adapter-boundary scenarios, injected HTTP calls are recorded by the scenario's
  `one bounded provider call` check; they are separate from the Python fixture counters.
- Explicit limitations and zero paid API calls/cost.

Day 4 adds a `trace` report to each result. It captures production-stage events and checks
that published statuses, submission attempts and cancellation requests match their trace
events, with no observed loss. The harness's temporary database is removed after each case;
the exported trace remains in the report. See [mission tracing](mission-tracing.md) for the
schema, persistent ROS setup and exporter.

The runner uses shorter virtual deadlines (5 s lookup, 3 s arrival, 2 s Nav2 response,
10 s trip) to exercise exact boundary conditions without waiting. These are harness
settings, not deployment defaults. Runtime UUIDs, wall timings and temporary paths in
exception messages can differ across runs; the scripted event order and expected contract
remain reproducible. Repeating sequential cases checks repeatability, not concurrent races.

Reports are local under the ignored `simulation/artifacts/` directory. The Python CI
matrix runs the suite and retains a separate report artifact for each Python version,
including a failed report when available.

## Evidence limits

This is synthetic software evidence, separate from the [mission evaluation](mission-evaluation.md)
dataset and real-model accuracy. It makes no network calls and does not measure physical
movement, stopping distance, live provider deadlines, ROS callback scheduling or Gazebo
sensor behavior. Named-place dispatch uses configured poses; memory retrieval uses an
offline hashing embedder and a scripted visual verdict over a tiny image fixture.

Capture-pose faults exercise localization admission and controller trust checks, not an
actual TF buffer outage. Depth faults exercise the synchronizer, not the full RGB-D node.
SQLite interruption tests rollback and history-only restart, not power loss, LanceDB index
recovery, or reconciliation with a Nav2 goal surviving a controller crash. Those need
separate integration/qualification runs. Passing this suite alone is not production qualification.

The [Day 3 validation record](validation/day-03.json) records the observed regression,
fix, source hashes and local check results.
The [Day 8 record](validation/day-08.json) records the expanded suite, concurrent callback
repairs and controlled ROS latency checks. Startup goal reconciliation remains outstanding.
