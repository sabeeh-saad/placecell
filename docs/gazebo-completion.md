# Gazebo follow-up repairs — 22 September 2026

This follows the [earlier repair checkpoint](gazebo-repairs.md). Its historical reports
remain unchanged. The new runs are under
`simulation/artifacts/gazebo-completion-20260922/`.

## Changes

- Object scan timestamps have their own effect on scheduling; only actual object/view
  mutations invalidate a checked arrival evidence version. Identity alternatives and
  newer evidence are still rechecked before accepting success.
- The full-scene request check runs alongside object identity verification. Arrival
  uses image embeddings without an unused caption-embedding request. Expired captures
  can be retried up to three times in ROS, without extending the original deadline or
  accepting a stale result. Library callers retain a one-attempt default.
- A full ingestion queue no longer prevents the controller receiving an arrival image.
  The background queue stays bounded and rejected ingestion retains its cleanup path.
- Object positions retain bounded observed surface samples. Removal/movement checks
  project this surface instead of requiring the supporting furniture inside a bounding
  sphere to disappear. Invalid depth, foreground occlusion, inconsistent samples and
  out-of-view support still prevent a geometric absence claim; visual absence must agree.
  Existing collections upgrade to schema 9; old positions retain the legacy geometry path.
- The deterministic fixture rejects small JPEG color artifacts while retaining actual
  second displays. A dedicated ROS executor keeps test sensor relays running during
  blocking Gazebo world-change service calls.

## Validation

The first targeted run passed all five affected cases with continuous ingestion:
normal arrival and a moved printer matched, removal produced `missing`, occlusion
produced `unobserved`, and a true lookalike produced `ambiguous`. Each trace retained its
critical events and reported zero drops or write errors.

The full Python suite passed 1,209 tests with 95.53% coverage. The offline fault suite
passed 246/246 runs across 82 scenarios. Ruff and mypy passed. The ROS sensor suite
passed 14 checks, including queue saturation during object arrival. The rebuilt image
passed all 11 Gazebo cases with continuous ingestion and no dropped trace events,
plus 23 ROS operator checks. All 100 controlled cancellation samples passed; p99 was
1.914 ms from command receipt to the asynchronous cancel call returning. This is not
a measurement of physical stopping time.

The [validation manifest](validation/gazebo-completion-2026-09-22.json) records each
attempt separately. The final live-model **printer-then-home mission passed** with the
unchanged five-second image freshness limit. The first verification attempt expired;
its result was discarded. A new capture succeeded on attempt two in 4.22 seconds,
and only then did the robot return home. It traveled 3.924 m during the mission, retained
one printer identity, rejected the duplicate command, and lost no trace events.
The final live run made 74 provider requests with reported cost $0.01950393.
The first live startup was blocked before making any provider calls: the long-idle AMCL
estimate had a negative yaw variance (about −3.4e−14). The localization gate remained
strict. The live rerun uses a fresh simulator; stationary endurance remains unqualified.
An overlapping simulator startup also encountered Gazebo clock discovery interference;
subsequent runs use a unique `GZ_PARTITION`. An earlier isolated run was stopped after
its temporary credential expired (HTTP 401), before learning a target or dispatching a
mission. The final successful run validated its credential before launching a fresh
simulator and performed startup, the live mission and cleanup in one bounded lifetime.
The temporary credential files and all task-owned containers were removed after testing.
Deterministic fixture results establish integration behavior;
they do not establish hosted-model recognition accuracy or hardware readiness. One
successful live mission also does not establish repeatability or stationary endurance.
