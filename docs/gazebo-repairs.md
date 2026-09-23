# Gazebo repairs before Day 13

The three fixes below address the identity, arrival-capture and trace failures in the
[first checkpoint](gazebo-checkpoint.md). They do not make the full integration run a
passing qualification. The default live mission still refused success when hosted
verification exceeded its five-second image-age limit. The continuous-ingestion matrix
also exposed an arrival verdict invalidated by a concurrent object update.

The [repair manifest](validation/gazebo-repairs-2026-09-22.json) records the exact sources,
images, profiles and results. Local evidence is under
`simulation/artifacts/gazebo-repairs-20260922/`. The earlier checkpoint manifest remains
unchanged historical evidence.

## Implemented repairs

1. **Incomplete captures cannot create another RGB-D object identity.** The ROS tracker
   requires localized, usable geometry when `depth_topic` is configured. Missing depth,
   invalid depth pixels and excessive position uncertainty leave object records unchanged.
   A following usable capture can run without waiting another object scan interval.
   Scene ingestion continues. Explicit RGB-only library pipelines retain their existing
   behavior. No old ambiguous records are deleted, and true positioned lookalikes remain
   separate identities.
2. **Object arrival waits for a complete capture.** An RGB-only observation cannot start
   object verification. The controller stays within its original arrival deadline until
   a fresh aligned-depth observation arrives. This does not extend image lifetime,
   refresh sensor trust, or relax identity and geometry checks.
3. **Progress cannot crowd critical events out of the trace queue.** Nav2 distance
   feedback is limited to five updates per second per trip; acceptance, cancellation and
   results remain immediate. Pending progress coalesces within a mission/request/step/
   stage and retains ordering around critical events and flush barriers. Critical events
   can evict queued progress. Critical-only saturation still drops diagnostics without
   blocking motion callbacks, and increments `dropped_critical_events`.

Thirteen new regressions cover these contracts, including real lookalikes, unchanged
arrival deadlines, feedback floods, flush ordering and critical-only saturation. The
initial ten-test artifact failed before the implementation: five failures exercise
previous behavior, and five require the new geometry-policy option. The final full suite
passes 1,200 tests with 95.68% branch-inclusive coverage. Ruff and mypy pass. The offline
fault runner passes 246/246 executions across 82 scenarios; the rebuilt image passes
23 operator and 13 sensor checks over real DDS with scripted action/model components.
The separate 100-trial cancellation benchmark passes at 1.66 ms p99 (1.78 ms maximum),
measuring command-callback receipt to async cancel API return under synthetic load.
It does not measure physical braking or full Gazebo sensor load.

## Live-provider evidence

The default five-second capture-age run learned one localized printer, rejected a
duplicate command, and traveled 1.952 m to its approach pose. It retained the same single
object identity and accepted a fresh complete arrival capture. Its trace had no drops or
write errors. The hosted detection, embedding and paired comparison took about 5.69
seconds, so the image expired and the mission ended `destination_unverified` / `geometry`.
It did not dispatch the return-home leg. This is a failed mission, not a successful
arrival qualification.

A separate ten-second diagnostic profile on the final image also retained one identity,
skipped an incomplete capture, accepted the next complete capture and lost no trace
events. It traveled 2.000 m, but a paired comparison took 15.19 seconds (18.26 seconds
for the verification span), exceeding even that experimental budget. It also ended
unverified and did not return home. Production defaults remain unchanged. The two
provider-using attempts made 102 requests with $0.02636488 in reported usage; all request
records supplied costs. These are provider-reported amounts, not an account bill.

An earlier attempt timed out before provider calls because stationary AMCL covariance
had collapsed to an invalid estimate. Restarting the isolated simulator restored
readiness. Localization thresholds were not relaxed.

## Continuous Gazebo fault matrix

With background object ingestion active, 7/11 strict expectations passed: the ordered
mission with duplicate suppression, cancellation during motion and planning, malformed
planning output, camera loss, depth loss and a true lookalike. Both sensor-loss cases
confirmed the simulated robot stopped. All eleven traces flushed with zero dropped events,
zero critical drops and zero write errors; dispatched-goal counts matched retained traces.

The single-object trial failed on the concurrent-update guard. Removal returned
`unobserved` instead of the required `missing`. Movement returned `ambiguous` instead
of `matched`. The occlusion trial encountered object ambiguity before dispatch and did
not reach its injection point. All four remain failures in the denominator. In particular,
the depth-admission repair does not eliminate every kind of identity ambiguity or detector
error from continuous ingestion.

The separate final-image `--isolate-arrival` profile passed 0/3 strict cases. The
single-object case still found competing detections; occlusion lost sensor provenance
before dispatch, and the lookalike trial canceled after sensor provenance was lost.
Deferring background object refresh did not make these trials pass. These results are
preserved separately, not counted as successes in the continuous matrix. Their traces
also flushed without dropped events. The harness logs include DDS destruction warnings
at process teardown; they are retained with the reports.

## Remaining limits

- A concurrent background object update can invalidate the final request check. The
  current conservative revision guard is preserved; continuous-ingestion availability
  still needs a design that handles new evidence without accepting stale verdicts.
- Removing a printer did not prove that its old occupied region was visibly empty.
  `unobserved` remains a safe refusal but fails the harness's stricter `missing`
  expectation. Movement likewise requires actual geometric clearance, not merely a
  similar-looking object at another position.
- Hosted provider latency must fit the configured capture-age budget. The longer
  diagnostic profile does not qualify the default five-second profile.
- These are individual trials in one authored office. Deterministic providers use an
  office-specific pixel detector; they do not establish general recognition quality.
  No physical robot stopping, long-duration operation or crash reconciliation is qualified.

Day 13 remains the memory/history-bounds work in the [roadmap](roadmap.md). Integration
failures remain visible and should not be treated as passing release evidence.

The task's two simulator containers and temporary credential file were removed after
testing. Source files and test artifacts were scanned for the credential without printing
it. No unrelated deployment or physical robot was changed. The repairs remain local;
they have not been committed or pushed.
