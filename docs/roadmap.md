# PlaceCell: 30-day production-readiness plan

Baseline: [v0.1.0-alpha.1](https://github.com/sabeeh-saad/placecell/releases/tag/v0.1.0-alpha.1).
Updated 18 September 2026 after the maintainer prioritized production readiness within
one month. This replaces the previous feature-expansion plan.

Validation resources: **Gazebo and saved recordings; no physical robot**. The objective is
to qualify the existing software for a narrow, documented reference deployment. A one-month
deadline is the target, not evidence that the software meets its release criteria.

The [production-readiness contract](production-readiness.md) defines scope, proposed
acceptance targets, evidence requirements and release blockers. It must remain explicit
that physical-robot behavior is unqualified. Simulation results cannot establish braking,
contact behavior, physical sensor performance or hardware reliability.

Day 1, 21 September: the [mission reference deployment](reference-deployment.md) now defines
the opt-in profile, operator workflow and available local recordings. Validation is tracked
in the [Day 1 record](validation/day-01.json). The monthly model budget is undecided;
compute/storage allowances are provisional until Day 2 measurements. No live-model
mission quality or production qualification is claimed by these setup checks.

Day 2, 21 September: a versioned draft dataset, split validation, scripted planning baseline
and mission-outcome scorer are described in the [evaluation guide](mission-evaluation.md).
The [Day 2 record](validation/day-02.json) distinguishes the offline software baseline from
unassessed model/execution quality. Human label review, independent held-out data, model
spending and a measured ingestion-age budget remain dependencies; no paid run is scheduled.

Day 3, 21 September: the [offline fault runner](fault-injection.md) now connects the actual
mission controller and Nav2 adapter to controlled providers/transport. It reproduced and
fixed a late Nav2 success advancing a mission after transport timeout. The
[Day 3 record](validation/day-03.json) records 36 scenarios and validation evidence.
These sequential software checks do not satisfy live ROS/Gazebo fault coverage, concurrent
race testing, physical stopping measurements or post-crash active-goal reconciliation.

## Scope for this month

The supported reference is a single robot, one aligned RGB-D stream, trusted localization
in a known versioned map, ROS 2 Jazzy/Nav2 and the bundled Linux/Docker simulation path.
The ROS reference uses Python 3.12; the Python core remains tested on 3.10–3.12. Record the
actual dependency versions, image digests, compute profile and model configuration used
for qualification. Broader combinations remain outside the qualified profile until tested.

Keep the existing planning/review agents, single and ordered multi-goal instructions,
visual/object memory, candidate clarification, arrival verification and ROS feedback.
Concentrate on making those behaviors reproducible, bounded and observable.

Defer reference-image instructions, additional agent roles, room/relational language,
new autonomous recovery policies, pause/resume, conversational plan editing, map-free
navigation and multiple robots. Reconsider them after the readiness gates are met.
Operational additions such as diagnostics and a mission-state snapshot are in scope.

## Week 1: Define the supported deployment and expose failures

**Day 1 — Freeze the deployment contract.** Record the reference OS/ROS/Nav2/Python stack,
model adapters, map/camera assumptions, compute budget and operator workflow. Inventory
available recordings and model access. Done when an unfamiliar developer can distinguish
supported, experimental and untested configurations.

**Day 2 — Freeze evaluation and acceptance.** Label normal missions, chains, lookalikes,
missing/moved targets, invalid requests and follow-up references. Separate tuning and
held-out sessions/layouts. Record baseline results and finalize the proposed thresholds
before optimization. Done when success, uncertainty and failure have unambiguous denominators.

**Day 3 — Build the fault-injection harness.** Exercise delayed/malformed model responses,
missing sensors, stale TF/localization, Nav2 rejection and lost results, storage errors
and process interruption. Reuse existing regression tests. Done when each injected fault
has a reproducible setup, expected state and saved outcome.

**Day 4 — Add complete mission traces.** Correlate input, reviewed plan, retrieval candidates,
verification, Nav2 requests/results and terminal state. Capture stage timings and model
usage without credentials. Done when a failed mission can be explained from its artifact.

**Day 5 — Stabilize the operator contract.** Specify versioned command/status payloads and
provide a current mission snapshot for reconnecting clients while preserving the existing
status stream. Done when a new subscriber can recover current state without replaying commands.

**Day 6 — Make CI cover the supported path.** Ensure relevant planner, memory, controller
and ROS changes trigger appropriate integration checks, not just depth-file changes.
Add package installation and offline smoke checks. Keep paid-model trials separate from
ordinary CI. Done when release checks cannot silently omit an affected integration.

**Day 7 — Review readiness gaps.** Run the baseline and fault harness and rank blockers.
Keep evidence for every failure. Done when weeks 2–4 have a concrete repair order, and
missing data or compute/model budgets are recorded as dependencies rather than assumed.

## Week 2: Harden execution and destination grounding

**Day 8 — Verify cancellation and goal ownership.** Cover stop during planning, submission,
navigation and arrival checks, including missing cancellation acknowledgements. Done when
stale callbacks cannot restart movement or release uncertain Nav2 ownership incorrectly.
Measure cancel-request latency independently of physical stopping.

**Day 9 — Handle duplicates and late events.** Introduce or validate stable event IDs at
supported input boundaries, scoped deduplication and terminal-state rules. Preserve deliberate
repeated visits. Done when replayed transport messages cannot duplicate a trip and the
limits of legacy text-only input are documented.

**Day 10 — Harden model/input contracts.** Exercise invalid fields, overlong inputs,
unsupported actions, untrusted text in observations and provider errors. Keep cancellation
independent of model completion. Done when every rejected result has a visible reason
and cannot create an executable destination.

**Day 11 — Exercise sensor and clock faults.** Test lost camera/depth, mismatched timestamps,
stale/invalid localization, delayed TF and simulated-clock resets. Done when invalid
provenance blocks new goals and interrupted missions retain explicit, correct state.

**Day 12 — Verify target freshness and identity.** Use moved, removed, occluded and lookalike
targets to harden retrieval, stale-choice checks and fresh arrival verification. Done when
failures are attributed to retrieval, identity, geometry or execution rather than hidden
in one success score. Keep uncertainty explicit.

**Day 13 — Bound persistent memory and conversation history.** Validate queue limits,
retention, evidence references and history pruning under repeated visits and corrections.
Done when configured limits have measurable effects and deleted context cannot silently
resolve a follow-up to the wrong destination.

**Day 14 — Run the execution gate.** Repeat the fault matrix, deterministic mission runs and
held-out cases affected by fixes. Done when all critical execution invariants pass or their
failures remain explicit release blockers. Reserve unfinished repairs before new work.

## Week 3: Prove recovery and operational behavior

**Day 15 — Test crash/restart recovery.** Interrupt processes around persistence and model
work. Verify acknowledged state, retained evidence, job recovery and no automatic movement
replay. Done when repeated crash points have machine-readable recovery reports.

**Day 16 — Test backup, restore and upgrades.** Back up consistent state plus images, restore
into a fresh instance, and exercise the supported migration/rollback path. Corruption must
be detected; a backup has a documented recovery point. Done when an operator can recover
without editing internal tables or accepting silent data loss.

**Day 17 — Exercise saturation and backpressure.** Feed more observations/requests than the
configured capacity and interrupt providers. Done when queues stay bounded, busy/drop
outcomes are visible, cancellation remains responsive and recovery does not create a retry storm.

**Day 18 — Run held-out real-model evaluations.** Test the pinned planning/review/vision
configuration against independent labels and compare caption/image/combined retrieval.
Record versions, repetitions, uncertainty, cost and latency. Done when results measure real
model behavior; scripted responses cannot substitute for this gate. Calls require a budget.

**Day 19 — Run multi-layout Gazebo missions.** Vary object placements, appearances and routes
across held-out layouts, including multi-goal and ambiguous missions. Simulator labels are
available to the evaluator, not the robot's perception pipeline. Done when aggregate results
and per-scenario failures are reproducible from recorded seeds/configuration.

**Day 20 — Add operational diagnostics.** Provide a proposed `placecell doctor` interface,
health/readiness information and actionable errors for the supported configuration. Done when
wrong map/frame settings, missing services and unhealthy storage are diagnosed before a trip.
A read-only report is enough; a full dashboard is not a release requirement.

**Day 21 — Start endurance qualification.** Launch a documented 24-hour software workload
with fixed input rates, retention limits and resource budgets. Combine a longer replay
workload with repeated Gazebo missions, reporting their durations separately. Use scripted
providers for continuous fault/load checks; retain separate live-provider trials. Done when
resource, error, timing and recovery evidence is being captured for later review.

## Week 4: Fix evidence-backed gaps and qualify a release

**Day 22 — Inspect endurance results.** Review memory/disk growth, queue age, dropped work,
latency, timeouts and recovery. Done when each breach has a reproduction and an owner;
a completed process alone does not mean the endurance test passed.

**Day 23 — Repair the measured bottleneck.** Fix the highest-impact resource or responsiveness
failure, then rerun the affected workload. Optimize based on stage measurements, preserving
retrieval and verification quality. Done when the improvement and its tradeoffs are quantified.

**Day 24 — Make deployment reproducible.** Record supported dependency constraints, image
and model identifiers, configuration validation and startup/shutdown behavior. Verify wheel,
source package and reference container. Done when a fresh environment reproduces the same
supported setup without relying on the maintainer's workspace.

**Day 25 — Verify the command and data boundaries.** Review trusted command entry points,
configuration/secrets handling, model output validation, dependency findings and diagnostic
artifacts. The reference has an isolated local ROS graph; remotely exposed or multi-user
control needs separate qualification. Done when untrusted evidence cannot issue actions
and shared reports do not expose credentials or unapproved recordings.

**Day 26 — Run an operator onboarding drill.** Follow only the published setup and runbooks
in a clean environment, including a failed mission, backup restore and diagnosis. Invite
another developer if available. Done when missing steps and unclear errors are fixed and
the supported deployment can be operated from the documentation.

**Day 27 — Freeze a release candidate.** Stop feature additions, classify remaining defects
and build candidate artifacts from one commit. Done when every readiness gate has evidence
or a named blocker, and no critical correctness/data-loss issue is open.

**Day 28 — Run candidate acceptance.** Execute the Python matrix, affected ROS/Gazebo checks,
held-out model trials, installation checks and supported migration tests on the candidate.
Reuse earlier evidence only where the relevant code/configuration is unchanged and record
that provenance. Done when the candidate has a complete qualification report.

**Day 29 — Repair and requalify.** Use this buffer for failures; repeat endurance or other
checks invalidated by fixes. Update runbooks and known limitations. Done when the report
matches the actual candidate, not an earlier passing commit.

**Day 30 — Make the release decision from evidence.** Publish the candidate as a qualified
release only if its declared gates pass. Otherwise publish or retain an explicitly marked
prerelease with the blockers and next steps. State exactly which software configuration was
qualified in simulation/replay and that hardware qualification is outstanding.

## Working agreement

Work together in this task: pick a measurable outcome, explain the change, implement it,
run relevant checks, review the result and push a tested increment. Keep unfinished work
on a branch. The day numbers are an allocation of effort, not a commitment to manufacture
one feature or commit daily. Reassign days when a release blocker needs more work.

This plan does not create unattended automations, authorize spending on model providers
or assume access to a physical robot. Live evaluation and endurance workloads need an agreed
resource budget. Published performance claims must identify their evaluated environment.
