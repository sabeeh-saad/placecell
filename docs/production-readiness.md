# Production-readiness contract

Status: **proposed and unassessed**, 18 September 2026. This document defines what must
be demonstrated; it does not certify the current alpha or turn a deadline into a guarantee.
Implementation sequencing is in the [30-day roadmap](roadmap.md).

## Release claim and supported profile

The one-month objective is operationally reliable PlaceCell software for one documented
reference configuration, qualified using Gazebo and saved recordings. Physical-robot
qualification remains outstanding and must be prominent in the release notes.

The reference profile is a single robot and controller process, one aligned RGB-D stream,
a known versioned map, external localization, ROS 2 Jazzy/Nav2 on the bundled Ubuntu
24.04-based Docker path, and Python 3.12 for the ROS process. The core test matrix covers
Python 3.10–3.12. Freeze the actual package versions, container digest, model configuration,
compute resources, input rates and retention settings used for qualification.

Supported user behavior is the existing text/final-transcript interface for single or
ordered multi-goal visits, clarification/selection, cancellation and scoped follow-up
context. Memory goals require candidate and fresh arrival checks. Named places retain
Nav2 completion semantics and must be reported separately. Camera ingestion continues
independently of a mission.

Live planning and perception must be evaluated with the actual supported provider setup.
Provider interfaces alone do not qualify every compatible model. Hosted-model versions
that cannot be fixed require recorded evaluation dates/configurations and a defined
requalification policy when behavior changes.

Pause/resume, plan editing, reference-image instructions, extra autonomous recovery,
map-free operation, manipulation and multi-robot coordination are outside this month's
qualification scope. Remote/public command endpoints and multi-user authorization are
also outside the isolated local ROS reference deployment.

## Evidence rules

Freeze the evaluation protocol and final acceptance targets on day 2, before tuning against
the held-out set. The numbers below are initial engineering targets for review, not industry
standards or measured results. Record any change to the contract and its rationale; do not
lower a threshold at release time to convert a failure into a pass.

Each report must identify its commit, package and model versions, configuration, scenario,
labels, run/seed, hardware, timestamps and evidence artifacts. Separate scripted-provider
software checks, saved-image model evaluations and complete Gazebo navigation. State
what was tested with live models and what was not. Tests using the same scene/configuration
are not independent evidence of generalization.

Keep failures, abstentions and timeouts in the report. Count all eligible trials and show
per-scenario results; refusal on a feasible unambiguous request is not a successful mission.
Ground-truth simulator labels may score a trial but must not leak into robot decisions.

## Gate 1: Execution correctness

Proposed minimum: 1,000 deterministic mission/fault executions spanning at least 100
separately specified cases. Repeating a case exercises races; it does not create a new
perception example. Existing tests count where their scenario and assertions meet this contract.

Release blockers include any observed:

- Goal execution before required plan review, localization or destination checks.
- Duplicate trips caused by replayed events with the same supported request identity.
- Incorrect ordering, silent skipping of a failed goal or substitution of a requested target.
- Late callbacks advancing a canceled/finished mission or crossing mission/map/session scope.
- A success report without the goal's required completion evidence.
- Automatic movement replay after process restart.

Cover cancellation during model work, submission, travel, arrival checks and ambiguity.
Measure wall-clock time from command acceptance to a cancellation request being issued.
The initial target is p99 at most 500 ms on the declared reference machine under supported
load, independent of model responsiveness. Measure Nav2 acknowledgement separately against
its configured deadline. Missing acknowledgement must retain uncertain goal ownership,
block a replacement trip and produce an operator-visible failure.

This gate measures software commands and simulated action behavior. It does not establish
physical stopping time or replace an independent robot stop mechanism.

## Gate 2: Language, grounding and mission outcomes

Proposed minimum: 200 held-out labelled instruction/perception cases across at least three
separately configured layouts or recording sessions, plus 50 complete Gazebo missions
covering single goals, chains and clarification. Record the sampling and repetitions;
multiple frames of the same object do not count as independent scene diversity.

Initial targets:

- At least 95% complete success on feasible, unambiguous supported missions, with the full
  requested order and all required arrival checks. Report counts and per-scenario rates.
- At least 95% appropriate clarification or rejection on the labelled ambiguous,
  unsupported or unavailable-target cases. Report unnecessary refusal separately.
- Zero observed wrong-destination motion or wrong-instance acceptance in the qualification
  set; any such event is investigated as a blocker even if aggregate success remains high.

Include lookalikes, changed viewpoints, moved/removed/occluded targets, negation, questions,
references to prior outcomes and malformed model replies. Keep named-place and visually
verified memory-goal results separate. Reuse human-labelled data for caption-only,
image-only and combined retrieval comparisons without tuning on the held-out split.

Report uncertainty and sample size. Zero observed critical errors in a finite simulated
set is not a guarantee of zero real-world error. If the live-model budget or labelled data
is unavailable, this gate remains unassessed; scripted answers cannot satisfy it.

## Gate 3: Persistence and recovery

Fault trials must cover process termination around accepted writes/jobs, provider failures,
write denial, disk exhaustion, missing evidence, damaged storage and interrupted upgrades.

- Process-restart trials preserve acknowledged committed state and recover accepted jobs
  according to the documented retry policy, without replaying movement.
- Storage failures are surfaced; no failed write is reported as a successful persistence
  operation. An affected mission cannot silently continue with missing required context.
- Backups include authoritative state and referenced images at a consistent point. Restore
  succeeds in a fresh instance and verifies record/evidence integrity.
- Corruption is detected and produces a controlled refusal/repair workflow. Recovery from a
  backup states its recovery point; this is not a claim that corruption cannot lose data.
- The documented upgrade and rollback path is demonstrated. Rollback may require restoring
  a compatible backup rather than running an old binary against a newer schema.

## Gate 4: Endurance and bounded resources

Run at least one 24-hour continuous software workload after critical fixes, using recorded
observations and scripted provider responses for predictable load and fault injection.
Also run repeated complete Gazebo missions for a declared duration. Report real wall-clock
and simulated time separately. Live-model evaluation is a separate, budgeted workload.

Pass only if:

- There are no unexplained process exits, deadlocks or permanently stuck missions.
- Queue depths remain within configured bounds; drops, busy responses and exhausted work
  are visible, and accepted-work age meets the documented budget under supported load.
- RSS, disk use and retained evidence remain within the limits frozen on day 2 for a specified
  workload. After retention stabilizes, unexplained sustained growth is a failure.
- Provider outages, overload, restarts and reconnects preserve execution and persistence gates.
- Stage latency, CPU, memory, disk and model usage are recorded, with breaches and recovery
  visible rather than omitted from averages.

A shorter soak does not satisfy the 24-hour target. Changes affecting scheduling,
retention, persistence or recovery require repeating affected endurance checks.

## Gate 5: Deployment and operation

- A clean installation from the candidate wheel/source and the reference container succeeds.
- Health/readiness checks explain invalid configuration, unavailable inputs/services and
  unhealthy storage before goals are accepted.
- A newly connected client can obtain current mission state; logs and evidence correlate
  with the same mission/step identities as live status.
- Commands, status payloads, configuration defaults, migration policy and supported model
  capabilities are documented and versioned with compatibility checks.
- A documented startup, shutdown, backup, restore and troubleshooting drill is reproducible.
- Input/model output validation, secret handling and the local command-authority boundary
  have been reviewed. Published diagnostic bundles contain no credentials or recordings
  that lack permission to share.
- Dependencies and container contents are recorded; relevant vulnerability findings are
  assessed and resolved or explicitly justified for the supported deployment.

## Gate 6: Release evidence and decision

The candidate commit must pass its Python matrix, applicable ROS/Gazebo integration,
package checks and gates above. Preserve links and checksums for the tested artifacts.
Earlier evidence can be reused only with a recorded explanation of why subsequent changes
do not invalidate it. Maintain an explicit list of unresolved defects and exclusions.

There must be no unresolved critical execution, silent data-loss or supported-deployment
security issue. Other known issues need documented impact and a supported workaround.
Missing evidence is not a pass.

If the gates pass, the release claim must name the qualified software configuration and
the simulation/replay evidence, while stating that physical-robot qualification remains
outstanding. If gates fail or remain unassessed on day 30, retain a prerelease designation
and publish the blockers. Do not relabel the build as generally production-ready to meet
the date.

Hardware qualification later needs measurements of actual sensor calibration/timing,
localization, motion/stop behavior, onboard resource and power constraints, environmental
variation and the robot's independent safety integration. This contract cannot substitute
for those deployment-specific trials.
