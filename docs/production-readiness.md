# Production-readiness contract

Status: **version 1 targets frozen; Day 18 development diagnostics added, no gate fully satisfied**,
24 September 2026. The [Day 7 review](readiness-review.md) separates partial software
evidence from unassessed model/endurance quality and ranks the remaining release blockers.
This document defines what must be demonstrated; it does not certify the current alpha
or turn a deadline into a guarantee. Provider spending and the ingestion-age budget remain
explicit open dependencies; this is not a declaration that every resource target is finalized.
Implementation sequencing is in the [30-day roadmap](roadmap.md).

The [Day 1 reference deployment](reference-deployment.md) specifies the initial configuration
and workflow. Its offline/setup evidence does not mark the acceptance gates below as passed.

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

The Day 2 protocol freezes case identities, grouped splits, order/instance scoring,
missing-trial handling and the numerical targets below before live-model tuning. These
are engineering targets, not industry standards or measured results. The initial 24 cases
are assistant-authored drafts; there is no approved held-out dataset yet. Require human
review and independent data before claiming quality. Record any change to the contract and its rationale; do not
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

Qualification minimum: 1,000 deterministic mission/fault executions spanning at least 100
separately specified cases. Repeating a case exercises races; it does not create a new
perception example. Existing tests count where their scenario and assertions meet this contract.

Day 3 adds 36 offline scenarios with controlled provider/action faults and SQLite
interruption. Repetitions are sequential with fixed event ordering; they establish
repeatability, not race coverage. See the [validation record](validation/day-03.json).
These checks leave the qualification gate unassessed, including live ROS scheduling,
actual TF outages, active-goal reconciliation after a process restart and measured stop latency.

Day 4 adds correlated stage traces and read-only mission export, including incomplete
capture indicators and usage fields that remain unknown when unreported. A blocked trace
writer/full queue does not block the tested cancellation callback, and shutdown idle feedback
does not overwrite a completed mission's trace. The real-ROS checks cover admission and an
unavailable action server; they do not qualify moving Gazebo missions or stop latency under load.

Day 5 adds the versioned [operator interface](operator-interface.md), retained mission state
and a read-only snapshot service. Real ROS checks exercise late subscriptions, volatile
command delivery and disabled/enabled node startup; controller tests cover mission/choice
state and rejected commands. Event sequence numbers are not command deduplication keys,
and a fresh idle snapshot is not proof that Nav2 has reconciled a pre-crash goal.

Day 8 fixes concurrent timeout/result and stale-trip callback failures and expands the
offline suite to 46 scenarios. The [cancellation contract](cancellation-ownership.md)
documents real ROS action tests, paused-time deadlines, and 100 cancellation trials with
a synthetic slow observation callback: 1.38 ms p99, 1.70 ms maximum against the 500 ms
target for that workload. Full RGB-D/provider saturation and post-crash reconciliation
remain unqualified. See the [exact sources, image and reports](validation/day-08.json).

Day 9 adds [scoped command identity](command-identity.md), bounded durable reservations,
retry/conflict receipts, targeted stop/choice commands and stop during pending admission.
The [Day 9 record](validation/day-09.json) contains 969 passing tests, 138/138 existing
fault runs and 14 real-ROS operator checks, including explicit retry after reopening the
journal in a replacement controller. The legacy-stop benchmark remains below its 500 ms
target at 2.26 ms p99 for the controlled workload. Journal restart suppression is separate
from Nav2 goal reconciliation; legacy input, replaced/restored journals, physical motion
and candidate release qualification retain their documented limits.

Day 10 repairs eight reproduced model/input boundary failures and adds strict bounded
decision parsing, completion/refusal checks and untrusted-observation data separation.
The [contract](model-input-contracts.md) and [validation record](validation/day-10.json)
describe 60 fault scenarios, malformed-response refusal over ROS and continued cancellation
under blocked providers. Scripted protocol checks do not qualify live-model instruction
following, unsupported-action detection or resistance to misleading observations; those
remain held-out evaluation requirements under Gate 2.

Day 11 adds [sensor and clock contracts](sensor-clock-contracts.md): live RGB/TF and aligned
depth requirements for affected goals, trust generations that survive recovery between
polls, and a latched clock-reset fault. The [Day 11 record](validation/day-11.json) includes
regressions and real DDS sensor/clock input through the production node. Clock-reset
recovery requires confirmed Nav2 quiescence and a fresh collection/keyframe directory;
automatic post-crash ownership recovery, physical sensor timing and endurance remain open.

Day 12 adds [target freshness and identity checks](target-freshness.md), including current
references at dispatch/completion, capture-age limits through provider work, live lookalike
comparisons and explicit failure-stage attribution. The [Day 12 record](validation/day-12.json)
keeps synthetic object cases, deterministic faults and ROS delivery checks separate from
live-model identity accuracy and held-out mission qualification.

Day 14's [execution checkpoint](execution-gate.md) distinguishes mission executions from
component checks and planning-only scores. CI now enforces at least 100 distinct mission
cases and 1,000 executions, alongside all scenario and trace assertions. The
[validation record](validation/day-14.json) records 1,020 passing mission executions across 102
cases, 30 passing component checks, and reused ROS evidence. This closes only the documented deterministic coverage requirement when its
report passes; surviving Nav2 goals after a process crash and cancellation under the full
supported workload remain unqualified. There are no independent held-out mission cases.

Day 15 adds [durable goal ownership and startup reconciliation](crash-recovery.md).
The [validation record](validation/day-15.json) covers repeated controller deaths during
reservation, planning/review, submission, acceptance and terminal persistence. Forty-two
DDS trials preserve an unrelated client's goal, and 18 real Nav2/Gazebo trials confirm
that a surviving goal is canceled without replaying later mission steps. Missing/unknown
ownership remains blocked; an explicit operator-established server reset is the fallback.
This closes the tested process-restart gap. Full supported-workload cancellation and
physical-robot stopping remain unqualified.
An earlier remapped stress run retained uncertain ownership after result confirmation
timed out; later instrumented and deliberate-reply-loss runs passed. This safe blocked
outcome remains recorded, without claiming intermittent transport delays are eliminated.

Release blockers include any observed:

- Goal execution before required plan review, localization or destination checks.
- Duplicate trips caused by replayed events with the same supported request identity.
- Incorrect ordering, silent skipping of a failed goal or substitution of a requested target.
- Late callbacks advancing a canceled/finished mission or crossing mission/map/session scope.
- A success report without the goal's required completion evidence.
- Automatic movement replay after process restart.

Cover cancellation during model work, submission, travel, arrival checks and ambiguity.
Measure wall-clock time from command acceptance to a cancellation request being issued.
The version 1 target is p99 at most 500 ms on the declared reference machine under supported
load, independent of model responsiveness. Measure Nav2 acknowledgement separately against
its configured deadline. Missing acknowledgement must retain uncertain goal ownership,
block a replacement trip and produce an operator-visible failure.

This gate measures software commands and simulated action behavior. It does not establish
physical stopping time or replace an independent robot stop mechanism.

## Gate 2: Language, grounding and mission outcomes

Qualification minimum: 200 held-out labelled instruction/perception cases across at least three
separately configured layouts or recording sessions, plus 50 complete Gazebo missions
covering single goals, chains and clarification. Record the sampling and repetitions;
multiple frames of the same object do not count as independent scene diversity.

Version 1 targets:

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

Day 18's [development evaluation](live-evaluation.md) exercises actual planning/review
models and two vision adapter calls. Its [record](validation/day-18.json) preserves 28/48
exact-label matches across two repetitions of 24 draft cases. Article/alias mismatches and
two repeated clarification disagreements require label/behavior review. There are still
zero independently reviewed held-out cases; vision identity and retrieval comparisons are
unassessed. This preparation does not close Gate 2 or resume the deferred product missions.

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

Day 13 adds transactional memory admission and cleanup limits, a durable sighting-age
cutoff, whole-request conversation pruning and atomic correction-file replacement.
The [retention checks](memory-retention.md) cover write failure, evidence ownership and
clean node recreation without movement replay. They do not qualify abrupt process death,
active Nav2 goal reconciliation, corruption recovery or upgrade/rollback.

Day 15 adds 42 repeated SIGKILL persistence checks, preserving acknowledged state,
accepted-job evidence, failure counts and idempotent recovery across model work,
SQLite transactions, cleanup and LanceDB projection writes. Together with the separate
DDS/Gazebo restart trials, this qualifies the documented process-crash boundaries.
It does not cover power loss, restoring backups, repairing damaged storage, disk exhaustion
or interrupted schema upgrades. Those remaining requirements keep this gate partial.

Day 16 adds [offline verified backup and fresh-instance restore](backup-restore.md),
including image references, failed work and command/context journals. Repeated process
deaths exercise atomic publication and the version-0-to-1 SQLite upgrade. A pre-upgrade
backup is restored and read by the actual previous main revision for offline data
rollback. ROS startup checks verify maintenance exclusion, relocated retrieval data,
unknown navigation ownership and a fresh command session. The
[validation record](validation/day-16.json) separates these results from earlier
development failures. Power-loss durability, damaged-table repair, runtime disk
exhaustion and movement under downgraded controller software remain unqualified;
this gate is still partial.

## Gate 4: Endurance and bounded resources

Run at least one 24-hour continuous software workload after critical fixes, using recorded
observations and scripted provider responses for predictable load and fault injection.
Also run repeated complete Gazebo missions for a declared duration. Report real wall-clock
and simulated time separately. Live-model evaluation is a separate, budgeted workload.

The reference campaign has an eight-CPU container quota, a 16 GiB container memory ceiling,
and at most 20 GiB of per-run data/artifacts. The first two limits were used for Day 1's
bounded Gazebo checks; they are engineering ceilings, not minimum hardware recommendations.
The disk ceiling still needs runtime enforcement/retention qualification. Use the reference
5 Hz simulated camera input and 8–20-second sampling settings, and retain their configuration
hash. Automatic-phase deadlines remain the profile's 90-second planning/lookup and arrival
limits, 10-second Nav2 response limit and 600-second trip limit; human clarification wait is
reported separately. The 500 ms cancellation target is a wall-clock command-latency target.

Measure accepted ingestion-work age with the selected provider workload before freezing
that budget; the model budget is still undecided. Do not use the scripted evaluator's tiny
latency or memory footprint to fill this gap. The endurance gate cannot pass while that
budget is missing. Existing setup evidence does not establish 24-hour resource behavior.

Day 13's [validation record](validation/day-13.json) measures configured row/content
limits under repeated visits and corrections. Scene sightings, persistent conversation
content, corrections, refinement requests and cleanup work now have explicit caps. These
logical bounds do not enforce the 20 GiB filesystem ceiling or establish 24-hour endurance.

Day 17 adds [bounded overload handling](overload.md), durable ingestion cooldowns and
queue accounting. The [validation record](validation/day-17.json) includes production
RGB ingestion and question/maintenance saturation alongside 100 controlled DDS cancellation
trials. Authored poses, scripted providers and a controlled action server limit this evidence;
it does not qualify the complete RGB-D/model workload, physical stopping or 24-hour endurance.

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
