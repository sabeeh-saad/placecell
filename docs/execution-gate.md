# Day 14: deterministic execution checkpoint

The campaign exercises the production planner/reviewer, destination resolver, command
journal, operator routing, mission controller and Nav2 adapter with scripted providers
and action transport. It requires at least 100 separately specified mission cases and
1,000 executions. Passing this checkpoint supplies execution evidence; it does not
qualify live-model language or vision, physical motion, endurance or crash recovery.

## Run the checkpoint

```bash
placecell-check-faults --repeat 10 --execution-gate \
  --output simulation/artifacts/execution-gate/run-01.json
```

Use a new report path. No API key is needed. The default matrix contains 105 scenarios:
102 mission cases and three component cases. Ten repetitions run 1,050 contracts,
of which 1,020 count toward the execution target. Each starts from a fresh controller,
journal, memory store and trace database. All repeats and failures remain in the report.

The two depth-synchronization checks and standalone SQLite interruption check remain
required component contracts, but do not count toward mission case diversity or execution
volume. Planning-only dataset scores and unrelated unit tests also do not count.

`--execution-gate` returns nonzero if any contract fails, fewer than 100 distinct mission
cases were exercised, or fewer than 1,000 mission executions were run. A small selected
suite can pass its contracts and still fail this checkpoint. The JSON `execution_gate`
section reports both denominators and the excluded component runs. Without the flag,
the CLI retains its existing selected-contract exit behavior.

The separate Python 3.12 `execution` CI job runs the complete ten-repeat campaign and
uploads the report even on failure. Repository merge-protection rules remain a separate
setting. Local results do not claim that hosted CI has already run.

## Added execution contracts

Day 14 adds 23 scenarios to the existing 82:

- Three-step chains preserve a deliberate return visit. Middle-leg rejection or abort,
  stop between legs and a full next-step worker queue prevent the third destination.
- A named-place, remembered-target, named-place chain requires fresh visual arrival
  before its final leg. A successful ambiguity choice continues the reviewed sequence.
- Full planning queues and instructions received during planning cannot create a second
  executable mission. Old feedback and result errors cannot alter a replacement trip.
- Expired, deleted or stopped ambiguity choices cannot dispatch their old destination.
- Identical command retries during motion, after completion and after journal reopening
  cannot duplicate a trip. Conflicting payloads preserve the original mission. A stop
  from another map or for another request cannot cancel it. Stop between reservation and
  routing prevents the newly reserved instruction from starting.
- A full context budget blocks dispatch. A deleted latest destination supplies an explicit
  history boundary to planning, excluding an older destination; scripted clarification
  produces no goal. This checks context construction, not model interpretation quality.

The command identity scenarios invoke the production JSON parser and operator routing;
only ROS object construction and receipt publication are replaced by fixtures. They do
not claim DDS coverage. Every scenario includes expected dispatch/state/ownership checks,
and the runner also checks status, dispatch and cancellation trace capture.

## Evidence and remaining blockers

The completed campaign passed all 1,050 contracts: 1,020 mission executions across
102 cases and 30 component checks. The full Python suite passed 1,306 tests with 95.68%
coverage. A flaky redaction test was corrected after a three-letter fake token happened
to match a random session ID; production redaction and navigation behavior were unchanged.

The [Day 14 validation record](validation/day-14.json) identifies exact sources, image,
case catalog, results, exclusions and evidence reuse. The campaign uses a network-disabled
container with eight CPUs and a 16 GiB memory ceiling. Deadlines are virtual; repetitions
check deterministic contracts, not independent perception samples or concurrent scheduling.

Day 13's ROS retention, sensor, operator and cancellation evidence is reusable only where
its exercised runtime sources and scripts still match. Day 14 changes the offline runner,
scenario fixtures and CI; it does not change controller or ROS runtime behavior. The
record explicitly audits those hashes and keeps reused evidence separate from new runs.

There are still zero independent held-out cases in the draft evaluation dataset. The
24 development planning cases are reported separately, and cannot satisfy that dependency.
The five deferred live-model Gazebo missions remain paused. No paid calls are made.

Gate 1 remains partial until post-crash active-goal reconciliation and cancellation under
the supported full RGB-D/provider workload are qualified. Held-out data and live-model
qualification remain Gate 2 dependencies. Day 15 addresses crash/restart recovery; no
automatic replay or recovery behavior is introduced by this checkpoint.
