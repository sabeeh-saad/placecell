# Day 7 readiness review

Reviewed 22 September 2026 against the local Days 4–6 implementation and the
[version 1 qualification contract](production-readiness.md). **Day 7 is complete;
the software remains a prerelease. No release gate is fully satisfied.**

The [validation record](validation/day-07.json) contains results, checksums, blockers
and evidence-reuse decisions. This review changes documentation and sequencing, not
runtime behavior or acceptance targets.

## Evidence checked

- **Fresh planning baseline:** 24/24 draft development cases passed with scripted
  replies. All 24 execution outcomes remain unassessed. This checks the planner,
  reviewer and scorer, not model understanding or navigation accuracy.
- **Fresh fault suite:** 108/108 runs passed: 36 distinct scenarios, three repetitions
  each, including 34 fault scenarios and two successful controls. No failed cases.
  Repetition does not add scenario diversity or test concurrent ROS callbacks.
- **Held-out guard:** scoring the held-out split failed explicitly with “no cases in
  held_out; do not substitute development data.” No score report was created. This
  expected refusal records a data dependency, not a qualified model result.
- **Day 6 reuse:** all 74 recorded source hashes and 28 artifact hashes matched,
  including all 65 current runtime Python files. Reuse the local 911-test result
  (95.83% coverage), separate wheel/source installs, eight real-ROS operator checks
  and Gazebo sensor/direct-Nav2 checks. These were not rerun on Day 7. Hosted CI and
  Python 3.10/3.11 remain unvalidated for the current changes.
- **Earlier evidence:** Days 1–5 were reviewed as historical evidence. Some source
  hashes differ after later work; do not add their execution counts to a current
  qualification total. All 26 artifact hashes listed by Days 2–5 matched. Day 1's
  source-manifest digest matches its sorted-JSON representation, not raw file bytes;
  the audit records the encoding explicitly.
- **Data:** six registered recording manifests still match their development hashes
  and contain 60 frames. The existing inventory records 54 depth frames; media
  validity was not rechecked here. All belong to one office development group.
  There are zero held-out cases and no approved human-reviewed mission/instance labels.

Raw reports are under ignored `simulation/artifacts/day-07/`; compact results and
checksums are in the validation record. Preserve raw reports with candidate artifacts.
Day 7 made no paid calls and started no simulator or endurance workload.

## Gate assessment

1. **Execution — partial:** 36 sequential scenarios pass. At least 100 separately
   specified cases and 1,000 deterministic executions, required integration faults
   and wall-clock cancel-request p99 ≤500 ms are still needed. The 24 planning cases
   and 911 unit tests are not automatically additional execution qualification cases.
2. **Language/grounding/outcomes — unassessed:** no approved held-out campaign or
   complete language-to-grounded-Gazebo qualification. Fixed Nav2 routes do not
   establish the 95% mission completion and appropriate-abstention targets.
3. **Persistence/recovery — partial:** SQLite interruption/storage checks pass;
   multi-store crash recovery, surviving Nav2 goals, backup/restore, corruption and
   upgrade/rollback need end-to-end drills.
4. **Endurance/bounds — unassessed:** no qualifying 24-hour workload. Accepted-work
   age is unspecified; the 20 GiB run ceiling needs enforcement and qualification.
5. **Deployment/operation — partial:** local installs, traces and reconnect snapshots
   have evidence. Diagnostics, recovery runbooks, dependency/boundary review and a
   clean operator drill remain outstanding.
6. **Release evidence — blocked:** no candidate commit containing the current work,
   hosted matrix result or complete gate report.

## Ranked blockers and exit criteria

Ranks are repair order, not claims that every missing test represents a reproduced
defect. Implementation work belongs to this development task. Human label review,
spending and resource decisions belong to the maintainer. Prepare R5 alongside R1–R4.

### R1 — Cancellation and goal ownership · Days 8, 15 · gates 1, 3

**Implementation gap and missing evidence.** The Nav2 adapter keeps its trip in
process memory; startup reports idle even though a crashed controller's goal may
survive. The review did not reproduce a new motion failure. Existing cancellation
cases are sequential and do not measure command latency.

Day 8 defines ownership through submission, cancellation, terminal results and restart.
Exercise stop in every phase, delayed acceptance, stale/duplicate callbacks and real
ROS callback races. Exit: uncertain ownership blocks replacement trips; stale results
cannot advance a sequence; stop receipt to cancel request meets p99 ≤500 ms under a
declared load, independently of acknowledgement or physical stopping. Report unavailable
goal handles separately. Specify startup admission now; prove reconciliation or controlled
refusal for surviving goals in Day 15 crash drills before release.

### R2 — Command identity and retries · Day 9 · gate 1

**Implementation gap.** Version 1 JSON has no caller command ID. Status sequences and
generated request IDs do not deduplicate input. Add a versioned identity/retry contract,
bounded retention and explicit scope/restart behavior. Exit: supported retries cannot
duplicate trips, conflicting ID reuse is rejected, and deliberate repeated visits remain
valid. Document legacy text limits; equal text does not establish duplicate intent.

### R3 — Inputs, sensor provenance and identity · Days 10–12 · gates 1, 2

**Missing evidence; repair defects found by trials.** Extend malformed/overlong and
adversarial model/input cases, full-node RGB-D/TF/localization outages and clock resets,
then moved/removed/occluded/lookalike targets. Existing synchronizer/capture-pose tests
do not reproduce actual TF outages. Exit: invalid provenance cannot admit goals, missing
identity evidence cannot produce success, and failures have interpretation/retrieval/
identity/geometry/execution attribution. Visual quality still requires R5's real labels.

### R4 — Execution qualification matrix · build Days 8–13; close Day 14 · gate 1

**Coverage gap.** Map each scenario to an invariant, injected event, expected dispatch/
state and report. Exit: ≥100 separately specified cases, ≥1,000 deterministic executions,
required ROS integrations and no unresolved critical invariant failure. Repeating 36 cases
or counting unrelated unit tests cannot satisfy diversity. Reserve Day 14 for repairs.

### R5 — Labels, live runner and model campaign · prepare now; Days 18–19 · gate 2

**Data/budget dependency and runner work.** Obtain ≥200 human-reviewed held-out instruction/
perception cases across ≥3 independently configured layouts/sessions; freeze grouped splits
before tuning. Review the 24 draft development labels separately. Prepare ≥50 complete
Gazebo missions covering single goals, chains and clarification, with physical instance IDs.

Build a runner joining actual mission events with evaluator-only ground truth in the
existing trial format. Diagnostic trace export exists; that scoring adapter does not.
Record model/provider versions and dates. Exit: ≥95% feasible completion, ≥95% appropriate
abstention, zero observed wrong-destination motion/wrong-instance acceptance, and all
failures/timeouts, costs and uncertainty retained. Paid trials require a maintainer-agreed
per-run/total budget and stop policy. Unavailable budget/data leaves Gate 2 unassessed.

### R6 — Persistent state and recovery · Days 15–16 · gate 3

**Missing end-to-end evidence.** Test termination around accepted work and multi-store
writes, disk full/write denial, missing images, corruption and interrupted upgrades.
Mission data currently lives inside a disposable container; the documented stopped-process
copy is not a qualified backup/restore procedure. Exit: acknowledged work and image
references survive declared crash points; fresh-instance restore verifies a consistent
backup; upgrade/rollback and corruption refusal are reproducible. Never replay movement;
R1 covers surviving Nav2 goals.

### R7 — Retention, overload and endurance · Days 13, 17, 21–23 · gate 4

**Implementation gap and resource dependency.** Mission context bounds prompt reads but
does not prune persisted events. Object/trace limits are not a global scene/history/
recording disk bound. Add retention preserving live evidence, enforce 20 GiB per run,
and measure/freeze accepted-work age for the selected provider workload. Agree a host
window, eight-CPU quota, 16 GiB memory ceiling and artifact allocation with the maintainer.

Exit: after critical fixes, ≥24 hours of continuous software load plus separately reported
complete Gazebo mission duration, with bounded queues/age, visible drops/busy outcomes,
CPU/RSS/disk, stage latency and recovery evidence meeting every Gate 4 limit. Repeat affected
endurance after scheduling/persistence/retention changes. No run is launched by this review.

### R8 — Diagnostics and operating boundaries · Days 20, 24–26 · gate 5

**Missing operational evidence.** Diagnose invalid map/frame, missing services and unhealthy
storage before motion. Inventory dependencies/container contents, assess relevant findings,
and review command authority, untrusted evidence and secrets. Exit: fresh-environment
startup/shutdown, failure/restore and onboarding drills pass using published instructions;
shared evidence is reviewed for credentials and recording permissions. This review does
not claim a vulnerability scan or physical-robot qualification.

### R9 — Candidate and hosted checks · Days 27–30 · gate 6

**Release evidence gap.** Publish a tested increment when requested, then run the hosted
Python 3.10/3.11/3.12 matrix, clean wheel/source installs and ROS/Gazebo checks. Inspect
required-check settings rather than assuming workflow YAML enforces them. Exit: one
candidate commit, checksummed artifacts, complete gate report, justified evidence reuse
and no unresolved critical execution/data-loss/security issue. Retain prerelease status
if any gate remains unmet on Day 30.

## Revised sequence and dependencies

- **Week 2:** R1, then R2, then R3; expand R4 with each repair. Keep retention on Day 13
  and the Day 14 repair reserve. Define startup ownership now, before Day 15. Start R5
  label/runner preparation alongside this work instead of waiting until Day 18.
- **Week 3:** prove recovery before saturation. Days 18–19 require approved labels and
  model budget; otherwise advance offline recovery/diagnostics and retain the missing
  Gate 2 evidence. Day 21 requires critical execution fixes, enforceable retention,
  a measured accepted-work-age limit and an agreed workload/resource window.
- **Week 4:** repair measured breaches and repeat invalidated checks before candidate
  freeze. Complete R8 and R9. Days 27–30 remain qualification/fix time; no feature expansion
  or reduced thresholds to satisfy a date.

## Reproduce fresh checks

Use new output paths on another run; existing reports are never overwritten.

```bash
.venv/bin/python -m placecell.mission_evaluation validate \
  --dataset evaluation/missions/baseline-v1.json \
  --output simulation/artifacts/day-07/dataset.json
.venv/bin/python -m placecell.mission_evaluation scripted \
  --dataset evaluation/missions/baseline-v1.json \
  --replies evaluation/missions/scripted-replies-v1.json \
  --run-id day-07-offline-01 \
  --save-trials simulation/artifacts/day-07/trials.json \
  --output simulation/artifacts/day-07/baseline.json
.venv/bin/python -m placecell.fault_injection --repeat 3 \
  --output simulation/artifacts/day-07/faults.json
```
