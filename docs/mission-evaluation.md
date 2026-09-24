# Evaluate navigation missions

The mission evaluator keeps interpretation and execution separate. It scores an ordered
plan against explicit destination aliases, then scores any supplied execution trace against
physical target IDs and required confirmations. It does not send robot commands or call
model services. The initial dataset contains 24 **assistant-authored, draft synthetic cases**;
they are not human-reviewed visual ground truth or a model-accuracy benchmark.

The separate [budgeted live-model runner](live-evaluation.md) now executes actual planning
and review calls with request/cost limits. Its development reports use this scoring contract;
grounded execution and independently labelled visual/retrieval quality remain unassessed.

## Run the offline baseline

From a checkout with the development dependencies installed:

```bash
mkdir -p simulation/artifacts/day-02
python -m placecell.mission_evaluation validate \
  --dataset evaluation/missions/baseline-v1.json \
  --output simulation/artifacts/day-02/dataset.json
python -m placecell.mission_evaluation scripted \
  --dataset evaluation/missions/baseline-v1.json \
  --replies evaluation/missions/scripted-replies-v1.json \
  --run-id day-02-offline-01 \
  --save-trials simulation/artifacts/day-02/trials.json \
  --output simulation/artifacts/day-02/report.json
```

Use the active environment's Python, for example `.venv/bin/python`. A fresh installation
also exposes `placecell-evaluate-missions` with the same arguments. Output files must be
new; choose a different name for another run. Failed scored cases produce a report and
exit code 1. A zero exit code means the requested check passed, not that release gates passed.

The `scripted` mode runs the actual `MissionPlanner` and `PlanReviewAgent`. Only the
instruction and historical context enter their prompts. Replies come from a separate
fixture file; expected labels are read only by the scorer. No language understanding,
visual recognition, robot movement or API availability is measured by this mode.
Execution remains explicitly `unassessed`, even when all scripted plans match their labels.
The initial cases are deliberately easy to inspect, not statistically representative.

The baseline covers single destinations, chains, repeated visits, distinguishing attributes,
lookalikes, unavailable/removed targets, negation, questions, quoted/hypothetical instructions,
unsupported actions and follow-up references. A plan may correctly say `ready` for “go to
the cupboard” while grounding must later return `not_found`: the planning agent has not
been given an inventory of the scene.

## Label and split cases before evaluating models

`evaluation/missions/baseline-v1.json` is the versioned label contract. Each case has a
stable ID, category, group, instruction, context and scenario assumptions. The expected
fields are:

- `decisions`: accepted planning decisions. A ready plan cannot be interchangeable with
  rejection. Unsupported requests may allow either clarification or rejection.
- `destinations`: ordered lists of explicitly permitted textual aliases, one list per visit.
  Repeated visits stay repeated. Matching ignores case and whitespace only; it is not an
  LLM judge or a fuzzy semantic match. Unexpected phrasing needs independent label review.
- `outcomes`: permitted final execution outcomes, separate from the planning decision.
- `target_ids`: physical target IDs in the permitted dispatch order. For a target that
  cannot be grounded, this can be empty even though the planned description is nonempty.
- `visual_required`: whether success requires confirmations of every dispatched target.
  Named-place navigation can be scored separately without claiming visual identity.

Scenario assumptions and expected target IDs are evaluator information. Do not add them to
the model input as a shortcut. They become visual labels only after an annotator checks the
actual images and identities. Review the draft instructions and expected decisions as well;
change `label_status` to `human_reviewed` only after real review, recording who reviewed what
in `annotation_notes`. Do not alter aliases after seeing results merely to improve a score.
A label correction creates a new dataset hash and requires rescoring and an explanation.

Splits belong to groups, never individual frames. The loader rejects a layout appearing
in both splits, a recording digest registered twice, and identical instruction/context
pairs crossing splits. Assign each real layout a stable ID; the validator cannot detect
leakage hidden by renamed layout IDs or re-encoded images. The six existing recordings
are all registered under **one development group, `gazebo-office-v1`**. They contain 60
images, 54 with depth; none currently has approved mission-case labels.

There are **zero held-out cases** in the initial dataset. Requesting `--split held_out`
fails explicitly. It never substitutes development cases. Acquire independently labelled
layouts/sessions, register their groups, and freeze the split before tuning. The existing
[object-label format](object-evaluation.md) remains the way to label boxes, visibility and
physical instance continuity in saved RGB-D observations.

## Import actual runner outcomes

The live planning runner saves this versioned trial format directly. Other runners,
including a future labelled Gazebo adapter, can save matching outcomes and use:

```bash
python -m placecell.mission_evaluation score \
  --dataset evaluation/missions/baseline-v1.json \
  --trials /path/to/actual-trials.json \
  --split development \
  --output /path/to/new-scored-report.json
```

The trial document identifies `schema_version: 1`, the exact `dataset_sha256`, a unique
`run_id`, `runner` (`scripted`, `live_model` or `gazebo_live_model`), a `configuration`
object and a list of `trials`. Record model/provider versions, software revision,
configuration hashes, evaluation time, scene/seed, hardware and evidence paths in that
configuration. The importer trusts runner-supplied evidence and does not attest that a
declared live model or robot actually ran. This command is an importer, not a live runner.

Each trial uses this shape (the values here illustrate one completed case):

```json
{
  "case_id": "single",
  "plan": {
    "status": "ok",
    "decision": "ready",
    "destinations": ["printer"],
    "latency_ms": 1200.0
  },
  "execution": {
    "outcome": "succeeded",
    "dispatched_targets": ["printer-a"],
    "confirmed_targets": ["printer-a"],
    "latency_ms": 18000.0
  },
  "cost_usd": null
}
```

Use `plan.status: error` or `timeout`, `decision: null` and empty destinations when
planning fails; retain elapsed time and optionally `error_type`. Do not omit that case.
Use `execution: null` only when execution was not assessed. For a `gazebo_live_model` run,
missing execution is a failure. `dispatched_targets` records every dispatched physical
target identity, including wrong goals, duplicates and visits before a later failure.
It does not prove physical motion happened. `confirmed_targets` records actual successful
completion evidence in order, not intended destinations. Day 4 provides
[diagnostic mission trace export](mission-tracing.md). A live scoring adapter must still
join those events with evaluator-only physical target/instance labels and emit this trial
format; the diagnostic exporter does not provide that ground truth or automatically produce
qualified evaluation trials. This is tracked in the [Day 7 review](readiness-review.md).

The scorer rejects duplicate or unexpected trial IDs, labels with the wrong hash, nonfinite
or negative timings/costs, and inconsistent plan records. Missing trial rows remain failed
cases in the plan denominator. Wrong dispatch order, extra visits, dispatch after a rejected
plan, phantom confirmations and success without required confirmations are failures.
Stopping at the correct first destination before a later failure is not a wrong-destination
dispatch, but it is not a completed successful mission either.

## Interpret the report

Each stage reports eligible, assessed, passed, failed and unassessed counts. `pass_rate`
uses all eligible cases as the denominator and is null when none were assessed. It is
not the rate on a hand-picked subset of completed requests. Feasible mission completion
and appropriate abstention have separate counts; refusing every request cannot earn a
good mission-completion score. Categories and individual failed cases remain visible.

Latency includes failed and timed-out assessed cases and uses nearest-rank p50/p95/p99.
Small-sample percentiles are descriptive, not proof of an SLA. Planning and execution
timings are separate. Human response time, if included by an external runner, must be
identified explicitly. Unknown cost is null; `known_cost_usd` is only the known subtotal.
No requests or billing estimates are produced by the offline runner.

All reports state that release qualification is unassessed. In particular, zero observed
wrong dispatches with zero execution trials supplies no evidence about navigation quality.
The [Day 2 baseline record](validation/day-02.json) records the actual scope, source hashes,
results and outstanding data/model/resource dependencies for this checkout.
