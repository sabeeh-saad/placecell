# Budgeted live-model evaluation

Day 18 preparation adds an executable planning/review runner and a development camera
protocol check. The [validation record](validation/day-18.json) retains actual model
outcomes and spending. **Held-out qualification remains pending:** the current dataset
contains 24 draft development cases, with no independently reviewed held-out cases.
The maintainer authorized development checks while leaving that requirement open.

## Run a development check

Prepare a private credential file with mode `0600`, outside the repository. Pass its path,
never the key itself, to the command. The runner does not copy it into artifacts.

```bash
python -m placecell.live_evaluation preflight \
  --dataset evaluation/missions/baseline-v1.json --split development \
  --output /tmp/placecell-evaluation-preflight

python -m placecell.live_evaluation run \
  --dataset evaluation/missions/baseline-v1.json --split development \
  --output /tmp/placecell-evaluation-run \
  --key-file /private/path/openrouter-key \
  --max-usd 0.50 --max-requests 1600 --max-seconds 1200 --repeats 2
```

An installed distribution also provides `placecell-evaluate-live`. Each output directory
must be new. Paid execution requires explicit cost and request limits. The default split
is `held_out`; it rejects missing cases and draft labels before reading the credential or
making a request. It never substitutes development data. A `human_reviewed` field is an
annotation attestation, not automated proof of independence.

The runner invokes the production `MissionPlanner` and `PlanReviewAgent`, with eight
destinations maximum, temperature zero, 2,048 completion tokens, 30-second request timeouts
and one client HTTP attempt. The requested model for each stage defaults to
`google/gemini-2.5-flash`; `--model` and `--review-model` are explicit overrides.
Model IDs are hosted aliases, not immutable weight snapshots. Response IDs, returned
model/provider names, token counts, timestamps and runtime source hashes are retained.

Only the instruction and supplied historical context reach the planner. Scenario text,
categories, expected aliases, physical target IDs and labels remain with the scorer.
Each repetition starts fresh model conversations. The reviewer sees the proposed plan
through the production review path; it is called only when the planner proposes movement.
No coordinates, robot commands or success-at-arrival claims are produced by this runner.

## Spending and interrupted runs

All stages share one sequential transport and one flushed request ledger. Before each
HTTP request, it records a spending reservation. Routing price ceilings, bounded text and
images, completion limits and conservative token allowances determine the reservation.
It refuses a request if the remaining budget cannot cover that reservation. Completed
requests replace reservations with valid provider-reported costs.

Unknown/malformed cost, transport failure, any non-200 status, an exceeded reservation,
or a failed accounting write stops further requests. Unknown cost remains `null`, with
the reservation retained; it never becomes zero. The request limit and wall-clock deadline
also stop admission. There is no automatic retry or resume, and a prior ledger cannot be
overwritten. Creating another output directory does not reset the authorized checkpoint
budget: subtract prior charges and unresolved reservations when setting its limits.
The runner does not aggregate spending across separate output directories.

Price ceilings and reservations limit client admission; they cannot guarantee a remote
provider's billing. The response's `usage.cost` supplies reported charges, as described
by [OpenRouter usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting).
The runner uses documented
[provider price filters](https://openrouter.ai/docs/guides/routing/provider-selection#max-price).
It does not change an API key's account-level credit limit.

`requests.jsonl` contains reservation and completion events, correlated by sequence,
trial and stage. It omits authentication headers, prompts, image payloads and raw errors.
An interrupted request with only a reservation is unresolved billing, not a free request.
The live run recorded on Day 18 used 74 model requests and $0.0169171 in reported cost,
with no unknown-cost requests. Its temporary credential file was removed afterward.

## Reports and interpretation

`trials-01.json`, `report-01.json` and subsequent repetitions use the existing
[mission scoring contract](mission-evaluation.md). They are checkpointed after every
attempt. An empty report is written before calls begin, so interruption never erases
the intended denominator. Missing trials count as failures. `attempted_trials` and
`unrun_trials` distinguish actual attempts from denominator entries; the scorer's
`assessed` count alone is not a model-call count.

`summary.json` records completion separately from matching the labels. Exit code 1 means
a mismatch, an incomplete/budget-stopped run or a failed optional camera check; inspect
the reports before treating it as a software exception. Recorded errors contain types,
not raw provider messages. Model outputs and label files must remain unchanged when
reviewing a run. Label corrections require independent review, a new dataset hash and
an explanation; they must not silently improve the published score.

The first live development run completed two repetitions of all 24 cases. Each scored
14/24 against the frozen draft aliases. Eight failures per repetition differed only by
the leading article `the`, which the scorer does not normalize. Two decisions disagreed
with the labels: `hypothetical` returned `clarify` instead of `reject`, and
`known-reference` returned `clarify` instead of the expected visit. These are review items,
not 20 demonstrated navigation failures. Both decision disagreements withheld movement.
No labels or prompts were adjusted after observing the results, and no alternative score
is substituted for the recorded 28/48 total.

Repetitions are correlated and do not increase independent case diversity. These draft,
development-only scores are not a generalization estimate or release acceptance result.
The two runs' per-case decisions and destinations were identical; per-run latencies and
all outcomes remain in their individual reports.

## Vision, retrieval and the remaining qualification work

For development only, `--recording /path/to/observations.jsonl` adds two calls on the first
saved frame: the production captioner and the scene verifier's `printer` query. Frame and
recording hashes identify the input. A valid negative or uncertain verdict can pass this
protocol check; no expected visual answer or independent accuracy label is invented.
`camera-smoke.json` explicitly identifies that limit.

The Day 18 frame returned a caption and a valid `matched` verdict. That proves the adapters
worked with the provider on that input; it does not establish object identity accuracy,
lookalike handling, depth geometry or Gazebo mission completion.

Human-labelled image/instance cases and channel comparisons remain pending. Use the
existing [RGB-D object evaluator](object-evaluation.md) and retrieval evaluator with reviewed
labels when available; the new budget transport currently covers OpenRouter chat/vision
requests, not embedding requests. Do not use the older evaluation CLIs under this spending
authorization: their separate accounting does not enforce this run's shared cap.

Gate 2 still needs 200 held-out labelled instruction/perception cases across at least three
independent layouts or recording sessions, and 50 complete Gazebo missions. Reusing these
office frames, changing their group names, or marking generated labels as human-reviewed
cannot supply that evidence. The five deferred live product missions remain paused.
