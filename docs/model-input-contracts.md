# Model and input contracts

Day 10 hardens the boundaries between operator input, model responses and executable
navigation. Eight new regression cases initially failed, including malformed/refused
planner responses dispatching a scripted goal. The repaired boundaries and validation
are recorded in [Day 10 evidence](validation/day-10.json).

## Accepted responses

Chat adapters require exactly one completed assistant response. `finish_reason` must be
explicitly `stop` or, for tool-enabled chat, `tool_calls`. Missing reasons, truncation,
content filtering, extra choices, non-assistant roles, provider refusal and legacy
`function_call` responses are rejected. Ordinary provider metadata remains allowed.

Tool calls must use `type: "function"`, a nonblank string ID of at most 128 characters,
and a 1–128 character ASCII function name. IDs cannot repeat within a reply. Arguments
must be an object, never an array, null or a coerced string. The chat adapter accepts at
most eight tool calls; the mission planner and reviewer each require **exactly one**
call to their own decision tool, with no accompanying prose.

Planner and reviewer arguments retain exact field sets, permitted decision values,
bounded explanations and bounded destination lists. A review can approve, clarify or
reject; it cannot rewrite the plan. Unsupported tools and action/coordinate fields
cannot reach the navigator. Model-generated destination text still goes through named
place or memory resolution; it does not gain the direct coordinate-command capability.

Visual verification and object comparison require exactly `result` and `reason`.
Object absence requires exactly `result`; detections require exactly `label`,
`description` and `box_2d` for each item. Extra fields invalidate the entire result.
Native detector replies require one completed candidate and bounded text-only parts;
tool/function parts, blocked replies and incomplete output are refused. Invalid
detection responses are errors, never empty-scene evidence.

JSON decoding rejects duplicate fields at every nesting level, non-finite numeric
values (including overflowing exponents), and nesting beyond 32 levels. The operator
JSON boundary uses the same checks. No decision is inferred by dropping malformed
fields or choosing the last occurrence of a repeated key.

## Budgets and provider failures

- Operator instructions: nonblank strings, at most 2,000 characters; legacy direct
  movement grammar remains limited to 500. Operator JSON envelopes remain capped at 16,384.
- Mission history: at most 20 objects and 16,000 serialized JSON characters. Complete
  model task data is capped at 32,768, and each mission decision at 16,384.
- Chat content/tool arguments and detector JSON: at most 65,536 characters each.
  Visual verdicts and captions are limited to 16,384. Text-part arrays allow at most 64 parts.
  Existing semantic field limits still apply: descriptions 500, explanations 1,000, and
  the configured mission destination count (default eight).
- Standard-library HTTP transport: reads at most 8 MiB plus one byte from both success
  and error responses, refusing oversize bodies and closing the stream.
- Retry configuration: 1–32 attempts with finite nonnegative bounded delays. Invalid,
  negative or non-finite `Retry-After` values use configured backoff. Timeout configuration
  must be finite and positive. HTTP error text shown to callers is capped at 200 characters.

The navigation mission planner/reviewer and visual verifier retain their one-attempt
policies. HTTP errors and malformed outputs cannot become approvals. Custom injected
models are also checked at the mission boundary; their typed Python objects do not bypass
decision field validation. Planning and lookup exceptions produce bounded, visible
`rejected` or `not_found` messages without dispatching a goal. Arrival errors retain the
existing `destination_unverified` outcome rather than declaring success.

The HTTP timeout bounds blocking network operations, not an entire mission or arbitrary
custom model code. The controller's monotonic deadline independently invalidates late
work. Model calls stay on the existing bounded worker; stop does not wait for them to
return. A slow or stuck worker can limit subsequent availability, but its late reply
cannot revive a canceled request. There is no thread-killing or automatic mission replay.

## Untrusted observations and semantic limits

The current operator instruction, historical context and proposed plan are serialized
as task data in a user message. Historical `role` or `content` fields cannot create new
chat messages or change the system role. Prompts explicitly identify historical captions,
signs and provider explanations as observations, never authorization to execute commands.
Captioning now separates its trusted description instruction from the camera observation.

Destination verification receives the requested description and image pixels, not the
stored caption as proof. Tests insert command-like text into a saved caption, history
and verifier explanation and check that it cannot alter the message roles, dispatch
coordinates, target, review requirement or controller state by direct interpretation.
No code evaluates model prose as a command or supplies a navigation tool to these models.

These are deterministic data-boundary checks with scripted replies. They do **not** prove
that a live model cannot be persuaded by text in an image/history to emit a schema-valid
but incorrect plan or verdict. Independent review and destination verification remain
required, and can still share semantic errors. Held-out intent, unsupported-action,
misleading-observation and wrong-instance evaluation remains a separate qualification
gate. No new regex intent classifier or claim of complete prompt-injection resistance is made.

## Compatibility and validation

Providers must now supply explicit completion reasons, unique valid tool IDs, and strict
decision JSON. A compatibility endpoint that omits completion evidence or returns prose
alongside a mission tool call will produce a visible refusal; update the endpoint response
format rather than bypassing validation. Operator command/status schema versions remain
unchanged, and valid named, coordinate and ordered-visit flows remain supported at their
existing input boundaries.

Validation includes malformed shape/field/size matrices, HTTP errors, bounded reads,
retry configuration, untrusted observation text and blocked planner/reviewer calls that
return invalid output or errors after stop. The fault runner now has 60 distinct scenarios,
including 14 provider-contract cases and two successful controls. ROS operator checks add
four malformed-provider cases plus a valid ordered-mission control over real DDS topics.
The existing controlled action benchmark rechecks cancellation under callback load.

Reproduce with:

```bash
.venv/bin/pytest tests/test_model_contracts.py tests/test_missions.py tests/test_verification.py
.venv/bin/python -m placecell.fault_injection --repeat 3 --output /tmp/day-10-faults-new.json
simulation/sim build
simulation/sim check-operator
simulation/sim check-cancel
```

Use a fresh fault-report path. Local artifacts are ignored by Git; the validation record
contains their hashes. Real-model spending, physical robot commands and endurance runs
are outside these checks. Post-crash active Nav2 ownership reconciliation remains Day 15 work.
