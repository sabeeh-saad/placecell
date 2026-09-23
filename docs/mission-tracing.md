# Inspect a mission trace

Mission traces connect an instruction to planning, review, retrieved candidates, visual
checks, navigation and the final status. They are separate from conversation history and
never authorize, queue or replay movement. Tracing is separate from the
[versioned ROS operator interface](operator-interface.md) and its live mission snapshot.

File-backed tracing currently supports the Linux/Unix deployment. It is optional in the
general node configuration and enabled in `simulation/config/missions.yaml`.

## Enable and export

The mission profile writes `/home/simulator/placecell-missions/traces.sqlite3`. To enable
tracing for another existing node configuration, add these ROS parameter overrides to
that node's launch command:

```bash
-p mission_trace_path:=/path/to/traces.sqlite3 \
-p mission_trace_max_events:=10000 \
-p mission_trace_max_bytes:=16777216 \
-p mission_trace_queue_size:=256
```

An empty `mission_trace_path` disables tracing. Use a dedicated file, separate from the
mission-context or memory database. Invalid limits, an incompatible database, or another
writer using the same trace file fail startup with an error. Runtime write failures are
reported through trace-health counters and node diagnostics; they do not alter navigation
decisions. The core API accepts `trace_store=TraceStore(path)` in `NavigationCommands`.

Use `mission_id` from `/placecell/navigation_status` to select an agent mission. For a
single command or a request rejected before a mission exists, use its `request_id`.
Selection replies and stop instructions retain their own request IDs and link to the
original mission when one is active. Separate visits in a chain have distinct request IDs
under one mission ID.

From a sourced container shell or an environment with PlaceCell installed:

```bash
python -m placecell.trace_export \
  --database /home/simulator/placecell-missions/traces.sqlite3 \
  --mission-id YOUR_MISSION_ID \
  --output /home/simulator/placecell-missions/mission-report-01.json
```

A fresh package installation also exposes `placecell-export-trace` with these arguments.
Omit `--mission-id` to export all retained missions and discover their IDs in the summary.
Export is read-only and works after the node exits. It returns a committed snapshot while
the node is running, so recent queued events may not yet appear. Existing output files
are never overwritten. Exit code 2 indicates invalid arguments or an unavailable mission/database.

The simulation container's mission directory is not a persistent host volume. Copy the
export or the closed trace database out before recreating the container. Export filenames
and copies have no automatic retention; include them in the run's artifact budget.

## What the report records

Each event has a schema version, persistent sequence number, process-session ID, mission
and request IDs, step, stage, kind, wall timestamp and monotonic timestamp. Stage spans
pair `start` and `end` events with a span ID, duration and outcome. Nested model/review
durations are inclusive: adding all spans would double-count time. Durations use the
process's wall-clock monotonic timer, independently of ROS simulation/capture time.

The timeline includes:

- The bounded, redacted instruction and whether admission succeeded.
- The proposed destination sequence, reviewer verdict and short returned explanation.
  These are structured decisions, not hidden model reasoning or complete model prompts.
- Retrieved scene/object IDs, similarities, confidence where available, eligibility policy,
  eligible IDs, visual verdicts and the destination selected for dispatch.
- Approach, lookup and arrival-verification spans, accepted/rejected arrival observation
  metadata, image digests when available, and the arrival verdict. Image bytes, vectors
  and HTTP request/response bodies are not copied into the trace.
- Nav2 submission attempts, acceptance/rejection, cancellation requests and acknowledgements,
  action results, transport events and mission status transitions. Distance-only progress
  is sampled and may be coalesced. Acknowledgement is
  distinct from a terminal result. Shutdown's unrelated `idle` broadcast does not replace
  the recorded outcome of a completed mission.
- Instrumented HTTP request durations, attempt counts, model name when supplied in the
  request, HTTP status, reported token usage and explicitly USD-labelled cost when supplied.

For the Day 3 late-success regression, a trace shows Nav2 requesting cancellation after
the trip deadline, its eventual successful action result, and the mission ending
`canceled` without a second dispatch. Transport success and mission success are recorded
separately, so that sequence can be explained from the artifact.

Context is explicitly captured at worker/action dispatch. A late response from an old
mission remains in that mission's trace even after another request starts. An unfinished
span may indicate an active operation, process interruption, or lost/retained-away events;
it is not automatically classified as a timeout.

## Usage and evidence limits

The built-in HTTP adapters retain reported token counts from compatible `usage` fields
and Gemini `usageMetadata`. Missing or malformed fields stay `null`. Cost is accepted only
from an explicitly named `cost_usd` field; a generic `cost` field is not assumed to be USD,
and token prices are not guessed. Many providers therefore leave cost unknown.

The summary separates the known subtotal from a total for the recorded final responses.
Retries, interrupted calls and custom providers can add unknown usage. The retry-attempt
count and this scope are explicit: these figures are not an account bill or a guaranteed
whole-mission cost. Custom chat/verifier implementations still get stage timings, but
their internal calls need their own instrumentation to supply usage.

Instruction and returned explanation text can contain sensitive operational information.
Configured API-key values are registered for redaction, common credential patterns and
credential fields are removed, and endpoint URLs are suppressed. Raw exception strings
are excluded from span errors; published status explanations pass through the same
redactor. Arbitrary unlabelled sensitive text cannot be classified reliably: review an
export before sharing it. New trace databases are created with owner-only permissions;
exports use the environment's normal file permissions.

## Bounds, loss and recovery

Callbacks sanitize bounded data and enqueue without waiting for disk writes. A background
writer commits SQLite events. Defaults retain at most 10,000 events in a 16 MiB main
database, with a 256-event queue. A payload budget leaves room for indexes/metadata;
SQLite also enforces a page cap. The rollback journal can temporarily require comparable
additional disk space. The exporter loads the retained snapshot into memory.

Retention removes the oldest events, including parts of old or long-running missions.
Texts and collections have field limits; oversized events are replaced by an omission
marker using a 16 KiB size guard. Truncation is marked on the event and in health counters.
Nav2 distance feedback is forwarded at most once per 0.2 seconds per trip. Acceptance,
cancellation and terminal results bypass that limiter. Pending distance-only events
coalesce by mission, request, step and stage, without crossing a flush barrier. When the
queue is full, a critical event can displace queued progress. Critical events and flush
barriers are never displaced by progress. If the queue contains only critical work,
overflow still drops diagnostics instead of blocking cancellation.

Exports show global dropped, write-error, trimmed, truncated and unclean-shutdown counters,
plus `coalesced_events` and `dropped_critical_events`. Deliberate coalescing is reported in
health but excluded from `loss_counters_nonzero`; evicted progress counts as dropped.
Zero critical drops does not guarantee completeness if writes failed or retention trimmed
events. Nonzero counters apply to the database history, not necessarily only the selected
mission.

`capture_status: open_or_unclean` means the writer is still open or did not record a clean
shutdown; it does not alone prove a crash. On reopening, a previous open session increments
the unclean-shutdown counter. Pending in-memory events can be lost on a crash. Traces are
diagnostic evidence, not an audit log with guaranteed durability. Missing evidence is
visible as loss indicators or unfinished spans and must not be treated as a successful stage.

## Validate offline

The [fault runner](fault-injection.md) embeds traces in its JSON results and checks that
every published status, dispatch attempt and cancellation request has a matching event:

```bash
placecell-check-faults \
  --case nav_timeout_late_success \
  --case control_visual_arrival \
  --output simulation/artifacts/traces/fault-report-01.json
```

Tests also cover writer failure, a blocked writer/full queue during cancellation, retention,
redaction, process interruption, ambiguity selection and late-provider correlation.
The [Day 4 record](validation/day-04.json) separates these software checks from the bounded
real-ROS check. Neither establishes live-model accuracy, physical stopping time or endurance.
