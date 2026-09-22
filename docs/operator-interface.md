# Operator commands and reconnect state

The operator interface accepts version 1 and version 2 JSON commands and exposes an atomic,
read-only version 1 mission snapshot. Existing text commands and the live status topic remain
available. All names below assume the default node name `/placecell`; ROS remapping and
namespaces apply normally. This interface is for the reference deployment's trusted local
ROS graph.

Version 2 adds [durable command identity and bounded retries](command-identity.md),
targeted stop/choice commands and `/placecell/command_receipt`. Use it for clients that
need to retry uncertain delivery. The version 1 examples below retain their original behavior.

## Send a version 1 command

Publish a `std_msgs/msg/String` on `/placecell/command_json`. Its `data` must be one JSON
object with exactly the fields for one of these three commands:

```json
{"schema_version": 1, "command": "instruction", "text": "Visit the printer, then the cupboard"}
{"schema_version": 1, "command": "choose", "option": 2}
{"schema_version": 1, "command": "stop"}
```

Each line above is a separate message. `schema_version` must be integer `1`; strings,
booleans and floating-point versions are rejected. `text` must be nonblank and contain at
most 2,000 characters; `option` must be an integer from 1 through 3. Duplicate or unknown
fields, unsupported commands/versions, malformed JSON and envelopes longer than 16,384
characters produce an `invalid` status without reaching a model or changing the mission.

For example:

```bash
ros2 topic pub --once /placecell/command_json std_msgs/msg/String \
  "{data: '{\"schema_version\":1,\"command\":\"instruction\",\"text\":\"Visit the printer, then the cupboard\"}'}"

ros2 topic pub --once /placecell/command_json std_msgs/msg/String \
  "{data: '{\"schema_version\":1,\"command\":\"stop\"}'}"
```

Validated commands use the same controller as `/placecell/command`: `instruction` requires
the existing grammar when mission mode is off, and planning/review when it is on. Existing
stop/choice phrases inside an instruction retain their direct path. `stop` and `choose`
never require a model. A choice is accepted only while that option is still valid.

Both command subscriptions are reliable, volatile, keep-last depth 1. They are live inputs,
not durable jobs. Reconnecting version 1/text clients must **read state without resending a command**.
There is no client command ID, idempotency key or automatic retry guarantee in version 1.
A repeated instruction after a completed mission can deliberately start another mission;
delivery during an active trip produces `busy`. Do not use either topic as a command queue.

## Read current state

For an immediate read, call the standard `std_srvs/srv/Trigger` service:

```bash
ros2 service call /placecell/get_mission_snapshot std_srvs/srv/Trigger '{}'
```

`success: true` means that the snapshot was read, including when navigation is disabled or
the last mission failed. `message` contains the JSON snapshot. The call takes no command,
does not poll the controller, and cannot dispatch or replay work.

For a display that updates continuously, subscribe to the retained snapshot topic:

```bash
ros2 topic echo /placecell/mission_snapshot std_msgs/msg/String \
  --qos-durability transient_local --qos-reliability reliable --qos-depth 1
```

This publisher is reliable, transient-local, keep-last depth 1. It publishes at startup and
refreshes every 0.2 seconds using a steady clock, including while simulated time is paused.
A matching late subscriber receives the most recently published snapshot without waiting
for another command. The service reads current state under the controller lock; the topic
reflects the latest completed refresh. Executor scheduling can delay refreshes.

The snapshot object always has:

- `schema_version: 1` and `type: "mission_snapshot"`.
- `instance_id`: an opaque controller-lifetime ID; a fresh process/controller gets a new ID.
- `sequence`: the latest status event number observed by this read, including rejected inputs.
- `captured_at_unix_s`: wall-clock Unix time of serialization, independent of simulated time.
- `navigation_enabled` and `closed`: configuration and controller-shutdown flags.
- `busy`: whether the controller still owns a request or mission. This stays true during
  cancellation with an unconfirmed Nav2 outcome. It is not a physical-motion measurement.
- `active_request_id`: the current step/request ID, or `null` without an active request.
  An ambiguous multi-goal mission can be busy while this field is `null`.
- `awaiting_choice` and `choice_remaining_s`: whether a selection can still be made and its
  remaining monotonic-time allowance. The allowance is `null` without pending choices.
  At expiry it can be zero until the next controller poll records the terminal outcome.
- `status`: the complete latest mission-state status, using the status schema below.
- `command_identity`: the version 2 scope, retry window, record limit and durability flag,
  or null when identified admission is unavailable. Read this before constructing a version 2 envelope.

The retained status includes the reviewed destination sequence, current step, actual goal,
choices, distance, object-verification result and latest outcome. A completed mission stays
visible until another state-changing request. Invalid/busy input responses and an idle stop
remain events; they do not replace a running mission or its completed outcome. An ambiguous
single-goal request is not `busy`, but still has `awaiting_choice: true`.

Snapshot reads and heartbeats do not increment `sequence`. The snapshot's `status.sequence`
can be lower than its top-level `sequence` when later events only reject inputs. Countdown,
capture time and shutdown flags can change without a new event, so do not discard snapshots
just because their sequence is unchanged. Use snapshots for current state and events for
feedback; an `invalid` event does not mean the current mission failed.

Treat a missing heartbeat/service as an unavailable connection. A retained message alone
does not prove that the publisher or robot is still responsive. Compare elapsed receipt
time locally rather than assuming synchronized clocks between machines.

After restart, the new controller starts at `idle` (or `disabled`), sequence zero and a new
instance ID. It never restores a previous executable mission or replays commands from
history. Historical outcomes remain in mission context/traces. Startup `idle` does not
establish that a Nav2 goal left by a crashed process has stopped; active-goal reconciliation
after crashes remains a separate qualification item.
The [Day 8 ownership contract](cancellation-ownership.md#startup-contract-for-day-15)
defines the required startup admission rule; cross-process reconciliation is Day 15 work.

## Versioned live status

`/placecell/navigation_status` remains a reliable, volatile `std_msgs/msg/String` event
stream with depth 10. Its JSON retains every previous field and adds:

- `schema_version: 1`, `type: "navigation_status"`.
- `instance_id`: the same ID as the snapshot.
- `sequence`: a strictly increasing status event number within that instance.

The existing fields are `request_id`, `state`, `message`, `destination`, `choices`,
`distance_remaining`, `object_result`, `search_attempt`, `mission_id`, `mission_step` and
`mission_destinations`. IDs are opaque strings, absent mission IDs are empty strings,
and `mission_step` is zero before a reviewed plan and one-based during its execution.
Only a final `succeeded` completes a sequence; `step_succeeded` is an intermediate visit.
`verifying_arrival` now explicitly reports that a fresh arrival image is being checked.

`destination` is an object or `null`. `choices` is an array of destination objects, each
with a one-based `option`. Destination objects contain `label`, `source`, `memory_id`
(string or `null`), `object_id`, `goal_kind`, `target`, numeric `x`, `y`, `yaw`, `frame_id`
and `map_id`. `distance_remaining` is finite metres or `null`; missing or nonfinite
transport values are not reported as a number. `message` is human-readable text, not a
field for parsing decisions. Use `state` and the structured fields.

Version 1 readers should ignore unknown output fields and display unrecognized states
without inferring success. An incompatible schema change requires a new schema version;
new optional output fields do not. Command inputs remain strict to avoid executing a
misspelled or unsupported request. Event sequences are diagnostic ordering, not durable
delivery or cross-process deduplication IDs.

## Python and validation

`NavigationCommands.snapshot()` returns a frozen `NavigationSnapshot` under the same lock
used by command handling and navigation callbacks. `placecell.operator` exposes
`parse_operator_command`, `navigation_payload` and `snapshot_payload` without importing
ROS. `placecell.ros2.node.navigation_payload` remains available for existing callers.

Run the middleware check in a sourced ROS 2 Jazzy environment with PlaceCell installed:

```bash
python simulation/scripts/operator_test.py --output /tmp/placecell-operator-check
```

It covers late transient-local subscribers with the refresh timer stopped, paused simulated
time, read-only services, multi-step completion, invalid inputs, cancellation, new-instance
behavior, durable command retries/restart and the actual PlaceCell node with navigation enabled/disabled. Navigation and
model results in the multi-step cases are scripted. This is operator-contract evidence;
it does not establish real-model mission quality, Gazebo movement or physical stopping.

Day 8 adds `simulation/sim check-cancel` for actual ROS actions and cancellation timing.
Each command subscription now has its own callback group; steady deadline timers remain
responsive with paused simulation time. See the [measured scope](cancellation-ownership.md).
