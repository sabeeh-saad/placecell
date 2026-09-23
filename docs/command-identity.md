# Command identity and retry contract

Day 9 adds version 2 JSON input on `/placecell/command_json`. A caller assigns one ID
per intent and resends the **same envelope**, including its timestamp, when delivery is
uncertain. The scoped SQLite journal commits a reservation before the controller sees
the command. Retries cannot repeat planning, resolve another choice or dispatch another
trip, including after completion or a controller restart with the same journal.

This is at-most-once routing within a bounded retry window. A reservation does not prove
that a command was accepted, dispatched or completed. It is not an exactly-once execution
claim. The [Day 9 evidence](validation/day-09.json) records the tested sources and limits.

## Envelope and scope

Read `/placecell/get_mission_snapshot` first. Its additive `command_identity` object gives
the configured `scope`, `retry_window_s`, `max_records` and `durable` flag. It is `null`
when identified admission is unavailable, including when navigation is disabled.
For example, with a scope copied from that snapshot:

```json
{
  "schema_version": 2,
  "command_id": "c556699b-40e4-49dd-8433-f3c2db8997ef",
  "scope": {"robot_id": "robot", "map_id": "office-v1", "conversation_id": "default"},
  "issued_at_unix_s": 1790064000.0,
  "command": "instruction",
  "text": "Visit the printer, then visit the printer again"
}
```

The example timestamp is illustrative: generate current UTC Unix seconds for a **new**
intent. Never refresh it on retry. Use a fresh UUID for each deliberate new instruction;
equal text with different IDs is allowed, and repeated destinations inside a mission
remain separate visits. IDs allow 1–128 ASCII letters, digits, underscores and hyphens.
Scope values are nonblank strings up to 256 characters. Scope must exactly match the
node's `robot_id`, versioned `map_id` and `mission_conversation_id`.

Version 2 retains the version 1 instruction/option limits and strict JSON field checking.
For `stop` and `choose`, omit `text` and add `target_request_id` using the snapshot's
`status.request_id` (32 lowercase hexadecimal characters). A choice also has `option`:

```json
{"schema_version":2,"command_id":"choice-1","scope":{"robot_id":"robot","map_id":"office-v1","conversation_id":"default"},"issued_at_unix_s":1790064001,"command":"choose","option":2,"target_request_id":"ba2ebd818d974943bc3537fdba0158d83"}
```

The target is checked under the controller lock. A changed target emits `stale_command`
without modifying the mission. Version 2 rejects stop/choice phrases disguised as
`instruction` commands; use their explicit command forms. A stop does not require a model.

## Receipts and client behavior

`/placecell/command_receipt` is a reliable, volatile `std_msgs/msg/String` topic, depth 10.
Its JSON has `schema_version: 1`, `type: "command_receipt"`, `command_id`, `scope`,
`request_id` (or null), current `instance_id`, and a `disposition`:

- `recorded`: the reservation committed. The controller may accept or refuse the command.
  Match subsequent status events to `request_id`; mission steps may receive new IDs.
  Stop outcomes describe the targeted active request. Read the snapshot for current state.
- `duplicate`: this exact parsed command was reserved previously; no controller call occurs.
  The original `request_id` is returned. It is not evidence of success or current ownership.
- `conflict`: the retained ID was used with different content, target or timestamp. No work
  occurs; the receipt identifies the original request. JSON field order and equivalent numeric
  timestamp representations do not cause conflicts, but text whitespace is significant.
- `wrong_scope`, `expired`, `future_timestamp`: the scope or time is unacceptable.
- `capacity`: the journal has no free record slot. Live records are never evicted to admit work.
- `unavailable`: no journal is configured, or persistence/clock validation failed.
- `disabled`: navigation is disabled.

Malformed envelopes produce the existing `invalid` status and never reserve an ID. All
dispositions except `recorded` avoid controller work. Rejected admission (`capacity`, for
example) has no reservation and could be admitted by a later retry after the condition
changes; clients should stop automatic retries on these refusals and reconcile state.

A `busy`, `invalid`, `unavailable` or `stale_command` **controller response after reservation**
consumes that ID. Retrying it later cannot turn the refusal into movement. A fresh intentional
request needs a new ID. A lost receipt is recovered by resending the unchanged envelope
within its window. Receipts are not retained jobs or a queryable execution history.

## Retention, clocks and storage

ROS defaults are `command_journal_path: ~/.placecell/commands.sqlite3`,
`command_retry_window_s: 86400.0`, and `command_max_records: 10000`. The reference mission
profile stores the journal alongside its mission context and traces in the persistent
`placecell-missions` directory. An empty path selects memory-only storage and the snapshot
reports `durable: false`; restart suppression then does not apply.

An envelope expires at `issued_at_unix_s + retry_window_s`. Expired envelopes are always
refused, even after their records are removed. Issue times more than five seconds ahead
of server UTC are refused. Commands require synchronized UTC clocks; ROS simulated time
is not used. The journal persists a time high-water mark, so clock rollback cannot revive
an expired envelope after pruning or restart. A bad forward clock jump can refuse valid
work until the clock catches up; deleting the journal is not a safe clock repair.

Cleanup happens during admission. Capacity is global across all scopes in the file;
fixed field limits and a row limit bound record growth. SQLite reuses freed pages and
does not promise immediate file shrinkage. Retry windows can be 1 second–7 days and
capacity 1–1,000,000 records. The stored window is immutable for that journal; opening
it with a different window or incompatible schema fails. Never reuse an ID: conflict
detection lasts only while its record is retained. A reused ID with a new timestamp
after expiry cannot be distinguished from a fresh intent.

Claims use a SQLite transaction, FULL synchronous commits and a 50 ms lock timeout.
Concurrent connections racing on one ID produce at most one reservation. This does not
authorize multiple active controllers: mission/transport ownership remains single-controller.
Keep the journal on reliable local persistent storage. Corruption prevents startup; write
or lock failure refuses identified input. Copies restored from an older backup, deleted
journals, new paths and memory-only journals do not preserve unseen reservations. Recovery
must prevent old-window input from being reused; backup/recovery qualification is later work.

## Stop and restart boundaries

Legacy `/placecell/command` and version 1 JSON still have no identity guarantee. Publish
each intentional instruction once; repeating its text may start another trip. The plain
text `stop` path has a separate ROS callback group from JSON admission, so a blocked
journal write does not occupy its callback group. An intervening stop invalidates an
instruction still waiting for reservation; a later successful write cannot start it.
Version 2 stops need journal admission like other identified commands. Use legacy text
`stop` when the journal is unavailable or blocked. This is an application cancellation
path, not a physical emergency-stop guarantee.

A crash after committing but before routing leaves a consumed ID with an unknown outcome;
retry never executes it. No journal row is automatically replayed. Reopening the same
journal/scope suppresses retries; changing scope rejects old-scope envelopes, and changing
back finds the original reservations while retained. A controller's new `instance_id`
does not change the durable deduplication scope.

Deduplication does not reconcile a surviving Nav2 goal. Follow the
[startup ownership contract](cancellation-ownership.md#startup-contract-for-day-15): establish
that the old goal has terminated before issuing new movement after a crash. Machine-enforced
cross-process ownership reconciliation remains Day 15 work.

## Validation

`tests/test_command_identity.py` covers strict input, conflicts, concurrent connections,
retention, capacity, scope changes, rollback, lock/corruption failures and abrupt process
exit after reservation. `tests/test_ros2_operator.py` exercises retries across mission
states, consumed refusals, choices, stop targeting, controller exceptions, and stop during
a blocked journal write. Existing terminal-event/race tests remain in the full suite.

`simulation/sim check-operator` now includes explicit DDS retries, deliberate repeat visits,
conflicts, stale stops, expiry/scope refusal and a new controller reopening the same journal.
`simulation/sim check-cancel` rechecks the legacy cancellation path and its controlled ROS
latency. These use scripted providers; they do not qualify real-model quality, physical
stopping, storage durability under power loss, or post-crash Nav2 reconciliation.
