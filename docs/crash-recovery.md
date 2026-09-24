# Crash and restart recovery

The ROS controller persists the exact Nav2 goal UUID **before submitting it**. Startup
reconciles that goal without resuming its mission. The ingestion journal independently
recovers accepted camera work and retains its evidence. No hosted model or credential is
needed for these recovery checks.

## Navigation ownership

`navigation_ownership_path` defaults to `~/.placecell/navigation.sqlite3`. Keep it on
persistent local storage with the command journal, mission context and memory database.
It is bound to the robot ID, versioned map ID and fully resolved action name. A lifetime
file lock refuses a second controller using the same journal; a changed scope or damaged
journal fails startup. All controllers for this deployment must use the same path. Separate
paths or restored older files cannot coordinate ownership across processes.

Recovery resolves action-name remappings before constructing the result and cancel service
names, so an action alias still reconciles the same scoped server used for goal submission.

On startup:

- A previously confirmed terminal goal, or explicit operator attestation, permits admission.
- A recorded pending goal produces an `uncertain`, busy snapshot. The controller queries
  its result and requests cancellation using **only that UUID**, with a zero timestamp.
  New instructions and choices are refused before planning or destination lookup.
- Only a succeeded, canceled or aborted action result clears pending ownership. Cancel
  acknowledgement, `UNKNOWN`, missing status and an unavailable server keep it blocked.
  Requests are bounded to one result request and one cancel request at a time; timed-out
  requests are removed locally and retried after a delay using the same UUID.
- A missing journal starts unknown and remains blocked. It is not evidence of a clean server.

Recovery never submits a movement goal. After confirmation, the snapshot becomes idle and
the operator must issue a fresh instruction. Durable command IDs retain their existing
deduplication semantics: even a crash between reservation and routing consumes that ID.
Retained mission context is history, not executable work. A terminal-result journal write
failure retains uncertain ownership instead of releasing the trip.

## First use or unresolved ownership

Stop the controller and all other command sources. Independently stop/reset the scoped
Nav2 server and establish that the robot is stopped; an action-client disconnect is not
enough. With the controller still stopped, record that operator-established condition:

```bash
python -m placecell.navigation_ownership attest-clean \
  --journal ~/.placecell/navigation.sqlite3 \
  --robot-id robot --map-id office-v1 --action-name /navigate_to_pose \
  --confirm-nav2-stopped --reason 'Nav2 reset with all command sources stopped; stationary robot confirmed'
```

Use the actual configured scope and journal path. The mission simulation profile instead
uses `/home/simulator/placecell-missions/navigation.sqlite3` and robot ID `office_robot`.
Run the command inside the same container/storage environment as the controller. Start the
fresh Nav2 server and controller afterward. This command records an **operator attestation**;
it does not stop the robot, reset Nav2 or infer safety from silence. Its exclusive lock
prevents use against a running controller. `inspect` with the same arguments reads the
journal while the controller is stopped. Do not delete the journal to clear a block.

If a server has forgotten a goal result, automatic recovery intentionally remains blocked.
Use the supervised reset procedure. Do not restore an older journal while retaining a
running action server. The [Day 16 backup/restore workflow](backup-restore.md) always
restores unknown navigation ownership and requires a new command session.

## Ingestion and evidence

SQLite commits are authoritative. Accepted work keeps its image across a process death,
including interruption during captioning/embedding. Uncommitted memory writes roll back;
committed observations are recognized before another model call and do not double-count
sightings. Exhausted jobs retain their failure count and image until explicit retry or
discard. Creation markers recover unacknowledged images; cleanup rechecks memory,
object-view and job references. Interrupted unlink operations are safe to repeat.
LanceDB projections rebuild from the committed SQLite state and dirty-vector journal.

In-flight external requests may be charged more than once when retried after a crash;
the provider protocol does not supply exactly-once execution. These tests use local fixtures.

## Reproduce

```bash
.venv/bin/pytest tests/test_navigation_recovery.py tests/test_nav2.py tests/test_navigation_races.py
.venv/bin/python scripts/check_recovery.py --repeat 3 --lancedb --output /tmp/recovery-checks
simulation/sim build
simulation/sim check-recovery
simulation/sim check-recovery-gazebo
simulation/sim check-recovery --remap-action
simulation/sim check-recovery-gazebo --remap-action
simulation/sim check-recovery-gazebo --remap-action --drop-first-result
```

Each output directory must be new. The two simulation commands use isolated, network-disabled
containers; the Gazebo command starts and stops its own office and Nav2 instance. Reports
include each crash point, repetition, recovery outcome and zero paid API calls. The persistence
runner sends real `SIGKILL` after a child confirms the selected boundary. ROS checks cover
command reservation, planning, review, durable goal reservation, submission, acceptance and
terminal-result persistence. A separate DDS goal verifies that recovery does not cancel other
clients. Gazebo checks retain the real action server while killing/replacing its controller.

The [Day 15 record](validation/day-15.json) separates these campaigns and their exact source
and image hashes. Final campaigns pass 102 crash trials and 1,323 unit tests. An earlier
stress trial using a one-second response deadline stayed safely blocked after confirmation
timed out; it is preserved separately from the passing instrumented and dropped-reply runs.
Do not treat later successful retries as proof that intermittent transport delays cannot recur.
The claims cover process crashes on the tested Linux filesystem and ROS
profile. Power loss, disk corruption repair, filesystem exhaustion, full-workload latency,
physical stopping and live-model mission quality remain separate qualification work.
