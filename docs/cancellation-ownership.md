# Cancellation and goal ownership

Day 8 hardens ownership within one controller lifetime. A stop or transport deadline
must prevent a later success from advancing a mission, even when callbacks arrive in a
different order. [Day 8 validation](validation/day-08.json) records the tested sources,
regressions and measured scope. Cross-process reconciliation remains Day 15 work.

## Ownership rules

- Planning, lookup, clarification, search planning and arrival verification have no
  outstanding motion goal. Stop clears the active request/choices; late worker replies
  cannot submit a goal or finish a replacement request.
- Submission owns the trip before acknowledgement. Stop records cancellation intent.
  Without a goal handle there is no goal-specific cancel request to send; a late accepted
  handle is canceled. A missing response keeps ownership uncertain.
- Once accepted, issue the asynchronous cancel request before publishing or persisting
  cancellation status. Stop is independent of model completion. An acknowledgement is
  not a terminal action result and does not release the trip.
- Missing/rejected cancel acknowledgements retain ownership, publish `uncertain` or
  `cancel_failed`, and block replacement instructions. No timer releases uncertain ownership.
- Release only on a known terminal result/rejection or known pre-dispatch failure. A send
  exception is uncertain delivery, not proof that no goal exists.
- Cancellation intent accompanies terminal callbacks atomically. If a result overtakes
  the timeout event, it still cannot advance the mission. Named-goal success after stop
  becomes `canceled`; memory-goal success after transport cancellation stays
  `destination_unverified` without starting a new visual check.
- Validate response/trip deadlines when callbacks arrive as well as when polling. A delayed
  timer cannot admit an expired success. A known terminal goal needs no additional cancel.
- Internal callbacks address the exact trip object. Reused textual transport IDs cannot
  make an old error cancel a new trip. Duplicate acceptance cannot register another result
  request. Controller request/search-leg checks discard late feedback and repeated results.

Callbacks run outside the adapter lock. The terminal event carries captured cancellation
intent rather than depending on another callback arriving first. Traces retain that intent
alongside raw Nav2 outcomes; raw successful navigation need not mean mission success.

## ROS scheduling and time

Each command subscription has its own mutually exclusive callback group, separate from
camera/TF work. Day 9 separates text stop from JSON journal admission and invalidates
pending admission when a stop intervenes; see [command identity](command-identity.md).
The action client uses a reentrant group; transport/controller locks protect
ownership across concurrent callbacks. Each deadline timer has a separate mutually exclusive
group and a 100 ms **steady-clock** period. Pausing simulation time does not freeze wall-clock
command or transport deadlines. Observation freshness retains its existing clock rules.

The node uses four executor threads. This provides room for commands, action events and
deadlines alongside a blocked default-group callback; it is not a hard real-time guarantee
under arbitrary starvation or blocked storage. Context writes still use the controller
lock. Stop-status persistence now follows the transport call. Later saturation/endurance
checks must qualify the full supported workload.

## Startup contract for Day 15

Restarting PlaceCell does not establish that its previous Nav2 goal stopped. Before using
the current implementation after a crash, the operator must establish that Nav2 has no
surviving goal from the old controller. Do not restart to bypass uncertain ownership.
An idle snapshot, missing status or missing server connection is not proof of termination.

The required Day 15 admission rule is: **start with ownership unknown and refuse movement
until surviving ownership is reconciled or an independently established clean Nav2 instance
is available.** Persist enough robot/map/action-server/goal identity to distinguish the old
trip, query or cancel that specific goal, and retain uncertainty when termination cannot be
confirmed. Do not cancel other clients indiscriminately, replay a saved mission, or infer
completion from silence. Test crashes around submission/acceptance with a surviving goal.

This cross-process rule is specified, **not yet machine-enforced** by Day 8. Startup still
creates the documented idle snapshot. It remains a release blocker; Day 8 results apply
within one process lifetime.

## Reproduce and interpret validation

```bash
.venv/bin/pytest tests/test_navigation_races.py tests/test_nav2.py tests/test_missions.py
.venv/bin/python -m placecell.fault_injection --repeat 3 \
  --output simulation/artifacts/day-08/faults.json
simulation/sim build
simulation/sim check-cancel
simulation/sim check-operator
```

Use new offline output paths for repeats. Docker checks create unique artifact directories.
`check-cancel` runs a controlled `NavigateToPose` action server and production command/
controller/adapter/timer components in a network-isolated container with eight CPUs and
16 GiB RAM. It needs neither the office nor provider credentials.

Checks cover blocked planner/reviewer replies, delayed acceptance, rejected/missing cancel
acknowledgements, concurrent feedback/results, and a successful two-goal control. One hundred
stop trials run with SQLite context and persistent traces while a 5 Hz synthetic observation
stream drives a 750 ms callback in the default callback group. ROS time stays paused at zero.
The client and controlled server each use four executor threads.
Trace capture uses a 4,096-event queue. The harness drains it after each ten-trial batch
outside the measurement intervals and verifies clean close, zero loss and the complete
status sequence. Earlier timing runs exposed that the original harness ignored a timed-out
trace close; their reports and the capture finding are retained in the validation record.

Latency uses monotonic command-callback receipt to return from the asynchronous cancel API.
Reports retain every trial, nearest-rank p50/p95/p99/max, publish-to-cancel timing and separate
acknowledgement timing. Delayed acceptance has a separate acceptance-to-cancel measurement;
it is excluded from the accepted-handle denominator. Model-phase stops send no goal and
report local handling time. Any measured accepted-handle request over 500 ms fails the
check. Failures/timeouts remain in the report.

This measures controlled ROS software behavior, not physical stopping, Gazebo motion or
the full RGB-D/provider workload. No live-model accuracy, endurance, hosted CI result or
cross-process recovery is claimed. Cite the image/source hashes and sample counts from
the validation record.
