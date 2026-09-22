# Target freshness and identity

Day 12 hardens the transition from a remembered target to a selected destination and
then to fresh arrival evidence. The [validation record](validation/day-12.json) preserves
the reproduced failures, local software checks and exact source/artifact hashes.

## Selection and dispatch

A remembered target must remain current through lookup, operator choice, approach
planning and the final check immediately before submission to Nav2. A deleted, expired,
missing or revised object invalidates an old choice. The operator must request the
destination again; the controller cannot silently substitute a different target.

Once object retrieval finds plausible candidates, rejection by their current-view visual
checks stops that lookup. An older scene cannot override those rejected object views.
Scene retrieval remains available when object retrieval has no plausible candidate.
This favors refusal when evidence conflicts and can reduce successful recall.

## Arrival evidence

Reaching a remembered pose starts an observation wait. It does not prove the target is
there. The capture must be later than arrival, localized in the original scope and at the
reached viewpoint. Its source timestamp and elapsed monotonic age must remain within
`navigation_max_observation_age_s`, default **5 seconds**, through queueing and completion
of verification. A paused source clock cannot extend the lifetime of an accepted capture.
A pre-arrival or already stale image is ignored while the controller waits for usable
evidence; expiration during verification ends that attempt without success.

This capture-age limit is separate from `navigation_arrival_timeout_s`, which bounds the
whole wait/check phase (30 seconds by default, 90 in the simulation profile). The same
capture-age parameter configures scene and object arrival checks in the ROS node. Slow
providers can exhaust the 5-second budget, even when their request timeout is longer.
The budget is configurable and has not been calibrated against held-out live-provider
latencies. Python integrations configure `NavigationCommands.max_observation_age_s` and
`ObjectArrivalPolicy.max_observation_age_s` consistently and supply the correct source
clock and a monotonic clock.

Object arrival retains pre-departure crops as the identity reference. It compares fresh
appearance with those crops, fresh depth or a matching recorded viewpoint, and a paired
image verdict. It also checks the requested description. Verification itself never
writes a sighting, absence claim or updated target location into memory.

- A stationary target can match when appearance, geometry and visual checks agree.
- A moved target additionally needs strong appearance agreement, bounded movement and
  a visible, empty old location. Occlusion cannot establish that location is empty.
- A visible, empty old location with a positive absence check yields `missing`.
- An occluded target remains `unobserved`; a lookalike or uncertain comparison remains
  `ambiguous`. Neither is a success or evidence of confirmed removal.
- Rival appearances include both the departure snapshot and current objects in the same
  robot/camera/frame/map scope, even if their category labels differ. They are checked
  again after the paired model response. Comparisons refuse scopes above 1,000 objects.
- Deletion, identity changes and target evidence newer than the arrival image invalidate
  that image's result. Scene references are also rechecked before accepting success.

After object identity verification, a journal-generation token guards the final request
check and controller completion. Any object-journal write during that interval causes a
conservative refusal, including a write about an unrelated object. This avoids accepting
a verdict across changing evidence but may increase refusals during continuous ingestion.
A new observation/attempt is required; there is no automatic repeated provider loop.
Existing bounded local search still applies to eligible `missing`/`unobserved` outcomes.

## Failure attribution

The additive `failure_stage` field is available in navigation status, current snapshots,
mission history and trace status events. Its values are:

- `retrieval`: the selected reference is missing, stale, changed or no longer usable.
- `identity`: appearance/description does not establish a unique requested identity.
- `geometry`: capture freshness, localization, depth, visibility or movement is insufficient.
- `execution`: provider, worker, persistence, transport or execution deadline failures;
  cancellation also uses this stage.
- Empty string: no failure stage is being reported, including successful completion and
  normal progress. This is not an independent success indicator; inspect `state`.

Keep `object_result`, `state` and `message` alongside this attribution. For example,
`missing/geometry` and `unobserved/geometry` carry different evidence. A stage identifies
the check that prevented completion; it is not a calibrated physical root-cause diagnosis.
Status schema version remains 1 because the field is additive.

## Reproduce the checks

```bash
pytest tests/test_target_freshness.py tests/test_object_arrival.py tests/test_object_navigation.py
python -m placecell.fault_injection --repeat 3 --output /tmp/target-faults.json
simulation/sim build
simulation/sim check-operator
simulation/sim check-sensors
simulation/sim check-cancel
```

The target regressions use synthetic RGB-D pixels, actual object tracking/storage,
retrieval and arrival/controller code, with scripted detectors, embeddings and comparators.
They include moved, removed, occluded and lookalike targets, stale choices, changing
references during provider calls, paused clocks and positive controls. Nine additional
fault-runner cases exercise scene reference changes, stale/late images, uncertain identity
and transport aborts. Reports retain each final state and failure stage separately from
whether the software contract passed.

The ROS operator check verifies delivery of all four stages through status, snapshot
service and retained DDS history using scripted resolution/transport outcomes. Sensor
and cancellation checks cover their existing middleware contracts. These runs make no
paid calls and do not qualify live-model recognition, Gazebo target identity, physical
motion or the held-out mission gate. Those remain separate roadmap requirements.
