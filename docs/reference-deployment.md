# Mission reference deployment

This is the Day 1 reference candidate for the existing mission workflow, defined on
21 September 2026. It makes that workflow repeatable; it is not a
production qualification. The [local inventory and validation record](validation/day-01.json)
separates measured checks from work still outstanding.

## Reference configuration

Use the bundled office on Linux x86-64 with Docker Engine and Compose v2. The container
uses Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, Nav2 and Python 3.12. The Python core
continues to support the existing 3.10–3.12 test matrix. The image uses CPU software
rendering; a GPU and host ROS installation are not required for this profile.

The Dockerfile pins its ROS base image by digest and currently fixes LanceDB to 0.38.0
and PyArrow to 21.0.0. Apt dependencies are not individually pinned. Record the final image
ID and installed versions for each evaluation; rebuilding the same source may install
different system packages. A historical image does not qualify newly mounted source.

The reference is one controller process and one simulated robot, `office_robot`, with
camera ID `front` and map ID `office-v1`. RGB, aligned depth and CameraInfo use
`camera_optical_frame`; the base is `base_footprint`, and navigation uses `map`.
Camera frames target 640 × 480 at 5 Hz in simulation time. AMCL supplies localization
from laser scans and odometry on the map generated from the office's static geometry.
Object names and ground-truth poses are not supplied to perception.

Load these ROS parameter files **in order**:

1. `simulation/config/placecell.yaml`: sensors, localization, embeddings, captions,
   object memory, checked approaches and navigation.
2. `simulation/config/missions.yaml`: planning/review, explicit verification models,
   mission history and separate storage for interactive missions.

The second file is an overlay, not a standalone configuration. The launcher below loads
both. Existing `check-pipeline` behavior is unchanged and does not enable mission planning.

The candidate model setup uses OpenRouter's `google/gemini-embedding-2` with 768-dimensional
vectors, and `google/gemini-2.5-flash` for captions, detection, planning, plan review and
visual verification. These IDs are carried forward from the existing perception profile;
live tool-calling accuracy for planning/review remains unassessed. The two agents have
separate conversations but share a model. This is not evidence of independent errors.
Hosted IDs are not immutable model snapshots: record provider metadata, dates and any
routing changes with each evaluation and rerun affected evaluations when they change.

`OPENROUTER_API_KEY` is forwarded only to the explicitly launched process. No credential
belongs in YAML, command arguments or committed artifacts. Camera ingestion makes API
calls even while the robot is idle. The monthly API budget is **undecided**; Day 1 checks
make no paid calls. The launcher does not enforce a monetary spending cap.

## Behavior and resource envelope

This profile enables ordered visits, independent plan review, visual destination checks,
fresh arrival/instance checks, clarification, cancellation and scoped conversation history.
Only text or completed speech transcripts enter the command topic. Memory must be learned
before a destination can be grounded. The office contains a printer, workstation, chair
and bookshelf; there is no guaranteed learned `cupboard` destination.

The current configuration specifies:

- One active mission, at most eight destinations, and a 30-second timeout per planning
  or review request with no automatic retries. The overall planning/lookup limit is 90 seconds.
- A 90-second arrival limit, 30-second visual provider timeouts, 10-second Nav2 response
  limit and 600-second trip limit. These are software deadlines, not measured stopping times.
- At most four queued ingestion jobs, batch size one, and 8–20-second scene sampling
  intervals. Object scanning has an eight-second minimum interval. Arrival captures may
  bypass ordinary sampling. Sensor/observation ages use simulation time; cloud work uses wall time.
- Localization age at most five seconds, position standard deviation at most 0.3 m,
  yaw standard deviation at most 0.35 rad, and RGB-D stamp skew at most 80 ms. RGB-D delivery
  may wait up to one wall-clock second under software rendering.
- The existing 0.75 retrieval threshold, calibrated in this office only. Different scenes
  require evaluation, not an assumption that the threshold transfers.
- Up to 1,000 object records and four retained views per object; recent prompt context is
  limited to 20 events and 16,000 serialized characters. These are not global disk limits.

Automatic scene curation/refinement/consolidation, extra search viewpoints and the separate
question-answering agent are disabled in this profile. Persistent scene history, conversation
history and optional recordings do not yet have a demonstrated disk bound. Retention and
endurance qualification remain later work.

The inspected development host has 16 logical CPUs, an Intel Core Ultra 9 285H and about
62 GiB RAM, with other workloads running. This describes available hardware, not a minimum
requirement or dedicated allocation. The Day 2 qualification contract fixes engineering
ceilings of eight CPUs, 16 GiB RAM and 20 GiB of data/artifacts per run. CPU/RAM quotas were
used for the bounded Day 1 checks; disk enforcement, retention qualification and a measured
accepted ingestion-work age limit remain open. These ceilings are not minimum hardware
recommendations or evidence of endurance. See the [Day 7 review](readiness-review.md).

## Operator walkthrough

Run host commands from the repository root. Use a fresh world for an independent run.
`start-nav` recreates the container and resets simulation time; export any previous mission
data before using it. Do not restore old simulated-time memory into a reset world as if it
were fresh.

### 1. Check simulation without model calls

```bash
./simulation/sim build
./simulation/sim start-nav
./simulation/sim check --sensors-only
./simulation/sim check-nav
```

`check-nav` moves the simulated robot through real Nav2 goals without a model. It does
not exercise natural-language planning or recognition. After it finishes, use
`./simulation/sim start-nav` again to start the interactive trial from the office's origin.

### 2. Start the mission process when a model budget is agreed

In terminal A:

```bash
read -rs -p 'OpenRouter API key: ' OPENROUTER_API_KEY
echo
export OPENROUTER_API_KEY
./simulation/sim missions
unset OPENROUTER_API_KEY
```

The launcher stays in the foreground and refuses to start without a key. Exit code 73
means another profile launcher holds the lock. Run only one PlaceCell node: the lock
does not coordinate independently launched nodes or the other simulation test commands.
Do not run `check-pipeline`, `record-commands` or navigation tests alongside this session.

The launcher uses `/home/simulator/placecell-missions/` for scene/object memory, retained
images, corrections, `missions.sqlite3` and `traces.sqlite3`. Mission trace capture was added
on Day 4; the Day 1 record retains hashes of its earlier profile. See [mission tracing](mission-tracing.md)
for export, retention and evidence limits. A node restart within the same running world
preserves context but does not resume a mission. Recreating/removing the container removes
that directory; it is not a persistent host volume.

For an optional saved RGB-D session, pass a **new** container directory at startup:

```bash
./simulation/sim missions -p recording_dir:=/home/simulator/placecell-missions/recording-01
```

Use this instead of the earlier launcher command, not as a second process. Choose a new
name on every restart. Recording has no automatic retention and must be included in the
storage allowance.

### 3. Watch feedback and learn the destinations

In terminal B, open a sourced container shell and start feedback before publishing commands:

```bash
./simulation/sim shell
ros2 topic echo /placecell/navigation_status
```

In terminal C, open another `./simulation/sim shell`. Confirm the configured mode and
save the effective parameters (they contain environment-variable names, not API keys):

```bash
ros2 param get /placecell mission_enabled
ros2 param get /placecell navigation_enabled
ros2 param dump /placecell > /home/simulator/placecell-missions/effective-parameters.yaml
```

Keep the camera facing the printer initially. The foreground node periodically reports
ingestion job counts and learned object counts. To inspect learned labels and positions,
run this read-only check in terminal C:

```bash
python3 - <<'PY'
import json
import sqlite3

database = '/home/simulator/placecell-missions/db/office_missions.state.sqlite3'
with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as connection:
    for (payload,) in connection.execute('SELECT payload FROM objects'):
        obj = json.loads(payload)
        print(obj['label'], obj.get('position'))
PY
```

A stored label alone does not prove a navigable goal; retrieval, geometry and fresh
verification still apply. To learn a second destination, use `./simulation/sim teleop`
in another host terminal **only while no mission is active**. Face the workstation or
bookshelf, wait for ingestion, then exit teleop before sending a mission. Use descriptions
supported by the saved views. Keep this learning pass separate from later held-out trials.

### 4. Send a single visit or an ordered mission

In terminal C, publish each command once and wait for its terminal status before another:

```bash
ros2 topic pub --once /placecell/command std_msgs/msg/String "{data: 'Go to the printer'}"
```

After learning both destinations and completing the first command:

```bash
ros2 topic pub --once /placecell/command std_msgs/msg/String \
  "{data: 'First go to the printer, then visit the bookshelf'}"
```

`mission_id` identifies the whole instruction; `mission_step` and `request_id` identify
the current visit. `step_succeeded` completes an intermediate visit. Only the final visit
produces mission `succeeded`. A Nav2 result by itself does not establish visual arrival.
Record rejected, failed and timed-out requests as outcomes, not successful demonstrations.
The status topic is an event stream. Day 5 added a retained `/placecell/mission_snapshot`
topic and read-only `/placecell/get_mission_snapshot` service for reconnecting clients;
see the [operator contract](operator-interface.md) for freshness and restart semantics.
Clients that retry delivery should use [version 2 command IDs](command-identity.md).
The reference profile persists `commands.sqlite3` in the same mission volume as context
and traces. Reuse the complete envelope on retry; use a new ID for another intentional visit.

“First go to the printer, then the cupboard” is also a valid input form, but it cannot be
promised to complete when no cupboard was observed. The correct outcome is clarification
or a visible grounding failure, with no invented destination.

If feedback offers numbered visual candidates, inspect the choices and publish, for example:

```bash
ros2 topic pub --once /placecell/command std_msgs/msg/String "{data: 'option two'}"
```

Only select an option actually offered. A planner's language clarification instead needs
a new, complete instruction. The office does not guarantee an ambiguous example on every
run; controlled lookalike scenes belong in the evaluation dataset.

### 5. Cancel, stop and retain the run

Cancellation bypasses the model:

```bash
ros2 topic pub --once /placecell/command std_msgs/msg/String "{data: 'stop'}"
```

Wait for the cancellation result before requesting another trip. If ownership remains
uncertain, inspect Nav2; do not start a replacement controller to bypass the busy state.
Stop ends the remaining sequence. There is no pause/resume or mid-mission editing.

Once the mission is terminal, Ctrl+C in terminal A stops PlaceCell and its camera API
work. Wait for process exit, then copy the closed databases and their images from a host
terminal before removing the container:

```bash
mkdir -p simulation/artifacts/day-01-missions
docker compose -f simulation/compose.yaml cp \
  simulator:/home/simulator/placecell-missions/. simulation/artifacts/day-01-missions/
./simulation/sim stop
```

Use a new host artifact directory for each run. This is a stopped-process export, not a
qualified online-backup procedure. Keep recordings and diagnostics local until reviewed.

## What Day 1 establishes

The profile, launcher, storage boundaries, model configuration and operator workflow are
defined. Offline parameter and regression checks can establish wiring and execution logic.
Gazebo sensor/Nav2 checks can establish the reference transport and simulated motion path.
Neither establishes real-model chained-mission accuracy, generalization, resource limits
under endurance load, or behavior on a physical robot.

Existing recordings from the same office are development/baseline data. Independent
human labels, additional layouts/sessions, a frozen held-out split and a model budget
remain dependencies. Acceptance targets are frozen in the qualification contract;
the accepted ingestion-work age budget still needs measurement. See the inventory and
[Day 7 review](readiness-review.md) for available evidence and remaining work.
