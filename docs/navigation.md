# Spoken navigation with continuous memory

The intended loop is:

```text
"robot go to the printer"
    → completed speech transcript
    → explicit movement command
    → named place or remembered robot viewpoint
    → Nav2 navigation goal

camera + stamped robot pose → sampling → durable ingestion → memory updates
                                                           → caption refinement
```

The camera pipeline runs independently of speech, destination lookup and navigation. It
stays active before, during and after a trip. By default, it admits observations after at
least two seconds and sufficient movement or turning, plus a stationary refresh after
60 seconds. Updates become available when captioning, embedding and persistence finish;
they are not synchronous with every camera frame. Repeated sightings reinforce existing
memories, new scenes create memories, and changed retained images schedule refinement.
Arriving at a navigation goal does not itself increase memory confidence or prove an
object is still there.

## Robot prerequisites

Run the robot's existing Nav2 stack and localization against a known map. Supply the
camera topic and timestamped TF transform from the map frame to the robot base. The
default base frame is `base_footprint`; use `base_frame:=base_link` if that is your robot's
frame. Configure the same `map_id` for stored observations and destinations. Use a new map
ID when a map's coordinate system changes.

The action adapter uses asynchronous goal, feedback, result and cancellation interfaces
shared by Humble and Jazzy. Nav2 handles path planning, obstacle avoidance and recovery
through its configured [NavigateToPose action](https://api.nav2.org/actions/humble/navigatetopose.html).
The adapter also reads newer result error fields when available. Real robot commissioning
is still required; automated tests use an injected action client, not a live Nav2 server.

Install with the Python interpreter matching your sourced ROS distribution:

```bash
source /opt/ros/humble/setup.bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[video,lancedb]'
```

Use the corresponding setup file for Jazzy. `rclpy`, `nav2_msgs`, camera messages and TF
come from the robot's ROS installation. Configure actual caption and embedding providers
for semantic landmark lookup; the default hashing embedder is for offline development.

Start the memory node with your existing provider and camera parameters, adding:

```bash
placecell-ros2 --ros-args \
  -p navigation_enabled:=true \
  -p map_id:=office \
  -p places_file:=/absolute/path/places.json \
  -p image_topic:=/camera/color/image_raw \
  -p embed_model:="$EMBED_MODEL" \
  -p caption_model:="$VISION_MODEL" \
  -p embed_base_url:="$EMBED_URL" \
  -p caption_base_url:="$VISION_URL"
```

Set those environment variables to your provider configuration and `PLACECELL_API_KEY`
when its endpoint requires a key. `places_file` is optional for memory-based destinations.
Navigation defaults to disabled so existing memory-only installations keep working. In
simulation, set `use_sim_time:=true`: memory ranking, retention and destination freshness
then use the ROS clock, matching observation timestamps. Transport deadlines use monotonic
wall time so a paused simulation does not leave a request pending indefinitely.

## Define or learn destinations

An exact named place takes precedence over a semantic memory search. Names such as
"kitchen" or "station three" need configured poses if camera evidence cannot identify
them. A places file contains navigation poses measured in the current map, for example:

```json
{
  "kitchen": {"x": 2.4, "y": -1.0, "yaw": 1.57, "frame_id": "map", "map_id": "office"},
  "station three": {"x": 6.0, "y": 3.2, "yaw": 0.0, "frame_id": "map", "map_id": "office"}
}
```

Replace these example coordinates with your robot's measured poses. The file is loaded
at startup. Other destinations are retrieved from episodic visual memories for the same
robot and map. Summary averages are excluded from navigation. The goal is the robot's
recorded observation pose and heading, not a measured object coordinate.

Default gates require similarity of at least 0.5, effective confidence of at least 0.2,
and a sighting within seven days. These are configurable starting values, not calibrated
probabilities. Evaluate them with your embedding model and scenes. Different places within
10% of the top score produce up to three numbered choices; nearby views within one metre
are treated as one location. A choice or pending lookup expires after 30 seconds. Memory
content, map, age and operator verdicts are checked again before dispatch.

Supported commands include:

- `go to the printer`, `navigate to kitchen`, `can you take me to station three`
- `go to 2.4, -1.0` or `go to 2.4, -1.0, 1.57` for explicit numeric coordinates
- `option one`, `option two`, `option three` after an ambiguous result
- `stop` or `cancel navigation`

The movement grammar currently uses English. Questions, negated requests and conditional
or compound movement requests are rejected by this command interface. Send questions to
`/placecell/ask`. Coordinate commands require numeric literals and commas; the speech
adapter does not infer coordinates from spoken number words.

## Use an existing speech-to-text system

Publish each completed command once as `std_msgs/String` on `/placecell/command`. Gate
your recognizer's partial results and background speech before publishing. Use volatile
QoS, depth one and a short publisher lifespan so disconnected clients do not deliver old
movement commands later. Commands are never added to the durable camera ingestion queue
or replayed after restarting this node.

You can test the same interface with text:

```bash
ros2 topic echo /placecell/navigation_status
ros2 topic pub --once /placecell/command std_msgs/msg/String "{data: 'go to kitchen'}"
ros2 topic pub --once /placecell/command std_msgs/msg/String "{data: 'stop'}"
```

Run the echo in a separate terminal. Status JSON includes a request ID, state, destination,
numbered choices and distance remaining when available. For a namespaced robot, remap
these topics and configure `nav2_action` to its action name.

## Use a microphone directly

An optional entry point uses a local [Vosk recognition model](https://alphacephei.com/vosk/models)
and `sounddevice`. Download and extract a suitable model before starting the robot; the
entry point requires its local directory and does not download models at startup.

```bash
pip install -e '.[speech]'
placecell-listen --model /absolute/path/to/vosk-model \
  --command-topic /placecell/command \
  --wake-word robot
```

Say **"robot go to the printer"**, **"robot option two"**, or **"robot stop"**. Use `--device`
to choose a microphone and `--sample-rate` to match supported input settings. Audio is
mono signed 16-bit PCM, defaults to 16 kHz, and is not saved. This adapter follows the
provider's [streaming microphone interface](https://github.com/alphacep/vosk-api/blob/master/python/example/test_microphone.py).

Only final utterances whose complete word scores meet `--min-confidence` (default 0.8)
and whose text begins with the wake phrase are forwarded. The publisher gives commands a
two-second lifespan. The audio queue holds eight chunks; stale, dropped or interrupted
audio causes the affected utterance to be discarded. Please repeat after an interruption.
Microphone recognition quality and latency have not been measured on the robot. Other
recognizers can replace this adapter through the same command topic.

## Trip ownership and cancellation

One trip runs at a time. New destination requests while busy are rejected; say stop before
choosing another trip. Stop bypasses slow destination lookup. If a goal is still waiting
for Nav2 acceptance, cancellation is remembered and sent as soon as it is accepted.
Cancellation acceptance is not reported as completion: the final action result must
confirm it. Speech stop is a Nav2 cancellation request, not a hardware emergency stop.

The defaults allow 10 seconds for a goal response and 600 seconds for a trip. A missing
response or expired trip requests cancellation. Transport uncertainty keeps the trip
owned and blocks another goal; inspect Nav2 if cancellation cannot be confirmed. Node
shutdown requests cancellation and briefly spins for its result. It logs any remaining
uncertainty rather than claiming the robot stopped.

The main parameters are `navigation_enabled`, `nav2_action`, `places_file`,
`navigation_min_similarity`, `navigation_min_confidence`, `navigation_max_memory_age_s`,
`navigation_ambiguity_margin`, `navigation_lookup_timeout_s`,
`navigation_response_timeout_s`, and `navigation_timeout_s`.
