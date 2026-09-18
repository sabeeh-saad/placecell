# Spoken navigation with continuous memory

The intended loop is:

```text
"robot go to the printer"
    → completed speech transcript
    → explicit movement command
    → named place or visually checked remembered robot viewpoint
    → Nav2 navigation goal
    → fresh arrival image → destination verified or unverified (memory goals)

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

The optional [multimodal embedding adapter](multimodal.md) searches both images and captions.
It uses the same verified destination path. Changing embedding models requires a new
collection and evaluation of similarity thresholds on your recordings.

Optional [object memory](objects.md) adds instance-level retrieval and RGB-D change tracking.
It retains distinct object choices even when they share an observation pose. Object goals
use the latest recorded robot viewpoint or a checked approach pose. Their
[arrival check](object-arrival.md) compares the selected object's saved views with fresh
RGB-D evidence. Optional nearby viewpoint search can try another checked pose after a miss.

## Robot prerequisites

Run the robot's existing Nav2 stack and localization against a known map. Supply the
camera topic and timestamped TF transform from the map frame to the robot base. The
default base frame is `base_footprint`; use `base_frame:=base_link` if that is your robot's
frame. Configure the same `map_id` for stored observations and destinations. Use a new map
ID when a map's coordinate system changes.

Navigation requires a nonempty, versioned `map_id` and `localization_required:=true`.
Publish `geometry_msgs/PoseWithCovarianceStamped` estimates in the map frame on
`localization_topic` (default `/amcl_pose`). Capture and dispatch require a recent estimate
with planar position standard deviation at most 0.3 m and yaw standard deviation at most
0.35 rad. The covariance must be finite, symmetric and positive semidefinite, with positive
planar variances. All-zero covariance is treated as unknown quality. A capture's TF pose
must also agree with the estimate within 0.5 m and 0.5 rad.

The default maximum estimate age is five seconds, checked against both ROS time and
monotonic receipt time. Configure `localization_max_age_s` for the localization publisher's
actual update rate, including while stationary. AMCL can stop publishing new estimates
when the robot is stationary; if its estimate ages out, captures and new goals wait for a
fresh estimate and an active trip requests cancellation. Replaying an old estimate does
not refresh its age. Alternative localization systems can provide the same message type.
Memory-only recordings may explicitly set `localization_required:=false`; unchecked views
remain ineligible for navigation. These checks cannot detect every localization failure,
including an incorrectly confident estimate or a map changed without updating `map_id`.

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
  -p localization_topic:=/amcl_pose \
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
recorded observation pose and heading by default. Optional [object approach planning](approach.md)
uses fresh RGB-D object geometry, the costmap and Nav2 path queries to select a stopping
pose near an object. It requires object memory to be enabled.

Default gates require similarity of at least 0.5, effective confidence of at least 0.2,
and a retained image captured within seven days with checked localization and a known
image–pose pairing. These are configurable starting values, not calibrated probabilities.
Evaluate them with your embedding model and scenes.

The resolver checks up to three distinct candidate places with a vision model, comparing
the actual image with the user's destination. Stored captions are not supplied to this
check. The model must return a valid `matched`, `not_matched`, or `uncertain` verdict with
visual evidence. Uncertainty, provider failures, missing images, or too many possible
places produce a request for more detail or a failed lookup. Multiple visually matched
places produce numbered choices even if their retrieval scores differ. Views from the
same camera within one metre and 0.5 rad are grouped as one place. The ROS resolver uses
only the currently configured camera. The previous score-based `navigation_ambiguity_margin`
parameter has been removed. A choice or pending lookup expires after 30 seconds. Image
identity, pose, map, age, localization and operator verdicts are checked again before dispatch.

Set `verification_model` and optionally `verification_base_url` to a vision endpoint.
They default to `caption_model` and `caption_base_url`. Each verification request has an
eight-second timeout and no automatic retries. Without a verifier, memory destinations
are unavailable; configured named places and numeric coordinates still work. Using the
same model for captioning and verification can repeat the same mistake; query-specific
pixel checks reduce reliance on caption retrieval but do not establish ground truth.

After Nav2 reaches a memory goal, status changes to `awaiting_observation`. The next
suitable camera frame bypasses the ordinary sampling interval and must have been captured
after arrival, by the same robot and camera, within 0.35 m and 0.35 rad of the requested
viewpoint, with valid localization. Its image is snapshotted before background ingestion
can replace it. A matching visual check produces `succeeded`; a mismatch, uncertain result,
missing frame or expired deadline produces `destination_unverified`. The default overall
arrival deadline is 30 seconds. Stop also cancels pending verification, and late answers
cannot complete a canceled trip. Verification does not increase memory confidence or
sighting counts. Camera ingestion still updates memories from the actual observation.
Named places and coordinate goals report Nav2's result without claiming visual identity.

Object goals additionally require a match against the saved object reference. Ambiguous
identity produces `destination_ambiguous`. With `object_search_enabled`, a clear absence
or an unobserved target can enter `planning_search` and `searching` before another arrival
check. Status includes `object_result` and `search_attempt`; search is disabled by default.
See [object arrival and search](object-arrival.md) for thresholds, limits and setup.

Schema 5 records retained-image capture time and localization provenance. Earlier memories
remain searchable for questions but cannot become navigation goals until a new checked
observation establishes their pairing. Updating captions alone cannot repair an unknown
capture pose. Whole-scene memories return the robot to a recorded viewpoint. Object
approach poses require a fresh measured position; manipulation remains outside this interface.

Supported commands include:

- `go to the printer`, `navigate to kitchen`, `can you take me to station three`
- `go to 2.4, -1.0` or `go to 2.4, -1.0, 1.57` for explicit numeric coordinates
- `option one`, `option two`, `option three` after an ambiguous result
- `stop` or `cancel navigation`

The default movement grammar uses English. Questions, negated requests and conditional
or compound movement requests are rejected by this default interface. Optional
[agent-planned missions](missions.md) accept natural single or chained navigation requests,
review the proposed order, preserve conversation context and report each goal's status.
Send questions to
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
`navigation_lookup_timeout_s`, `navigation_response_timeout_s`, `navigation_timeout_s`,
`navigation_arrival_timeout_s`, `verification_model`, `verification_base_url`,
`verification_request_timeout_s`, `localization_topic`, `localization_max_age_s`,
`localization_max_position_std_m`, and `localization_max_yaw_std_rad`.
