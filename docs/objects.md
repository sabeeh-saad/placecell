# Object memory with RGB-D change tracking

Object memory is optional. It adds persistent object identities alongside the existing
whole-scene memories. Detection and embedding use your configured providers; enabling
this feature does not download a local model. `GeminiObjectDetector` uses the native
Gemini vision endpoint, and crops use the collection's multimodal embedder, including
`GeminiEmbedder` or an explicitly configured `ClipEmbedder`.

## What is stored

Each object has an ID, category, robot/camera/map scope, first/last sighting times, state,
and an optional map-space surface position with a heuristic uncertainty and extent.
Each retained view contains a PNG crop, bounding box, separate image and description
vectors, capture time, localization flag and the robot's observation pose. Full-frame
files remain referenced until both scene memory and object memory release them.

Objects, crops, change events and the parent scene observation commit in the same SQLite
transaction. Detection and embedding happen outside that transaction. Queued observations
contain compressed depth and the original camera transform, so retries cannot accidentally
pair an old RGB image with new depth or a new pose. Both built-in stores support these
records; LanceDB remains the derived index for scene memories. Object retrieval scans
bounded pages of SQLite vectors without loading all crop images into memory.
Collections upgrade to schema 7, so older clients reject them instead of deleting shared
keyframes without accounting for object references. Back up the collection before upgrading.

Defaults allow 1,000 objects, four recent views and 32 change events per object. The ROS
curator removes objects not seen for 30 days. Detection runs at most once every 15 seconds
per robot/camera/map, independently of scene sampling. Stationary scene sampling still
controls when a new frame is available, normally every 60 seconds. Crop embeddings are
batched. At most four visual absence checks run per scan. These limits bound stored data
and provider work, but API latency and cost must be measured on your camera recordings.
At capacity, ingestion reports an error and retains the failed job; prune old objects or
raise the limit before retrying. Failed jobs also occupy the bounded ingestion queue.

## Identity and changes

A nearby detection needs a strong crop similarity and a unique best match on both sides
of the association. Several lookalike objects are not automatically merged. Unresolved
nearby associations create separate `ambiguous` hypotheses, which cannot authorize a
navigation goal. Camera and map versions partition identities; cross-camera identity
matching is outside this milestone.

A larger move can preserve identity only when the appearance match is strong and unique,
and the old location is demonstrably empty in the same RGB-D observation. Otherwise it
becomes a separate observed instance. This conservative rule can fragment identities;
it does not prove that two visually indistinguishable objects are the same physical item.
Movement events include the new position; earlier events retain the previous position.

A missing detection alone is never a miss. The previous object's extent must project
fully into the image, every sampled depth ray through it must show background behind
it, and a vision check using the previous crop and current full frame must confirm the
old location is empty. Foreground obstruction, invalid depth, image boundaries, missing
localization or uncertain vision results do not count. Three confirmations at least ten
minutes apart mark it `missing`; a new matched observation clears misses. Pending absence
evidence already blocks navigation to that object.

Without usable depth, crops and identities can still update from nearly the same robot
viewpoint and image region. RGB-only data cannot establish disappearance or a larger move.
Very small objects, reflective surfaces, overlapping objects and large localization error
can leave positions unknown or associations ambiguous. Uncertainty is a conservative
engineering estimate, not a calibrated probability; tune it using measured camera and
localization errors. Stored coordinates describe a visible surface, not an object's hidden
centre or footprint.

## ROS 2 setup

Install the crop dependency alongside the usual robot extras:

```bash
pip install -e '.[objects,video,lancedb]'
```

The camera must provide **rectified RGB**, depth aligned to that RGB optical frame, and
matching zero-distortion `CameraInfo`. The default topic names are examples and do not
establish alignment by themselves. Raw distorted images, cropped/binned calibration,
mismatched sizes/frames and excessive timestamp skew are rejected for geometry. Publish
rectified, aligned streams with calibrated intrinsics; the bridge currently does not
undistort images. Depth accepts `16UC1` millimetres or `32FC1` metres, including row padding
and either byte order. Depth and dynamic CameraInfo must be within 80 ms of the RGB frame;
a zero-stamped static CameraInfo is also accepted. RGB callbacks use the nearest already
received depth/CameraInfo from an eight-message buffer. If the matching data arrive later,
that capture remains RGB-only. The camera optical frame must have timestamped TF into the
versioned map, including camera height, tilt and mounting offset.

Set `GEMINI_API_KEY` and `GEMINI_VISION_MODEL` to your key and an available vision model
that supports image bounding boxes and structured JSON. The vision model is separate from
`gemini-embedding-2`. Add these parameters to your existing node launch:

```bash
-p objects_enabled:=true \
-p object_model:="$GEMINI_VISION_MODEL" \
-p object_api_key_env:=GEMINI_API_KEY \
-p embed_backend:=gemini \
-p embed_model:=gemini-embedding-2 \
-p embed_api_key_env:=GEMINI_API_KEY \
-p image_topic:=/your/rectified_rgb/image \
-p depth_topic:=/your/aligned_rectified_depth/image \
-p camera_info_topic:=/your/rectified_rgb/camera_info \
-p object_min_interval_s:=15.0 \
-p object_max_records:=1000 \
-p object_max_views:=4 \
-p object_retention_s:=2592000.0
```

The ROS depth error floor is the larger of `object_position_error_m` (default 0.1 m)
and `localization_max_position_std_m`. It also includes a distance-dependent contribution
from `localization_max_yaw_std_rad`. Loose localization limits can therefore prevent
association or disappearance decisions; use measured limits appropriate to your robot.
Keep `localization_required:=true` and set a versioned `map_id`. Existing scene captioning
and navigation verification parameters still apply; `object_model` does not configure
the arrival verifier. See [navigation setup](navigation.md) and [multimodal setup](multimodal.md).

When depth is unavailable or rejected, logs explain why and RGB memory continues. Do not
assume that `objects_enabled` alone means a 3D position was measured. Inspect the record's
`position` field and the reported uncertainty.

## Library API

```python
import os
from placecell import Ingester, ObjectRecall, ObjectTracker
from placecell.providers import GeminiObjectDetector

# store and embedder use the same model/dimension, as in the multimodal guide.
tracker = ObjectTracker(
    store,
    embedder,
    GeminiObjectDetector(os.environ["GEMINI_VISION_MODEL"], api_key=os.environ["GEMINI_API_KEY"]),
)
ingester = Ingester(embedder, store, captioner=captioner, objects=tracker)
ingester.ingest(observations)

hits = ObjectRecall(store, embedder).similar(
    "red printer",
    robot_id="robot",
    camera_id="front",
    frame_id="map",
    map_id="office-v1",
)
for hit in hits:
    record = hit.object
    print(record.id, record.status, record.position, record.last_seen)
    print(store.objects.history(record.id))
    print(hit.view.memory.pose)  # robot observation viewpoint

store.objects.delete(object_id)  # explicitly remove one identity and all of its crops/history
store.objects.prune(before_timestamp)  # library applications schedule their own retention
```

`Curator.forget(filter)` also removes an entire object identity when a retained view matches
the filter, including its crops and history. Automatic scene decay and contradiction do
not remove object identities; their retention and visibility evidence are independent.

Supply optional `Observation.depth` with
`DepthSnapshot.capture(depth_metres, (fx, fy, cx, cy), map_from_camera)` for recordings.
The transform is a rigid 4x4 matrix from the RGB optical frame into the observation's
versioned map. A trusted capture sets `localization_checked=True`; the library does not
independently verify an arbitrary caller's transform or localization claim.

`reembed(source, target, embedder)` also copies object IDs, crops, positions, views and
history, re-embedding the crops and descriptions. Object collections require a model
supporting both images and text. Pause writers and use a separate target collection;
full-frame files remain shared references, as described in the multimodal guide.

## Navigation scope and validation

Object lookup precedes scene lookup when enabled. It searches retained crops/descriptions
but offers the latest observation viewpoint. Distinct objects remain separate choices
even if their robot poses coincide. Missing, ambiguous, stale, unlocalized or visually
unverified candidates cannot authorize movement. The selected object revision is checked
again before dispatch. Memory goals retain the existing Nav2 cancellation, localization
and fresh-image arrival verification flow.

This milestone **does not generate a new stopping pose beside an object's estimated
coordinates**. Nav2 receives the previously observed robot pose. Costmap-aware approach
planning, active searches for moved objects, cross-camera re-identification, manual
resolution of identity hypotheses and physical-instance verification at arrival remain
future work. Arrival verification currently checks the requested visible destination,
not a guaranteed physical identity match against the stored crop. Crop-only lookup can
also reject relational requests such as “the printer next to the window” as uncertain.

Automated tests use synthetic RGB-D scenes, pixel-based test embeddings, injected Gemini
responses and a mocked Nav2 client. They cover motion, identical instances, occlusions,
invalid depth, replay, rollback, durable history, cleanup, migration and stale-goal guards.
No model weights are downloaded by these tests. Live Gemini accuracy, your ROS topic
alignment, latency and real robot behavior still need commissioning with recordings.
