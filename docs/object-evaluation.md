# Evaluate object memory on RGB-D recordings

The evaluator replays camera observations through a fresh object tracker and compares
its output against independent human labels. It does not read or change a live collection.
It measures detection, identity continuity, disappearance decisions, surface position
error, processing latency and reported API usage. Synthetic test results establish that
the evaluator works; real tracking accuracy requires labeled robot recordings.

## Capture a session

With the [aligned RGB-D inputs](objects.md) configured, set the ROS node parameter
`recording_dir` to a **new** directory, for example `/data/evaluation/office-session-01`.
The directory must not already exist. Leave it empty to disable recording, the default.

The recorder copies sampled RGB images and writes `observations.jsonl` containing capture
time, robot pose, localization provenance and compressed depth with its calibration and
capture transform. Images have independent ownership, so normal memory cleanup cannot
delete the recording. Frames captured without usable depth remain explicitly RGB-only.
The recorder follows camera sampling and arrival captures; it does not save every incoming
camera frame. Recording performs local file writes in the capture callback, so measure
capture latency on your storage device.

Use a bounded commissioning session and monitor free space. Recordings have no automatic
retention policy. A write failure is logged and disables recording for that node process;
ordinary memory ingestion continues. Restart with a new session directory to resume capture.
The portable format currently handles one robot/camera stream in increasing timestamp
order. Existing rosbag files need conversion into observations; there is no direct rosbag
reader in this release.

Library applications can use `RecordingWriter(directory).append(observation)` and
`read_recording(".../observations.jsonl")`. Read observations retain their original capture
times. Do not replace replay timestamps or camera transforms with the current robot pose.

## Label the recording

Create a separate JSONL file with one row per labeled frame. Copy `observation_id` from
the recording. Give each physical object a stable human ID across frames, including
when it moves or is temporarily hidden. A visible example:

```json
{"observation_id":"robot:front:1000000","complete":true,"objects":[{"id":"printer-a","visibility":"visible","box":[0.4,0.3,0.6,0.7],"position":[2.0,2.0,0.8]}]}
```

`box` is normalized `[left, top, right, bottom]` in the saved image. Optional `position`
is a measured visible surface point in map-space metres, not the hidden object centre.
Use `"visibility":"occluded"` for a known object hidden behind something, or `"absent"`
when independent ground truth establishes it is gone; omit the box in either case.
Objects outside the annotated scope can simply be omitted. Do not label an object absent
solely because a detector missed it.

`complete` defaults to true: every visible object within the detector's intended scope
must then be labeled. Use false for partial annotations; unmatched predictions on these
frames will not be counted as false detections. Label before inspecting tracking output
where possible. The evaluator never passes these labels to the detector or embedder.

## Run with hosted Gemini models

Install `pip install -e '.[objects]'` and configure `GEMINI_API_KEY` and your available
`GEMINI_VISION_MODEL`. This command makes hosted API requests; it downloads no CLIP weights:

```bash
placecell-evaluate-objects \
  --recording /data/evaluation/office-session-01/observations.jsonl \
  --labels /data/evaluation/office-session-01-labels.jsonl \
  --output /data/evaluation/office-session-01-report.json \
  --vision-model "$GEMINI_VISION_MODEL" \
  --embedding-model gemini-embedding-2 \
  --dimension 768 \
  --scan-interval 15 \
  --max-frames 10000
```

The output must be a new file. The scan interval matches the default tracker policy;
reports explicitly count labeled frames skipped by that interval. Every label must
reference a recording frame and at least one labeled scan must run. For other providers
or policies, call `evaluate_objects(tracker, read_recording(path), load_object_labels(labels))`
with a fresh empty store and your configured `ObjectTracker`.

Visible predictions are matched one-to-one to human boxes by descending intersection
over union, with a default threshold of 0.5. Reports include visible detection recall,
unmatched detections in complete frames, ambiguous hypotheses, identity switches, false
merges, fragmentations, and up to 32 identity failure examples. An ambiguous detection
may count as detected while still being unusable for navigation. The matching is a
bounded greedy assignment, not a standard HOTA or IDF1 benchmark implementation.

False-missing rates count known objects labeled visible or occluded. Absence confirmation
rates count linked objects labeled absent. Unknown identities have a separate count and
are excluded from these rate denominators; inspect it alongside detection recall so a
tracker that never learned an object cannot appear successful. Position errors use only
fresh depth estimates on matched detections. Empty denominators and unavailable position
measurements produce `null`, not zero error.

Latency includes preparation and commit for each scan. API meters report request counts,
failures, latency and token usage when supplied by the provider. Optional
`--embedding-input-usd-per-million`, `--vision-input-usd-per-million` and
`--vision-output-usd-per-million` use your own rates for a rough cost estimate. Cost is
`null` if rates or token counts are missing, as can happen with embedding responses.
These estimates do not account for every billing tier or discount. Reports retain no
provider prompts, API keys or image payloads; recordings themselves contain camera data.

Start with revisits, moved objects, two lookalike objects, temporary occlusion, genuine
removal and a return after removal. Keep recordings used to tune thresholds separate
from recordings used for the final evaluation. Compare configurations on the same labels
and review failure examples before enabling approach planning on the robot.
