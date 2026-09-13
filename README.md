# placecell

Long-term visual memory for mobile robots, named after the hippocampal neurons that
fire when an animal is at a particular place. The robot records what it sees while it
drives, keeps it together with time and map position, and answers questions like
*"where did I see the fire extinguisher?"* or *"what was near the door this morning?"*
with the robot's observation position, which can serve as a navigation viewpoint.
That position is not a measured object location.

With Nav2 enabled, spoken commands such as **"robot go to the printer"** can resolve a
destination from a named place or visual memory and start a navigation goal. Camera
ingestion continues throughout the trip, so revisiting a place updates its memory.
See [spoken navigation setup](docs/navigation.md) for microphone input, existing speech
topics, destination selection and cancellation.

Inspired by NVIDIA's ReMEmbR, built from scratch around three goals the original does
not have:

- **Provider-agnostic.** Embeddings and the reasoning model come from any backend behind
  one small interface: OpenAI-compatible APIs (OpenAI, OpenRouter, Gemini's compatibility
  endpoint, local servers), with more adapters to come. A provider declares which media it
  can embed, so clips only go to models that understand video.
- **Store-agnostic.** An in-process store by default, so a robot needs no database server.
  Filters on time and place are part of the contract and pushed into the backend.
- **Works without a robot.** The core takes frames, poses and timestamps. A video file plus
  a pose log is enough to build and query a memory on a laptop. ROS 2 is a thin wrapper,
  not a dependency of the core.

## How it works

1. **Memory building** runs while the robot drives in a known map. A segmenter keeps one
   observation every few seconds after movement or turning, plus periodic stationary refreshes. Each kept
   frame is captioned and embedded; the entry stores the caption, a keyframe reference, the
   timestamp and the robot pose in the map frame. Seeing the same thing at the same place
   again reinforces the existing memory instead of adding a row.
2. **Querying** offers three tools: similarity, time range and position radius. Results are
   ranked by similarity times decayed confidence, so a memory seen often last week outranks
   one seen once a month ago.
3. **Forgetting** is a background curator: confidence decays with time, memories that faded
   or aged out are removed together with their evidence, superseded ones after a grace
   period, and explicit deletion by time, area or camera is one call.
4. **Navigation** accepts explicit movement commands, resolves a map-scoped destination,
   and sends it to Nav2. Ambiguous destinations require a choice. Stop requests cancel the
   current trip, including a goal still waiting for acceptance. Questions remain read-only.

Memories are built after mapping, not during it, because the map frame shifts while SLAM
is still closing loops.

## A memory that maintains itself

- **Reinforcement.** Seeing the same thing at the same place again strengthens the existing
  memory instead of adding one. Sighting ids and timestamps are retained for replay detection
  and time queries, even when the newest keyframe replaces the previous one. A superseded
  memory revives with its misses cleared if the object comes back.
- **Contradiction.** When the robot looks at a place from the same spot and heading and no
  longer sees what memory expects, that is a miss. Misses on separate visits add up, and after
  enough of them the memory is superseded. One person blocking the view does not count.
- **Corrections.** An operator can mark an answer right or wrong, on `/placecell/correct` in
  ROS 2 or through `CorrectionLog` in the library. Wrong verdicts halve a memory's rank, and
  repeated ones get it superseded by the curator. The log is append-only and mergeable.
- **Decay and curation.** Confidence halves every week unless reinforced. Faded, aged-out and
  superseded memories are removed with their keyframes.
- **Consolidation.** Clusters of similar sightings in one map cell are summarised into one
  sentence by a chat model and stored as a summary memory carrying the combined observation
  count. Members stay, marked with the summary's id, for time questions and evidence.
- **Re-embedding.** `reembed(source, target, embedder)` rebuilds a collection under a new
  embedding model from the stored captions and keyframes, lifecycle fields intact.
- **Refinement.** When a retained keyframe changes, a background recheck can refresh the
  memory's caption and embedding from that evidence. Captionless memories and explicit
  recheck requests also enter the queue. Refinement preserves confidence, sighting counts,
  misses and operator verdicts; a rewrite never counts as another observation.

## Improving stored memories

The feedback loop is observation → refinement → consolidation → retrieval → feedback.
For example, a memory first described as "a cabinet" can be rechecked against a later,
clearer frame and described as "a red fire equipment cabinet". The description and its
embedding change together. Summaries based on a changed memory are invalidated and become
eligible for rebuilding on the next consolidation pass.

```python
from placecell import MemoryRefiner, RefinementPolicy

refiner = MemoryRefiner(
    store,
    embedder,
    careful_captioner,
    RefinementPolicy(max_memories=8, max_attempts=3, keep_revisions=3),
    producer="vision-model-version",
)
report = refiner.run()  # process a bounded set of queued memories

# Revisit an existing memory, including after changing the captioning model.
store.refinements.request(memory_id, "operator correction")
report = refiner.run()
revisions = store.refinements.history(memory_id)
undone = refiner.rollback(memory_id)
```

Use a captioner that examines the source image carefully; it receives the evidence alone,
without previous captions, summaries or correction notes. Text embeddings are rebuilt from
the new caption; a provider with media support embeds the image, matching ingestion. This
is evidence-driven maintenance, not model training, and a different caption is not proof of
better accuracy. Evaluate the descriptions against actual robot scenes. Operator verdicts
remain in force until later operator feedback changes their effect.

The queue and revision history persist with a disk-backed store. Requests for one memory
coalesce, attempts are reserved before model calls, and failures wait at least five minutes
before retrying. After three attempts, a request stays available for inspection through
`store.refinements.pending()`; another explicit request or changed image resets its budget.
Completed work is not repeated automatically. Equal nonempty image digests suppress rechecks
when a new filename contains the same pixels; without a digest, evidence identity is based
on its URI and metadata. Caller-owned files should be immutable or carry updated digests.

Each successful change retains its previous caption and vector, source evidence reference,
producer and reason. The default keeps three revisions per memory, deleted with that memory.
Undo applies to the latest refinement only, provided its caption, vector and evidence are
still current. It preserves subsequent lifecycle updates and cancels pending rechecks.
Revision history does not retain old image files. A concurrent sighting, deletion or other
memory update prevents an outdated refinement from committing.

In ROS, refinement runs hourly when a caption model is configured, processing at most eight
requests per pass on the bounded maintenance worker. `refine_model` optionally selects a
different vision model at `caption_base_url`; otherwise it uses `caption_model` with a
careful-description prompt and high image detail. Set `refine_interval_s:=0.0` to disable
execution, or adjust `refine_interval_s` and `refine_batch_size` to budget provider work.
A wrong verdict on `/placecell/correct` requests an evidence recheck for that episodic memory.
You can also publish `{"memory_id":"...","action":"recheck"}` or
`{"memory_id":"...","action":"rollback"}` on `/placecell/refine`. Rechecks run on the next
maintenance pass; rollback requires refinement to be enabled. Summary rebuilding remains
controlled separately by `consolidate_interval_s` and requires a chat model.

## Quickstart, offline

```python
from placecell import CollectionInfo, Curator, Ingester, InMemoryStore, Recall
from placecell.providers import HashingEmbedder
from placecell.sources import PoseTrack
from placecell.sources.video import iter_video_observations  # pip install placecell[video]

embedder = HashingEmbedder()  # swap for OpenAICompatibleEmbedder(...) when you have a key
store = InMemoryStore(CollectionInfo("office", embedder.model_name, embedder.dimension))

track = PoseTrack.from_csv("poses.csv")  # timestamp,x,y,yaw[,frame_id,map_id]
observations = iter_video_observations(
    "tour.mp4", track, robot_id="mipa-01", camera_id="front", start_time=track.span[0], out_dir="frames"
)
report = Ingester(embedder, store, captioner=my_captioner).ingest(observations)

recall = Recall(store, embedder)
for hit in recall.similar("fire extinguisher", k=3):
    print(hit.score, hit.memory.pose, hit.memory.caption)

Curator(store).run()  # expire what faded
```

`my_captioner` is anything with a `caption(items) -> list[str]` method, for example
`OpenAICompatibleCaptioner("gpt-4o-mini", api_key=...)` or any vision-language model behind
an OpenAI-compatible server. Swap `InMemoryStore` for `LanceDBStore("~/.placecell", info)`
to keep the memory on disk (`pip install placecell[lancedb]`).

## Asking questions

```python
from placecell import Agent
from placecell.providers import OpenAICompatibleChat

agent = Agent(recall, OpenAICompatibleChat("gpt-4o-mini", api_key=...))
answer = agent.ask("where did you last see a fire extinguisher?")
print(answer.text, [m.memory.pose for m in answer.evidence])
```

The model gets three tools, similarity, time range and position radius, and must finish by
citing the memory ids it used. `answer.grounded` is True only for a nonempty set of citations
that all name retrieved memories. Prose, empty citations and unknown ids return False.

## ROS 2

```bash
pip install "placecell[video,lancedb]"
export PLACECELL_API_KEY=...
placecell-ros2 --ros-args -p image_topic:=/camera_front/color/image_raw -p robot_id:=mipa-01 \
  -p embed_model:=text-embedding-3-small -p caption_model:=gpt-4o-mini -p chat_model:=gpt-4o-mini
```

The node subscribes to a `sensor_msgs/Image` (or `CompressedImage` with `compressed:=true`),
looks up `map -> base_footprint` at each image stamp, writes keyframes, and ingests in a
background thread. Publish a `std_msgs/String` question on `/placecell/ask` and read the JSON
answer, with the cited memories and their map positions, on `/placecell/answer`. All settings
are ROS parameters; the API key comes only from the environment.

Time queries match actual sighting timestamps, not the interval between the first and last
visit. Results expose matching times through `RankedMemory.observed_at`; agent tool results
and ROS answers include `observed_at` and `last_seen`. Nearby agent queries honor `map_id`.

Existing schema 2 and 3 collections are upgraded to schema 4 when opened. The upgrade retains
stored rows, captions, evidence and lifecycle counts. It can preserve the recorded first
and last times, but cannot reconstruct intermediate sightings or observation ids that the
older schema discarded. Replay detection for merged observations is complete for sightings
ingested after the upgrade.

Keyframes produced by the video and ROS sources are marked as managed files. Ingestion
removes rejected or replaced managed files once no memory references them; caller-supplied
evidence is unmanaged by default. The ROS curator removes evidence after deleting its final
memory reference. Failed library ingestion keeps pending keyframes and rolls back segmentation
so the same observations can be retried; completed writes are recognized before captioning.

## Layout

```
src/placecell/
  memory.py        Pose, Evidence, Memory: the data model and its invariants
  providers/       embedding, captioning and chat contracts; hashing embedder; OpenAI-compatible adapters
  store/           store contract with push-down filters; in-memory reference backend; LanceDB backend
  retrieval.py     the three query tools and their ranking
  lifecycle.py     reinforcement, decay, retention curator, supersede, forget
  refinement.py    bounded evidence rechecks, caption revisions and undo
  pipeline.py      segment -> caption -> embed -> persist, batched, idempotent
  agent.py         the tool-calling reasoning loop that ends in a cited answer
  navigation.py    movement commands, destination resolution and trip ownership
  speech.py        final-transcript filtering and bounded offline audio recognition
  sources/         pose tracks from CSV, keyframes from video files
  ros2/            message conversion (testable without ROS) and the rclpy node
```

## Design principles

- **One embedding model per collection.** Vectors from different models are not comparable.
  The collection records its model and refuses others; switching models means re-indexing.
- **Evidence lives outside the store.** The store holds a reference and a content hash. Raw
  video is not kept unless a retention setting asks for it.
- **Deterministic ids.** A memory's id derives from robot, camera and timestamp, so retries
  and replayed recordings never create duplicates.
- **Slow work stays outside transactions.** Sampling keeps a checkpoint per stream.
  Captioning and embedding operate on batches; persistence and contradiction updates
  commit together for each observation. One ingestion worker preserves their order.
- **Filters are pushed down.** Time and position constraints belong to the store contract;
  a backend that fetches everything and filters in Python is wrong, not slow.
- **Ready for clips.** Evidence has a kind, providers declare capabilities, and the schema
  is versioned, so video-native embeddings become a new ingestion mode, not a migration.

## Status

Pre-alpha; no release on PyPI yet. Library and worker behavior are covered by automated
tests, including persistent-store upgrades and failure recovery. Live ROS integration,
hardware endurance and retrieval quality still need validation on representative robot
recordings. See `CHANGELOG.md`.

## Continuous operation

Use a persistent `db_path` for restart recovery. Each collection now has a
`<collection>.state.sqlite3` file containing authoritative memory metadata, observation
history, ingestion jobs and cleanup intents. LanceDB supplies a derived vector index.
Schema 2 and 3 collections import into schema 4 in bounded batches when opened. The
original sighting history is retained during import; older clients reject schema 4.
Stop writers and back up the entire database directory and keyframe directory together
before an upgrade. Do not remove the state file when rebuilding a vector index.

The supported deployment is one process per collection, with serialized memory updates.
Questions and maintenance use bounded background workers, and provider calls never hold
the memory transaction. Camera callbacks check sampling and queue capacity before JPEG
encoding or disk writes. Defaults are a two-second minimum interval and a 60-second
stationary refresh (`min_interval_s`, `max_interval_s`). Configure the latter to match how
quickly stationary scene changes need to be noticed.

The ingestion journal holds up to `max_queue` jobs (64 by default), including in-flight
and failed work. Accepted jobs and their images survive restarts. Failures retry with
bounded backoff using `ingest_attempts` (5) and `ingest_retry_delay_s` (1 second). Exhausted
jobs retain their images for inspection and continue to count toward capacity. Inspect
`store.jobs.failed()`, retry with `store.jobs.retry_failed()`, or explicitly discard selected
jobs with `store.jobs.complete(ids)` and drain cleanup. Startup recovers incomplete image
creation; cleanup rechecks both memory and job references before unlinking evidence.
Keep each collection's managed keyframes in its own directory.

Questions use `question_workers` (2) and `question_queue` (8). Overflow receives an explicit
busy response. Maintenance runs in one background worker with one waiting slot. The node
logs queued and failed jobs, oldest job age and dropped observations every 30 seconds.

Memory records contain at most 64 recent sightings. To read older retained events, page
through `store.sightings(memory_id, limit=64, after=(timestamp, observation_id))` until empty.
Time filters consult the full retained history, including gaps. `store.iter_query()` pages
memories in ID order for maintenance or migration; `query(limit=...)` orders and limits
inside the state store. Calling `query()` without a limit explicitly requests all matches.

Default retention expires memories after 90 days without a sighting, including reinforced
memories (`RetentionPolicy.max_idle_s`). Detailed sightings older than 90 days are pruned
in bounded passes (`history_age_s`), while retaining each memory's latest sighting. These
settings limit history and inactivity, not bytes on disk; size them for the robot's storage
and observation rate. Disable the idle cap explicitly with `max_idle_s=None` if required.
Replay deduplication of merged observations is guaranteed within retained history.

Consolidation processes bounded groups and retains underlying observations. Updating or
deleting a supporting memory invalidates its summary and releases remaining members for
reconsolidation. Retrieval groups summaries with their members and combines semantic and
recent candidates before confidence and feedback ranking. This bounded candidate strategy
is approximate; evaluate recall and false contradictions on your own scenes.

The ROS maintenance pass calls `store.maintain()` to synchronize vectors, build an index
once the collection reaches 1,000 rows, and compact database versions. Standalone callers
should schedule it themselves. `store.rebuild_index()` reconstructs the vector projection
from authoritative state after an indexing failure. Back up or operate through the store
API rather than editing its underlying tables.

## Development

```bash
pip install -e ".[dev,video,lancedb]"
ruff check . && ruff format --check . && mypy && pytest --cov
```

## License

Apache-2.0
