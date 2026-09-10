# placecell

Long-term visual memory for mobile robots, named after the hippocampal neurons that
fire when an animal is at a particular place. The robot records what it sees while it
drives, keeps it together with time and map position, and answers questions like
*"where did I see the fire extinguisher?"* or *"what was near the door this morning?"*
with a map position the navigation stack can drive to.

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
   observation every few seconds, and only when the robot has moved or turned. Each kept
   frame is captioned and embedded; the entry stores the caption, a keyframe reference, the
   timestamp and the robot pose in the map frame. Seeing the same thing at the same place
   again reinforces the existing memory instead of adding a row.
2. **Querying** offers three tools: similarity, time range and position radius. Results are
   ranked by similarity times decayed confidence, so a memory seen often last week outranks
   one seen once a month ago.
3. **Forgetting** is a background curator: confidence decays with time, memories that faded
   or aged out are removed together with their evidence, superseded ones after a grace
   period, and explicit deletion by time, area or camera is one call.

Memories are built after mapping, not during it, because the map frame shifts while SLAM
is still closing loops.

## A memory that maintains itself

- **Reinforcement.** Seeing the same thing at the same place again strengthens the existing
  memory instead of adding one. A superseded memory revives if the object comes back.
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
citing the memory ids it used. `answer.grounded` is False when it answered in prose without
citing anything, so a caller can refuse ungrounded answers.

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

## Layout

```
src/placecell/
  memory.py        Pose, Evidence, Memory: the data model and its invariants
  providers/       embedding, captioning and chat contracts; hashing embedder; OpenAI-compatible adapters
  store/           store contract with push-down filters; in-memory reference backend; LanceDB backend
  retrieval.py     the three query tools and their ranking
  lifecycle.py     reinforcement, decay, retention curator, supersede, forget
  pipeline.py      segment -> caption -> embed -> persist, batched, idempotent
  agent.py         the tool-calling reasoning loop that ends in a cited answer
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
- **Stages exchange batches, nothing else.** Segment, caption, embed and persist are pure
  functions over lists. They run in one process today and can run as separate workers
  behind queues tomorrow without changing what they do.
- **Filters are pushed down.** Time and position constraints belong to the store contract;
  a backend that fetches everything and filters in Python is wrong, not slow.
- **Ready for clips.** Evidence has a kind, providers declare capabilities, and the schema
  is versioned, so video-native embeddings become a new ingestion mode, not a migration.

## Status

Pre-alpha. The library, the LanceDB backend, the agent and the ROS 2 node exist and are
tested; no release on PyPI yet. Next: a Gemini Embedding 2 adapter for clip mode, a queue-based
distributed runner for the pipeline stages, and consolidation of repeated memories into
summaries. See `CHANGELOG.md`.

## Development

```bash
pip install -e ".[dev,video,lancedb]"
ruff check . && ruff format --check . && mypy && pytest --cov
```

## License

Apache-2.0
