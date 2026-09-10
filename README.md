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

`my_captioner` is anything with a `caption(items) -> list[str]` method; a vision model
adapter is the next piece on the roadmap.

## Layout

```
src/placecell/
  memory.py        Pose, Evidence, Memory: the data model and its invariants
  providers/       embedding contract, offline hashing embedder, OpenAI-compatible adapter
  store/           store contract with push-down filters, in-memory reference backend
  retrieval.py     the three query tools and their ranking
  lifecycle.py     reinforcement, decay, retention curator, supersede, forget
  pipeline.py      segment -> caption -> embed -> persist, batched, idempotent
  sources/         pose tracks from CSV, keyframes from video files
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

Pre-alpha. The core library and its tests exist; captioning adapters, a persistent store
backend, the reasoning agent and the ROS 2 wrapper are next. See `CHANGELOG.md`.

## Development

```bash
pip install -e ".[dev,video]"
ruff check . && ruff format --check . && mypy && pytest --cov
```

## License

Apache-2.0
