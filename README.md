# placecell

Long-term visual memory for mobile robots, named after the hippocampal neurons that
fire when an animal is at a particular place. The robot records what it sees while it
drives, keeps it in a vector store together with time and map position, and answers
questions like *"where did I see the fire extinguisher?"* or *"what was near the door
this morning?"* with a map position the navigation stack can drive to.

Inspired by NVIDIA's ReMEmbR (Retrieval-augmented Memory for Embodied Robots), built
from scratch with three goals the original does not have:

- **Provider-agnostic.** Embeddings and the reasoning model come from any backend
  behind one small interface: OpenAI-compatible APIs, Gemini, OpenRouter, local models.
- **Store-agnostic.** In-process vector stores (LanceDB, sqlite-vec) by default, so a
  robot needs no database server. Milvus and Qdrant as optional backends.
- **Works without a robot.** The core library takes frames, poses and timestamps. A
  video file plus a pose log is enough to build and query a memory on a laptop.
  ROS 2 is a thin wrapper, not a dependency of the core.

## How it works

1. **Memory building** runs while the robot drives in a known map. Every few seconds a
   frame is captioned and embedded; the entry stores the caption, a keyframe
   reference, the timestamp and the robot pose in the map frame. Near-duplicate
   entries are dropped, so a robot standing still produces one memory, not hundreds.
2. **Querying** is an agent loop. The model has three retrieval tools, similarity,
   time range and position radius, and calls them until it can answer with evidence.
   The answer carries the position and time it is based on.

Memories are built after mapping, not during it, because the map frame shifts while
SLAM is still closing loops.

## Status

Design stage. Interfaces first, adapters next, ROS 2 wrapper last.

## License

Apache-2.0
