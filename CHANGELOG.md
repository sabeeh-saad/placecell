# Changelog

## Unreleased

- Data model: `Pose`, `Evidence`, `Memory` with deterministic ids, decayed confidence and a schema version.
- Provider contract with capabilities per media kind; `HashingEmbedder` for offline use; `OpenAICompatibleEmbedder` with batching, retries and an injectable transport.
- Store contract with push-down filters; `InMemoryStore` as the reference backend.
- Retrieval tools: similarity, time range, position radius, ranked by similarity times decayed confidence.
- Lifecycle: reinforcement of repeated observations, decay, retention curator, supersede and forget.
- Ingestion pipeline with segmentation, captioning, embedding and persistence stages.
- Sources: interpolated pose tracks from CSV, keyframes from video files.
- `OpenAICompatibleCaptioner`: frame captions from a vision-language model over chat completions.
- `LanceDBStore`: persistent backend with filters pushed into scans and vector searches; the store test suite runs as a contract over both backends.
- `Agent` and `OpenAICompatibleChat`: tool-calling reasoning loop that ends in an answer citing memory ids.
- ROS 2: `placecell-ros2` node with TF pose lookup, background ingestion, and an ask/answer topic pair; conversions are testable without ROS.
