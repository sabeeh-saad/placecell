# Changelog

## Unreleased

- Data model: `Pose`, `Evidence`, `Memory` with deterministic ids, decayed confidence and a schema version.
- Provider contract with capabilities per media kind; `HashingEmbedder` for offline use; `OpenAICompatibleEmbedder` with batching, retries and an injectable transport.
- Store contract with push-down filters; `InMemoryStore` as the reference backend.
- Retrieval tools: similarity, time range, position radius, ranked by similarity times decayed confidence.
- Lifecycle: reinforcement of repeated observations, decay, retention curator, supersede and forget.
- Ingestion pipeline with segmentation, captioning, embedding and persistence stages.
- Sources: interpolated pose tracks from CSV, keyframes from video files.
