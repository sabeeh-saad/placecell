# Changelog

## Unreleased

- Keep retained images, capture poses, captions and embeddings together; separate incompatible camera views and prevent place drift during repeated merges.
- Schema 5 records image capture time and localization provenance. Legacy views need a new observation before navigation can use them.
- Start memory weights conservatively and require separated visits for bounded confidence increases.
- Verify destination images against the requested place and check a fresh view after reaching a memory goal.
- Gate captures and navigation on recent localization covariance, cancel trips when localization is lost, and require a versioned map ID.

- Schema 4 separates indexed observation history from current memory state and imports older collections in pages.
- Make memory updates transactional and recover the vector projection from committed state after indexing failures.
- Bound scalar queries, migration and maintenance memory use; add idle expiry and incremental history retention.
- Invalidate summaries when members change and include recent candidates before retrieval reranking.
- Persist accepted ingestion jobs, retry failures, retain failed work and recover interrupted keyframe creation.
- Sample before encoding, refresh stationary views and bound question and maintenance workers.
- Defer evidence deletion until memory updates commit; protect images referenced by queued jobs.

- Require valid retrieved citations for grounded answers and honor the configured map in nearby queries.
- Validate store batches before writes and give summaries identities derived from their members.
- Schema 3 retains sighting identities and times, records supersession separately from last sighting,
  and upgrades existing schema 2 collections on open.
- Clear misses when observations reinforce a memory and keep shared evidence until its final reference is removed.
- Clean up generated keyframes rejected by segmentation, dropped by the worker, or replaced by reinforcement.
- Restore uncommitted segmentation state after ingestion failures and skip completed observations before captioning retries.

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
- Schema 2: memories carry `misses`, `last_miss`, `role` and `consolidated_into`.
- `Observer`: contradiction detection; memories not seen again from the same place and heading on several visits are superseded.
- `Correction`, `CorrectionLog`: operator verdicts weigh into retrieval and the curator; `/placecell/correct` topic in ROS 2.
- `Consolidator`, `ChatSummarizer`: clusters of sightings in one map cell become one summary memory.
- `reembed`: migrate a collection to another embedding model from captions and keyframes.
