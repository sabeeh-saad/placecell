# Changelog

## Unreleased

- Add 23 deterministic mission scenarios and a CI execution checkpoint requiring 100 distinct mission cases and 1,000 executions; report component checks separately.

- Bound memory admission, detailed sightings, refinement requests and evidence cleanup; keep job-owned images and externally owned recordings safe during retention.
- Persist conversation row/content/age limits with whole-request pruning and explicit missing-history boundaries; exclude unavailable target references from follow-up context.
- Bound correction logs, commit files before changing verdicts, and prune only feedback for deleted memories; add a no-cost ROS retention/restart check to CI.

- Ground configured place names and purpose-based search queries without relaxing destination checks; prefer the learned viewing side and visually verify the unique crop selected by embedding, rival and geometry checks.
- Use reliable RGB-D transport in simulation, refresh object geometry through normal arrival ingestion, and start Nav2 after lifecycle service discovery settles.
- Add a selectable three-product Gazebo scene and bounded live multi-stop evaluation, scoring exact visit order, purpose-based descriptions, repeated visits and an absent target with camera and decision-trace evidence.
- Distinguish object evidence changes from scan scheduling, run independent arrival checks together, omit unused arrival caption embeddings, and retry expired captures within the original deadline.
- Keep arrival capture available under ingestion backpressure; retain observed surface samples for removal/movement checks and upgrade existing collections to schema 9.
- Reject tiny color artifacts in the deterministic Gazebo detector and keep sensor forwarding active during world-change service calls.
- Require usable localized geometry for object updates in configured RGB-D pipelines; incomplete captures remain scene-only and do not consume the object scan interval.
- Wait for aligned depth before starting object arrival verification, retaining the original arrival deadline and capture-age checks.
- Rate-limit Nav2 distance feedback and coalesce pending trace progress; prioritize critical events and expose critical-drop counts under bounded queue saturation.
- Add bounded live-provider and deterministic-provider Gazebo mission checkpoints, including duplicate commands, stops, sensor loss and changed targets.
- Prefer newer complete RGB-D captures after a dropped pair's wait expires; keep recent depth trust until its existing freshness deadline without refreshing it from RGB-only frames.
- Recheck target references immediately before dispatch and after arrival checks; bound arrival capture age through queueing and provider work, including paused source time.
- Compare current and pre-departure lookalikes across category labels, reject conflicting object-to-scene fallback, and invalidate identity verdicts when evidence changes during final checks.
- Expose retrieval, identity, geometry and execution failure stages in operator status, snapshots, mission history and traces; add target regressions and nine repeatable fault scenarios.
- Guard navigation with live sensor freshness and trust generations; lost/recovered localization or camera/depth cannot turn an interrupted mission into success.
- Latch backward/source-clock changes, block new captures/goals until a fresh run, validate capture-time TF and image timestamps, and use steady RGB-D waits.
- Add 13 sensor-provenance fault scenarios and an isolated DDS sensor/clock check through the production node, included in CI.
- Reject ambiguous or malformed provider replies: duplicate JSON fields, refusals beside decisions, missing completion evidence, unsupported actions, extra visual fields and oversized content.
- Bound HTTP responses and model task data, validate retry/timeouts, and keep observation text separate from trusted model instructions. Live-model semantic robustness remains unqualified.
- Add 14 provider-contract fault scenarios, malformed-response checks over ROS, and regressions proving late invalid/failed model replies cannot revive stopped work.
- Add version 2 scoped command IDs, durable bounded retry suppression, conflicting-reuse refusal and command receipts. Preserve deliberate repeated visits and legacy input.
- Bind identified stop/choice commands to the current request; keep text stop available during journal admission and prevent a pending write from starting work after stop.
- Exercise command expiry, capacity, clock rollback, concurrent claims, crash reservations and real DDS retries/restart; active Nav2 reconciliation remains separate work.
- Preserve cancellation intent atomically with terminal Nav2 results, enforce deadlines even when polling is delayed, and prevent stale trip callbacks from canceling a replacement trip.
- Send cancellation before stop-status persistence; isolate ROS commands/action callbacks and use steady-clock deadline timers with a four-thread executor.
- Add controlled ROS action/latency checks to CI and expand offline fault coverage to 46 scenarios. Define the startup ownership contract; crash reconciliation remains outstanding.
- Run Python and ROS/Gazebo CI for all pull requests and main pushes, with merge-queue/manual triggers; add offline ROS operator checks and retain test reports on failure.
- Build wheel/source distributions and validate both in separate clean environments across the Python CI matrix, including installed entry points, memory, mission evaluation, fault checks and trace export.
- Add strict version 1 JSON commands, versioned status events, and a retained mission snapshot with a read-only ROS service for reconnecting operators. Preserve text commands and expose controller instance/event ordering.
- Keep rejected inputs from replacing mission state; report arrival verification explicitly and clear single-goal choices on stop or expiry.
- Add optional persistent mission traces with correlated planning/review, retrieval and visual checks, Nav2 events, stage timings and reported provider usage; enable bounded trace capture in the reference mission profile.
- Add a read-only trace exporter and trace completeness checks to the offline fault suite. Diagnostic write failures and queue overflow remain separate from navigation decisions.
- Add an offline fault runner covering planning, review, visual verification, localization, Nav2, context persistence and interrupted SQLite writes, with per-case JSON reports and CI artifacts.
- Preserve transport-initiated cancellation in the mission controller: a late Nav2 success after a timeout no longer launches the next destination.

## [0.1.0-alpha.1](https://github.com/sabeeh-saad/placecell/releases/tag/v0.1.0-alpha.1) - 2026-09-18

First public alpha prerelease. Python package version: `0.1.0a1`.
See the [release notes](docs/releases/v0.1.0-alpha.1.md) for installation and current limitations.

- Add optional agent-planned navigation missions with independent plan review, ordered goal execution, clarification, and cancellation.
- Persist conversation context and mission outcomes by robot, map, and conversation; never replay movement automatically after restart.
- Publish mission and per-goal progress over `/placecell/navigation_status` while camera ingestion continues.
- Add mission documentation and an editable architecture diagram with a reproducible PNG exporter.
- Align Ruff versions across local development, pre-commit, and CI; fix Python 3.10 NumPy typing compatibility.

- Add AMCL/Nav2 office navigation and a live camera-to-memory-to-navigation test with saved RGB-D recordings and failure reports.
- Support Gemini image/text embeddings and structured object detection through OpenRouter.
- Let TF update while image callbacks wait, synchronize RGB-D arrivals, and allow brief transform delivery skew for footprints.
- Verify selected object crops with their original scene context; reject truncated verification replies.
- Preserve an object's identity across detector label changes only after strong visual, geometric and paired-image agreement; keep its established category searchable.
- Account for compact objects' surface relief in depth estimates while preserving uncertainty and discontinuity checks.
- Require LanceDB 0.38 or later for the scalar-index API used by durable memory.
- Add a bundled Gazebo office and RGB-D/lidar robot with Docker launch, keyboard driving, a velocity watchdog and live sensor/motion checks.
- Use inspectable RGB-D subscription callbacks for compatibility with ROS 2 Jazzy's callback validation.

- Add a native Gemini multimodal adapter with query/document formatting, bounded batches and retries; keep local CLIP as an optional adapter.
- Expose embedding backend selection and separate API-key configuration in ROS.
- Store independent image and caption embeddings in schema 6; retrieve candidates from both channels and expose their scores.
- Preserve both channels through reinforcement, refinement rollback, migration and index recovery; upgrade old collections without guessing vector modality.
- Add a labeled-recording evaluator comparing caption-only, image-only and combined retrieval, with per-query results and ranking metrics.

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
