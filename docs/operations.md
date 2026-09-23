# Memory operations and persistence

Operational behavior for long-running PlaceCell collections. For robot setup, see
[navigation](navigation.md); conversation and mission history are documented separately
in [missions](missions.md).

## Memory lifecycle

- **Reinforcement.** Seeing the same thing at the same place again strengthens the existing
  memory instead of adding one. Sighting ids and timestamps are retained for replay detection
  and time queries, even when the newest keyframe replaces the previous one. A superseded
  memory revives with its misses cleared if the object comes back.
- **Contradiction.** When the robot looks at a place from the same spot and heading and no
  longer sees what scene memory expects, that is a miss. Misses on separate visits add up,
  and after enough of them the scene memory is superseded. Scene similarity alone cannot
  distinguish occlusion from disappearance. Optional object tracking adds depth visibility
  checks and explicit visual absence verification before counting object misses.
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

## Continuous operation

Use a persistent `db_path` for restart recovery. Each collection now has a
`<collection>.state.sqlite3` file containing authoritative memory metadata, observation
history, ingestion jobs and cleanup intents. LanceDB supplies a derived vector index.
Schema 2–8 collections import into schema 9 in bounded batches when opened. The
original sighting history is retained during import; older clients reject schema 9.
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

## Collection compatibility and evidence

Time queries match actual sighting timestamps, not the interval between the first and last
visit. Results expose matching times through `RankedMemory.observed_at`; agent tool results
and ROS answers include `observed_at` and `last_seen`. Nearby agent queries honor `map_id`.

Existing schema 2–8 collections are upgraded to schema 9 when opened. The upgrade retains
stored rows, captions, evidence and lifecycle counts. It can preserve the recorded first
and last times, but cannot reconstruct intermediate sightings or observation ids that the
older schema discarded. Replay detection for merged observations is complete for sightings
ingested after the upgrade.

Schema 5 keeps the retained image, capture pose, timestamp, caption and embedding together.
Nearby views merge only for the same robot and camera with compatible headings, within a
fixed place anchor. Legacy image–pose pairings and localization quality cannot be recovered;
those memories need a new checked observation before navigation can use them.

Schema 6 adds vector modality and independent caption vectors. Older vectors remain
searchable through the primary channel; re-embed saved frames and captions to populate
both channels. The upgrade does not infer vector modality from stored image references.

Schema 8 records the age of measured object positions separately from RGB sightings.
Older object positions keep an unknown measurement time until a new depth observation.
Optional [Nav2 approach planning](approach.md) checks a stopping pose near a localized
object against the costmap and planner. The [RGB-D recording evaluator](object-evaluation.md)
measures tracking against human labels without changing the robot's live collection.

New memories start with an evidence weight of 0.5. Repeat frames from the same visit do not
increase it; a revisit after a gap of at least ten minutes can increase it toward a ceiling
of 0.8. This weight is a retention and ranking heuristic, not a probability that a caption
is correct. Visual destination checks and fresh arrival checks are described in the
[navigation guide](navigation.md), together with the required localization topic.

Keyframes produced by the video and ROS sources are marked as managed files. Ingestion
removes rejected or replaced managed files once no memory references them; caller-supplied
evidence is unmanaged by default. The ROS curator removes evidence after deleting its final
memory reference. Failed library ingestion keeps pending keyframes and rolls back segmentation
so the same observations can be retried; completed writes are recognized before captioning.
