# Memory operations and persistence

Operational behavior for long-running PlaceCell collections. For robot setup, see
[navigation](navigation.md); conversation and mission history are documented separately
in [missions](missions.md).

Use the [backup, restore and upgrade runbook](backup-restore.md) before replacing
storage or software. `placecell-backup` requires stopped writers, validates the
configured journals and image references, and restores only into a fresh directory.
Restored command sessions and navigation admission have explicit recovery gates.

See [saturation and backpressure](overload.md) for admission limits, queue diagnostics,
maintenance coalescing and persistent provider cooldowns. These checks cover controlled
overload; the full provider workload and 24-hour endurance gate remain unqualified.

## Memory lifecycle

- **Reinforcement.** Seeing the same thing at the same place again strengthens the existing
  memory instead of adding one. Sighting ids and timestamps are retained for replay detection
  and time queries, even when the newest keyframe replaces the previous one. A superseded
  memory revives with its misses cleared if the object comes back.
- **Contradiction.** When the robot looks at a place from the same spot and heading and no
  longer sees what scene memory expects, that is a miss. Only memories from the same camera
  on the same robot are judged, and only against a vector of the same kind (image with image,
  caption with caption). Misses on separate visits add up, and after enough of them the
  scene memory is superseded. Scene similarity alone cannot distinguish occlusion from
  disappearance. Optional object tracking adds depth visibility checks and explicit visual
  absence verification before counting object misses.
- **Corrections.** An operator can mark an answer right or wrong, on `/placecell/correct` in
  ROS 2 or through `CorrectionLog` in the library. Wrong verdicts halve a memory's rank, and
  repeated ones get it superseded by the curator. The log is bounded and replaced
  atomically; feedback for retained memories is never silently evicted.
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

Navigation has a separate persistent `navigation_ownership_path` journal. First startup
with a missing journal refuses movement until an operator establishes a clean Nav2 server.
After a crash with a recorded goal, recovery queries/cancels only that UUID and waits for
a terminal result. See [crash recovery](crash-recovery.md) for setup, supervision and tests.
Keep this journal with the deployment's other persistent state; do not delete it to unblock
navigation or restore an older copy while Nav2 continues running.

Questions use `question_workers` (2) and `question_queue` (8). Overflow receives an explicit
busy response. Each question runs at most eight model steps and `chat_max_tool_calls` (16)
retrieval calls, returning at most 20 memories per call with captions cut to 300 characters
and five sighting times (the first and the four most recent). `chat_max_context_chars`
(40,000) bounds the transcript that is resent on every step: a result that does not fit is
cut behind an explicit `truncated` marker, its omitted memories cannot be cited, and the
model is then offered only the `answer` tool. A provider failure is published as an `error`
reply rather than an answer. Without `chat_model`, the best-matching caption is returned only
when its similarity reaches `answer_min_similarity` (0.5); otherwise the reply is
`No confident answer.` See the [answer schema](operator-interface.md#question-answers). Maintenance runs in one background worker with one waiting slot. The node
logs queued and failed jobs, oldest job age and dropped observations every 30 seconds.

Scene admission defaults to 10,000 records, and each memory retains at most 1,024 detailed
sightings. At the record limit a new memory evicts the least valuable one in the same
transaction, keeping each robot within an equal share; `memory_evict_at_capacity: false`
restores transactional refusal. Updates to existing records never evict.
See [memory retention](memory-retention.md) for configuration and cleanup limits.

Memory records contain at most 64 recent sightings. To read older retained events, page
through `store.sightings(memory_id, limit=64, after=(timestamp, observation_id))` until empty.
Time filters consult the full retained history, including gaps. `store.iter_query()` pages
memories in ID order for maintenance or migration; `query(limit=...)` orders and limits
inside the state store. Calling `query()` without a limit explicitly requests all matches.

Default retention expires memories after 90 days without a sighting, including reinforced
memories (`RetentionPolicy.max_idle_s`). Detailed sightings older than 90 days are pruned
in bounded passes (`history_age_s`), while retaining each memory's latest sighting. A stored
cutoff prevents stale metadata updates from restoring expired rows. These
settings limit history and inactivity, not bytes on disk; size them for the robot's storage
and observation rate. Disable the idle cap explicitly with `max_idle_s=None` if required.
Replay deduplication of merged observations is guaranteed within retained history.

Consolidation processes bounded groups and retains underlying observations. Updating or
deleting a supporting memory invalidates its summary and releases remaining members for
reconsolidation. Retrieval groups summaries with their members and combines semantic and
recent candidates before confidence and feedback ranking. This bounded candidate strategy
is approximate; evaluate recall and false contradictions on your own scenes.

The ROS maintenance pass calls `store.maintain()` to synchronize vectors, size the indexes
to the collection and compact database versions. Each vector column gets an IVF index with
scalar quantization and about √n partitions once it holds 1,000 vectors, and the index is
retrained whenever that count has doubled; `<collection>.index.json` records the size it
was trained at. Robot, camera, map, frame, role and superseded columns carry bitmap
indexes. A pass with no changed rows returns without work. Standalone callers should
schedule it themselves. `store.rebuild_index()` reconstructs the vector projection from
authoritative state after an indexing failure; the next `maintain()` retrains the indexes.
Back up or operate through the store API rather than editing its underlying tables.

A search whose filter leaves at most 2,048 rows, such as a place or time filter, is scored
exactly from the state store. Larger searches probe a tenth of the index partitions (at
least 32), rescore five times k candidates with full-precision vectors, and probe further
when a selective filter leaves fewer than k matches. `benchmarks/store_scaling.py` measures
latency and recall@10 at a given collection size.

Searches never write the vector index. Memories added, changed or deleted since the last
index sync are scored exactly from the state store and their possibly stale index entries
are skipped, so a write is searchable at once. The ROS node copies up to 1,024 changed rows
into the index every 2 s on its own worker (`store.sync_index()`), and `maintain()` copies
all of them. When more than 256 changed rows wait, a search first syncs 256 of them; if
another sync is running, or the backlog is still longer, it scores its filtered set exactly
from the state store instead.

Ingestion searches for merge candidates before it opens the write transaction, so the
transaction holds the store only for the write. Inside it the chosen candidate is read
again; if it changed or was deleted meanwhile, the merge is decided again. A memory that
became similar in between is not reconsidered and the observation is stored separately.

The state store is a SQLite database in WAL mode with a single writer connection. Reads
outside a transaction (`get`, `query`, `iter_query`, `search`, `count`, `sightings`) use a
pool of read-only connections, one per concurrently reading thread: they see committed
state, never wait for a write transaction and never hold one up. A thread inside
`store.transaction()` reads through the writer connection and sees its own uncommitted
changes. `InMemoryStore` cannot share its in-memory database between connections, so its
reads still wait for a running transaction.

## Provider credentials

The ROS node reads API keys only from environment variables named by parameters. The
shared `api_key_env` key (default `PLACECELL_API_KEY`) goes to the chat endpoint and to
any other endpoint whose base URL has the same scheme, host and port as `chat_base_url`.
An endpoint on another origin gets no key unless its own variable is named:

| Endpoint | Base URL | Key variable |
| --- | --- | --- |
| Chat, consolidation | `chat_base_url` | `chat_api_key_env` |
| Captioning, refinement | `caption_base_url` | `caption_api_key_env` |
| Visual verification | `verification_base_url`, else captioning | `verification_api_key_env` |
| Mission planning | `mission_base_url`, else chat | `mission_api_key_env` |
| Plan review | `mission_review_base_url`, else planning | `mission_review_api_key_env` |
| Embeddings | `embed_base_url`, else the backend default | `embed_api_key_env` |
| Object detection and arrival | `object_base_url` | `object_api_key_env` (`GEMINI_API_KEY`) |

Key variable parameters default to empty, except `object_api_key_env`. An endpoint with
an empty base URL uses the URL and key of the endpoint it falls back to. A named variable always wins; if it is
unset, no key is sent. The Gemini embedding backend tries `GEMINI_API_KEY` before the
shared key. For a single provider, set `chat_base_url` to it and export
`PLACECELL_API_KEY`. A second provider needs its own variable:

```bash
export REVIEW_API_KEY="your-key"
placecell-ros2 --ros-args \
  -p mission_review_base_url:=https://reviewer.example.com/v1 \
  -p mission_review_api_key_env:=REVIEW_API_KEY
```

Keys travel only over https, or over plain http to `localhost`, 127.0.0.0/8 or `::1`.
Configuring a key for any other http endpoint fails at startup; keyless http endpoints,
such as a local model server, still work. Provider redirects are never followed: a 3xx
response fails the request with its status and target, so no key reaches another host.
Mission traces redact every configured key.

## Model request options

Chat and mission requests send `max_tokens` and `temperature: 0` by default. Reasoning
models such as OpenAI's o-series and gpt-5 reject both: set the group's token parameter to
`max_completion_tokens` and a negative temperature, which omits the field. The model is never
guessed from its name. Reasoning tokens count against the budget, so raise it too:

| Parameter | Default | Applies to |
| --- | --- | --- |
| `chat_max_tokens`, `mission_max_tokens` | 400, 2048 | chat and consolidation; planning and review |
| `chat_token_parameter`, `mission_token_parameter` | `max_tokens` | or `max_completion_tokens` |
| `chat_temperature`, `mission_temperature` | 0.0 | a negative value omits `temperature` |
| `caption_max_tokens` | 1024 | captioning and refinement |
| `verification_max_tokens` | 2048 | visual verification and search-query grounding |
| `verification_structured_output` | true | request strict JSON-schema verdicts |

```bash
placecell-ros2 --ros-args \
  -p mission_model:=o4-mini \
  -p mission_token_parameter:=max_completion_tokens \
  -p mission_temperature:=-1.0 \
  -p mission_max_tokens:=8192
```

Caption and verification budgets are caps, not targets: a caption needs about 80 tokens and
a verdict about 100, so ordinary models stop far below them. Thinking models spend hidden
reasoning tokens from the same budget first, so the earlier 120 and 512 could end a reply at
`length` before any answer. A reply cut off at the limit still fails, and a failed caption is
retried by the ingestion queue, so a too-small budget costs more than a generous one.

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
evidence is unmanaged by default. Automatic curation removes managed evidence after its
final memory, object-view and job reference is gone; externally owned recordings remain.
Failed library ingestion keeps pending keyframes and rolls back segmentation
so the same observations can be retried; completed writes are recognized before captioning.
