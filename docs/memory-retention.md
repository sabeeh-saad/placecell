# Day 13: bounded memory and conversation history

The reference deployment now enforces admission and history limits in the storage
transactions, independently of the maintenance timer. A full collection admits a new
identity by evicting its least valuable memory in the same transaction (see
[eviction at capacity](#eviction-at-capacity)); updates to existing identities never
evict. With eviction disabled, a full collection refuses new identities and the
refusal reaches the bounded ingestion retry queue, where failed jobs retain their
evidence and capacity.

## Memory and work limits

`StoreLimits` configures both `InMemoryStore` and `LanceDBStore`. ROS exposes the same
settings:

- `memory_max_records`: 10,000 scene/summary records, including superseded records.
  The limit is shared by every robot writing to the collection.
- `memory_evict_at_capacity` (`StoreLimits.evict_at_capacity`): `true`. At the record
  limit, a new memory replaces the least valuable one. `false` restores refusal.
- `memory_max_sightings`: 1,024 detailed sightings per memory. The object returned by
  `get()` still contains a preview of at most 64; use `sightings()` to page the rest.
- `refine_max_pending`: 256 refinement requests, including exhausted requests. Repeat
  requests for an existing entry coalesce even when the queue is full.
- `cleanup_max_pending`: 2,048 unique pending evidence deletions. At capacity, mutations
  that need another cleanup entry roll back. Cleanup must recover before more media
  can replace existing references. Maintenance drains between bounded batches and
  reserves room for evidence held by jobs. Object expiration releases one identity
  at a time and drains cleanup between identities, up to 128 per pass.

The existing `max_queue` bounds queued, in-flight and failed ingestion jobs together
(default 64). `object_max_records`, `object_max_views` and the bounded object event
history continue to limit object identities, crops and tracking events. These are
separate from the scene-memory limit. Set cleanup capacity large enough for the
retained views of one object and evidence held by unfinished jobs; an atomic release
that cannot fit is refused without losing its references.

A legacy collection above a newly lowered record limit remains readable and updatable,
and a LanceDB collection imported on open keeps every record. Each new memory then
replaces exactly one existing memory, or is refused with eviction disabled, until
maintenance or explicit deletion makes space. Sighting counts are enforced when that
memory receives sightings. The code does not silently delete an existing collection
to satisfy a smaller startup limit.

## Eviction at capacity

Admission reads per-robot record counts that SQLite triggers maintain in the writing
transaction, so its cost does not grow with the collection, and every delete path and
rollback keeps the counts exact. Existing state files gain the counts once on open.

Every robot that holds memories is entitled to an equal share of `memory_max_records`.
A robot at or above its share gives up one of its own memories; a robot below it takes
one from the robot holding the most. One robot therefore never pushes another below an
equal share, and a robot that joins later is not starved by an older one. Within that
robot, the victim is:

1. a superseded memory, which is already scheduled for deletion;
2. an episodic memory folded into a summary; its summary stays valid and keeps
   representing it, unlike an explicit deletion, which invalidates the summary;
3. otherwise, among its 32 least recently seen memories, the one with the lowest
   confidence after a seven-day half-life decay.

Each tier takes the least recently seen memory first. Memories written by the same
batch are never evicted; if nothing else is available, the write is refused. Eviction
uses the ordinary deletion path: sightings, refinement requests and the vector
projection are removed with the memory, a summary is invalidated when it is evicted
itself, and managed evidence enters the cleanup journal and is deleted after commit
only once no memory, object view or job references it. If the cleanup journal is
full, the whole write rolls back. An evicted navigation target is revalidated like any
other deleted target. Eviction makes no provider calls.

## Aging and evidence ownership

`memory_max_idle_s` defaults to 90 days without a sighting. Confidence and superseded
record expiry still apply. `memory_history_age_s` defaults to 90 days; each bounded
pruning pass retains the latest sighting. A persistent cutoff prevents an older
in-memory snapshot, caption correction or metadata update from restoring expired
sighting rows. Observation totals retain their lifetime counts. Replay suppression
for merged observations covers retained detailed history, not arbitrary old replays.

Automatic retention deletes only managed files, and only after the final scene,
object-view and job reference is gone. Caller-owned recordings survive automatic
expiry. Explicit `Curator.forget()` with an evidence remover remains an explicit
file-deletion operation. Cleanup failures retain their journal entries. In-memory
store closure retains its explicit cleanup contract: it queues evidence without deleting
files and rolls back if the cleanup queue cannot fit. Use `Curator.forget()` with a
remover to clear a large reference store in bounded batches before closing it.

The mission profile enables local maintenance every 60 seconds. This performs no
model calls. Maintenance reports removed memories, discredited memories and pruned
sightings. Existing library users must schedule their own curator and vector-store
maintenance passes; ROS exposes the two age settings and `curator_interval_s`.

## Conversation continuity

`MissionContext` defaults to 1,000 events, 2 MiB of accounted retained content and
30 days of retention. ROS uses `mission_context_max_events`,
`mission_context_max_bytes` and `mission_context_retention_s`. Limits apply to the
whole database across scopes, so opening more conversations cannot bypass capacity.
Read windows remain isolated by robot, map and conversation.

Insertion and pruning share one SQLite transaction. Old requests are removed as
whole groups rather than leaving their statuses without an instruction. If the
current request alone cannot fit, its new event is refused and the transaction rolls
back. Existing persistence-failure handling still blocks new motion and preserves stop
handling. Reads and startup also enforce aging; expired context need not wait for the
timer. `stats()` reports retained events, accounted bytes and cumulative pruned events.

Planning and review retain their twenty-event, 16,000-character prompt budget.
Truncation and pruning now leave an explicit history boundary. A newer event that
cannot fit never causes an older destination to be substituted. The ROS adapter also
checks referenced scene/object identities: an unavailable reference removes that
request and older events from the prompt, leaving a boundary instead of an older
"last destination". Both agent prompts require clarification when the request depends
on missing context. Current target checks still apply after planning. This validates
history construction and refusal contracts; live-model handling of arbitrary language
remains part of the separate model-quality gate.

## Corrections

Correction logs default to 10,000 records and 4 MiB, exposed as
`correction_max_records` and `correction_max_bytes`. Memory IDs, questions and notes
are individually bounded. Files are replaced atomically before in-memory verdicts
change. A failed write cannot make an answer appear more trustworthy until restart.

At capacity, feedback is visibly refused instead of dropping old negative verdicts
for a retained memory. Maintenance removes feedback only for deleted scene memories;
all verdicts for retained or superseded scene memories remain. Unknown scene IDs are
refused by the ROS correction callback. A saved correction can outlive a refused
refinement enqueue, which produces its own warning. An existing JSONL file larger
than configured limits is refused intact; raise the limits or perform an explicit
review/import rather than silently discarding feedback. One process owns each log.

## Reproduction and evidence

Run `simulation/sim check-retention` after building the image. It is also in ROS CI.
The disposable container has networking disabled, blocks provider calls and uses
actual ROS parameters, DDS correction delivery, SQLite/LanceDB and node recreation.
Authored memory observations make the limits reproducible; it does not drive Nav2 or
claim camera/model accuracy.

The test makes 120 revisits and retains one identity, eight sightings and one managed
image. It submits 100 two-event mission histories and retains three whole requests:
six events, 738 accounted bytes and 194 pruned events. It also exercises capacity
refusal (with eviction disabled), failed-job ownership, correction overflow, evidence
release, aging and restart. Eviction itself is covered by the store unit tests.
Six expired objects are removed with a four-entry cleanup limit; the earlier bulk
expiration stalled at that limit, and its failed reproduction is retained.
The [Day 13 validation record](validation/day-13.json) records exact sources, image,
reports and the regression suite. Reproductions against the preceding commit showed:

- A stale metadata update restored all five sightings after pruning had retained one.
- 1,100 mission instructions produced 1,100 persistent rows despite the prompt window.
- A failed correction write changed the live weight to 1.0 while reopening retained 0.5.
- Automatic expiry deleted an externally owned recording.

Logical row/content limits are not a filesystem quota. SQLite may keep allocated free
pages, and vector versions, recordings and evaluation artifacts have separate storage
lifetimes. This checkpoint does not certify the 20 GiB campaign ceiling, 24-hour
endurance, process-crash recovery or live-model mission quality. Those gates remain
separate. The five deferred live product missions were not rerun for Day 13.
