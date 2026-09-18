# Memory refinement and revision history

Refinement revisits retained visual evidence to update captions and embeddings. It
preserves the distinction between a new observation and a new interpretation of an
existing observation.

The feedback loop is observation → refinement → consolidation → retrieval → feedback.
For example, a memory first described as "a cabinet" can be rechecked against a later,
clearer frame and described as "a red fire equipment cabinet". The description and its
embedding change together. Summaries based on a changed memory are invalidated and become
eligible for rebuilding on the next consolidation pass.

## Library API

Given an existing `store`, compatible `embedder`, configured image captioner
(`careful_captioner`), and the `memory_id` to revisit:

```python
from placecell import MemoryRefiner, RefinementPolicy

refiner = MemoryRefiner(
    store,
    embedder,
    careful_captioner,
    RefinementPolicy(max_memories=8, max_attempts=3, keep_revisions=3),
    producer="vision-model-version",
)
report = refiner.run()  # process a bounded set of queued memories

# Revisit an existing memory, including after changing the captioning model.
store.refinements.request(memory_id, "operator correction")
report = refiner.run()
revisions = store.refinements.history(memory_id)
undone = refiner.rollback(memory_id)
```

Use a captioner that examines the source image carefully; it receives the evidence alone,
without previous captions, summaries or correction notes. Text embeddings are rebuilt from
the new caption; a provider with media support also embeds the image, matching ingestion. This
is evidence-driven maintenance, not model training, and a different caption is not proof of
better accuracy. Evaluate the descriptions against actual robot scenes. Operator verdicts
remain in force until later operator feedback changes their effect.

## Persistence and rollback

The queue and revision history persist with a disk-backed store. Requests for one memory
coalesce, attempts are reserved before model calls, and failures wait at least five minutes
before retrying. After three attempts, a request stays available for inspection through
`store.refinements.pending()`; another explicit request or changed image resets its budget.
Completed work is not repeated automatically. Equal nonempty image digests suppress rechecks
when a new filename contains the same pixels; without a digest, evidence identity is based
on its URI and metadata. Caller-owned files should be immutable or carry updated digests.

Each successful change retains its previous caption and vector, source evidence reference,
producer and reason. The default keeps three revisions per memory, deleted with that memory.
Undo applies to the latest refinement only, provided its caption, vector and evidence are
still current. It preserves subsequent lifecycle updates and cancels pending rechecks.
Revision history does not retain old image files. A concurrent sighting, deletion or other
memory update prevents an outdated refinement from committing.

## ROS configuration

In ROS, refinement runs hourly when a caption model is configured, processing at most eight
requests per pass on the bounded maintenance worker. `refine_model` optionally selects a
different vision model at `caption_base_url`; otherwise it uses `caption_model` with a
careful-description prompt and high image detail. Set `refine_interval_s:=0.0` to disable
execution, or adjust `refine_interval_s` and `refine_batch_size` to budget provider work.
A wrong verdict on `/placecell/correct` requests an evidence recheck for that episodic memory.
You can also publish `{"memory_id":"...","action":"recheck"}` or
`{"memory_id":"...","action":"rollback"}` on `/placecell/refine`. Rechecks run on the next
maintenance pass; rollback requires refinement to be enabled. Summary rebuilding remains
controlled separately by `consolidate_interval_s` and requires a chat model.

See [memory operations](operations.md) for retention and recovery, and [image and caption retrieval](multimodal.md) for embedding configuration.
