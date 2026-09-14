# Image and caption retrieval

`GeminiEmbedder` provides hosted image/text embeddings through Gemini Embedding 2.
`ClipEmbedder` is an optional local alternative. When a captioner is configured, ingestion
stores two independent vectors: one from the pixels and one from the caption. A frame
showing a printer can therefore match “printer” even if its caption only mentions a desk.
Accuracy still depends on the model, image, query and environment.

## Use Gemini

Install from this repository (the package has not been published to PyPI):

```bash
pip install -e '.[lancedb,video]'
export GEMINI_API_KEY="your-key"
placecell-ros2 --ros-args \
  -p embed_backend:=gemini \
  -p embed_model:=gemini-embedding-2 \
  -p embed_dimension:=768 \
  -p embed_api_key_env:=GEMINI_API_KEY \
  -p collection:=office_gemini
```

Add your existing camera, map, localization and caption-model parameters. Gemini uses the
native API endpoint automatically when `embed_base_url` is empty. `embed_api_key_env` lets
embedding authentication differ from the caption/chat provider; otherwise the Gemini
backend checks `GEMINI_API_KEY`, then the existing `api_key_env`. Credentials stay in the
environment. No SDK, PyTorch, CLIP weights or local model download is required.

```python
import os
from placecell.providers import GeminiEmbedder

embedder = GeminiEmbedder(api_key=os.environ["GEMINI_API_KEY"], dimension=768)
```

The adapter uses Google's [native multimodal embeddings API](https://ai.google.dev/gemini-api/docs/embeddings).
It supports `gemini-embedding-2` and the explicit `gemini-embedding-2-preview` model name,
with 768 (default), 1536 or 3072 dimensions. Each saved PNG/JPEG is uploaded as a separate
request inside a batch, so different frames never become one aggregated embedding.
Captions and search queries use the documented document/query prefixes. Recall calls the
optional `embed_queries` method automatically; `embed_text` embeds stored captions.

Requests are bounded by item count and serialized size, with configurable timeouts and
retry backoff for rate limits/server errors. Defaults are 16 items, 8 MB per image and
12 MB per batch. Unsupported formats, oversized files and malformed responses fail
explicitly. This adapter currently accepts frames, not audio, PDFs or video clips. The
collection identity includes the model, dimension and retrieval-format version. Switching
between Gemini and CLIP, or changing Gemini dimensions, requires re-embedding.

## Optional local CLIP

From this repository (the package has not been published to PyPI):

```bash
pip install -e '.[clip,lancedb,video]'
```

For a CPU deployment, install the CPU build of PyTorch for your platform before the extra,
using the [official installation selector](https://pytorch.org/get-started/locally/).
The optional extra adds Sentence Transformers, PyTorch and Pillow; the base installation
continues to work without them. Weights download on the first nonempty embedding request.
Preload them before starting the robot, then set `local_files_only=True` to use the cache
without network access. `cache_folder` controls where weights are stored.

```python
from placecell.providers import ClipEmbedder

embedder = ClipEmbedder()  # CPU, sentence-transformers/clip-ViT-B-32, 512 dimensions
embedder.embed_text(["printer"])  # preload model before accepting camera work
```

Supported checkpoints are `sentence-transformers/clip-ViT-B-32` (512 dimensions),
`sentence-transformers/clip-ViT-B-16` (512), and `sentence-transformers/clip-ViT-L-14` (768).
Choose `device="cuda"` explicitly to use a compatible GPU. For repeatable deployments,
set `revision` to the checkpoint's full commit hash when preloading, migrating, evaluating
and running ROS. The checkpoint and supplied revision form the collection identity;
changing either requires a separate collection. An omitted revision uses the upstream
default, which can change. Custom remote model code is disabled.

The adapter uses the documented [Sentence Transformers image-search interface](https://sbert.net/examples/sentence_transformer/applications/image-search/README.html).
It accepts local image paths and `file://` paths, applies EXIF orientation, converts to RGB,
and batches decoding/inference. Network image URLs and video clips are rejected; sample
video into frames with the existing video source. CLIP has a short text context: keep
queries and captions concise. Calls share one model instance and run serially, so measure
camera throughput and query latency on the robot's hardware.

To use it in ROS, add these parameters to your existing launch configuration:

```bash
placecell-ros2 --ros-args \
  -p embed_backend:=clip \
  -p embed_model:=sentence-transformers/clip-ViT-B-32 \
  -p embed_device:=cpu \
  -p embed_batch_size:=16 \
  -p collection:=office_clip
```

Keep your camera, map, localization and caption-model parameters from the existing setup.
`embed_revision`, `embed_cache_folder` and `embed_local_files_only` expose the corresponding
adapter options. `embed_dimension:=0` derives the dimension; an explicit incorrect dimension
fails startup. `embed_backend:=auto` preserves the existing text-provider behavior.
Without a captioner, image retrieval works, but caption retrieval has no caption to index.
With a captioner, captions are still generated automatically.

## Retrieval and stored state

```python
from placecell import Recall

recall = Recall(store, embedder)
combined = recall.similar("printer", k=5)  # default mode="combined"
image_only = recall.similar("printer", k=5, mode="image")
caption_only = recall.similar("printer", k=5, mode="caption")

for hit in combined:
    print(hit.memory.id, hit.image_similarity, hit.caption_similarity, hit.score)
```

Each channel contributes its own candidate list before confidence and correction weights
are applied. Combined retrieval uses the strongest cosine match for each memory, multiplied
by its decayed evidence weight and operator correction weight. It does not average image
and caption vectors. Recent candidates still enter before reranking, and results retain
the existing consolidation deduplication behavior. Scores from different modalities can
have different distributions even within one model: this maximum-score rule is a baseline
to evaluate, not a calibrated probability. Increasing `oversample` can improve reranking
coverage at additional query cost. Large LanceDB collections use approximate vector indexes.

`RankedMemory.image_similarity` and `caption_similarity` expose both raw cosine scores;
an absent channel is `None`. The ROS answer payload includes those fields and the selected
`similarity`. Direct store searches accept `channel="primary"`, `"image"`, or `"caption"`;
custom store implementations must support that keyword and apply filters before limiting.

Schema 6 records `embedding_kind` and an optional separate `caption_embedding`. The primary
embedding remains the media vector when supported, so reinforcement and scene-change
comparison continue to compare the same kind of observation. Both vectors move with the
retained image, caption and pose. Refinement rebuilds both and records both for guarded
rollback. Summaries carry caption vectors. SQLite owns committed vectors; LanceDB maintains
separate primary and caption indexes and can rebuild both after interruption.

Collections from schemas 2–6 upgrade when opened. Older untyped vectors have unknown modality
and remain available through combined retrieval's primary-vector fallback. Opening an old
collection does not infer modality from the presence of a JPEG or manufacture new vectors.
Explicit image-only/caption-only searches require known provenance. For manually constructed
memories, pass `kind="caption"` or `kind="image"` to `Memory.with_embedding`; its default
preserves unknown provenance.

## Re-embed existing recordings

Stop ingestion and maintenance first and back up the database and keyframe directory.
Create a new collection with the multimodal provider:

```python
from placecell import CollectionInfo, reembed
import os
from placecell.providers import GeminiEmbedder
from placecell.store.lancedb_store import LanceDBStore

embedder = GeminiEmbedder(api_key=os.environ["GEMINI_API_KEY"], dimension=768)
source = LanceDBStore.open("/path/to/db", "office")
target = LanceDBStore("/path/to/db", CollectionInfo("office_gemini", embedder.model_name, embedder.dimension))
try:
    report = reembed(source, target, embedder, batch_size=16)
    print(report)
finally:
    target.close()
    source.close()
```

Both captions and supported evidence are embedded. With Gemini, re-embedding sends those
captions and images to the configured API. Missing or unreadable frame files fail
the migration instead of silently producing a caption-only “multimodal” collection. Earlier
batches remain committed and can be retried. Memories with neither supported media nor text
are listed as skipped. IDs, capture poses, localization provenance and sighting history are
preserved; migration does not turn unchecked legacy views into navigation-ready evidence.

Migration copies media references, not image files or pending jobs. Keep the old collection
inactive: independent curators must not delete files shared across collections. Preserve a
separate image backup if the old collection must remain a usable archive. Change the ROS
collection only after verifying the report and evaluating the new collection.

## Measure retrieval on recordings

Label queries independently by inspecting recorded images. Use memory IDs from
`store.iter_query(Filter(role="episodic"), batch_size=256)` and record every acceptable
destination view for each query. Include objects omitted from captions, visually similar
objects, multiple instances and changed environments. Do not derive labels from generated
captions or retrieval results. Use separate sessions for tuning and final evaluation.

Create a JSON file with this shape, replacing the example IDs with actual recorded IDs:

```json
[
  {"query": "printer", "relevant_ids": ["robot1:front:1780000000000"]},
  {"query": "red cabinet", "relevant_ids": ["robot1:front:1780000030000"]}
]
```

```bash
placecell-evaluate \
  --db-path /path/to/db --collection office_gemini \
  --queries /path/to/queries.json --output /path/to/retrieval-report.json \
  --robot-id robot1 --map-id office-v1 --k 5
```

The evaluator defaults to `--backend gemini`, using `GEMINI_API_KEY`. Pass the deployment
`--model` and `--dimension`; `--api-key-env` selects a different environment variable. For
CLIP, select `--backend clip` explicitly and pass matching `--model`, `--revision`,
`--device`, `--cache-folder` and `--local-files-only` settings. The report contains per-query ranked IDs and
channel scores, hit rate@k (any relevant view found), recall@k (fraction of labeled views
found), MRR@k (reciprocal rank of the first relevant view), mean latency and channel coverage.
It also reports the combined-minus-caption hit-rate difference. Run with `--k 1` as well
as `--k 5`. Report files are never overwritten.

All three modes share the same collection, labels, filters and fixed clock, defaulting to
the collection's latest observation. The normal confidence and correction-free ranking
is evaluated, with one warm-up query per mode; model download/load time is excluded from
latency. The Python `evaluate_retrieval` function also accepts an explicit `now`. Evaluation
rejects missing/out-of-scope label IDs, legacy vectors of unknown modality, and collections
without image vectors. It measures known-destination retrieval; it does not measure
unknown-destination rejection, object visibility or navigation success.

The automated tests validate native API request/response handling and use controlled vectors to demonstrate omitted-caption recovery and
storage correctness. They are not evidence of improved accuracy on real robot recordings.

## Navigation remains verified

Image matches enter the existing destination-resolution path and return the corresponding
robot capture pose. They do not estimate an object's map position. Localization, age,
ambiguity, source-image verification and fresh arrival verification still apply. The
default navigation cosine threshold is unchanged and may be too strict for text-to-image
CLIP scores. Evaluate and tune it with labeled robot scenes before enabling motion; also
recheck reinforcement and contradiction thresholds for the new model. See the
[navigation guide](navigation.md).
