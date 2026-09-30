# Benchmarks

Measurement harnesses that stay out of the installed `placecell` package. Run them from
the repository root with the source tree on the path.

## Store scaling

`store_scaling.py` loads a synthetic collection of each requested size and measures:

- **ingest**: observations per second through `Ingester` with contradiction checks, as the
  ROS node runs it; about 40 % are revisits that merge into an existing memory;
- **search**: p50/p95 latency of `search(k=10)` without a filter and with a robot, a
  place (5 m radius) and a one-day time filter, outside and inside `store.transaction()`;
- **recall@10** of each search against exact brute force over a snapshot of the store;
- **reads beside a writer**: search latency while another thread ingests, and that
  writer's throughput.

The data is reproducible from `--seed`: clustered unit vectors (members of a cluster have
cosine similarity near 0.64, so nearest neighbours are not trivial), poses on a
100 x 100 m map per robot and one to three sightings per memory over 30 days.

```bash
PYTHONPATH=src python benchmarks/store_scaling.py --backend lancedb --sizes 10k,100k --output scaling.json
PYTHONPATH=src python benchmarks/store_scaling.py --backend memory --sizes 10k --dim 512 --robots 4
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--backend` | `lancedb` | `memory` (`InMemoryStore`) or `lancedb` (`LanceDBStore`) |
| `--sizes` | `10k,100k` | Comma-separated collection sizes; `k` and `m` suffixes work |
| `--dim` | 512 | Vector dimension |
| `--robots` | 4 | Robots writing to the collection, each with its own map |
| `--clusters` | 256 | Vector clusters |
| `--seed` | 0 | Data and query seed |
| `--queries` | 50 | Queries per filter type |
| `--ingest` | 500 | Observations in the ingest measurement |
| `--writes` | 300 | Observations ingested while the readers run |
| `--output` | none | Write the full report as JSON |
| `--workdir` | system temp | Where LanceDB collections are created and removed |

The script prints a Markdown table. Latencies are in milliseconds. LanceDB runs call
`maintain()` once after loading, like the ROS maintenance timer, and report its duration.
Run on an otherwise idle machine and compare runs made on the same one; pin the process
with `taskset` when other work shares the host.
