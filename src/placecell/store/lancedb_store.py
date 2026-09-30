"""Persistent store on LanceDB: a directory on disk, one table per collection, no server.

SQLite owns current metadata, observation history and mutation journals. LanceDB is a
recoverable vector projection. Filtered scalar queries use SQLite indexes and return only
the requested rows. Existing Lance collections are imported in bounded batches on open.

Needs `pip install placecell[lancedb]`.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from placecell.errors import ModelMismatchError, PlacecellError, ValidationError
from placecell.memory import SCHEMA_VERSION, SearchChannel, Vector
from placecell.store.base import CollectionInfo, Filter, Hit
from placecell.store.codec import from_row as _from_row
from placecell.store.codec import to_row as _to_row
from placecell.store.limits import StoreLimits
from placecell.store.schema import check_file
from placecell.store.state import StateStore

_MAX_IN_LIST = 500
_SCALAR_COLUMNS = ("robot_id", "camera_id", "map_id", "frame_id", "superseded", "role")
"""Low-cardinality filter columns, indexed as bitmaps."""
EXACT_SEARCH_ROWS = 2048
"""Filtered sets up to this size are scored exactly in the state store instead of the vector index."""
REFINE_FACTOR = 5
"""Indexed searches rescore this many times k candidates with full-precision vectors."""
SEARCH_BACKLOG_ROWS = 256
"""Rows changed since the last index sync that a search scores exactly beside the index.

With more waiting, a search first syncs this many, or scores its whole filtered set exactly
when another sync is running."""


class LanceDBStore(StateStore):
    def __init__(self, path: str | Path, info: CollectionInfo, *, limits: StoreLimits | None = None) -> None:
        if info.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"this code expects schema version {SCHEMA_VERSION}, got {info.schema_version}")
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise PlacecellError("LanceDBStore needs lancedb: pip install placecell[lancedb]") from e
        self._path = Path(path).expanduser().resolve()
        self._path.mkdir(parents=True, exist_ok=True)
        check_file(self._path / f"{info.name}.state.sqlite3")
        self._db = lancedb.connect(str(self._path))
        self._meta_path = self._path / f"{info.name}.collection.json"
        if self._meta_path.exists():
            stored = CollectionInfo(**json.loads(self._meta_path.read_text()))
            if stored.model != info.model or stored.dimension != info.dimension:
                raise ModelMismatchError(
                    f"collection {info.name!r} at {self._path} is bound to {stored.model!r}/{stored.dimension}, "
                    f"not {info.model!r}/{info.dimension}"
                )
            self._table = self._db.open_table(info.name)
            if stored.schema_version in (2, 3, 4, 5, 6, 7, 8) and info.schema_version == SCHEMA_VERSION:
                if stored.schema_version == 2:
                    self._upgrade_sightings()
                else:
                    self._table.update(values={"schema_version": SCHEMA_VERSION})
            elif stored.schema_version != info.schema_version:
                raise ValidationError(
                    f"collection {info.name!r} has schema version {stored.schema_version}, "
                    f"this code expects {info.schema_version}"
                )
            missing = {
                k: v
                for k, v in {
                    "view_timestamp": "CAST(NULL AS DOUBLE)",
                    "localization_checked": "false",
                    "anchor_x": "x",
                    "anchor_y": "y",
                    "anchor_yaw": "yaw",
                    "embedding_kind": "'legacy'",
                    "caption_vector": "vector",
                }.items()
                if k not in self._table.schema.names
            }
            if missing:
                self._table.add_columns(missing)
        else:
            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("robot_id", pa.string()),
                    pa.field("camera_id", pa.string()),
                    pa.field("timestamp", pa.float64()),
                    pa.field("x", pa.float64()),
                    pa.field("y", pa.float64()),
                    pa.field("yaw", pa.float64()),
                    pa.field("frame_id", pa.string()),
                    pa.field("map_id", pa.string()),
                    pa.field("evidence_kind", pa.string()),
                    pa.field("evidence_uri", pa.string()),
                    pa.field("evidence_digest", pa.string()),
                    pa.field("evidence_duration", pa.float64()),
                    pa.field("caption", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), info.dimension)),
                    pa.field("caption_vector", pa.list_(pa.float32(), info.dimension)),
                    pa.field("embedding_kind", pa.string()),
                    pa.field("model", pa.string()),
                    pa.field("confidence", pa.float64()),
                    pa.field("observations", pa.int64()),
                    pa.field("last_seen", pa.float64()),
                    pa.field("superseded", pa.bool_()),
                    pa.field("misses", pa.int64()),
                    pa.field("last_miss", pa.float64()),
                    pa.field("role", pa.string()),
                    pa.field("consolidated_into", pa.string()),
                    pa.field("schema_version", pa.int64()),
                    pa.field("sighting_ids", pa.list_(pa.string())),
                    pa.field("sighting_times", pa.list_(pa.float64())),
                    pa.field("superseded_at", pa.float64()),
                    pa.field("evidence_managed", pa.bool_()),
                    pa.field("view_timestamp", pa.float64()),
                    pa.field("localization_checked", pa.bool_()),
                    pa.field("anchor_x", pa.float64()),
                    pa.field("anchor_y", pa.float64()),
                    pa.field("anchor_yaw", pa.float64()),
                ]
            )
            self._table = self._db.create_table(info.name, schema=schema)
            self._write_info(info)
        # Rows in any other column order make merge_insert rewrite whole fragments and drop their index coverage.
        self._columns = self._table.schema.names
        self._projection_lock = threading.RLock()
        self._index_path = self._path / f"{info.name}.index.json"
        self._indexes: dict[str, dict[str, int]] = (
            json.loads(self._index_path.read_text()) if self._index_path.exists() else {}
        )
        self._maintained: tuple[int, int] | None = None
        super().__init__(info, self._path / f"{info.name}.state.sqlite3", limits=limits)
        if not self._conn.execute("SELECT 1 FROM settings WHERE key='imported'").fetchone():
            # Existing memories are imported whole; capacity applies only to new observations.
            for batch in self._table.search().limit(None).to_batches(batch_size=256):
                self._write((_from_row(row) for row in batch.to_pylist()), admit=False)
            self._conn.execute("INSERT INTO settings VALUES ('imported','1')")
        self._sync_index()
        if not any("id" in index.columns for index in self._table.list_indices()):
            from lancedb.index import BTree

            self._table.create_index("id", config=BTree())
        self._write_info(info)

    def _write_info(self, info: CollectionInfo) -> None:
        pending = self._meta_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(asdict(info), indent=2))
        pending.replace(self._meta_path)

    def _write_indexes(self) -> None:
        """Record the size each vector index was trained at; a lost record only means an early rebuild."""
        pending = self._index_path.with_suffix(".json.tmp")
        pending.write_text(json.dumps(self._indexes, indent=2))
        pending.replace(self._index_path)

    @classmethod
    def open(cls, path: str | Path, name: str) -> LanceDBStore:
        """Open an existing collection using the identity it was created with."""
        meta = Path(path) / f"{name}.collection.json"
        if not meta.exists():
            raise ValidationError(f"no collection {name!r} at {path}")
        info = CollectionInfo(**json.loads(meta.read_text()))
        return cls(path, replace(info, schema_version=SCHEMA_VERSION))

    def _upgrade_sightings(self) -> None:
        """Upgrade version 2 in place, resuming safely if only the metadata write was interrupted."""
        expressions = {
            "sighting_ids": "make_array(id, '')",
            "sighting_times": "make_array(timestamp, last_seen)",
            "superseded_at": "CAST(NULL AS DOUBLE)",
            "evidence_managed": "false",
        }
        missing = {name: expr for name, expr in expressions.items() if name not in self._table.schema.names}
        if missing:
            self._table.add_columns(missing)
        self._table.update(
            where="superseded = true AND superseded_at IS NULL", values_sql={"superseded_at": "last_seen"}
        )
        self._table.update(values={"schema_version": SCHEMA_VERSION})

    def sync_index(self, limit: int = 1024) -> int:
        """Copy up to `limit` changed rows into the vector projection and return how many were copied.

        Searches never wait for it; they score rows changed since the last sync exactly. Call it
        every few seconds outside camera callbacks to keep that set small. `maintain()` copies all.
        """
        if type(limit) is not int or limit < 1:
            raise ValidationError("sync limit must be a positive integer")
        return self._sync_index(limit)

    def _sync_index(self, limit: int | None = None) -> int:
        """Replay committed vector changes. SQLite remains authoritative after an index failure."""
        synced = 0
        with self._projection_lock:
            while limit is None or synced < limit:
                batch = 256 if limit is None else min(256, limit - synced)
                with self._reading() as conn:
                    pending = conn.execute("SELECT id,generation FROM dirty_vectors LIMIT ?", (batch,)).fetchall()
                    if not pending:
                        break
                    rows, deleted = [], []
                    for item in pending:
                        memory = self.get(item[0])
                        if memory is None:
                            deleted.append(item[0])
                        else:
                            row = _to_row(memory)
                            rows.append({name: row[name] for name in self._columns if name in row})
                if rows:
                    self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
                if deleted:
                    self._table.delete("id IN (" + ",".join(_quote(i) for i in deleted) + ")")
                with self.transaction():
                    self._conn.executemany(
                        "DELETE FROM dirty_vectors WHERE id=? AND generation=?", ((p[0], p[1]) for p in pending)
                    )
                synced += len(pending)
        return synced

    def search(
        self, vector: ArrayLike, k: int, where: Filter | None = None, *, channel: SearchChannel = "primary"
    ) -> list[Hit]:
        if channel not in {"primary", "image", "caption"}:
            raise ValidationError("search channel must be primary, image or caption")
        scope = where or Filter()
        query = np.asarray(vector, dtype=np.float32)
        if k < 1 or query.shape != (self.info.dimension,) or not np.all(np.isfinite(query)):
            raise ValidationError("invalid search size or query vector")
        if not np.linalg.norm(query):
            return []
        # A writer must see its uncommitted updates, and history filters and fields that change
        # without rewriting vector rows belong to the state store. Small sets are scored exactly.
        if (
            self._in_transaction()
            or scope.time_from is not None
            or scope.time_to is not None
            or scope.observation_id is not None
            or scope.evidence_uri is not None
            or scope.unconsolidated
            or self._count_upto(scope, channel, EXACT_SEARCH_ROWS + 1) <= EXACT_SEARCH_ROWS
        ):
            return super().search(vector, k, scope, channel=channel)
        # Rows changed since the last index sync are scored from state. A search never waits for
        # another sync; it shortens a long backlog itself or scores its whole set exactly.
        unit = query / np.linalg.norm(query)
        unsynced = self._unsynced(unit, k, scope, channel)
        if unsynced is None and self._projection_lock.acquire(blocking=False):
            try:
                self._sync_index(SEARCH_BACKLOG_ROWS)
            finally:
                self._projection_lock.release()
            unsynced = self._unsynced(unit, k, scope, channel)
        if unsynced is None:
            return super().search(vector, k, scope, channel=channel)
        changed, fresh = unsynced
        column = "caption_vector" if channel == "caption" else "vector"
        expr = _sql(scope)
        if channel == "image":
            expr = f"({expr or 'true'}) AND embedding_kind = 'image'"
        elif channel == "caption":
            expr = f"({expr or 'true'}) AND embedding_kind <> 'legacy' AND caption_vector IS NOT NULL"
        if changed:
            # Their index entries may be stale or deleted.
            expr = f"({expr or 'true'}) AND id NOT IN ({','.join(_quote(i) for i in sorted(changed))})"
        partitions = self._indexes.get(column, {}).get("partitions", 0)
        probes = _probes(partitions) if partitions else None
        rows = self._nearest(query, column, expr, k, probes)
        while probes is not None and len(rows) < k and probes < partitions:
            # A selective filter can leave fewer than k matches in the partitions probed.
            probes = min(partitions, 4 * probes)
            rows = self._nearest(query, column, expr, k, probes)
        ranked = sorted(
            [*fresh, *((row["id"], 1.0 - float(row["_distance"])) for row in rows)], key=lambda v: (-v[1], v[0])
        )
        with self._reading():
            return [
                Hit(memory, score)
                for identity, score in ranked[:k]
                if (memory := self.get(identity)) is not None and scope.matches(memory)
            ]

    def _unsynced(
        self, query: Vector, k: int, where: Filter, channel: SearchChannel
    ) -> tuple[set[str], list[tuple[str, float]]] | None:
        """Rows changed since the last sync and the k best of them the filter keeps, scored from state.

        None when more than `SEARCH_BACKLOG_ROWS` rows wait.
        """
        with self._reading() as conn:
            changed = {
                row[0] for row in conn.execute("SELECT id FROM dirty_vectors LIMIT ?", (SEARCH_BACKLOG_ROWS + 1,))
            }
            if len(changed) > SEARCH_BACKLOG_ROWS:
                return None
            return changed, self._best_changed(conn, query, k, where, channel)

    def _nearest(
        self, query: Vector, column: str, expr: str | None, k: int, probes: int | None
    ) -> list[dict[str, Any]]:
        from lancedb.query import LanceVectorQueryBuilder

        builder = self._table.search(query.tolist(), query_type="vector", vector_column_name=column)
        if not isinstance(builder, LanceVectorQueryBuilder):  # pragma: no cover
            raise PlacecellError("unexpected vector query builder")
        builder = builder.distance_type("cosine").select(["id", "_distance"]).limit(k)
        if probes is not None:
            # Quantized distances pick candidates; the stored vectors rank them.
            builder = builder.nprobes(probes).refine_factor(REFINE_FACTOR)
        if expr:
            builder = builder.where(expr, prefilter=True)
        return builder.to_list()

    def maintain(self, vector_index_min_rows: int = 1000) -> None:
        """Copy every changed row into the projection, size its indexes to the collection, and compact old versions.

        A vector index is built once a column holds `vector_index_min_rows` vectors and
        retrained whenever that count has doubled since. Nothing is rebuilt or compacted when
        no row changed since the last pass. Schedule this outside camera callbacks.
        """
        with self._projection_lock:
            self._sync_index()
            if self._maintained == (self._table.version, vector_index_min_rows):
                return
            from lancedb.index import Bitmap, IvfSq

            indexed = {column for index in self._table.list_indices() for column in index.columns}
            for column in _SCALAR_COLUMNS:
                if column not in indexed:
                    self._table.create_index(column, config=Bitmap())
            for column in ("vector", "caption_vector"):
                rows = self._table.count_rows(f"{column} IS NOT NULL")
                built = self._indexes.get(column, {}).get("rows", 0) if column in indexed else 0
                if rows >= max(1, vector_index_min_rows) and rows >= 2 * built:
                    partitions = _partitions(rows)
                    self._table.create_index(
                        column, config=IvfSq(distance_type="cosine", num_partitions=partitions), replace=True
                    )
                    self._indexes[column] = {"rows": rows, "partitions": partitions}
                    self._write_indexes()
            self._table.optimize()
            self._maintained = (self._table.version, vector_index_min_rows)

    def rebuild_index(self) -> None:
        """Rebuild the derived vector table without changing authoritative memories or jobs.

        The vector indexes are retrained by the next `maintain()`.
        """
        with self._projection_lock:
            self._indexes.clear()
            self._write_indexes()
            # Marked before the rows go, so searches meanwhile score every memory from state.
            with self.transaction():
                self._conn.execute(
                    "INSERT INTO dirty_vectors SELECT id,1 FROM memories WHERE 1 "
                    "ON CONFLICT(id) DO UPDATE SET generation=generation+1"
                )
            self._table.delete("id IS NOT NULL")
            self._sync_index()

    def close(self) -> None:
        self._sync_index()
        with self._lock:
            super().close()
            del self._table
            del self._db


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _sql(where: Filter) -> str | None:
    """The filter as a DataFusion SQL predicate, or None when it keeps everything."""
    clauses: list[str] = []
    if not where.include_superseded:
        clauses.append("superseded = false")
    if where.robot_id is not None:
        clauses.append(f"robot_id = {_quote(where.robot_id)}")
    if where.camera_id is not None:
        clauses.append(f"camera_id = {_quote(where.camera_id)}")
    for field in ("role", "frame_id", "map_id"):
        value = getattr(where, field)
        if value is not None:
            clauses.append(f"{field} = {_quote(value)}")
    if where.unconsolidated:
        clauses.append("consolidated_into = ''")
    if where.observation_id is not None:
        clauses.append(f"array_has(sighting_ids, {_quote(where.observation_id)})")
    if where.evidence_uri is not None:
        uri = where.evidence_uri.removeprefix("file://")
        clauses.append(f"evidence_uri IN ({_quote(uri)}, {_quote('file://' + uri)})")
    if where.time_from is not None and where.time_to is not None:
        # Inserting a bound into the sorted times gives its lower-bound position.
        # Different positions mean at least one sighting lies in [from, to), including gaps correctly.
        def position(bound: float) -> str:
            return f"array_position(array_sort(array_append(sighting_times, {float(bound)!r})), {float(bound)!r})"

        clauses.append(f"{position(where.time_to)} > {position(where.time_from)}")
    elif where.time_from is not None:
        clauses.append(f"array_max(sighting_times) >= {float(where.time_from)!r}")
    elif where.time_to is not None:
        clauses.append(f"array_min(sighting_times) < {float(where.time_to)!r}")
    if where.near is not None and where.radius is not None:
        p = where.near
        clauses.append(f"frame_id = {_quote(p.frame_id)}")
        clauses.append(f"map_id = {_quote(p.map_id)}")
        clauses.append(f"((x - {p.x!r}) * (x - {p.x!r}) + (y - {p.y!r}) * (y - {p.y!r})) <= {where.radius**2!r}")
    return " AND ".join(clauses) if clauses else None


def _partitions(rows: int) -> int:
    """IVF partitions for a column of `rows` vectors: about the square root, so each holds about as many."""
    return max(1, min(4096, round(math.sqrt(rows))))


def _probes(partitions: int) -> int:
    """Partitions searched per query: a tenth of them, and at least 32."""
    return min(partitions, max(32, math.ceil(partitions / 10)))
