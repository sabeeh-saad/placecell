"""Persistent store on LanceDB: a directory on disk, one table per collection, no server.

SQLite owns current metadata, observation history and mutation journals. LanceDB is a
recoverable vector projection. Filtered scalar queries use SQLite indexes and return only
the requested rows. Existing Lance collections are imported in bounded batches on open.

Needs `pip install placecell[lancedb]`.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from placecell.errors import ModelMismatchError, PlacecellError, ValidationError
from placecell.memory import SCHEMA_VERSION
from placecell.store.base import CollectionInfo, Filter, Hit
from placecell.store.codec import from_row as _from_row
from placecell.store.codec import to_row as _to_row
from placecell.store.state import StateStore

_MAX_IN_LIST = 500


class LanceDBStore(StateStore):
    def __init__(self, path: str | Path, info: CollectionInfo) -> None:
        if info.schema_version != SCHEMA_VERSION:
            raise ValidationError(f"this code expects schema version {SCHEMA_VERSION}, got {info.schema_version}")
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise PlacecellError("LanceDBStore needs lancedb: pip install placecell[lancedb]") from e
        self._path = Path(path).expanduser().resolve()
        self._path.mkdir(parents=True, exist_ok=True)
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
            if stored.schema_version in (2, 3, 4) and info.schema_version == SCHEMA_VERSION:
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
        self._projection_lock = threading.RLock()
        super().__init__(info, self._path / f"{info.name}.state.sqlite3")
        if not self._conn.execute("SELECT 1 FROM settings WHERE key='imported'").fetchone():
            for batch in self._table.search().limit(None).to_batches(batch_size=256):
                super().upsert(_from_row(row) for row in batch.to_pylist())
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

    def _sync_index(self) -> None:
        """Replay committed vector changes. SQLite remains authoritative after an index failure."""
        with self._projection_lock:
            while True:
                with self.transaction():
                    pending = self._conn.execute("SELECT id,generation FROM dirty_vectors LIMIT 256").fetchall()
                    if not pending:
                        return
                    rows, deleted = [], []
                    for item in pending:
                        memory = self.get(item[0])
                        if memory is None:
                            deleted.append(item[0])
                        else:
                            rows.append(_to_row(memory))
                if rows:
                    self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
                if deleted:
                    self._table.delete("id IN (" + ",".join(_quote(i) for i in deleted) + ")")
                with self.transaction():
                    self._conn.executemany(
                        "DELETE FROM dirty_vectors WHERE id=? AND generation=?", ((p[0], p[1]) for p in pending)
                    )

    def search(self, vector: ArrayLike, k: int, where: Filter | None = None) -> list[Hit]:
        scope = where or Filter()
        # A writer must see its uncommitted updates, and history filters belong to the state store.
        # Exact scans are paged and spatial filters are indexed before vectors are loaded.
        with self._lock:
            if (
                self._depth
                or scope.time_from is not None
                or scope.time_to is not None
                or scope.observation_id is not None
            ):
                return super().search(vector, k, scope)
        with self._projection_lock:
            query = np.asarray(vector, dtype=np.float32)
            if k < 1 or query.shape != (self.info.dimension,) or not np.all(np.isfinite(query)):
                raise ValidationError("invalid search size or query vector")
            if not np.linalg.norm(query):
                return []
            self._sync_index()
            from lancedb.query import LanceVectorQueryBuilder

            builder = self._table.search(query.tolist(), query_type="vector")
            if not isinstance(builder, LanceVectorQueryBuilder):  # pragma: no cover
                raise PlacecellError("unexpected vector query builder")
            builder = builder.distance_type("cosine")
            # These fields change without rewriting vector rows, so use the authoritative scan.
            if scope.evidence_uri is not None or scope.unconsolidated:
                return super().search(vector, k, scope)
            expr = _sql(scope)
            if expr:
                builder = builder.where(expr, prefilter=True)
            return [
                Hit(memory, 1.0 - float(row["_distance"]))
                for row in builder.limit(k).to_list()
                if (memory := self.get(row["id"])) is not None and scope.matches(memory)
            ]

    def maintain(self, vector_index_min_rows: int = 1000) -> None:
        """Refresh the projection, create an index at the threshold, and compact old versions.

        Schedule this outside camera callbacks. Small collections use exact search.
        """
        with self._projection_lock:
            self._sync_index()
            if self._table.count_rows() >= vector_index_min_rows and not any(
                "vector" in index.columns for index in self._table.list_indices()
            ):
                from lancedb.index import IvfFlat

                self._table.create_index(
                    "vector",
                    config=IvfFlat(distance_type="cosine", num_partitions=max(1, self._table.count_rows() // 4096)),
                )
            self._table.optimize()

    def rebuild_index(self) -> None:
        """Rebuild the derived vector table without changing authoritative memories or jobs."""
        with self._projection_lock:
            self._table.delete("id IS NOT NULL")
            with self.transaction():
                self._conn.execute(
                    "INSERT INTO dirty_vectors SELECT id,1 FROM memories WHERE 1 "
                    "ON CONFLICT(id) DO UPDATE SET generation=generation+1"
                )
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
