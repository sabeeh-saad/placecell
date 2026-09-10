"""Persistent store on LanceDB: a directory on disk, one table per collection, no server.

Filters are translated to SQL and pushed into LanceDB, for scans and as a pre-filter of
vector searches, so a fleet-sized table is never pulled into Python. Collection identity
(model, dimension, schema version) lives in a small JSON file next to the table and is
checked on every open, so a collection can never be fed vectors from another model.

Needs `pip install placecell[lancedb]`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from placecell.errors import ModelMismatchError, PlacecellError, ValidationError
from placecell.memory import Evidence, EvidenceKind, Memory, Pose
from placecell.store.base import CollectionInfo, Filter, Hit

_MAX_IN_LIST = 500


class LanceDBStore:
    def __init__(self, path: str | Path, info: CollectionInfo) -> None:
        try:
            import lancedb
            import pyarrow as pa
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise PlacecellError("LanceDBStore needs lancedb: pip install placecell[lancedb]") from e
        self._path = Path(path)
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
            if stored.schema_version != info.schema_version:
                raise ValidationError(
                    f"collection {info.name!r} has schema version {stored.schema_version}, "
                    f"this code expects {info.schema_version}"
                )
            self._table = self._db.open_table(info.name)
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
                    pa.field("schema_version", pa.int64()),
                ]
            )
            self._table = self._db.create_table(info.name, schema=schema)
            self._meta_path.write_text(json.dumps(asdict(info), indent=2))
        self._info = info

    @classmethod
    def open(cls, path: str | Path, name: str) -> LanceDBStore:
        """Open an existing collection using the identity it was created with."""
        meta = Path(path) / f"{name}.collection.json"
        if not meta.exists():
            raise ValidationError(f"no collection {name!r} at {path}")
        return cls(path, CollectionInfo(**json.loads(meta.read_text())))

    @property
    def info(self) -> CollectionInfo:
        return self._info

    def upsert(self, memories: Iterable[Memory]) -> int:
        rows = []
        for memory in memories:
            self._check(memory)
            rows.append(_to_row(memory))
        if rows:
            self._table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)
        return len(rows)

    def get(self, memory_id: str) -> Memory | None:
        rows = self._table.search().where(f"id = {_quote(memory_id)}").limit(1).to_list()
        return _from_row(rows[0]) if rows else None

    def delete(self, ids: Iterable[str]) -> int:
        removed = 0
        batch = list(dict.fromkeys(ids))
        for start in range(0, len(batch), _MAX_IN_LIST):
            expr = f"id IN ({', '.join(_quote(i) for i in batch[start : start + _MAX_IN_LIST])})"
            removed += self._table.count_rows(expr)
            self._table.delete(expr)
        return removed

    def delete_where(self, where: Filter) -> int:
        expr = _sql(where) or "id IS NOT NULL"
        removed = self._table.count_rows(expr)
        if removed:
            self._table.delete(expr)
        return removed

    def query(self, where: Filter | None = None, limit: int | None = None) -> list[Memory]:
        if limit is not None and limit < 0:
            raise ValidationError("limit must not be negative")
        if limit == 0:
            return []
        query = self._table.search()
        expr = _sql(where or Filter())
        if expr:
            query = query.where(expr)
        rows = [_from_row(r) for r in query.limit(None).to_list()]
        rows.sort(key=lambda m: (m.timestamp, m.id))
        return rows[:limit] if limit is not None else rows

    def search(self, vector: ArrayLike, k: int, where: Filter | None = None) -> list[Hit]:
        if k < 1:
            raise ValidationError("k must be at least 1")
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (self._info.dimension,):
            raise ValidationError(f"query vector must have shape ({self._info.dimension},), got {query.shape}")
        if not np.linalg.norm(query):
            return []
        from lancedb.query import LanceVectorQueryBuilder

        builder = self._table.search(query.tolist(), query_type="vector")
        if not isinstance(builder, LanceVectorQueryBuilder):  # pragma: no cover - lancedb contract
            raise PlacecellError(f"unexpected query builder {type(builder).__name__}")
        builder = builder.distance_type("cosine")
        expr = _sql(where or Filter())
        if expr:
            builder = builder.where(expr, prefilter=True)
        return [Hit(_from_row(r), 1.0 - float(r["_distance"])) for r in builder.limit(k).to_list()]

    def count(self, where: Filter | None = None) -> int:
        expr = _sql(where or Filter())
        return int(self._table.count_rows(expr) if expr else self._table.count_rows())

    def close(self) -> None:
        """Release the connection. The data stays on disk; reopen with the same info or `open`."""
        del self._table
        del self._db

    def _check(self, memory: Memory) -> None:
        if memory.embedding is None:
            raise ValidationError(f"memory {memory.id} has no embedding")
        if memory.model != self._info.model:
            raise ModelMismatchError(
                f"collection {self._info.name!r} is bound to {self._info.model!r}, "
                f"memory {memory.id} was embedded by {memory.model!r}"
            )
        if memory.embedding.shape != (self._info.dimension,):
            raise ValidationError(
                f"memory {memory.id} has dimension {memory.embedding.shape[0]}, collection has {self._info.dimension}"
            )


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
    if where.time_from is not None:
        clauses.append(f"timestamp >= {where.time_from!r}")
    if where.time_to is not None:
        clauses.append(f"timestamp < {where.time_to!r}")
    if where.near is not None and where.radius is not None:
        p = where.near
        clauses.append(f"frame_id = {_quote(p.frame_id)}")
        clauses.append(f"map_id = {_quote(p.map_id)}")
        clauses.append(f"((x - {p.x!r}) * (x - {p.x!r}) + (y - {p.y!r}) * (y - {p.y!r})) <= {where.radius**2!r}")
    return " AND ".join(clauses) if clauses else None


def _to_row(m: Memory) -> dict[str, Any]:
    e = m.evidence
    return {
        "id": m.id,
        "robot_id": m.robot_id,
        "camera_id": m.camera_id,
        "timestamp": m.timestamp,
        "x": m.pose.x,
        "y": m.pose.y,
        "yaw": m.pose.yaw,
        "frame_id": m.pose.frame_id,
        "map_id": m.pose.map_id,
        "evidence_kind": e.kind.value if e else "",
        "evidence_uri": e.uri if e else "",
        "evidence_digest": e.digest if e else "",
        "evidence_duration": e.duration_s if e else 0.0,
        "caption": m.caption,
        "vector": m.embedding.tolist() if m.embedding is not None else None,
        "model": m.model,
        "confidence": m.confidence,
        "observations": m.observations,
        "last_seen": m.last_seen,
        "superseded": m.superseded,
        "schema_version": m.schema_version,
    }


def _from_row(r: dict[str, Any]) -> Memory:
    evidence = (
        Evidence(EvidenceKind(r["evidence_kind"]), r["evidence_uri"], r["evidence_digest"], r["evidence_duration"])
        if r["evidence_kind"]
        else None
    )
    return Memory(
        id=r["id"],
        robot_id=r["robot_id"],
        camera_id=r["camera_id"],
        timestamp=r["timestamp"],
        pose=Pose(r["x"], r["y"], r["yaw"], r["frame_id"], r["map_id"]),
        evidence=evidence,
        caption=r["caption"],
        embedding=np.asarray(r["vector"], dtype=np.float32),
        model=r["model"],
        confidence=r["confidence"],
        observations=int(r["observations"]),
        last_seen=r["last_seen"],
        superseded=bool(r["superseded"]),
        schema_version=int(r["schema_version"]),
    )
