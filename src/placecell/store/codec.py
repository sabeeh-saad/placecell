"""Serialization shared by the state store and vector projection."""

from __future__ import annotations

from typing import Any

import numpy as np

from placecell.memory import Evidence, EvidenceKind, Memory, Pose, Sighting


def to_row(m: Memory) -> dict[str, Any]:
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
        "misses": m.misses,
        "last_miss": m.last_miss,
        "role": m.role,
        "consolidated_into": m.consolidated_into,
        "schema_version": m.schema_version,
        "sighting_ids": [s.id for s in m.sightings],
        "sighting_times": [s.timestamp for s in m.sightings],
        "superseded_at": m.superseded_at,
        "evidence_managed": e.managed if e else False,
    }


def from_row(r: dict[str, Any]) -> Memory:
    evidence = (
        Evidence(
            EvidenceKind(r["evidence_kind"]),
            r["evidence_uri"],
            r["evidence_digest"],
            r["evidence_duration"],
            managed=r["evidence_managed"],
        )
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
        misses=int(r["misses"]),
        last_miss=r["last_miss"],
        role=r["role"],
        consolidated_into=r["consolidated_into"],
        schema_version=int(r["schema_version"]),
        sightings=tuple(Sighting(i, t) for i, t in zip(r["sighting_ids"], r["sighting_times"], strict=True)),
        superseded_at=r["superseded_at"],
    )
