"""JSON payloads published on `~/answer`, built without ROS so their schema can be tested directly."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from placecell.agent import Agent
from placecell.errors import PlacecellError
from placecell.retrieval import Recall

ANSWER_SCHEMA_VERSION = 1
NO_CONFIDENT_ANSWER = "No confident answer."


def answer_error(question: str, error: str, error_type: str = "") -> str:
    return json.dumps(
        {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "type": "answer",
            "question": question,
            "error": error,
            "error_type": error_type,
        }
    )


def answer_payload(
    question: str, text: str, citations_valid: bool, evidence: Sequence[Any], source: str = "agent"
) -> str:
    return json.dumps(
        {
            "schema_version": ANSWER_SCHEMA_VERSION,
            "type": "answer",
            "source": source,
            "question": question,
            "answer": text,
            "citations_valid": citations_valid,
            "grounded": citations_valid,  # Deprecated alias with the same meaning.
            "evidence": [
                {
                    "id": r.memory.id,
                    "x": r.memory.pose.x,
                    "y": r.memory.pose.y,
                    "yaw": r.memory.pose.yaw,
                    "time": r.observed_at[0] if r.observed_at else r.memory.timestamp,
                    "last_seen": r.memory.last_seen,
                    "observed_at": list(r.observed_at or r.memory.sighting_times),
                    "caption": r.memory.caption,
                    "confidence": r.confidence,
                    "similarity": r.similarity,
                    "image_similarity": r.image_similarity,
                    "caption_similarity": r.caption_similarity,
                }
                for r in evidence
            ],
        }
    )


def answer_question(question: str, agent: Agent | None, recall: Recall, min_similarity: float) -> str:
    """Build one `~/answer` payload. Without an agent the best caption is used only when confident."""
    try:
        if agent is not None:
            result = agent.ask(question)
            return answer_payload(question, result.text, result.citations_valid, result.evidence)
        hits = [h for h in recall.similar(question, k=5) if h.similarity is not None and h.similarity >= min_similarity]
        if not hits:
            return answer_payload(question, NO_CONFIDENT_ANSWER, False, [], source="retrieval")
        return answer_payload(question, hits[0].memory.caption, True, hits, source="retrieval")
    except PlacecellError as e:
        return answer_error(question, str(e), type(e).__name__)
