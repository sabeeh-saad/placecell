"""The reasoning loop: a chat model with the three retrieval tools, ending in a grounded answer.

The model never sees vectors. It calls tools, reads the memories they return, and finishes
by calling `answer` with the ids of the memories it relied on. The agent keeps no state
between questions, so any number of them can run side by side.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from placecell.chat import ChatMessage, ChatModel, ToolCall
from placecell.errors import ValidationError
from placecell.memory import Pose
from placecell.retrieval import RankedMemory, Recall


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    evidence: list[RankedMemory] = field(default_factory=list)
    steps: int = 0
    grounded: bool = False
    """True only when every citation names a retrieved memory and at least one is cited."""


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memories",
            "description": "Find memories whose content resembles a description. Best first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'red fire extinguisher'"},
                    "k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memories_between",
            "description": "Memories observed in a time window, oldest first. Times are unix seconds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "time_from": {"type": "number"},
                    "time_to": {"type": "number"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "required": ["time_from", "time_to"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memories_near",
            "description": "Memories observed within a radius of a map position, oldest first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "radius_m": {"type": "number", "exclusiveMinimum": 0, "default": 2.0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Finish. Give the answer and the ids of the memories it is based on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "memory_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "memory_ids"],
            },
        },
    },
]

SYSTEM_PROMPT = """You answer questions about what a mobile robot has seen, using its memory.
Each memory has an id, a time, a map position (x, y in metres, yaw in radians) and a caption.
Use the tools to look things up; do not guess. Convert relative times ("this morning") using the
current time given below. When you have enough, call `answer` with a short reply and the ids of
the memories it rests on. If nothing relevant exists, say so in `answer` with an empty id list.
Current time: {now_iso} (unix {now_unix:.0f}). Map frame: {frame}."""


class Agent:
    def __init__(
        self,
        recall: Recall,
        model: ChatModel,
        frame_id: str = "map",
        max_steps: int = 8,
        clock: Callable[[], float] = time.time,
        map_id: str = "",
    ) -> None:
        if max_steps < 1:
            raise ValidationError("max_steps must be at least 1")
        self._recall = recall
        self._model = model
        self._frame_id = frame_id
        self._map_id = map_id
        self._max_steps = max_steps
        self._clock = clock

    def ask(self, question: str) -> Answer:
        if not question.strip():
            raise ValidationError("question must not be empty")
        now = self._clock()
        system = SYSTEM_PROMPT.format(
            now_iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(timespec="seconds"),
            now_unix=now,
            frame=self._frame_id,
        )
        messages = [ChatMessage("system", system), ChatMessage("user", question)]
        seen: dict[str, RankedMemory] = {}
        for step in range(1, self._max_steps + 1):
            reply = self._model.complete(messages, TOOLS)
            if not reply.tool_calls:
                text = (reply.content or "").strip()
                return Answer(text or "No answer.", [], step, grounded=False)
            messages.append(ChatMessage("assistant", reply.content, reply.tool_calls))
            for call in reply.tool_calls:
                if call.name == "answer":
                    ids = call.arguments.get("memory_ids")
                    citations = (
                        list(dict.fromkeys(ids))
                        if isinstance(ids, list) and all(isinstance(i, str) for i in ids)
                        else []
                    )
                    evidence = [seen[i] for i in citations if i in seen]
                    grounded = bool(citations) and len(evidence) == len(citations)
                    return Answer(str(call.arguments.get("text", "")).strip(), evidence, step, grounded=grounded)
                result, found = self._run(call)
                for r in found:
                    seen.setdefault(r.memory.id, r)
                messages.append(ChatMessage("tool", result, tool_call_id=call.id))
        return Answer("I could not find an answer within the allowed number of steps.", [], self._max_steps, False)

    def _run(self, call: ToolCall) -> tuple[str, list[RankedMemory]]:
        """Execute one tool call. Errors go back to the model as text, never up the stack."""
        a = call.arguments
        try:
            if call.name == "search_memories":
                found = self._recall.similar(str(a["query"]), k=int(a.get("k", 5)))
            elif call.name == "memories_between":
                found = self._recall.between(float(a["time_from"]), float(a["time_to"]), limit=int(a.get("limit", 20)))
            elif call.name == "memories_near":
                pose = Pose(float(a["x"]), float(a["y"]), frame_id=self._frame_id, map_id=self._map_id)
                found = self._recall.near(pose, float(a.get("radius_m", 2.0)), limit=int(a.get("limit", 20)))
            else:
                return json.dumps({"error": f"unknown tool {call.name}"}), []
        except (KeyError, TypeError, ValueError) as e:
            return json.dumps({"error": f"bad arguments for {call.name}: {e}"}), []
        return json.dumps([_describe(r) for r in found]), found


def _describe(r: RankedMemory) -> dict[str, Any]:
    m = r.memory
    out: dict[str, Any] = {
        "id": m.id,
        "time": datetime.fromtimestamp(m.timestamp, tz=timezone.utc).isoformat(timespec="seconds"),
        "x": round(m.pose.x, 2),
        "y": round(m.pose.y, 2),
        "yaw": round(m.pose.yaw, 2),
        "caption": m.caption,
        "confidence": round(r.confidence, 3),
        "seen": m.observations,
    }
    if r.similarity is not None:
        out["similarity"] = round(r.similarity, 3)
    return out
