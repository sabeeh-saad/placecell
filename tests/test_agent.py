from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from placecell import (
    TOOLS,
    Agent,
    ChatMessage,
    ChatModel,
    ChatReply,
    CollectionInfo,
    InMemoryStore,
    Pose,
    Recall,
    ToolCall,
)
from placecell.errors import ProviderError, ValidationError
from placecell.providers import HashingEmbedder, OpenAICompatibleChat
from tests.conftest import FakeTransport, embedded


class ScriptedChat:
    """Replies in order; records what it was asked."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
        self.calls.append(list(messages))
        assert [t["function"]["name"] for t in tools] == [
            "search_memories",
            "memories_between",
            "memories_near",
            "answer",
        ]
        return self.replies.pop(0)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(f"call-{name}", name, arguments)


def test_agent_enforces_tool_and_result_limits(recall) -> None:
    agent = Agent(recall, ScriptedChat([]))
    text, found = agent._run(call("search_memories", query="printer", k=10000))
    assert "error" in json.loads(text) and found == []
    excessive = ChatReply(None, tuple(call("search_memories", query="printer") for _ in range(9)))
    answer = Agent(recall, ScriptedChat([excessive])).ask("printer?")
    assert not answer.grounded and "budget" in answer.text


@pytest.fixture
def recall(hashing: HashingEmbedder) -> Recall:
    store = InMemoryStore(__import__("placecell").CollectionInfo("t", hashing.model_name, hashing.dimension))
    store.upsert(
        [
            embedded(hashing, "red fire extinguisher on the wall", t=1000, x=4, y=2),
            embedded(hashing, "grey office chair", t=2000, x=9, y=9),
        ]
    )
    return Recall(store, hashing, clock=lambda: 3000.0)


def test_agent_runs_tools_and_returns_grounded_answer(recall: Recall) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("search_memories", query="fire extinguisher", k=2),)),
            ChatReply(None, (call("memories_near", x=4, y=2, radius_m=1),)),
            ChatReply(None, (call("answer", text="Near x=4, y=2.", memory_ids=["r1:front:1000000"]),)),
        ]
    )
    answer = Agent(recall, chat, clock=lambda: 3000.0).ask("where is the fire extinguisher?")
    assert answer.text == "Near x=4, y=2." and answer.grounded and answer.steps == 3
    assert [r.memory.id for r in answer.evidence] == ["r1:front:1000000"]
    # the model saw the system prompt with the clock, the question, and the tool results in order
    first = chat.calls[0]
    assert (
        first[0].role == "system"
        and "unix 3000" in (first[0].content or "")
        and first[1].content == "where is the fire extinguisher?"
    )
    third = chat.calls[2]
    assert [m.role for m in third] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    results = json.loads(third[3].content or "")
    assert results[0]["caption"].startswith("red fire") and "similarity" in results[0] and results[0]["x"] == 4
    near = json.loads(third[5].content or "")
    assert [r["id"] for r in near] == ["r1:front:1000000"] and "similarity" not in near[0]
    assert third[5].tool_call_id == "call-memories_near"


@pytest.mark.parametrize("ids", [[], ["ghost"], ["r1:front:1000000", "ghost"], "r1:front:1000000", [{}], None])
def test_agent_rejects_missing_unknown_and_malformed_citations(recall: Recall, ids: Any) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("search_memories", query="fire extinguisher"),)),
            ChatReply(None, (call("answer", text="At the door.", memory_ids=ids),)),
        ]
    )
    assert not Agent(recall, chat).ask("where?").grounded


def test_agent_deduplicates_valid_citations(recall: Recall) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("search_memories", query="fire extinguisher"),)),
            ChatReply(None, (call("answer", text="At the door.", memory_ids=["r1:front:1000000"] * 2),)),
        ]
    )
    answer = Agent(recall, chat).ask("where?")
    assert answer.grounded and len(answer.evidence) == 1


def test_agent_near_uses_the_configured_map(hashing: HashingEmbedder) -> None:
    store = InMemoryStore(CollectionInfo("maps", hashing.model_name, hashing.dimension))
    office = embedded(hashing, "a printer", t=100, pose=Pose(1, 2, map_id="office"))
    store.upsert([office, embedded(hashing, "a chair", t=200, pose=Pose(1, 2, map_id="warehouse"))])
    chat = ScriptedChat(
        [
            ChatReply(None, (call("memories_near", x=1, y=2, radius_m=1),)),
            ChatReply(None, (call("answer", text="A printer.", memory_ids=[office.id]),)),
        ]
    )
    answer = Agent(Recall(store, hashing), chat, map_id="office").ask("what is nearby?")
    assert answer.grounded and [r.memory.id for r in answer.evidence] == [office.id]
    tool_result = json.loads(chat.calls[1][-1].content or "")
    assert [r["id"] for r in tool_result] == [office.id]


def test_agent_reports_tool_errors_to_the_model_and_keeps_going(recall: Recall) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("teleport"), call("memories_between", time_from="x", time_to=1))),
            ChatReply(None, (call("memories_between", time_from=0, time_to=1500),)),
            ChatReply(None, (call("answer", text="At 1000.", memory_ids=["r1:front:1000000"]),)),
        ]
    )
    answer = Agent(recall, chat).ask("what did you see first?")
    assert answer.grounded and len(answer.evidence) == 1
    tool_messages = [m for m in chat.calls[2] if m.role == "tool"]
    assert "unknown tool" in (tool_messages[0].content or "")
    assert "bad arguments" in (tool_messages[1].content or "")
    assert json.loads(tool_messages[2].content or "")[0]["seen"] == 1


def test_agent_handles_prose_replies_and_step_limits(recall: Recall) -> None:
    prose = Agent(recall, ScriptedChat([ChatReply("  I do not know.  ")])).ask("hm?")
    assert prose.text == "I do not know." and not prose.grounded and prose.evidence == []
    empty = Agent(recall, ScriptedChat([ChatReply("")])).ask("hm?")
    assert empty.text == "No answer."
    looping = ScriptedChat([ChatReply(None, (call("search_memories", query="chair"),))] * 2)
    answer = Agent(recall, looping, max_steps=2).ask("where is the chair?")
    assert not answer.grounded and answer.steps == 2 and "steps" in answer.text
    with pytest.raises(ValidationError):
        Agent(recall, looping, max_steps=0)
    with pytest.raises(ValidationError):
        Agent(recall, looping).ask("  ")


def _reply(
    content: str | None = None, tool_calls: list[dict[str, Any]] | None = None
) -> tuple[int, dict[str, str], dict]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return 200, {}, {"choices": [{"message": msg}]}


def test_chat_adapter_encodes_history_and_decodes_tool_calls() -> None:
    transport = FakeTransport(
        [
            _reply(
                None,
                [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "search_memories", "arguments": '{"query": "door", "k": 3}'},
                    }
                ],
            ),
            _reply([{"type": "text", "text": "done"}], []),
        ]
    )
    chat = OpenAICompatibleChat("llm", base_url="https://x/v1", api_key="k", transport=transport)
    assert isinstance(chat, ChatModel) and chat.model_name == "llm"
    reply = chat.complete([ChatMessage("system", "s"), ChatMessage("user", "q")], TOOLS)
    assert reply.content is None and reply.tool_calls == (ToolCall("c1", "search_memories", {"query": "door", "k": 3}),)
    history = [
        ChatMessage("system", "s"),
        ChatMessage("user", "q"),
        ChatMessage("assistant", None, reply.tool_calls),
        ChatMessage("tool", "[]", tool_call_id="c1"),
    ]
    final = chat.complete(history, TOOLS)
    assert final.content == "done" and final.tool_calls == ()
    payload = transport.requests[1]["payload"]
    assert payload["tool_choice"] == "auto" and len(payload["tools"]) == 4 and payload["temperature"] == 0.0
    assert payload["messages"][2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search_memories", "arguments": '{"query": "door", "k": 3}'},
            }
        ],
    }
    assert payload["messages"][3] == {"role": "tool", "content": "[]", "tool_call_id": "c1"}
    plain = FakeTransport([_reply("x")])
    assert OpenAICompatibleChat("llm", transport=plain).complete([ChatMessage("user", "q")], []).content == "x"
    assert "tools" not in plain.requests[0]["payload"]


def test_chat_adapter_rejects_malformed_replies() -> None:
    bad = [
        _reply(None, [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}]),
        _reply(None, [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "[1, 2]"}}]),
        (200, {}, {"choices": []}),
    ]
    for response in bad:
        with pytest.raises(ProviderError):
            OpenAICompatibleChat("llm", transport=FakeTransport([response])).complete([ChatMessage("user", "q")], [])
    with pytest.raises(ValidationError):
        OpenAICompatibleChat("")
    with pytest.raises(ValidationError):
        OpenAICompatibleChat("llm", temperature=-1)


@pytest.mark.parametrize("reason", ["length", "content_filter"])
def test_interrupted_chat_responses_cannot_authorize_tool_execution(reason) -> None:
    status, headers, body = _reply(
        None, [{"id": "1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    )
    body["choices"][0]["finish_reason"] = reason
    with pytest.raises(ProviderError, match="interrupted"):
        OpenAICompatibleChat("llm", transport=FakeTransport([(status, headers, body)])).complete([], [])
