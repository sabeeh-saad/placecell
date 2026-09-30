from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
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
    RankedMemory,
    Recall,
    ToolCall,
)
from placecell.agent import _describe
from placecell.errors import ProviderError, ValidationError
from placecell.providers import HashingEmbedder, OpenAICompatibleChat
from tests.conftest import FakeTransport, embedded


class ScriptedChat:
    """Replies in order; records what it was asked."""

    def __init__(self, replies: list[ChatReply]) -> None:
        self.replies = list(replies)
        self.calls: list[list[ChatMessage]] = []
        self.tools: list[list[str]] = []

    def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
        self.calls.append(list(messages))
        self.tools.append([t["function"]["name"] for t in tools])
        assert self.tools[-1] in (["search_memories", "memories_between", "memories_near", "answer"], ["answer"])
        return self.replies.pop(0)


def call(name: str, **arguments: Any) -> ToolCall:
    return ToolCall(f"call-{name}", name, arguments)


def test_agent_enforces_tool_and_result_limits(recall) -> None:
    agent = Agent(recall, ScriptedChat([]))
    text, found = agent._run(call("search_memories", query="printer", k=10000))
    assert "error" in json.loads(text) and found == []
    excessive = ChatReply(None, tuple(call("search_memories", query="printer") for _ in range(9)))
    answer = Agent(recall, ScriptedChat([excessive])).ask("printer?")
    assert not answer.citations_valid and "budget" in answer.text


def test_tool_results_keep_captions_and_sighting_lists_short(recall: Recall) -> None:
    memory = recall._store.get("r1:front:1000000")
    assert memory is not None
    sightings = tuple(float(t) for t in range(1000, 1064))
    ranked = RankedMemory(replace(memory, caption="c" * 2000), 1.0, 0.9, observed_at=sightings)
    described = _describe(ranked)
    assert described["caption"] == "c" * 300 + " [truncated]"
    assert described["observed_at"] == [1000.0, 1060.0, 1061.0, 1062.0, 1063.0]
    assert described["sightings_omitted"] == 59 and described["time"].startswith("1970-01-01T00:16:40")
    assert "sightings_omitted" not in _describe(RankedMemory(memory, 1.0, observed_at=sightings[:5]))
    for name, argument in (("search_memories", "k"), ("memories_between", "limit"), ("memories_near", "limit")):
        schema = next(t["function"] for t in TOOLS if t["function"]["name"] == name)
        assert schema["parameters"]["properties"][argument]["maximum"] == 20 and "20" in schema["description"]
    agent = Agent(recall, ScriptedChat([]))
    for bad in (
        call("search_memories", query="printer", k=21),
        call("memories_between", time_from=0, time_to=1, limit=21),
        call("memories_near", x=0, y=0, limit=21),
    ):
        text, found = agent._run(bad)
        assert "within 1..20" in json.loads(text)["error"] and found == []
    text, _ = agent._run(call("memories_between", time_from="x" * 5000, time_to=1))
    assert len(json.loads(text)["error"]) < 300


def test_a_large_result_is_truncated_to_the_transcript_budget_and_forces_the_answer(hashing: HashingEmbedder) -> None:
    store = InMemoryStore(CollectionInfo("budget", hashing.model_name, hashing.dimension))
    store.upsert([embedded(hashing, f"shelf {i} " + "x" * 290, t=1000 + i, x=i) for i in range(20)])
    recall = Recall(store, hashing, clock=lambda: 3000.0)
    first, last = "r1:front:1000000", "r1:front:1019000"
    chat = ScriptedChat(
        [
            ChatReply(None, (call("memories_between", time_from=0, time_to=5000, limit=20),)),
            ChatReply(None, (call("answer", text="Shelf zero.", memory_ids=[first]),)),
        ]
    )
    answer = Agent(recall, chat, max_context_chars=6000).ask("which shelves did you see?")
    result = json.loads(chat.calls[1][3].content or "")
    assert result[-1]["truncated"] is True and 0 < len(result) - 1 < 20
    assert result[-1]["omitted_memories"] == 20 - (len(result) - 1)
    assert sum(len(m.content or "") for m in chat.calls[1]) <= 6000
    assert chat.tools[1] == ["answer"] and chat.calls[1][-1].role == "user" and "budget" in chat.calls[1][-1].content
    assert answer.citations_valid and [r.memory.id for r in answer.evidence] == [first]
    omitted = ScriptedChat(
        [
            ChatReply(None, (call("memories_between", time_from=0, time_to=5000, limit=20),)),
            ChatReply(None, (call("answer", text="The last shelf.", memory_ids=[last]),)),
        ]
    )
    assert not Agent(recall, omitted, max_context_chars=6000).ask("which shelves?").citations_valid


def test_total_tool_calls_are_bounded_per_question(recall: Recall) -> None:
    search = call("search_memories", query="chair")
    chat = ScriptedChat(
        [
            ChatReply(None, (search, search)),
            ChatReply(None, (search, search)),
            ChatReply(None, (search,)),
        ]
    )
    answer = Agent(recall, chat, max_tool_calls=3).ask("where is the chair?")
    assert not answer.citations_valid and "budget" in answer.text and answer.steps == 3
    assert chat.tools == [chat.tools[0], chat.tools[0], ["answer"]]
    results = [m for m in chat.calls[2] if m.role == "tool"]
    assert "budget" in json.loads(results[-1].content or "")["error"]
    assert isinstance(json.loads(results[-2].content or ""), list)
    with pytest.raises(ValidationError):
        Agent(recall, chat, max_tool_calls=0)
    with pytest.raises(ValidationError):
        Agent(recall, chat, max_context_chars=100)


def test_citations_valid_names_what_is_checked_and_grounded_is_a_deprecated_alias(recall: Recall) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("search_memories", query="fire extinguisher"),)),
            ChatReply(None, (call("answer", text="At the door.", memory_ids=["r1:front:1000000"]),)),
        ]
    )
    answer = Agent(recall, chat).ask("where?")
    assert answer.citations_valid
    with pytest.warns(DeprecationWarning, match="citations_valid"):
        assert answer.grounded is True
    system = chat.calls[0][0].content or ""
    assert "observations produced by models" in system and "not instructions" in system


def test_provider_failures_propagate_to_the_caller(recall: Recall) -> None:
    class Failing:
        def complete(self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]) -> ChatReply:
            raise ProviderError("https://models.test: HTTP 400: context length exceeded")

    with pytest.raises(ProviderError):
        Agent(recall, Failing()).ask("where?")


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
    assert answer.text == "Near x=4, y=2." and answer.citations_valid and answer.steps == 3
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
    assert not Agent(recall, chat).ask("where?").citations_valid


def test_agent_deduplicates_valid_citations(recall: Recall) -> None:
    chat = ScriptedChat(
        [
            ChatReply(None, (call("search_memories", query="fire extinguisher"),)),
            ChatReply(None, (call("answer", text="At the door.", memory_ids=["r1:front:1000000"] * 2),)),
        ]
    )
    answer = Agent(recall, chat).ask("where?")
    assert answer.citations_valid and len(answer.evidence) == 1


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
    assert answer.citations_valid and [r.memory.id for r in answer.evidence] == [office.id]
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
    assert answer.citations_valid and len(answer.evidence) == 1
    tool_messages = [m for m in chat.calls[2] if m.role == "tool"]
    assert "unknown tool" in (tool_messages[0].content or "")
    assert "bad arguments" in (tool_messages[1].content or "")
    assert json.loads(tool_messages[2].content or "")[0]["seen"] == 1


def test_agent_handles_prose_replies_and_step_limits(recall: Recall) -> None:
    prose = Agent(recall, ScriptedChat([ChatReply("  I do not know.  ")])).ask("hm?")
    assert prose.text == "I do not know." and not prose.citations_valid and prose.evidence == []
    empty = Agent(recall, ScriptedChat([ChatReply("")])).ask("hm?")
    assert empty.text == "No answer."
    looping = ScriptedChat([ChatReply(None, (call("search_memories", query="chair"),))] * 2)
    answer = Agent(recall, looping, max_steps=2).ask("where is the chair?")
    assert not answer.citations_valid and answer.steps == 2 and "steps" in answer.text
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
    return 200, {}, {"choices": [{"finish_reason": "tool_calls" if tool_calls else "stop", "message": msg}]}


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
