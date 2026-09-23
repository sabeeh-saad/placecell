import json
from dataclasses import replace
from pathlib import Path

import pytest

from placecell import ApproachPlanner, MissionPlanner, PlanReviewAgent, Pose
from placecell.errors import ProviderError, ValidationError
from placecell.object_types import ArrivalComparison, Detection
from placecell.providers.object_detection import ChatObjectDetector, GeminiObjectDetector
from placecell.verification import SceneVerdict, VisionVerifier
from tests.conftest import FakeTransport
from tests.test_approach import Environment, target
from tests.test_missions import Model, proposal, review
from tests.test_object_arrival import setup_arrival
from tests.test_object_detection import response
from tests.test_object_navigation import Harness
from tests.test_objects import LEFT, RIGHT, observation


def test_planning_and_review_receive_names_without_poses_or_extra_visits():
    model, critic = Model(proposal(("printer", "home"))), Model(review())
    plan = MissionPlanner(model, PlanReviewAgent(critic)).plan(
        "Go to the printer then home", configured_places=("home", "charging bay")
    )
    assert plan.destinations == ("printer", "home")
    for role in (model, critic):
        data = json.loads(role.calls[0][0][1].content)
        assert data["configured_places"] == ["home", "charging bay"]
        assert "coordinates" not in data
    with pytest.raises(ValidationError):
        MissionPlanner(model, PlanReviewAgent(critic)).plan("go home", configured_places=("x" * 101,))


def test_place_catalog_is_scoped_to_the_current_map(tmp_path):
    resolver = Harness(tmp_path).resolver
    resolver._places = {"home": Pose(0, 0, map_id="office-v1"), "foreign": Pose(0, 0, map_id="other")}
    assert resolver.configured_places == ("home",)


def test_approach_prefers_known_viewing_side_over_shorter_side_view():
    env = Environment()
    env.state = replace(env.state, robot_pose=Pose(3, 3, map_id="office-v1"))
    plan = ApproachPlanner(env, clock=lambda: 1000).plan(*target())
    assert abs(plan.pose.y) < 1e-6
    assert plan.pose.heading_difference(Pose(0, 0)) < 1e-6
    assert any(
        goal.distance_to(env.state.robot_pose) < plan.pose.distance_to(env.state.robot_pose) for goal in env.calls
    )


@pytest.mark.parametrize(
    "expanded,verdict,expected",
    [("printer", "matched", "resolved"), ("printer", "not_matched", "not_found"), ("invented", "matched", None)],
)
def test_grounded_expansion_keeps_original_request_and_visual_veto(tmp_path, expanded, verdict, expected):
    h = Harness(tmp_path)
    h.resolver._policy = replace(h.resolver._policy, min_similarity=0.75)
    original = h.resolver._objects.similar
    queries, checked = [], []

    def similar(query, **kwargs):
        queries.append(query)
        return [replace(hit, similarity=0.9 if query == "printer" else 0.7) for hit in original(query, **kwargs)]

    class Grounder:
        def search_query(self, target, labels):
            assert labels == ("printer",)
            return expanded

        def verify(self, target, _image):
            checked.append(target)
            return SceneVerdict(verdict, "Visible request attributes checked")

    h.resolver._objects.similar = similar
    h.resolver._verifier = Grounder()
    description = "where I can print my documents"
    result = h.resolver._resolve_object(description)
    assert (result.state if result else None) == expected
    assert queries == [description, "printer"] if expanded == "printer" else queries == [description]
    assert checked == ([description] if expanded == "printer" else [])
    if expected == "resolved":
        assert result.choices[0].target == description


@pytest.mark.parametrize(
    "value",
    [
        {"query": "printer", "reason": "prints documents"},
        {"query": "", "reason": "no category fits"},
        {"query": "unobserved refrigerator", "reason": "invented"},
        {"query": "printer", "reason": ""},
        {"query": "printer", "reason": "x", "pose": [1, 2]},
    ],
)
def test_semantic_query_provider_cannot_invent_categories(value):
    transport = FakeTransport(
        [(200, {}, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]})]
    )
    verifier = VisionVerifier("test", "https://example.test/v1", transport=transport)
    if value in ({"query": "printer", "reason": "prints documents"}, {"query": "", "reason": "no category fits"}):
        assert verifier.search_query("place for printing", ("printer",)) == value["query"]
    else:
        with pytest.raises(ProviderError):
            verifier.search_query("place for printing", ("printer",))


def comparison_value():
    return {
        "selected": 0,
        "identity": {"result": "matched", "reason": "Same panel and tray details"},
        "destination": {"result": "matched", "reason": "Selected printer satisfies the request"},
    }


@pytest.mark.parametrize("chat", [False, True])
def test_combined_comparison_sends_saved_pixels_current_scene_and_untrusted_target(tmp_path, chat):
    obs = observation(tmp_path)
    raw = Path(obs.evidence.uri).read_bytes()
    value = comparison_value()
    reply = (
        (200, {}, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}]})
        if chat
        else response(value)
    )
    transport = FakeTransport([reply])
    cls = ChatObjectDetector if chat else GeminiObjectDetector
    detector = cls("test", api_key="test", base_url="https://example.test/v1", transport=transport)
    result = detector.compare_arrival((raw, raw), (raw, raw), obs.evidence, "printer near the door")
    assert result.selected == 0
    assert result.identity.result == result.destination.result == "matched"
    payload = transport.requests[0]["payload"]
    if chat:
        parts = payload["messages"][1]["content"][1:]
        image_key = "image_url"
    else:
        parts = payload["contents"][0]["parts"]
        image_key = "inlineData"
    assert [parts[i]["text"] for i in (0, 2, 4, 6, 8)] == [
        "SAVED_REFERENCE index=0: the following image only.",
        "SAVED_REFERENCE index=1: the following image only.",
        "CURRENT_CANDIDATE index=0: the following image only.",
        "CURRENT_CANDIDATE index=1: the following image only.",
        "FULL_SCENE: context only, not a selectable candidate.",
    ]
    assert all(image_key in parts[i] for i in (1, 3, 5, 7, 9))
    assert "printer near the door" in parts[-1]["text"]
    assert len(transport.requests) == 1


@pytest.mark.parametrize("fault", ["index", "boolean", "missing_selection", "verdict", "extra", "extra_field"])
def test_malformed_combined_comparison_never_authorizes_arrival(tmp_path, fault):
    value = comparison_value()
    if fault in {"index", "boolean", "missing_selection"}:
        value["selected"] = {"index": 3, "boolean": True, "missing_selection": -1}[fault]
    elif fault == "verdict":
        value["identity"]["result"] = "yes"
    elif fault == "extra":
        value["destination"]["move"] = True
    else:
        value["objects"] = []
    obs = observation(tmp_path)
    detector = GeminiObjectDetector("test", api_key="test", transport=FakeTransport([response(value)]))
    with pytest.raises(ProviderError):
        detector.compare_arrival(
            (Path(obs.evidence.uri).read_bytes(),), (Path(obs.evidence.uri).read_bytes(),), obs.evidence, "printer"
        )


def test_combined_path_keeps_embeddings_geometry_and_capture_checks(tmp_path):
    verifier, reference, detector, comparator, now = setup_arrival(tmp_path)
    calls = []

    def inspect(references, candidates, image, target):
        calls.append((references, image, target))
        return ArrivalComparison(
            0,
            SceneVerdict("matched", "same details"),
            SceneVerdict("matched", "request satisfied"),
        )

    detector.compare_arrival = inspect
    assert verifier.verify(reference, observation(tmp_path, 1100), target="printer").result == "matched"
    assert len(calls) == 1 and detector.calls == 2 and not comparator.calls
    assert (
        verifier.verify(
            reference,
            observation(tmp_path, 1100),
            target="printer",
            request_check=lambda: SceneVerdict("not_matched", "Additional request check rejected the scene"),
        ).result
        == "unobserved"
    )
    assert (
        verifier.verify(reference, observation(tmp_path, 1100, colors=("blue",)), target="printer").result
        == "unobserved"
    )
    obs = observation(tmp_path, 1100)
    assert (
        verifier.verify(reference, replace(obs, depth=replace(obs.depth, position_error_m=1)), target="printer").result
        == "unavailable"
    )
    detector.compare_arrival = lambda *args: (now.__setitem__(0, 1106), inspect(*args))[1]
    with pytest.raises(ValidationError, match="expired"):
        verifier.verify(reference, obs, target="printer")


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("lookalikes", "ambiguous"),
        ("wrong_selection", "ambiguous"),
        ("identity", "ambiguous"),
        ("request", "unobserved"),
    ],
)
def test_combined_model_verdict_cannot_override_identity_or_original_request(tmp_path, fault, expected):
    verifier, reference, detector, _, _ = setup_arrival(tmp_path)
    detections = (Detection("printer", "red printer", LEFT), Detection("other", "other object", RIGHT))
    colors = ("red", "red" if fault == "lookalikes" else "blue")
    detector.detections = list(detections)
    detector.compare_arrival = lambda *_: ArrivalComparison(
        1 if fault == "wrong_selection" else 0,
        SceneVerdict("uncertain" if fault == "identity" else "matched", "identity evidence"),
        SceneVerdict("not_matched" if fault == "request" else "matched", "request evidence"),
    )
    obs = observation(tmp_path, 1100, (LEFT, RIGHT), colors)
    assert verifier.verify(reference, obs, target="red printer by window").result == expected


@pytest.mark.parametrize("lookalike", [False, True])
def test_arrival_compares_only_the_unique_candidate_after_embedding(tmp_path, monkeypatch, lookalike):
    verifier, reference, detector, _, _ = setup_arrival(tmp_path)
    detector.detections = [Detection("other", "other object", LEFT), Detection("printer", "red printer", RIGHT)]
    original = verifier.tracker.embed_crops
    embedded, compared = [], []

    def embed(crops, **kwargs):
        embedded.extend(crops)
        return original(crops, **kwargs)

    def compare(references, candidates, image, target):
        assert len(embedded) == 2
        assert candidates == (embedded[1].crop_png,)
        assert references == tuple(v.crop_png for v in reference.views)
        compared.append(candidates)
        return ArrivalComparison(0, SceneVerdict("matched", "same details"), SceneVerdict("matched", "requested"))

    detector.compare_arrival = compare
    monkeypatch.setattr(verifier.tracker, "embed_crops", embed)
    obs = observation(tmp_path, 1100, (LEFT, RIGHT), ("red" if lookalike else "blue", "red"))
    result = verifier.verify(reference, obs, target="printer")
    assert result.result == ("ambiguous" if lookalike else "matched")
    assert len(compared) == (0 if lookalike else 1)
