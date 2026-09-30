"""What the node creates on ROS, construction-time validation, and the shutdown sequence."""

from __future__ import annotations

import json

import pytest

from placecell.errors import ValidationError
from placecell.maintenance import StorageLease
from placecell.navigation_ownership import NavigationOwnership, NavigationScope
from placecell.ros2.node import IngestWorker
from tests.conftest import FakeMediaEmbedder
from tests.ros_fakes import hermetic_parameters, spy
from tests.ros_fakes import make_node as make_node
from tests.ros_fakes import ros as ros

SENSOR = "sensor_data"
VOLATILE = "10,RELIABLE,VOLATILE"
EXCLUSIVE = "MutuallyExclusiveCallbackGroup"
REENTRANT = "ReentrantCallbackGroup"


def group(value):
    return "default" if value is None else type(value).__name__


def topology(node):
    return {
        "subscriptions": [
            (s.topic, s.msg_type.__name__, str(s.qos), group(s.callback_group)) for s in node.subscriptions
        ],
        "publishers": [(p.topic, p.msg_type.__name__, str(p.qos)) for p in node.publishers],
        "timers": [
            (t.period, t.clock.clock_type if t.clock else "ROS_TIME", group(t.callback_group)) for t in node.timers
        ],
        "services": [(s.name, s.srv_type.__name__) for s in node.services],
        "clients": [(c.name, c.srv_type.__name__, group(c.callback_group)) for c in node.clients],
        "actions": [(a.action_name, a.action_type.__name__, group(a.callback_group)) for a in node.action_clients],
    }


OPERATOR_SUBSCRIPTIONS = [
    ("~/ask", "String", "10", "default"),
    ("~/correct", "String", "10", "default"),
    ("~/refine", "String", "10", "default"),
    ("~/command", "String", VOLATILE, EXCLUSIVE),
    ("~/command_json", "String", VOLATILE, EXCLUSIVE),
]
PUBLISHERS = [
    ("~/answer", "String", "10"),
    ("~/navigation_status", "String", "10"),
    ("~/command_receipt", "String", "10"),
    ("~/mission_snapshot", "String", "1,RELIABLE,TRANSIENT_LOCAL"),
]


def test_default_node_topics_timers_and_services(make_node):
    node = make_node()
    assert node.node_name == "placecell"
    assert topology(node) == {
        "subscriptions": [
            ("/amcl_pose", "PoseWithCovarianceStamped", SENSOR, "default"),
            ("/camera/color/image_raw", "Image", SENSOR, "default"),
            *OPERATOR_SUBSCRIPTIONS,
        ],
        "publishers": PUBLISHERS,
        "timers": [
            (0.2, "STEADY_TIME", "default"),  # odometry
            (0.04, "STEADY_TIME", "default"),  # RGB-D drain
            (0.2, "STEADY_TIME", "default"),  # operator snapshot
            (3600.0, "ROS_TIME", "default"),  # curator
            (30.0, "ROS_TIME", "default"),  # diagnostics
        ],
        "services": [("~/get_mission_snapshot", "Trigger")],
        "clients": [],
        "actions": [],
    }
    command, command_json = node.subscriptions[-2:]
    assert command.callback_group is not command_json.callback_group
    assert node._commands is None and node._navigator is None
    assert node.logger.messages("warning") == [
        "no caption_model and the embedder takes text only: frames cannot be stored. "
        "Set caption_model, or use an embedding model that accepts images."
    ]
    assert node.logger.messages("info")[-1] == (
        "placecell up: robot robot, camera front, model hashing-256, store memory, agent off"
    )


def test_full_node_topics_timers_services_and_actions(make_node, monkeypatch, tmp_path):
    monkeypatch.setenv("PLACECELL_TEST_KEY", "offline")
    monkeypatch.setattr("placecell.ros2.components.build_embedder", lambda *a, **k: FakeMediaEmbedder())
    node = make_node(
        db_path=str(tmp_path / "db"),
        compressed=True,
        rgbd_reliable=True,
        objects_enabled=True,
        object_backend="chat",
        object_model="scripted",
        object_api_key_env="PLACECELL_TEST_KEY",
        navigation_enabled=True,
        approach_enabled=True,
        object_search_enabled=True,
        mission_enabled=True,
        mission_model="scripted",
        mission_trace_path=str(tmp_path / "traces.sqlite3"),
        caption_model="scripted",
        chat_model="scripted",
        consolidate_interval_s=600.0,
        refine_interval_s=1800.0,
        nav2_action="robot/navigate_to_pose",
    )
    reliable = "8,RELIABLE"
    assert topology(node) == {
        "subscriptions": [
            ("/amcl_pose", "PoseWithCovarianceStamped", SENSOR, "default"),
            ("/camera/aligned_depth_to_color/image_raw", "Image", reliable, "default"),
            ("/camera/color/camera_info", "CameraInfo", reliable, "default"),
            ("/camera/color/image_raw", "CompressedImage", reliable, "default"),
            *OPERATOR_SUBSCRIPTIONS[:3],
            ("/global_costmap/costmap_raw", "Costmap", SENSOR, "default"),
            ("/local_costmap/published_footprint", "PolygonStamped", SENSOR, "default"),
            *OPERATOR_SUBSCRIPTIONS[3:],
        ],
        "publishers": PUBLISHERS,
        "timers": [
            (0.2, "STEADY_TIME", "default"),
            (0.04, "STEADY_TIME", "default"),
            (0.1, "STEADY_TIME", EXCLUSIVE),  # Nav2 transport deadlines
            (0.1, "STEADY_TIME", EXCLUSIVE),  # command deadlines
            (0.2, "STEADY_TIME", "default"),
            (3600.0, "ROS_TIME", "default"),
            (600.0, "ROS_TIME", "default"),  # consolidation
            (1800.0, "ROS_TIME", "default"),  # refinement
            (2.0, "ROS_TIME", "default"),  # vector index sync
            (30.0, "ROS_TIME", "default"),
        ],
        "services": [("~/get_mission_snapshot", "Trigger")],
        "clients": [
            ("/robot/navigate_to_pose/_action/get_result", "GetResultService", REENTRANT),
            ("/robot/navigate_to_pose/_action/cancel_goal", "CancelGoal", REENTRANT),
        ],
        "actions": [
            ("/compute_path_to_pose", "ComputePathToPose", "default"),
            ("robot/navigate_to_pose", "NavigateToPose", REENTRANT),
        ],
    }
    deadlines = [t.callback_group for t in node.timers if t.period == 0.1]
    assert deadlines[0] is not deadlines[1]
    assert node.logger.messages("info")[-1].endswith("store lancedb " + str(tmp_path / "db") + ", agent on")
    # The ownership journal is scoped to the resolved action name, so a remap cannot reuse it.
    node.destroy_node()
    with pytest.raises(ValidationError, match="scope"):
        NavigationOwnership(str(tmp_path / "navigation.sqlite3"), NavigationScope("robot", "test-v1", "/other"))
    NavigationOwnership(
        str(tmp_path / "navigation.sqlite3"), NavigationScope("robot", "test-v1", "/robot/navigate_to_pose")
    ).close()


@pytest.mark.parametrize(
    ("parameters", "periods"),
    [
        ({"odom_frame": "", "curator_interval_s": 0.0}, [0.04, 0.2, 30.0]),
        ({"caption_model": "scripted", "refine_interval_s": 0.0}, [0.2, 0.04, 0.2, 3600.0, 30.0]),
        ({"caption_model": "scripted"}, [0.2, 0.04, 0.2, 3600.0, 3600.0, 30.0]),
        ({"refine_model": "scripted"}, [0.2, 0.04, 0.2, 3600.0, 3600.0, 30.0]),
        ({"chat_model": "scripted"}, [0.2, 0.04, 0.2, 3600.0, 30.0]),
    ],
)
def test_optional_timers(make_node, parameters, periods):
    assert [t.period for t in make_node(**parameters).timers] == periods


@pytest.mark.parametrize(
    ("parameters", "message", "created"),
    [
        (
            {"navigation_enabled": True, "map_id": " "},
            "Navigation requires a versioned map_id and localization_required:=true.",
            set(),
        ),
        (
            {"navigation_enabled": True, "localization_required": False},
            "Navigation requires a versioned map_id and localization_required:=true.",
            set(),
        ),
        ({"embed_backend": "bogus"}, "embed_backend must be auto, gemini, openrouter or clip", set()),
        ({"object_backend": "bogus"}, "object_backend must be gemini or chat", set()),
        ({"answer_min_similarity": 0.0}, r"answer_min_similarity must be within \(0, 1\]", set()),
        ({"answer_min_similarity": 1.5}, r"answer_min_similarity must be within \(0, 1\]", set()),
        ({"max_queue": 0}, "invalid worker limits", set()),
        ({"question_workers": 0}, "task limits must be positive", set()),
        (
            {"navigation_enabled": True, "approach_enabled": True},
            "approach planning requires objects_enabled",
            {"keyframes", "commands.sqlite3"},
        ),
        (
            {"navigation_enabled": True, "object_search_enabled": True},
            "approach planning requires objects_enabled",
            {"keyframes", "commands.sqlite3"},
        ),
        (
            {"navigation_enabled": True, "mission_enabled": True},
            "mission_enabled requires mission_model with tool calling",
            {"keyframes", "commands.sqlite3", "missions.sqlite3", "navigation.sqlite3"},
        ),
    ],
)
def test_invalid_configuration_is_refused_at_its_construction_step(make_node, tmp_path, parameters, message, created):
    with pytest.raises(ValidationError, match=message):
        make_node(**parameters)
    # The storage lease is taken first; stores, keyframes and journals only as far as construction got.
    locks = {p.name for p in tmp_path.glob("*.maintenance.lock")}
    assert "corrections.jsonl.maintenance.lock" in locks
    produced = {p.name for p in tmp_path.iterdir() if not p.name.endswith((".lock", "-wal", "-shm", ".jsonl"))}
    assert produced - {"home"} == created


def test_running_node_holds_the_storage_lease(make_node, tmp_path):
    node = make_node()
    parameters = hermetic_parameters(tmp_path)
    with pytest.raises(ValidationError, match="Storage is in use"):
        StorageLease.for_parameters(parameters)
    assert node.destroy_node()
    StorageLease.for_parameters(parameters).close()


def navigation_node(make_node, tmp_path, **parameters):
    return make_node(
        navigation_enabled=True,
        mission_enabled=True,
        mission_model="scripted",
        mission_trace_path=str(tmp_path / "traces.sqlite3"),
        **parameters,
    )


SHUTDOWN = [
    ("_clock_jump", "unregister"),
    ("_commands", "close"),
    ("_command_tasks", "stop"),
    ("_worker", "stop"),
    ("_questions", "stop"),
    ("_maintenance", "stop"),
    ("_indexing", "stop"),
    ("_store", "close"),
    ("_mission_context", "close"),
    ("_mission_traces", "close"),
    ("_command_journal", "close"),
    ("_navigator", "close"),
    ("_storage_lease", "close"),
]


def record_shutdown(node, calls, **results):
    for owner, method in SHUTDOWN:
        target = getattr(node, owner)
        name = f"{owner}.{method}"
        setattr(target, method, spy(calls, name, getattr(target, method), results.get(name)))


def test_shutdown_stops_work_before_closing_storage_and_releases_the_lease_last(make_node, ros, tmp_path):
    node = navigation_node(make_node, tmp_path)
    record_shutdown(node, ros.calls)
    assert node.destroy_node() is True
    assert ros.calls == [f"{owner}.{method}" for owner, method in SHUTDOWN] + ["Node.destroy_node"]
    StorageLease.for_parameters(hermetic_parameters(tmp_path)).close()


@pytest.mark.parametrize("stuck", ["_worker.stop", "_questions.stop", "_maintenance.stop", "_command_tasks.stop"])
def test_stuck_worker_keeps_storage_open_and_locked(make_node, ros, tmp_path, stuck):
    node = navigation_node(make_node, tmp_path)
    record_shutdown(node, ros.calls, **{stuck: False})
    assert node.destroy_node() is True
    kept = {"_store.close", "_mission_context.close", "_storage_lease.close"}
    assert ros.calls == [f"{o}.{m}" for o, m in SHUTDOWN if f"{o}.{m}" not in kept] + ["Node.destroy_node"]
    with pytest.raises(ValidationError, match="Storage is in use"):
        StorageLease.for_parameters(hermetic_parameters(tmp_path))
    for owner in ("_store", "_mission_context", "_storage_lease"):
        getattr(node, owner).close()


def test_unfinished_trace_writer_is_reported_and_keeps_the_lease(make_node, ros, tmp_path):
    node = navigation_node(make_node, tmp_path)
    record_shutdown(node, ros.calls, **{"_mission_traces.close": False})
    node.destroy_node()
    assert "_store.close" in ros.calls and "_storage_lease.close" not in ros.calls
    assert node.logger.messages("warning")[-1] == "Mission trace writer did not finish before the shutdown deadline."
    node._storage_lease.close()


def test_shutdown_without_navigation_skips_its_parts(make_node, ros):
    node = make_node()
    node._worker.stop = spy(ros.calls, "_worker.stop", node._worker.stop)
    node._storage_lease.close = spy(ros.calls, "_storage_lease.close", node._storage_lease.close)
    assert node.destroy_node() is True
    assert ros.calls == ["_worker.stop", "_storage_lease.close", "Node.destroy_node"]


def test_worker_starts_with_the_node(make_node, monkeypatch, ros):
    monkeypatch.setattr(IngestWorker, "start", spy(ros.calls, "start", IngestWorker.start))
    node = make_node()
    assert ros.calls == ["start"] and not node._worker.health()["stopped"]


def test_operator_snapshot_reports_disabled_navigation(make_node):
    node = make_node()
    (service,) = node.services
    response = service.callback(None, type("Response", (), {})())
    snapshot = json.loads(response.message)
    assert response.success and not snapshot["navigation_enabled"] and snapshot["command_identity"] is None
    assert json.loads(node.publisher("~/mission_snapshot").messages[-1].data)["status"]["state"] == "disabled"
