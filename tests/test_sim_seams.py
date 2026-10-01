"""What the simulation scripts patch, overwrite and read must still exist and still be used.

The Gazebo scripts reach into production code: `patch("placecell...")`, attribute overwrites
such as `node._frames.capture = ...`, and reads of private node attributes. A patch on a
name that nothing calls any more passes silently and then exercises real providers or real
Nav2. Each seam found in simulation/scripts needs an entry below whose trigger proves it is
still used.
"""

from __future__ import annotations

import ast
import itertools
import pkgutil
import re
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from placecell import fault_injection as faults
from placecell.chat import ChatMessage, ChatReply, ToolCall
from placecell.navigation import Destination, DestinationResolver
from placecell.object_arrival import ObjectArrivalVerifier
from placecell.pipeline import Observation
from placecell.providers import HashingEmbedder, OpenAICompatibleChat
from placecell.providers._http import RetryPolicy, UrllibTransport
from placecell.providers.base import Capabilities
from placecell.ros2.node import IngestWorker
from tests.conftest import FakeMediaEmbedder
from tests.ros_fakes import CAMERA_FRAME, camera_info, depth_image, localization, raw_image, ros_type
from tests.ros_fakes import make_node as make_node
from tests.ros_fakes import ros as ros

SCRIPTS = sorted((Path(__file__).resolve().parents[1] / "simulation" / "scripts").glob("*.py"))
T0 = 1000
BUILTIN = {"use_sim_time"}


def chain(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def trees():
    return [(path.name, ast.parse(path.read_text())) for path in SCRIPTS]


def patch_targets() -> set[str]:
    """String targets of patch(), including (target, value) tuples iterated into patch(target, value)."""
    found = set()
    for _, tree in trees():
        loops = {
            loop.target.elts[0].id: loop.iter
            for loop in ast.walk(tree)
            if isinstance(loop, ast.For) and isinstance(loop.target, ast.Tuple)
        }
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call) or chain(call.func) != "patch":
                continue
            target = call.args[0]
            if isinstance(target, ast.Constant):
                found.add(target.value)
            else:
                found.update(pair.elts[0].value for pair in loops[target.id].elts)
    return found


def object_patches() -> set[str]:
    return {
        f"{call.args[0].id}.{call.args[1].value}"
        for _, tree in trees()
        for call in ast.walk(tree)
        if isinstance(call, ast.Call) and chain(call.func) == "patch.object"
    }


# Receivers in the scripts, by the production object they hold.
RECEIVERS = {
    "self.node": "node",
    "self.commands": "commands",
    "controller": "commands",
    "self.navigator": "navigator",
    "nav": "navigator",
    "node._navigator": "navigator",
    "embed": "embedder",
    "embedder": "embedder",
}


def normalized(path: str) -> str:
    for prefix, name in sorted(RECEIVERS.items(), key=lambda item: -len(item[0])):
        if path.startswith(prefix + "."):
            return name + path[len(prefix) :]
    return path


def overwritten() -> set[str]:
    """Assignments to a private attribute of a production object."""
    return {
        normalized(path)
        for _, tree in trees()
        for assign in ast.walk(tree)
        if isinstance(assign, ast.Assign)
        for target in assign.targets
        if (path := chain(target)) and any(part.startswith("_") for part in path.split(".")[1:])
    }


def node_paths() -> set[str]:
    """Every attribute path the scripts use below a private node attribute."""
    paths = set()
    for _, tree in trees():
        for attribute in ast.walk(tree):
            path = chain(attribute) if isinstance(attribute, ast.Attribute) else None
            if path and re.match(r"(self\.)?node\._", path):
                paths.add(path.removeprefix("self."))
    return paths


# Patch targets: the trigger builds whatever calls the target; `offline` sentinels never delegate.


class Offline(Exception):  # noqa: N818 - raised in place of a network call
    pass


def chat_call():
    chat = OpenAICompatibleChat("scripted", "http://offline.test/v1", None, retry=RetryPolicy(attempts=1))
    with pytest.raises(Offline):
        chat.complete([ChatMessage("user", "hi")], [])


def objects_node(make_node, monkeypatch, **parameters):
    monkeypatch.setenv("PLACECELL_TEST_KEY", "offline")
    monkeypatch.setattr("placecell.ros2.components.build_embedder", lambda *a, **k: FakeMediaEmbedder())
    return make_node(
        objects_enabled=True,
        object_backend="chat",
        object_model="scripted",
        object_api_key_env="PLACECELL_TEST_KEY",
        **parameters,
    )


def arrival_check(make_node, monkeypatch):
    node = objects_node(make_node, monkeypatch, navigation_enabled=True)
    monkeypatch.setattr(DestinationResolver, "arrival_available", lambda *a: True)
    destination = SimpleNamespace(object_reference=object(), target="printer")
    with pytest.raises(Offline):
        node._commands._resolver.verify_object_arrival(destination, None, "data:image/png;base64,", lambda: False)


PATCHES: dict[str, tuple[Callable, bool]] = {
    "placecell.ros2.components.build_embedder": (lambda make_node, monkeypatch: make_node(), False),
    "placecell.ros2.components.VisionVerifier": (
        lambda make_node, monkeypatch: make_node(navigation_enabled=True, verification_model="scripted"),
        False,
    ),
    "placecell.ros2.components.build_mission_planner": (
        lambda make_node, monkeypatch: make_node(navigation_enabled=True),
        False,
    ),
    "placecell.providers.OpenAICompatibleCaptioner": (
        lambda make_node, monkeypatch: make_node(caption_model="scripted"),
        False,
    ),
    "placecell.providers.object_detection.ChatObjectDetector": (objects_node, False),
    "placecell.providers._http.Endpoint.post": (lambda make_node, monkeypatch: chat_call(), True),
    "rclpy.node.Node.__init__": (lambda make_node, monkeypatch: make_node(), False),
    "UrllibTransport.post_json": (lambda make_node, monkeypatch: chat_call(), True),
    "ObjectArrivalVerifier.verify_image": (arrival_check, True),
}
CLASSES = {"UrllibTransport": UrllibTransport, "ObjectArrivalVerifier": ObjectArrivalVerifier}


def test_every_patch_target_in_the_scripts_has_a_trigger():
    assert patch_targets() | object_patches() == PATCHES.keys()


@pytest.mark.parametrize("target", sorted(PATCHES))
def test_patch_target_exists_and_is_called(make_node, monkeypatch, target):
    trigger, offline = PATCHES[target]
    owner, _, name = target.rpartition(".")
    resolved = CLASSES[owner] if owner in CLASSES else pkgutil.resolve_name(owner)
    original = getattr(resolved, name)
    calls = []

    def sentinel(*args, **kwargs):
        calls.append(args)
        if offline:
            raise Offline(target)
        return original(*args, **kwargs)

    with patch.object(resolved, name, sentinel):
        trigger(make_node, monkeypatch)
    assert calls, f"{target} is patched by a simulation script but was not called"
    if target == "rclpy.node.Node.__init__":
        assert [args[1] for args in calls] == ["placecell"]  # scripts select the node by this name


# Overwritten attributes: the trigger installs a sentinel and runs the path that must use it.

OVERWRITES: dict[str, Callable] = {}


def overwrite(*paths):
    def register(function):
        for path in paths:
            OVERWRITES[path] = function
        return function

    return register


def localized(node):
    node.subscription("/amcl_pose").callback(localization(T0))
    node._tf.set("map", "base_footprint")
    node._tf.set("map", CAMERA_FRAME)


def frame(node):
    node.subscription("/camera/color/image_raw").callback(raw_image(T0))


@overwrite("node._frames.capture")
def capture_is_replaceable(make_node, monkeypatch, tmp_path):
    node = make_node()
    calls = []
    node._frames.capture = lambda message, **kwargs: calls.append((message, kwargs))
    frame(node)
    assert len(calls) == 1


@overwrite("node._frames.depth_at")
def depth_lookup_is_replaceable(make_node, monkeypatch, tmp_path):
    monkeypatch.setattr(IngestWorker, "start", lambda self: None)
    node = objects_node(make_node, monkeypatch, rgbd_wait_s=60.0)
    localized(node)
    calls = []
    node._frames.depth_at = lambda message, dimensions: calls.append(dimensions)
    frame(node)
    node.subscription("/camera/aligned_depth_to_color/image_raw").callback(depth_image(T0))
    node.subscription("/camera/color/camera_info").callback(camera_info(T0))
    node.timer(0.04).callback()
    assert calls == [(4, 2)]


@overwrite("node._answer")
def answer_is_replaceable(make_node, monkeypatch, tmp_path):
    node = make_node()
    called = threading.Event()
    node._answer = lambda question: called.set()
    node.subscription("~/ask").callback(SimpleNamespace(data="where is the printer?"))
    assert called.wait(10)


@overwrite("node._run_refiner")
def refiner_is_replaceable(make_node, monkeypatch, tmp_path):
    node = make_node()
    called = threading.Event()
    node._run_refiner = called.set
    node._refine()
    assert called.wait(10)


@overwrite("node._worker.has_capacity", "node._worker.submit")
def worker_admission_is_replaceable(make_node, monkeypatch, tmp_path):
    node = make_node()
    localized(node)
    calls = []
    node._worker.has_capacity = lambda: calls.append("has_capacity") or True
    node._worker.submit = lambda observation: calls.append("submit") or True
    frame(node)
    assert calls == ["has_capacity", "submit"]


def navigation_node(make_node, tmp_path):
    from placecell.navigation_ownership import NavigationOwnership, NavigationScope

    ownership = NavigationOwnership(
        str(tmp_path / "navigation.sqlite3"), NavigationScope("robot", "test-v1", "/navigate_to_pose")
    )
    ownership.attest_clean("offline seam check")
    ownership.close()
    return make_node(navigation_enabled=True)


def printer():
    from placecell.memory import Pose

    return Destination("printer", Pose(1, 2, map_id="test-v1"), "named_place")


@overwrite("navigator._client.send_goal_async")
def goal_submission_is_replaceable(make_node, monkeypatch, tmp_path):
    node = navigation_node(make_node, tmp_path)
    assert node._navigator is node._commands._navigator  # scripts wrap the node's own transport
    client = node._navigator._client
    client.ready = True
    goals = []
    client.send_goal_async = lambda goal, **kwargs: (
        goals.append(goal) or client.__class__.send_goal_async(client, goal, **kwargs)
    )
    node._navigator.send("r", printer(), lambda event: None)
    assert len(goals) == 1


@overwrite("navigator._response_timeout")
def response_timeout_is_replaceable(make_node, monkeypatch, tmp_path):
    node = navigation_node(make_node, tmp_path)
    node._navigator._client.ready = True
    events = []
    node._navigator._response_timeout = 0.0
    node._navigator.send("r", printer(), events.append)
    node._navigator.poll()
    assert [event.state for event in events] == ["uncertain"]


@overwrite("embedder._capabilities")
def embedder_capabilities_are_replaceable(make_node, monkeypatch, tmp_path):
    embedder = HashingEmbedder()
    embedder._capabilities = Capabilities(text=True, image=True)
    monkeypatch.setattr("placecell.ros2.components.build_embedder", lambda *a, **k: embedder)
    node = make_node()
    assert not node.logger.messages("warning")  # an image-capable embedder needs no captioner


class Delegate:
    """Records calls to named methods and forwards everything to the wrapped object."""

    def __init__(self, inner, calls, name):
        self._inner, self._calls, self._name = inner, calls, name

    def __getattr__(self, attribute):
        value = getattr(self._inner, attribute)
        if not callable(value):
            return value

        def called(*args, **kwargs):
            self._calls.append(f"{self._name}.{attribute}")
            return value(*args, **kwargs)

        return called


RESOLVER = ("resolve", "current", "prepare_destination", "arrival_available", "verify")


@overwrite(
    "commands._navigator",
    "commands._submit_callback",
    "commands._mission_planner",
    *(f"commands._resolver.{method}" for method in RESOLVER),
)
def controller_collaborators_are_replaceable(make_node, monkeypatch, tmp_path):
    rig = faults._Rig(tmp_path)
    try:
        evidence = rig.memory_destination()
        rig.model.reply = ChatReply(
            None,
            (
                ToolCall(
                    "fixture",
                    "propose_navigation_plan",
                    {"decision": "ready", "destinations": ["printer"], "message": "One visual destination."},
                ),
            ),
        )
        commands, calls = rig.commands, []
        commands._navigator = Delegate(commands._navigator, calls, "_navigator")
        commands._mission_planner = Delegate(commands._mission_planner, calls, "_mission_planner")
        submit = commands._submit_callback
        commands._submit_callback = lambda task: calls.append("_submit_callback") or submit(task)
        resolver = commands._resolver
        for method in RESOLVER:
            original = getattr(resolver, method)
            setattr(resolver, method, lambda *a, _o=original, _m=method, **k: calls.append(_m) or _o(*a, **k))
        rig.start()
        rig.client.accept()
        rig.finish()
        rig.advance(0.1)
        commands.observe(Observation("fault-robot", "front", rig.stamp, rig.pose, evidence, localization_checked=True))
        rig.drain()
        assert rig.state == "succeeded"
        assert {"_navigator.send", "_mission_planner.plan", "_submit_callback", *RESOLVER} <= set(calls)
    finally:
        rig.close()


def test_every_overwritten_attribute_in_the_scripts_has_a_trigger():
    assert overwritten() == OVERWRITES.keys()


@pytest.mark.parametrize("path", sorted(OVERWRITES))
def test_overwritten_attribute_is_used(make_node, monkeypatch, tmp_path, path):
    OVERWRITES[path](make_node, monkeypatch, tmp_path)


def test_node_attributes_the_scripts_use_exist(make_node, monkeypatch, tmp_path):
    node = objects_node(
        make_node,
        monkeypatch,
        navigation_enabled=True,
        mission_enabled=True,
        mission_model="scripted",
        mission_trace_path=str(tmp_path / "traces.sqlite3"),
    )
    paths = node_paths()
    assert "node._frames.capture" in paths and "node._tf.can_transform" in paths
    missing = []
    for path in sorted(paths):
        value = node
        for name in path.split(".")[1:]:
            if not hasattr(value, name):
                missing.append(path)
                break
            value = getattr(value, name)
    assert not missing


# Parameters the scripts set, from dict literals, update() calls and `-p name:=value` arguments.


def literal_type(value: ast.AST) -> str | None:
    if isinstance(value, ast.Constant):
        return ros_type(value.value)
    if isinstance(value, ast.JoinedStr) or (isinstance(value, ast.Call) and chain(value.func) == "str"):
        return "PARAMETER_STRING"
    return None


def script_parameters() -> set[tuple[str, str, str | None]]:
    found = set()
    for script, tree in trees():
        for node in ast.walk(tree):
            items = []
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
                if any(isinstance(t, ast.Name) and t.id in {"params", "p"} for t in node.targets):
                    items = list(zip(node.value.keys, node.value.values, strict=True))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                receiver = ast.unparse(node.func.value)
                if receiver in {"p", "params"} or receiver.endswith("['ros__parameters']"):
                    items = [(ast.Constant(k.arg), k.value) for k in node.keywords]
                    for argument in node.args:
                        if isinstance(argument, ast.Dict):
                            items += list(zip(argument.keys, argument.values, strict=True))
            elif isinstance(node, ast.List):
                # `-p name:=value` node arguments; the value is formatted at run time in an f-string.
                for flag, argument in itertools.pairwise(node.elts):
                    if not (isinstance(flag, ast.Constant) and flag.value == "-p"):
                        continue
                    if isinstance(argument, ast.Constant):
                        name, value = argument.value.split(":=")
                        found.add((script, name, ros_type(yaml.safe_load(value))))
                    else:
                        found.add((script, argument.values[0].value.removesuffix(":="), None))
            for key, value in items:
                if isinstance(key, ast.Constant):
                    found.add((script, key.value, literal_type(value)))
    return found


def test_parameters_the_scripts_set_are_declared_with_their_types(make_node):
    declared = {d.name: d.type for d in make_node().declared}
    parameters = script_parameters()
    assert {"sensor_test.py", "overload_test.py", "checkpoint_control.py", "operator_test.py"} <= {
        script for script, _, _ in parameters
    }
    undeclared = sorted({(s, k) for s, k, _ in parameters if k not in declared and k not in BUILTIN})
    assert not undeclared
    mismatched = sorted((s, k, t) for s, k, t in parameters if k not in BUILTIN and t is not None and t != declared[k])
    assert not mismatched
