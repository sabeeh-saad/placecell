"""Stand-ins for the ROS 2 modules placecell.ros2 imports, recording what a node creates.

Only the surface the node and its adapters use is modelled. `install` puts the fakes in
`sys.modules` through `monkeypatch`, so they are removed again when the test ends.
"""

from __future__ import annotations

import math
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

MODULES = (
    "rclpy",
    "rclpy.action",
    "rclpy.callback_groups",
    "rclpy.clock",
    "rclpy.duration",
    "rclpy.node",
    "rclpy.qos",
    "rclpy.time",
    "tf2_ros",
    "action_msgs",
    "action_msgs.srv",
    "geometry_msgs",
    "geometry_msgs.msg",
    "nav2_msgs",
    "nav2_msgs.action",
    "nav2_msgs.msg",
    "sensor_msgs",
    "sensor_msgs.msg",
    "std_msgs",
    "std_msgs.msg",
    "std_srvs",
    "std_srvs.srv",
    "unique_identifier_msgs",
    "unique_identifier_msgs.msg",
)


def ros_type(value: Any) -> str:
    """The parameter type rclpy infers from a default value."""
    if isinstance(value, bool):
        return "PARAMETER_BOOL"
    if isinstance(value, int):
        return "PARAMETER_INTEGER"
    if isinstance(value, float):
        return "PARAMETER_DOUBLE"
    if isinstance(value, str):
        return "PARAMETER_STRING"
    raise TypeError(f"no ROS parameter type for {value!r}")


class Message:
    """A message whose fields are given as keywords."""

    def __init__(self, **fields: Any) -> None:
        self.__dict__.update(fields)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"


class AutoMessage(Message):
    """A request or goal message whose nested fields exist before they are assigned."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        value = AutoMessage()
        setattr(self, name, value)
        return value


def message_type(name: str, base: type = Message, **attributes: Any) -> type:
    return type(name, (base,), attributes)


# rclpy.time / rclpy.duration / rclpy.clock


class Time:
    def __init__(self, *, seconds: float = 0, nanoseconds: int = 0, clock_type: Any = None) -> None:
        self.seconds, self.extra_nanoseconds, self.clock_type = seconds, nanoseconds, clock_type
        self.nanoseconds = round(seconds * 1e9) + nanoseconds

    def to_msg(self) -> SimpleNamespace:
        return SimpleNamespace(sec=self.nanoseconds // 10**9, nanosec=self.nanoseconds % 10**9)

    def __repr__(self) -> str:
        return f"Time({self.nanoseconds}ns)"


class Duration:
    def __init__(self, *, seconds: float = 0, nanoseconds: int = 0) -> None:
        self.nanoseconds = round(seconds * 1e9) + nanoseconds

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Duration) and other.nanoseconds == self.nanoseconds

    def __hash__(self) -> int:
        return hash(self.nanoseconds)

    def __repr__(self) -> str:
        return f"Duration({self.nanoseconds}ns)"


class ClockType:
    ROS_TIME = "ROS_TIME"
    STEADY_TIME = "STEADY_TIME"
    SYSTEM_TIME = "SYSTEM_TIME"


class Clock:
    """A timer clock argument; only its type matters to the node."""

    def __init__(self, *, clock_type: str = ClockType.SYSTEM_TIME) -> None:
        self.clock_type = clock_type


@dataclass(frozen=True)
class JumpThreshold:
    min_forward: Any
    min_backward: Any
    on_clock_change: bool = True


@dataclass
class JumpHandle:
    threshold: JumpThreshold
    pre_callback: Callable[[], None] | None
    post_callback: Callable[[Any], None] | None
    registered: bool = True

    def unregister(self) -> None:
        self.registered = False


class NodeClock:
    """The node's ROS clock. Tests set its time and trigger jumps explicitly."""

    def __init__(self, seconds: float = 1000.0) -> None:
        self.seconds = seconds
        self.jumps: list[JumpHandle] = []

    def now(self) -> Time:
        return Time(seconds=self.seconds)

    def create_jump_callback(
        self,
        threshold: JumpThreshold,
        *,
        pre_callback: Callable[[], None] | None = None,
        post_callback: Callable[[Any], None] | None = None,
    ) -> JumpHandle:
        handle = JumpHandle(threshold, pre_callback, post_callback)
        self.jumps.append(handle)
        return handle

    def jump(self, seconds: float) -> None:
        """Move the clock like a /clock reset, calling registered jump callbacks."""
        for handle in self.jumps:
            if handle.registered and handle.pre_callback is not None:
                handle.pre_callback()
        self.seconds = seconds


# rclpy.qos / rclpy.callback_groups


class ReliabilityPolicy:
    RELIABLE = "RELIABLE"
    BEST_EFFORT = "BEST_EFFORT"


class DurabilityPolicy:
    TRANSIENT_LOCAL = "TRANSIENT_LOCAL"
    VOLATILE = "VOLATILE"


@dataclass(frozen=True)
class QoSProfile:
    depth: int
    reliability: str = ""
    durability: str = ""

    def __str__(self) -> str:
        return ",".join(str(v) for v in (self.depth, self.reliability, self.durability) if v)


class SensorDataQoS:
    def __str__(self) -> str:
        return "sensor_data"


qos_profile_sensor_data = SensorDataQoS()


class MutuallyExclusiveCallbackGroup:
    pass


class ReentrantCallbackGroup:
    pass


# tf2_ros


class TransformException(Exception):  # noqa: N818 - tf2_ros name
    pass


class Buffer:
    """Transforms keyed by (target, source); every lookup is recorded."""

    def __init__(self) -> None:
        self.transforms: dict[tuple[str, str], tuple[float, float, float, float, float]] = {}
        self.lookups: list[SimpleNamespace] = []

    def set(
        self, target: str, source: str, x: float = 0.0, y: float = 0.0, z: float = 0.0, yaw: float = 0.0, stamp=0.0
    ) -> None:
        """`stamp` is what a lookup of the latest transform, Time(), reports."""
        self.transforms[target, source] = (x, y, z, yaw, stamp)

    def lookup_transform(self, target: str, source: str, time: Time, timeout: Duration | None = None) -> Any:
        self.lookups.append(SimpleNamespace(target=target, source=source, time=time, timeout=timeout))
        if (target, source) not in self.transforms:
            raise TransformException(f"{target} -> {source} does not exist")
        x, y, z, yaw, latest = self.transforms[target, source]
        stamp = (time if time.nanoseconds else Time(seconds=latest)).to_msg()
        return SimpleNamespace(
            header=SimpleNamespace(stamp=stamp, frame_id=target),
            child_frame_id=source,
            transform=SimpleNamespace(
                translation=SimpleNamespace(x=x, y=y, z=z),
                rotation=SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
            ),
        )

    def can_transform(self, target: str, source: str, time: Time, timeout: Duration | None = None) -> bool:
        return (target, source) in self.transforms


class TransformListener:
    def __init__(self, buffer: Buffer, node: Any, **kwargs: Any) -> None:
        self.buffer, self.node = buffer, node


# rclpy.node


@dataclass
class Subscription:
    msg_type: type
    topic: str
    callback: Callable[[Any], None]
    qos: Any
    callback_group: Any = None


@dataclass
class Timer:
    period: float
    callback: Callable[[], None]
    clock: Clock | None = None
    callback_group: Any = None


@dataclass
class Service:
    srv_type: type
    name: str
    callback: Callable[[Any, Any], Any]


class Publisher:
    def __init__(self, msg_type: type, topic: str, qos: Any) -> None:
        self.msg_type, self.topic, self.qos = msg_type, topic, qos
        self.messages: list[Any] = []
        self._changed = threading.Condition()

    def publish(self, message: Any) -> None:
        with self._changed:
            self.messages.append(message)
            self._changed.notify_all()

    def wait_for(self, count: int, timeout: float = 10) -> bool:
        """Block until `count` messages were published; for messages sent from worker threads."""
        return self.wait_until(lambda messages: len(messages) >= count, timeout)

    def wait_until(self, predicate: Callable[[list[Any]], bool], timeout: float = 10) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: predicate(self.messages), timeout)

    def get_subscription_count(self) -> int:
        return 0


class Future:
    def __init__(self) -> None:
        self.callbacks: list[Callable[[Any], None]] = []

    def add_done_callback(self, callback: Callable[[Any], None]) -> None:
        self.callbacks.append(callback)

    def done(self) -> bool:
        return False


class ServiceClient:
    def __init__(self, node: FakeNode, srv_type: type, name: str, callback_group: Any = None) -> None:
        self.srv_type, self.name, self.callback_group = srv_type, name, callback_group
        self.ready = False
        self.requests: list[Any] = []

    def service_is_ready(self) -> bool:
        return self.ready

    def call_async(self, request: Any) -> Future:
        self.requests.append(request)
        return Future()

    def remove_pending_request(self, future: Future) -> None:
        pass


class ActionClient:
    def __init__(self, node: FakeNode, action_type: type, action_name: str, *, callback_group: Any = None) -> None:
        self.node, self.action_type, self.action_name = node, action_type, action_name
        self.callback_group = callback_group
        self.ready = False
        self.goals: list[Any] = []
        self.sent = threading.Event()
        node.action_clients.append(self)

    def server_is_ready(self) -> bool:
        return self.ready

    def send_goal_async(self, goal: Any, feedback_callback: Any = None, **kwargs: Any) -> Future:
        self.goals.append(goal)
        self.sent.set()
        return Future()


@dataclass
class Declared:
    name: str
    default: Any
    type: str
    value: Any


class Logger:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        self.records.append((level, message, kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("warning", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)

    def messages(self, level: str) -> list[str]:
        return [message for recorded, message, _ in list(self.records) if recorded == level]


class FakeNode:
    """rclpy.node.Node: declared parameters and created entities in creation order."""

    registry: ClassVar[FakeRos | None] = None

    def __init__(self, node_name: str, *, parameter_overrides: Any = None, **kwargs: Any) -> None:
        ros = FakeNode.registry
        assert ros is not None, "install() the ROS fakes first"
        self.node_name = node_name
        self.overrides = dict(ros.overrides)
        self.declared: list[Declared] = []
        self.subscriptions: list[Subscription] = []
        self.publishers: list[Publisher] = []
        self.timers: list[Timer] = []
        self.services: list[Service] = []
        self.clients: list[ServiceClient] = []
        self.action_clients: list[ActionClient] = []
        self.logger = Logger()
        self.clock = NodeClock(ros.start_s)
        self.destroyed = False
        ros.nodes.append(self)

    def declare_parameter(self, name: str, value: Any) -> SimpleNamespace:
        kind = ros_type(value)
        if any(d.name == name for d in self.declared):
            raise AssertionError(f"parameter {name} declared twice")
        chosen = self.overrides.get(name, value)
        if ros_type(chosen) != kind:
            # rclpy raises InvalidParameterTypeException for a statically typed parameter.
            raise TypeError(f"parameter {name} is {kind}, override {chosen!r} is not")
        self.declared.append(Declared(name, value, kind, chosen))
        return SimpleNamespace(value=chosen)

    def get_logger(self) -> Logger:
        return self.logger

    def get_clock(self) -> NodeClock:
        return self.clock

    def resolve_topic_name(self, topic: str, *, only_expand: bool = False) -> str:
        if topic.startswith("~/"):
            return f"/{self.node_name}/{topic[2:]}"
        return topic if topic.startswith("/") else "/" + topic

    def create_subscription(
        self, msg_type: type, topic: str, callback: Callable[[Any], None], qos: Any, *, callback_group: Any = None
    ) -> Subscription:
        subscription = Subscription(msg_type, topic, callback, qos, callback_group)
        self.subscriptions.append(subscription)
        return subscription

    def create_publisher(self, msg_type: type, topic: str, qos: Any) -> Publisher:
        publisher = Publisher(msg_type, topic, qos)
        self.publishers.append(publisher)
        return publisher

    def create_timer(
        self, period: float, callback: Callable[[], None], callback_group: Any = None, clock: Clock | None = None
    ) -> Timer:
        timer = Timer(period, callback, clock, callback_group)
        self.timers.append(timer)
        return timer

    def create_service(self, srv_type: type, name: str, callback: Callable[[Any, Any], Any]) -> Service:
        service = Service(srv_type, name, callback)
        self.services.append(service)
        return service

    def create_client(self, srv_type: type, name: str, *, callback_group: Any = None) -> ServiceClient:
        client = ServiceClient(self, srv_type, name, callback_group)
        self.clients.append(client)
        return client

    def destroy_node(self) -> bool:
        if FakeNode.registry is not None:
            FakeNode.registry.calls.append("Node.destroy_node")
        self.destroyed = True
        return True

    # Lookups for tests.

    def subscription(self, topic: str) -> Subscription:
        (match,) = [s for s in self.subscriptions if s.topic == topic]
        return match

    def publisher(self, topic: str) -> Publisher:
        (match,) = [p for p in self.publishers if p.topic == topic]
        return match

    def timer(self, period: float) -> Timer:
        (match,) = [t for t in self.timers if t.period == period]
        return match


@dataclass
class FakeRos:
    """Nodes created while installed, the parameter overrides for the next node, and a call log."""

    overrides: dict[str, Any] = field(default_factory=dict)
    start_s: float = 1000.0
    nodes: list[FakeNode] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)


def _modules() -> dict[str, ModuleType]:
    modules = {name: ModuleType(name) for name in MODULES}

    def define(name: str, **attributes: Any) -> None:
        for key, value in attributes.items():
            setattr(modules[name], key, value)

    for name in MODULES:
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, modules[name])
            modules[parent].__path__ = []
    define("rclpy.node", Node=FakeNode)
    define("rclpy.clock", Clock=Clock, ClockType=ClockType, JumpThreshold=JumpThreshold)
    define("rclpy.duration", Duration=Duration)
    define("rclpy.time", Time=Time)
    define(
        "rclpy.qos",
        QoSProfile=QoSProfile,
        ReliabilityPolicy=ReliabilityPolicy,
        DurabilityPolicy=DurabilityPolicy,
        qos_profile_sensor_data=qos_profile_sensor_data,
    )
    define(
        "rclpy.callback_groups",
        MutuallyExclusiveCallbackGroup=MutuallyExclusiveCallbackGroup,
        ReentrantCallbackGroup=ReentrantCallbackGroup,
    )
    define("rclpy.action", ActionClient=ActionClient)
    define("tf2_ros", Buffer=Buffer, TransformException=TransformException, TransformListener=TransformListener)
    define(
        "sensor_msgs.msg",
        Image=message_type("Image"),
        CompressedImage=message_type("CompressedImage"),
        CameraInfo=message_type("CameraInfo"),
    )
    define("std_msgs.msg", String=message_type("String"))
    define("std_srvs.srv", Trigger=message_type("Trigger"))
    define(
        "geometry_msgs.msg",
        PoseWithCovarianceStamped=message_type("PoseWithCovarianceStamped"),
        PolygonStamped=message_type("PolygonStamped"),
        PoseStamped=message_type("PoseStamped", AutoMessage),
    )
    get_result = message_type("GetResultService", Request=message_type("Request", AutoMessage))
    define(
        "nav2_msgs.action",
        NavigateToPose=message_type(
            "NavigateToPose",
            Goal=message_type("Goal", AutoMessage),
            Impl=SimpleNamespace(GetResultService=get_result),
        ),
        ComputePathToPose=message_type("ComputePathToPose", Goal=message_type("Goal", AutoMessage)),
    )
    define("nav2_msgs.msg", Costmap=message_type("Costmap"))
    define("action_msgs.srv", CancelGoal=message_type("CancelGoal", Request=message_type("Request", AutoMessage)))
    define("unique_identifier_msgs.msg", UUID=message_type("UUID"))
    return modules


def install(monkeypatch: pytest.MonkeyPatch) -> FakeRos:
    """Put the fakes in sys.modules for this test only; monkeypatch removes them at teardown."""
    stale = [name for name in MODULES if name in sys.modules and not hasattr(sys.modules[name], "__file__")]
    assert not stale, f"ROS stubs leaked from another test: {stale}"
    ros = FakeRos()
    for name, module in _modules().items():
        monkeypatch.setitem(sys.modules, name, module)
    # The node module binds the ROS classes it imports; load it again against these fakes.
    sys.modules.pop("placecell.ros2.placecell_node", None)
    monkeypatch.setattr(FakeNode, "registry", ros)
    return ros


# Building the production node.


def hermetic_parameters(root: Path) -> dict[str, Any]:
    """Storage inside the test directory and an in-memory store."""
    return {
        "db_path": "",
        "keyframe_dir": str(root / "keyframes"),
        "corrections_path": str(root / "corrections.jsonl"),
        "command_journal_path": str(root / "commands.sqlite3"),
        "mission_context_path": str(root / "missions.sqlite3"),
        "navigation_ownership_path": str(root / "navigation.sqlite3"),
        "map_id": "test-v1",
    }


class NodeFactory:
    """Builds nodes through `create_node()` and destroys them when the test ends."""

    def __init__(self, ros: FakeRos, root: Path) -> None:
        self.ros, self.root = ros, root
        self.built: list[Any] = []

    def __call__(self, **parameters: Any) -> Any:
        from placecell.ros2.node import create_node

        self.ros.overrides = {**hermetic_parameters(self.root), **parameters}
        before = len(self.ros.nodes)
        node = create_node()
        assert self.ros.nodes[before:] == [node]
        self.built.append(node)
        unknown = set(self.ros.overrides) - {d.name for d in node.declared}
        assert not unknown, f"overrides for undeclared parameters: {unknown}"
        return node

    def close(self) -> None:
        # A node that failed during construction has already released what it acquired.
        for node in self.built:
            if not node.destroyed:
                node.destroy_node()


@pytest.fixture
def ros(monkeypatch: pytest.MonkeyPatch) -> FakeRos:
    return install(monkeypatch)


@pytest.fixture
def make_node(ros: FakeRos, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[NodeFactory]:
    # Any default "~/.placecell" path lands in the test directory.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    before = set(threading.enumerate())
    factory = NodeFactory(ros, tmp_path)
    yield factory
    factory.close()
    # LanceDB keeps one process-wide event loop thread.
    leaked = [t.name for t in set(threading.enumerate()) - before if t.name != "LanceDBBackgroundEventLoop"]
    assert not leaked, f"threads still running after the node was destroyed: {leaked}"


# Incoming messages, shaped like the sensor_msgs/geometry_msgs fields the node reads.

CAMERA_FRAME = "camera_optical"


def header(sec: Any, nanosec: Any = 0, frame: str = CAMERA_FRAME) -> SimpleNamespace:
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec), frame_id=frame)


def raw_image(
    sec: Any, *, frame: str = CAMERA_FRAME, width: int = 4, height: int = 2, encoding: str = "rgb8"
) -> SimpleNamespace:
    step = width * 3
    data = bytes((i * 37) % 256 for i in range(step * height))
    return SimpleNamespace(
        header=header(sec, frame=frame), width=width, height=height, encoding=encoding, step=step, data=data
    )


def jpeg_image(sec: Any, *, frame: str = CAMERA_FRAME, width: int = 4, height: int = 2) -> SimpleNamespace:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (128, 64, 32)).save(buffer, "JPEG")
    return SimpleNamespace(header=header(sec, frame=frame), format="rgb8; jpeg compressed bgr8", data=buffer.getvalue())


def image(sec: Any, *, compressed: bool, **kwargs: Any) -> SimpleNamespace:
    return jpeg_image(sec, **kwargs) if compressed else raw_image(sec, **kwargs)


def localization(
    sec: float, x: float = 0.0, y: float = 0.0, yaw: float = 0.0, *, frame: str = "map", std: float = 0.1
) -> SimpleNamespace:
    covariance = [0.0] * 36
    for index in (0, 7, 35):
        covariance[index] = std**2
    position = SimpleNamespace(x=x, y=y, z=0.0)
    orientation = SimpleNamespace(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return SimpleNamespace(
        header=header(int(sec), round(sec % 1 * 1e9), frame),
        pose=SimpleNamespace(pose=SimpleNamespace(position=position, orientation=orientation), covariance=covariance),
    )


def depth_image(sec: Any, *, frame: str = CAMERA_FRAME, width: int = 4, height: int = 2) -> SimpleNamespace:
    """Aligned 16UC1 depth, one metre everywhere."""
    return SimpleNamespace(
        header=header(sec, frame=frame),
        width=width,
        height=height,
        encoding="16UC1",
        is_bigendian=0,
        step=width * 2,
        data=(1000).to_bytes(2, "little") * (width * height),
    )


def camera_info(sec: Any, *, frame: str = CAMERA_FRAME, width: int = 4, height: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        header=header(sec, frame=frame),
        width=width,
        height=height,
        d=[0.0] * 5,
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        k=[2.0, 0.0, width / 2, 0.0, 2.0, height / 2, 0.0, 0.0, 1.0],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(x_offset=0, y_offset=0),
    )


def spy(calls: list[str], name: str, function: Callable[..., Any], result: Any = None) -> Callable[..., Any]:
    """Record `name` and delegate; a non-None `result` replaces the delegated return value."""

    def recorded(*args: Any, **kwargs: Any) -> Any:
        calls.append(name)
        value = function(*args, **kwargs)
        return value if result is None else result

    return recorded
