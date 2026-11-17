"""Control pib4advanced servos through ROS 2 using degrees and milliamperes.

Requirements:
    Source the robot's ROS 2 environment, including its generated ``datatypes``
    package. Required packages are rclpy, sensor_msgs, diagnostic_msgs and
    trajectory_msgs. Run in the robot's ROS_DOMAIN_ID with DDS connectivity.
    This module does not install dependencies or start the motor backend.

Usage:
    The numbers below are illustrative, NOT measured safe robot limits::

        with ServoController() as controller:
            head = controller["head_turn"]
            head.set_limits(-10.0, 10.0)  # Replace with measured safe limits.
            angle_deg = head.get_position()
            current_ma = head.get_current()
            head.move(0.0)
            controller["head_tilt"].set_limits(-5.0, 5.0)
            controller.move_many({"head_turn": 5.0, "head_tilt": 0.0})

Safety:
    All physical servos start WITHOUT limits and cannot move until set_limits()
    has been called. Limits are held in memory only; a future configuration
    loader can use the setters. Cerebra seed limits are deliberately not used.
    Limits do not guarantee collision avoidance. No relay, torque or speed
    settings are changed. Closing the adapter is NOT an emergency stop.
    A successful move means backend acceptance, NOT arrival at the target.
    A timeout cannot undo a command already received; never blindly retry it.

ROS contract:
    Commands use /apply_joint_trajectory (datatypes/srv/ApplyJointTrajectory)
    with all joints in ONE JointTrajectoryPoint, positions in radians.
    Feedback uses /joint_states (sensor_msgs/msg/JointState), also in radians.
    /motor_current uses diagnostic_msgs/msg/DiagnosticStatus: name is the ROS
    motor name, values contains a matching key with a numeric mA string.
    This layout follows pib-rocks/pib-backend develop, motor_current.py.
    Except for /joint_states, these interfaces still require confirmation
    on the robot.

Execution:
    The controller owns an independent ROS context, node and executor. Calls
    are synchronous and serialized; callbacks run only while reading or moving.
    Do not call this blocking API from ROS callbacks. Feedback expires after
    max_age_s measured from local receipt, not the physical sampling time;
    queued DDS messages can be older. Missing/stale feedback raises TimeoutError,
    never a fabricated zero. There is no service-position fallback.
    Finger groups expand to the five stretch joints, excluding opposition.
    Read individual fingers for positions/currents; groups have no single value.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from math import degrees, isfinite, radians
from threading import RLock
from time import monotonic
from types import MappingProxyType, TracebackType

from datatypes.srv import ApplyJointTrajectory
from diagnostic_msgs.msg import DiagnosticStatus
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

SERVO_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "head_turn": "turn_head_motor",
        "head_tilt": "tilt_forward_motor",
        "shoulder_left_vertical": "shoulder_vertical_left",
        "shoulder_left_horizontal": "shoulder_horizontal_left",
        "upper_arm_left_rotation": "upper_arm_left_rotation",
        "elbow_left": "elbow_left",
        "lower_arm_left_rotation": "lower_arm_left_rotation",
        "wrist_left": "wrist_left",
        "thumb_left_opposition": "thumb_left_opposition",
        "thumb_left_stretch": "thumb_left_stretch",
        "index_left": "index_left_stretch",
        "middle_left": "middle_left_stretch",
        "ring_left": "ring_left_stretch",
        "pinky_left": "pinky_left_stretch",
        "shoulder_right_vertical": "shoulder_vertical_right",
        "shoulder_right_horizontal": "shoulder_horizontal_right",
        "upper_arm_right_rotation": "upper_arm_right_rotation",
        "elbow_right": "elbow_right",
        "lower_arm_right_rotation": "lower_arm_right_rotation",
        "wrist_right": "wrist_right",
        "thumb_right_opposition": "thumb_right_opposition",
        "thumb_right_stretch": "thumb_right_stretch",
        "index_right": "index_right_stretch",
        "middle_right": "middle_right_stretch",
        "ring_right": "ring_right_stretch",
        "pinky_right": "pinky_right_stretch",
    }
)
SERVO_GROUPS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "fingers_left": ("thumb_left_stretch", "index_left", "middle_left", "ring_left", "pinky_left"),
        "fingers_right": ("thumb_right_stretch", "index_right", "middle_right", "ring_right", "pinky_right"),
    }
)


_MOVE_TIMEOUT = "ROS operation timed out; a submitted move may still execute."


class ServoError(RuntimeError):
    """Report unavailable controller state or a rejected/invalid ROS response."""


def _finite(value: float, label: str) -> float:
    """Reject booleans, non-numbers and non-finite numeric values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number.")
    try:
        result = float(value)
    except OverflowError:
        raise ValueError(f"{label} must be finite.") from None
    if not isfinite(result):
        raise ValueError(f"{label} must be finite.")
    return result


def _positive(value: float, label: str) -> float:
    """Validate a positive duration."""
    result = _finite(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return result


class Servo:
    """Represent a named servo or command-only finger group owned by a controller."""

    def __init__(self, controller: ServoController, name: str) -> None:
        """Bind to an existing controller; reject unknown logical names."""
        controller._targets(name)
        self._controller = controller
        self._name = name

    @property
    def name(self) -> str:
        """Return the logical servo name."""
        return self._name

    def move(self, angle_deg: float) -> None:
        """Submit a target in degrees within configured limits, without waiting for arrival."""
        self._controller.move_many({self._name: angle_deg})

    def get_position(self) -> float:
        """Return fresh measured degrees, or raise TimeoutError; groups cannot be read."""
        return self._controller._get_feedback(self._name, current=False)

    def get_current(self) -> float:
        """Return fresh measured milliamperes, or raise TimeoutError; groups cannot be read."""
        return self._controller._get_feedback(self._name, current=True)

    def get_limits(self) -> tuple[float, float] | None:
        """Return inclusive degree limits, or None if unset.

        For groups, return the intersection of member limits; raise ServoError
        if configured member ranges have no common target.
        """
        return self._controller._get_limits(self._name)

    def set_limits(self, min_deg: float, max_deg: float) -> None:
        """Set measured inclusive degree limits in memory; groups update all members.

        Reject inverted/non-finite limits. This neither moves motors nor changes
        backend settings. The caller must supply physically verified limits.
        """
        self._controller._set_limits(self._name, min_deg, max_deg)


class ServoController:
    """Own synchronous ROS communication and servo objects.

    Args:
        timeout_s: Maximum wait for feedback or a complete service transaction.
        max_age_s: Maximum local age of a cached feedback value.

    Raises:
        ValueError: A duration is not finite and positive.
        ImportError: A required ROS package is missing (at module import).
    """

    def __init__(self, *, timeout_s: float = 2.0, max_age_s: float = 1.0) -> None:
        """Create a private ROS context; do not send commands or enable power."""
        self._timeout_s = _positive(timeout_s, "timeout_s")
        self._max_age_s = _positive(max_age_s, "max_age_s")
        self._lock = RLock()
        self._closed = False
        self._limits: dict[str, tuple[float, float]] = {}
        self._positions: dict[str, tuple[float, float]] = {}
        self._currents: dict[str, tuple[float, float]] = {}
        with ExitStack() as resources:
            self._open_ros(resources)
            self._subscribe()
            self._resources = resources.pop_all()

    def _open_ros(self, resources: ExitStack) -> None:
        """Create ROS resources and register their cleanup."""
        self._context = Context()
        self._context.init(args=[])
        resources.callback(self._context.try_shutdown)
        self._node = Node("pib_servo_adapter", context=self._context, use_global_arguments=False)
        resources.callback(self._node.destroy_node)
        self._executor = SingleThreadedExecutor(context=self._context)
        resources.callback(self._executor.shutdown)
        self._executor.add_node(self._node)
        self._client = self._node.create_client(ApplyJointTrajectory, "/apply_joint_trajectory")

    def _subscribe(self) -> None:
        """Subscribe to position and current feedback."""
        self._node.create_subscription(JointState, "/joint_states", self._on_positions, qos_profile_sensor_data)
        # Current is published per motor in bursts; retain more than one full cycle.
        current_qos = QoSProfile(depth=2 * len(SERVO_NAMES), reliability=ReliabilityPolicy.BEST_EFFORT)
        self._node.create_subscription(DiagnosticStatus, "/motor_current", self._on_current, current_qos)

    def __enter__(self) -> ServoController:
        """Return this open controller for use in a with statement."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Release only this controller's ROS resources, including after errors."""
        self.close()

    def __getitem__(self, name: str) -> Servo:
        """Return a servo handle for a logical name; raise KeyError for unknown names."""
        with self._lock:
            self._ensure_open()
            return Servo(self, name)

    def close(self) -> None:
        """Idempotently release resources; do not stop motors or shut down other ROS contexts."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._resources.close()

    def _ensure_open(self) -> None:
        """Raise ServoError if the controller or ROS context is closed."""
        if self._closed or not self._context.ok():
            raise ServoError("The servo controller is closed or its ROS context has stopped.")

    @staticmethod
    def _targets(name: str) -> tuple[str, ...]:
        """Resolve a logical name to physical servo names."""
        if name in SERVO_GROUPS:
            return SERVO_GROUPS[name]
        if name not in SERVO_NAMES:
            raise KeyError(f"Unknown logical servo name: {name!r}")
        return (name,)

    def _set_limits(self, name: str, min_deg: float, max_deg: float) -> None:
        """Validate and store degree limits for all named targets."""
        lower = _finite(min_deg, "min_deg")
        upper = _finite(max_deg, "max_deg")
        if lower > upper:
            raise ValueError("min_deg must not exceed max_deg.")
        with self._lock:
            self._ensure_open()
            for target in self._targets(name):
                self._limits[target] = (lower, upper)

    def _get_limits(self, name: str) -> tuple[float, float] | None:
        """Return the common degree limits, or None if any are unset."""
        with self._lock:
            self._ensure_open()
            limits = [self._limits.get(target) for target in self._targets(name)]
            if any(limit is None for limit in limits):
                return None
            lower = max(limit[0] for limit in limits if limit is not None)
            upper = min(limit[1] for limit in limits if limit is not None)
            if lower > upper:
                raise ServoError(f"No common limits for {name!r}.")
            return lower, upper

    def move_many(self, targets_deg: Mapping[str, float]) -> None:
        """Submit all targets in one trajectory point, after validating the whole batch.

        Args:
            targets_deg: Logical servo/group names mapped to absolute degrees.

        Raises:
            TypeError: targets_deg is not a mapping.
            KeyError: A logical name is unknown.
            ValueError: Targets are empty, invalid, overlapping or outside limits.
            ServoError: Limits are unset, the controller is closed or ROS rejects
                the command. Backend rejection can still mean partial execution.
            TimeoutError: Discovery/response timed out; execution is then unknown.
        """
        with self._lock:
            self._ensure_open()
            targets = self._expand_targets(targets_deg)
            request = self._make_request(targets)
            self._send(request)

    def _expand_targets(self, targets_deg: Mapping[str, float]) -> dict[str, float]:
        """Expand groups and validate every target before sending."""
        if not isinstance(targets_deg, Mapping):
            raise TypeError("targets_deg must be a mapping of servo names to degrees.")
        if not targets_deg:
            raise ValueError("At least one target is required.")
        expanded: dict[str, float] = {}
        for name, angle in targets_deg.items():
            physical = self._targets(name)
            value = _finite(angle, f"{name} angle_deg")
            for target in physical:
                if target in expanded:
                    raise ValueError(f"Overlapping targets for {target!r}.")
                self._validate_target(target, value)
                expanded[target] = value
        return expanded

    def _validate_target(self, name: str, angle_deg: float) -> None:
        """Reject targets without configured limits or outside their range."""
        limits = self._limits.get(name)
        if limits is None:
            raise ServoError(f"Set measured safe limits for {name!r} before moving.")
        if not limits[0] <= angle_deg <= limits[1]:
            raise ValueError(f"{name}: {angle_deg} degrees is outside {limits}.")

    @staticmethod
    def _make_request(targets: Mapping[str, float]) -> ApplyJointTrajectory.Request:
        """Build a single-point trajectory with ROS names and radians."""
        request = ApplyJointTrajectory.Request()
        point = JointTrajectoryPoint()
        request.joint_trajectory.joint_names = [SERVO_NAMES[name] for name in targets]
        point.positions = [radians(angle) for angle in targets.values()]
        request.joint_trajectory.points = [point]
        return request

    def _send(self, request: ApplyJointTrajectory.Request) -> None:
        """Submit a trajectory and validate its response within the timeout."""
        deadline = monotonic() + self._timeout_s
        self._wait_for_service(deadline)
        future = self._client.call_async(request)
        try:
            while not future.done():
                self._spin(deadline, _MOVE_TIMEOUT)
            self._check_response(future.result(), request.joint_trajectory.joint_names)
        finally:
            if not future.done():
                future.cancel()

    def _wait_for_service(self, deadline: float) -> None:
        """Wait for the trajectory service within the remaining timeout."""
        if not self._client.wait_for_service(timeout_sec=self._remaining(deadline, _MOVE_TIMEOUT)):
            raise TimeoutError("/apply_joint_trajectory is unavailable; no command sent.")

    @staticmethod
    def _check_response(response: ApplyJointTrajectory.Response | None, expected: list[str]) -> None:
        """Reject missing, failed or incomplete trajectory responses."""
        if response is None:
            raise ServoError("The trajectory service returned no response.")
        failures = [f"{result.motor_name}: {result.error_code} {result.detail}" for result in response.results if not result.successful]
        if not response.successful or failures:
            raise ServoError(f"Trajectory rejected (partial execution possible): {'; '.join(failures) or 'unspecified backend error'}")
        names = [result.motor_name for result in response.results]
        reported = set(names)
        unexpected = reported - set(expected)
        if unexpected or len(names) != len(reported):
            raise ServoError(f"Inconsistent trajectory response; unexpected {sorted(unexpected)} or duplicate motor results.")
        missing = set(expected) - reported
        if missing:
            raise ServoError(f"Incomplete trajectory response; execution unknown for {sorted(missing)}.")

    @staticmethod
    def _remaining(deadline: float, message: str) -> float:
        """Return seconds until the deadline, raising TimeoutError with message if expired."""
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError(message)
        return remaining

    def _spin(self, deadline: float, message: str) -> None:
        """Process ROS callbacks without exceeding the remaining wait."""
        self._ensure_open()
        self._executor.spin_once(timeout_sec=min(0.1, self._remaining(deadline, message)))

    def _get_feedback(self, name: str, *, current: bool) -> float:
        """Read fresh current or position feedback for one physical servo."""
        with self._lock:
            self._ensure_open()
            self._targets(name)
            if name in SERVO_GROUPS:
                raise ValueError("Finger groups have no single measurement; read individual fingers.")
            cache = self._currents if current else self._positions
            return self._wait_feedback(cache, SERVO_NAMES[name])

    def _drain(self) -> None:
        """Process already queued callbacks without waiting."""
        for _ in range(2 * len(SERVO_NAMES) + qos_profile_sensor_data.depth):
            self._executor.spin_once(timeout_sec=0)

    def _wait_feedback(self, cache: dict[str, tuple[float, float]], motor_name: str) -> float:
        """Wait for a fresh cached measurement or raise TimeoutError."""
        deadline = monotonic() + self._timeout_s
        message = f"No fresh feedback for {motor_name!r}; check motor power and ROS publishers."
        self._drain()
        while True:
            sample = cache.get(motor_name)
            if sample is not None and monotonic() - sample[1] <= self._max_age_s:
                return sample[0]
            if monotonic() >= deadline:
                raise TimeoutError(message)
            self._spin(deadline, message)

    def _on_positions(self, message: JointState) -> None:
        """Validate incoming joint states and update position feedback."""
        if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
            self._positions.clear()
            self._node.get_logger().error("Invalid /joint_states: duplicate names or mismatched position count.")
            return
        received = monotonic()
        for name, angle_rad in zip(message.name, message.position, strict=True):
            if name in SERVO_NAMES.values():
                self._store_position(name, angle_rad, received)

    def _store_position(self, name: str, angle_rad: float, received: float) -> None:
        """Cache a position in degrees or invalidate non-finite feedback."""
        angle_deg = degrees(angle_rad)
        if not isfinite(angle_deg):
            self._positions.pop(name, None)
            self._node.get_logger().error(f"Non-finite position for {name!r}.")
            return
        self._positions[name] = (angle_deg, received)

    def _on_current(self, message: DiagnosticStatus) -> None:
        """Cache valid motor current and log invalid feedback."""
        if message.name not in SERVO_NAMES.values():
            return
        self._currents.pop(message.name, None)
        try:
            value = self._parse_current(message)
        except ValueError as error:
            self._node.get_logger().error(f"Invalid current feedback for {message.name!r}: {error}")
            return
        self._currents[message.name] = (value, monotonic())

    @staticmethod
    def _level_code(level: object) -> int:
        """Normalize a ROS DiagnosticStatus level (single byte or int) to an integer."""
        if isinstance(level, (bytes, bytearray)) and len(level) == 1:
            return level[0]
        if not isinstance(level, bool) and isinstance(level, int):
            return level
        raise ValueError(f"Invalid diagnostic level: {level!r}")

    @staticmethod
    def _parse_current(message: DiagnosticStatus) -> float:
        """Extract a finite, non-negative current in milliamperes."""
        if ServoController._level_code(message.level) >= ServoController._level_code(DiagnosticStatus.ERROR):
            raise ValueError(f"Diagnostic level {message.level}: {message.message}")
        values = [entry.value for entry in message.values if entry.key == message.name]
        if len(values) != 1:
            raise ValueError("Expected exactly one current entry keyed by the ROS motor name.")
        current_ma = _finite(float(values[0]), "current_ma")
        if current_ma < 0:
            raise ValueError("Negative current is invalid or an unavailable-reading sentinel.")
        return current_ma
