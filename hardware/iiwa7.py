import threading
from dataclasses import dataclass
from enum import Enum

import numpy as np

from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
)
from pydrake.all import (
    Gain,
    OutputPort,
    Diagram,
    DiagramBuilder,
    IiwaCommandSender,
    IiwaControlMode,
    IiwaStatusReceiver,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    position_enabled,
    torque_enabled,
    Multiplexer,
    DrakeLcmInterface,
    DrakeLcm,
    Saturation,
    LcmInterfaceSystem,
    LeafSystem,
)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState


def AddLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm


class IiwaRobot(Diagram):
    """
    Drake Diagram providing an interface to a KUKA IIWA through the Drake LCM driver.
    """

    def __init__(
        self,
        lcm: DrakeLcmInterface,
        control_mode: IiwaControlMode = IiwaControlMode.kPositionAndTorque,
        lcm_channel_suffix: str = ""
    ):
        super().__init__()

        builder = DiagramBuilder()

        # Publish IIWA command.
        # IIWA driver won't respond faster than 1000Hz in torque_only mode and
        # 200Hz in other modes
        publish_period = 0.005
        if control_mode == IiwaControlMode.kTorqueOnly:
            publish_period = 0.001

        iiwa_command_sender = builder.AddSystem(
            IiwaCommandSender(control_mode=control_mode)
        )
        iiwa_command_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                channel="IIWA_COMMAND" + lcm_channel_suffix,
                lcm_type=lcmt_iiwa_command,
                lcm=lcm,
                publish_period=publish_period,
                use_cpp_serializer=True,
            )
        )
        builder.Connect(
            iiwa_command_sender.get_output_port(),
            iiwa_command_publisher.get_input_port(),
        )
        if position_enabled(control_mode):
            builder.ExportInput(
                iiwa_command_sender.get_position_input_port(),
                "position",
            )
        if torque_enabled(control_mode):
            builder.ExportInput(
                iiwa_command_sender.get_torque_input_port(),
                "torque",
            )
        # Receive IIWA status and populate the output ports.
        iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
        iiwa_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="IIWA_STATUS" + lcm_channel_suffix,
                lcm_type=lcmt_iiwa_status,
                lcm=lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10,
            )
        )
        builder.Connect(
            iiwa_status_subscriber.get_output_port(),
            iiwa_status_receiver.get_input_port(),
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_position_commanded_output_port(),
            "position_commanded",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_position_measured_output_port(),
            "position_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_velocity_estimated_output_port(),
            "velocity_estimated",
        )

        # These are negated as outlined in drake/manipulation/README.
        def NegatedPort(
            builder: DiagramBuilder, output_port: OutputPort
        ) -> OutputPort:
            negater = builder.AddNamedSystem(
                f"signflip_{output_port.get_name()}", Gain(-1,
                                                           size=output_port.size())
            )
            builder.Connect(output_port, negater.get_input_port())
            return negater.get_output_port()

        builder.ExportOutput(
            NegatedPort(
                builder=builder,
                output_port=iiwa_status_receiver.get_torque_commanded_output_port(),
            ),
            "torque_commanded",
        )
        builder.ExportOutput(
            NegatedPort(
                builder=builder,
                output_port=iiwa_status_receiver.get_torque_measured_output_port(),
            ),
            "torque_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_external_output_port(),
            "torque_external",
        )

        mux = builder.AddSystem(Multiplexer(input_sizes=[7, 7]))
        builder.Connect(
            iiwa_status_receiver.get_position_measured_output_port(),
            mux.get_input_port(0),
        )
        builder.Connect(
            iiwa_status_receiver.get_velocity_estimated_output_port(),
            mux.get_input_port(1),
        )
        builder.ExportOutput(
            mux.get_output_port(),
            "state_estimated",
        )

        builder.BuildInto(self)


class IntegratedVelocityController(LeafSystem):
    """
    Integrates a commanded joint velocity into a commanded joint position.

    q_cmd[k+1] = clip(
        q_cmd[k] + dt * v_cmd[k],
        joint_lower_limit,
        joint_upper_limit,
    )

    The initial state is taken from the initial_joint_position input during
    initialization.
    """

    def __init__(
        self,
        time_step: float,
        num_joints: int = 7,
        joint_lower_limit: float | list[float] = -np.inf,
        joint_upper_limit: float | list[float] = np.inf,
    ):
        super().__init__()

        self._num_joints = num_joints
        self._time_step = time_step

        lower = np.asarray(joint_lower_limit, dtype=float)
        upper = np.asarray(joint_upper_limit, dtype=float)

        if lower.ndim == 0:
            lower = np.full(num_joints, lower)

        if upper.ndim == 0:
            upper = np.full(num_joints, upper)

        assert lower.shape == (num_joints,)
        assert upper.shape == (num_joints,)

        self._lower = lower
        self._upper = upper

        self.velocity_input_port = self.DeclareVectorInputPort(
            "velocity", num_joints
        )

        self.initial_position_input_port = self.DeclareVectorInputPort(
            "initial_position", num_joints
        )

        state_index = self.DeclareDiscreteState(num_joints)

        self.DeclareStateOutputPort("position", state_index)

        self.DeclareInitializationDiscreteUpdateEvent(
            self._InitializeState
        )

        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._DiscreteUpdate,
        )

    def _InitializeState(self, context, discrete_state):
        q0 = self.initial_position_input_port.Eval(context)
        q0 = np.clip(q0, self._lower, self._upper)
        discrete_state.set_value(q0)
        print(f"[INFO] Initial joint position {q0}")

    def _DiscreteUpdate(self, context, discrete_state):
        q = context.get_discrete_state_vector().value()
        v = self.velocity_input_port.Eval(context)

        q_next = q + self._time_step * v
        q_next = np.clip(q_next, self._lower, self._upper)

        discrete_state.set_value(q_next)


class JointVelocityController(Diagram):
    """
    Clips desired joint velocities to joint velocity limits.
    """

    def __init__(
        self,
        max_velocity: float | list[float],
        num_joints: int = 7,
    ):
        super().__init__()

        max_velocity = np.asarray(max_velocity, dtype=float)
        if max_velocity.ndim == 0:
            max_velocity = np.full(num_joints, float(max_velocity))
        assert max_velocity.shape == (num_joints,)

        builder = DiagramBuilder()

        saturation = builder.AddSystem(
            Saturation(
                min_value=-max_velocity,
                max_value=max_velocity,
            )
        )

        builder.ExportInput(saturation.get_input_port(), "desired_velocity")

        builder.ExportOutput(saturation.get_output_port(), "commanded_velocity")

        builder.BuildInto(self)


class JointPositionController(LeafSystem):
    """
    The JointPositionController takes as input desired joint positions, and produces a
    joint velocity command as output to move the joint positions toward the desired state.

    This system is stateless, but is intended to be clocked at a known, fixed time step
    Δt by evaluating its output port at integer multiples of Δt.
    """

    def __init__(
        self,
        time_step: float,
        max_velocity: float | list[float],
        num_joints: int = 7,
    ):
        super().__init__()
        self._num_joints = num_joints
        self._time_step = time_step

        max_velocity = np.asarray(max_velocity, dtype=float)
        if max_velocity.ndim == 0:
            max_velocity = np.full(num_joints, float(max_velocity))
        assert max_velocity.shape == (num_joints,)
        self._max_velocity = max_velocity

        self.position_input_port = self.DeclareVectorInputPort(
            "position", num_joints
        )
        self.desired_position_input_port = self.DeclareVectorInputPort(
            "desired_position", num_joints
        )

        self.commanded_velocity_output_port = self.DeclareVectorOutputPort(
            "commanded_velocity", num_joints, self._CalcOutput
        )

    def _CalcOutput(self, context, output):
        position = self.position_input_port.Eval(context)
        desired_position = self.desired_position_input_port.Eval(context)

        if np.any(np.isnan(desired_position)):
            desired_position = position

        velocity = np.clip(
            (desired_position - position) / self._time_step,
            -self._max_velocity,
            self._max_velocity,
        )
        output.SetFromVector(velocity)


class IiwaOperatingMode(Enum):
    JOINT_VEL = "joint_vel"
    JOINT_POS = "joint_pos"


class IiwaRosInterface(Node):
    def __init__(self, namespace: str, operating_mode: IiwaOperatingMode):
        super().__init__(namespace)

        self._lock = threading.Lock()

        self._command_value = None
        self._new_command = False

        if operating_mode == IiwaOperatingMode.JOINT_VEL:
            self.create_subscription(
                JointState,
                f"/{namespace}/cmd_joint_vel",
                self._CmdJointVelCallback,
                1,
            )
        elif operating_mode == IiwaOperatingMode.JOINT_POS:
            self.create_subscription(
                JointState,
                f"/{namespace}/cmd_joint_pos",
                self._CmdJointPosCallback,
                1,
            )
        else:
            raise ValueError(f'Invalid control mode {operating_mode}')

        self._joint_state_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_states",
            10,
        )

    def _CmdJointPosCallback(self, msg: JointState):
        if len(msg.position) != 7:
            self.get_logger().warn(f"Expected 7 joint positions, got {np.array(msg.position)}")
            return

        with self._lock:
            self._command_value = np.array(msg.position)
            self._new_command = True
            self.get_logger().info(f"Received joint position command {self._command_value}")

    def _CmdJointVelCallback(self, msg: JointState):
        if len(msg.velocity) != 7:
            self.get_logger().warn(f"Expected 7 joint velocities, got {np.array(msg.velocity)}")
            return

        with self._lock:
            self._command_value = np.array(msg.velocity)
            self._new_command = True
            self.get_logger().info(f"Received joint velocity command {self._command_value}")

    def ConsumeCommand(self):
        with self._lock:
            command_value = (
                self._command_value.copy()
                if self._command_value is not None
                else None
            )
            new_command = self._new_command
            self._new_command = False

        return command_value, new_command

    def publish_joint_state(
        self,
        position,
        velocity,
        effort,
    ):
        msg = JointState()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"joint_{i}" for i in range(1, 8)]

        msg.position = list(position)
        msg.velocity = list(velocity)
        msg.effort = list(effort)

        self._joint_state_pub.publish(msg)


class RosJointPositionSource(LeafSystem):
    def __init__(self, ros_interface):
        super().__init__()

        self._ros_interface = ros_interface

        self.DeclareVectorOutputPort("desired_position", 7, self._CalcOutput)

    def _CalcOutput(self, context, output):
        qpos, new = self._ros_interface.ConsumeCommand()
        if qpos is None:
            qpos = np.full(7, np.nan)
        output.SetFromVector(qpos)


class RosJointVelocitySource(LeafSystem):
    def __init__(self, ros_interface):
        super().__init__()

        self._ros_interface = ros_interface

        self.DeclareVectorOutputPort("desired_velocity", 7, self._CalcOutput)

    def _CalcOutput(self, context, output):
        qvel, new = self._ros_interface.ConsumeCommand()
        if qvel is None:
            qvel = np.zeros(7)
        output.SetFromVector(qvel)


class Iiwa7System(Diagram):
    def __init__(
        self,
        lcm: DrakeLcmInterface,
        ros_interface: IiwaRosInterface,
        operating_mode: IiwaOperatingMode,
        max_joint_velocity: float | list[float],
        time_step: float,
        lcm_channel_suffix: str = "",
        joint_upper_limit = np.deg2rad(np.array([170, 120, 170, 120, 170, 120, 175])),
        joint_lower_limit = np.deg2rad(-np.array([170, 120, 170, 120, 170, 120, 175])),
    ):
        super().__init__()

        builder = DiagramBuilder()

        robot = builder.AddSystem(
            IiwaRobot(
                lcm=lcm,
                lcm_channel_suffix=lcm_channel_suffix,
                control_mode=IiwaControlMode.kPositionAndTorque,
            )
        )
        integrated_velocity = builder.AddSystem(
            IntegratedVelocityController(
                time_step=time_step,
                joint_lower_limit=joint_lower_limit,
                joint_upper_limit=joint_upper_limit,
            )
        )
        builder.Connect(
            integrated_velocity.GetOutputPort("position"),
            robot.GetInputPort("position"),
        )
        builder.Connect(
            robot.GetOutputPort("position_measured"),
            integrated_velocity.GetInputPort("initial_position"),
        )

        if operating_mode == IiwaOperatingMode.JOINT_VEL:
            controller = builder.AddSystem(
                JointVelocityController(max_velocity=max_joint_velocity)
            )
            command_source = builder.AddSystem(
                RosJointVelocitySource(ros_interface)
            )
            builder.Connect(
                command_source.get_output_port(),
                controller.GetInputPort("desired_velocity"),
            )

        elif operating_mode == IiwaOperatingMode.JOINT_POS:
            controller = builder.AddSystem(
                JointPositionController(
                    time_step=time_step,
                    max_velocity=max_joint_velocity,
                )
            )
            command_source = builder.AddSystem(
                RosJointPositionSource(ros_interface)
            )
            builder.Connect(
                integrated_velocity.GetOutputPort("position"),
                controller.GetInputPort("position"),
            )
            builder.Connect(
                command_source.get_output_port(),
                controller.GetInputPort("desired_position"),
            )

        else:
            raise ValueError(f'Invalid operating mode {operating_mode}')

        builder.Connect(
            controller.GetOutputPort("commanded_velocity"),
            integrated_velocity.GetInputPort("velocity"),
        )

        builder.BuildInto(self)


@dataclass
class Iiwa7SystemsConfig:
    operating_mode: IiwaOperatingMode
    enable_left_arm: bool
    enable_right_arm: bool
    max_joint_velocity: float | list[float] = 1.0
    time_step: float = 0.005


def AddIiwa7Systems(
    diagram_builder: DiagramBuilder,
    rclpy_executor: rclpy.executors.Executor,
    config: Iiwa7SystemsConfig,
) -> None:

    ros_namespaces = ["left_arm", "right_arm"]
    lcm_channel_suffixs = ["", "_2"]
    enabled = [config.enable_left_arm, config.enable_right_arm]

    if not any(enabled):
        return

    lcm = AddLcm(diagram_builder)

    for k in range(2):
        if not enabled[k]:
            continue

        ros_interface = IiwaRosInterface(
            namespace=ros_namespaces[k],
            operating_mode=config.operating_mode,
        )
        rclpy_executor.add_node(ros_interface)

        diagram_builder.AddSystem(
            Iiwa7System(
                lcm=lcm,
                ros_interface=ros_interface,
                operating_mode=config.operating_mode,
                max_joint_velocity=config.max_joint_velocity,
                time_step=config.time_step,
                lcm_channel_suffix=lcm_channel_suffixs[k],
            )
        )
