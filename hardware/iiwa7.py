import threading

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
        num_joints: int = 7,
        max_velocity=1.0,
    ):
        super().__init__()
        self._num_joints = num_joints
        self._time_step = time_step

        # Allow either a scalar (same limit for every joint) or a
        # per-joint array of velocity limits (rad/s).
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

        mask = np.isnan(desired_position)
        desired_position[mask] = position[mask]

        velocity = np.clip(
            (desired_position - position) / self._time_step,
            -self._max_velocity,
            self._max_velocity,
        )
        output.SetFromVector(velocity)


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
        joint_lower_limit=-np.inf,
        joint_upper_limit=np.inf,
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

        self.DeclareStateOutputPort("commanded_position", state_index)

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

    def _DiscreteUpdate(self, context, discrete_state):
        q = context.get_discrete_state_vector().value()
        v = self.velocity_input_port.Eval(context)

        q_next = q + self._time_step * v
        q_next = np.clip(q_next, self._lower, self._upper)

        discrete_state.set_value(q_next)


class IiwaRosInterface(Node):
    def __init__(self, namespace):
        super().__init__(namespace)

        self._lock = threading.Lock()

        self._desired_position = np.full(7, np.nan)
        self._new_command = False

        self.create_subscription(
            JointState,
            f"/{namespace}/cmd_joint_pos",
            self._command_callback,
            1,
        )

        self._joint_state_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_states",
            10,
        )

    def _command_callback(self, msg: JointState):
        if len(msg.position) != 7:
            self.get_logger().warn("Expected 7 joint positions.")
            return

        with self._lock:
            self._desired_position = np.array(msg.position)
            self._new_command = True

    def consume_command(self):
        with self._lock:
            q = self._desired_position.copy()
            new = self._new_command
            self._new_command = False

        return q, new

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


class RosCommandSource(LeafSystem):
    def __init__(self, ros_interface):
        super().__init__()

        self._ros = ros_interface
        self._command = np.zeros(7)

        self.DeclareVectorOutputPort(
            "desired_position",
            7,
            self._calc_output,
        )

    def _calc_output(self, context, output):
        q, new = self._ros.consume_command()

        self._command = q

        output.SetFromVector(q)


class Iiwa7System(Diagram):
    def __init__(
        self,
        lcm: DrakeLcmInterface,
        ros_interface: IiwaRosInterface,
        time_step: float,
        max_joint_velocity = 1.0,
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

        controller = builder.AddSystem(
            JointPositionController(
                time_step=time_step,
                max_velocity=max_joint_velocity,
            )
        )

        command_source = builder.AddSystem(
            RosCommandSource(ros_interface)
        )


        integrated_velocity = builder.AddSystem(
            IntegratedVelocityController(
                time_step=time_step,
                joint_lower_limit=joint_lower_limit,
                joint_upper_limit=joint_upper_limit,
            )
        )
        builder.Connect(
            controller.GetOutputPort("commanded_velocity"),
            integrated_velocity.GetInputPort("velocity"),
        )

        builder.Connect(
            robot.GetOutputPort("position_measured"),
            integrated_velocity.GetInputPort("initial_position"),
        )

        builder.Connect(
            integrated_velocity.GetOutputPort("commanded_position"),
            robot.GetInputPort("position"),
        )

        builder.Connect(
            integrated_velocity.GetOutputPort("commanded_position"),
            controller.GetInputPort("position"),
        )

        builder.Connect(
            command_source.get_output_port(),
            controller.GetInputPort("desired_position"),
        )

        builder.BuildInto(self)


def AddIiwa7Systems(
    diagram_builder: DiagramBuilder,
    rclpy_executor: rclpy.executors.Executor,
    ros_namespace: str = "left_arm",
    lcm_channel_suffix: str = "",
    time_step: float = 0.005,
) -> None:

    ros_interface = IiwaRosInterface(namespace=ros_namespace)

    rclpy_executor.add_node(ros_interface)

    lcm = AddLcm(diagram_builder)

    diagram_builder.AddSystem(
        Iiwa7System(
            lcm=lcm,
            ros_interface=ros_interface,
            time_step=time_step,
            lcm_channel_suffix=lcm_channel_suffix,
        )
    )
