import copy
import threading
from enum import Enum

import numpy as np

from drake import lcmt_iiwa_command, lcmt_iiwa_status
from pydrake.all import (AbstractValue, BusCreator, CollisionCheckerParams,
                         ConstantVectorSource, Diagram, DiagramBuilder,
                         DifferentialInverseKinematicsSystem, DofMask,
                         DrakeLcm, DrakeLcmInterface, FixedOffsetFrame, Frame,
                         Gain, IiwaCommandSender, IiwaControlMode,
                         IiwaStatusReceiver, InputPortIndex,
                         JacobianWrtVariable, JointLimits, LcmInterfaceSystem,
                         LcmPublisherSystem, LcmSubscriberSystem, LeafSystem,
                         ModelInstanceIndex, MultibodyPlant, OutputPort,
                         Parser, Quaternion, RigidTransform, RobotDiagramBuilder,
                         RotationMatrix, SceneGraphCollisionChecker,
                         SpatialForce, SpatialVelocity, ValueProducer,
                         position_enabled, torque_enabled)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped, WrenchStamped
from sensor_msgs.msg import JointState


class IiwaRobot(Diagram):
    """
    Drake Diagram providing an interface to a KUKA IIWA through the Drake LCM driver.
    """

    def __init__(
        self,
        lcm: DrakeLcmInterface,
        lcm_channel_suffix: str = "",
        control_mode: IiwaControlMode = IiwaControlMode.kPositionAndTorque,
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

        builder.BuildInto(self)


def AddIiwa7Model(
        plant: MultibodyPlant,
        end_effector_z_offset: float,
        joint_position_margin: np.ndarray | None = None,
        max_joint_velocity: np.ndarray | None = None,
    ) -> tuple[ModelInstanceIndex, Frame]:

    iiwa_instance = Parser(plant).AddModelsFromUrl(
        "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
    )[0]

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("iiwa_link_0", iiwa_instance),
        RigidTransform(),
    )

    ee_frame = plant.AddFrame(
        FixedOffsetFrame(
            name="iiwa_ee",
            P=plant.GetFrameByName("iiwa_link_7", iiwa_instance),
            X_PF=RigidTransform([0.0, 0.0, end_effector_z_offset]),
        )
    )

    if joint_position_margin is not None:
        assert joint_position_margin.shape == (7,)
        assert (joint_position_margin > 0).all()
    if max_joint_velocity is not None:
        assert max_joint_velocity.shape == (7,)
        assert (max_joint_velocity > 0).all()

    for i in range(7):
        joint = plant.GetJointByName(f"iiwa_joint_{i+1}", iiwa_instance)
        if joint_position_margin is not None:
            joint.set_position_limits(
                joint.position_lower_limits() + joint_position_margin[i],
                joint.position_upper_limits() - joint_position_margin[i],
            )
        if max_joint_velocity is not None:
            joint.set_velocity_limits([-max_joint_velocity[i]], [max_joint_velocity[i]])

    return iiwa_instance, ee_frame


class IiwaForwardKinematics(LeafSystem):
    """
    Converts joint positions into the end-effector pose via forward kinematics.
    """

    def __init__(self, end_effector_z_offset: float):
        super().__init__()

        plant = MultibodyPlant(0.0)
        _, ee_frame  = AddIiwa7Model(plant, end_effector_z_offset)
        plant.Finalize()

        self._plant = plant
        self._ee_frame = ee_frame

        self._position_input_port = self.DeclareVectorInputPort(
            "position", plant.num_positions()
        )

        self._velocity_input_port = self.DeclareVectorInputPort(
            "velocity", plant.num_velocities()
        )

        self._torque_input_port = self.DeclareVectorInputPort(
            "torque", plant.num_velocities()
        )

        self._plant_context_cache = self.DeclareCacheEntry(
            "plant_context",
            ValueProducer(
                lambda: AbstractValue.Make(self._plant.CreateDefaultContext()),
                self._CalcPlantContext,
            ),
            prerequisites_of_calc = {
                self._position_input_port.ticket(),
                self._velocity_input_port.ticket(),
            },
        )

        self.DeclareAbstractOutputPort(
            "cartesian_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._CalcCartesianPose,
        )

        self.DeclareAbstractOutputPort(
            "cartesian_velocity",
            lambda: AbstractValue.Make(SpatialVelocity()),
            self._CalcCartesianVelocity,
        )

        self.DeclareAbstractOutputPort(
            "cartesian_wrench",
            lambda: AbstractValue.Make(SpatialForce()),
            self._CalcCartesianWrench,
        )

    def _CalcPlantContext(self, context, value):
        plant_context = value.get_mutable_value()

        q = self._position_input_port.Eval(context)
        self._plant.SetPositions(plant_context, q)

        if self._velocity_input_port.HasValue(context):
            v = self._velocity_input_port.Eval(context)
            self._plant.SetVelocities(plant_context, v)

    def _CalcCartesianPose(self, context, output):
        plant_context = self._plant_context_cache.Eval(context)
        X_WT = self._ee_frame.CalcPoseInWorld(plant_context)
        output.set_value(X_WT)

    def _CalcCartesianVelocity(self, context, output):
        plant_context = self._plant_context_cache.Eval(context)
        V_WT = self._ee_frame.CalcSpatialVelocityInWorld(plant_context)
        output.set_value(V_WT)

    def _CalcCartesianWrench(self, context, output):
        plant_context = self._plant_context_cache.Eval(context)

        tau = self._torque_input_port.Eval(context)

        J = self._plant.CalcJacobianSpatialVelocity(
            context=plant_context,
            with_respect_to=JacobianWrtVariable.kV,
            frame_B=self._ee_frame,
            p_BoBp_B=np.zeros(3),
            frame_A=self._plant.world_frame(),
            frame_E=self._plant.world_frame(),
        )

        wrench = np.linalg.lstsq(J.T, tau)[0]
        F_WT = SpatialForce(tau=wrench[:3], f=wrench[3:])

        output.set_value(F_WT)


class IntegratedVelocitySwitch(LeafSystem):
    """
    Integrates one of several commanded joint velocities into a commanded joint
    position.

    The active velocity input is selected by an InputPortIndex received on an
    abstract input port.

    The initial joint position is loaded from the position_measured input port.
    """

    def __init__(
        self,
        num_velocity_inputs: int,
        max_velocity: np.ndarray,
        position_margin: np.ndarray,
        time_step: float,
    ):
        super().__init__()

        self._time_step = time_step

        plant = MultibodyPlant(0.0)
        AddIiwa7Model(
            plant=plant,
            end_effector_z_offset=0.0,
            joint_position_margin=position_margin,
            max_joint_velocity=max_velocity,
        )
        plant.Finalize()
        num_joints = plant.num_velocities()

        self._joint_limits = JointLimits(plant)

        self.active_input_port = self.DeclareAbstractInputPort(
            "active_input",
            AbstractValue.Make(InputPortIndex(0)),
        )

        self.velocity_input_ports = [
            self.DeclareVectorInputPort(
                f"velocity_{i}",
                num_joints,
            )
            for i in range(num_velocity_inputs)
        ]

        self.position_reset_input_port = self.DeclareVectorInputPort(
            "position_reset",
            num_joints,
        )

        self._position_state_index = self.DeclareDiscreteState(
            np.full(num_joints, np.nan)
        )

        self.DeclareStateOutputPort(
            "position",
            self._position_state_index,
        )

        self.DeclarePeriodicUnrestrictedUpdateEvent(
            period_sec=time_step,
            offset_sec=0.0,
            update=self._Update,
        )

    def _Update(self, context, state):
        q = context.get_discrete_state(self._position_state_index).value()

        if np.isnan(q).any():
            q = self.position_reset_input_port.Eval(context)

        active = self.active_input_port.Eval(context)

        v = self.get_input_port(active).Eval(context)
        v = np.clip(
            v,
            self._joint_limits.velocity_lower(),
            self._joint_limits.velocity_upper(),
        )

        if np.isnan(v).any():
            v = np.zeros_like(q)

        q_next = q + self._time_step * v
        q_next = np.clip(
            q_next,
            self._joint_limits.position_lower(),
            self._joint_limits.position_upper(),
        )

        state.get_mutable_discrete_state(self._position_state_index).set_value(q_next)


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
        k_vx: float = 0.5,
        num_joints: int = 7,
    ):
        super().__init__()
        self._time_step = time_step
        self._k_vx = k_vx

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

        velocity = self._k_vx * (desired_position - position) / self._time_step
        output.SetFromVector(velocity)


def IiwaDifferentialInverseKinematics(
        end_effector_z_offset: float,
        time_step: float,
        max_linear_velocity: np.ndarray,
        max_angular_velocity: np.ndarray,
        max_joint_velocity: np.ndarray | None = None,
        joint_position_margin: np.ndarray | None = None,
        K_VX: float = 0.5,
    ) -> tuple[DifferentialInverseKinematicsSystem, str]:

    max_cartesian_velocity = np.concatenate([max_angular_velocity, max_linear_velocity])
    assert max_cartesian_velocity.shape == (6,)
    assert (max_cartesian_velocity > 0).all()

    robot_builder = RobotDiagramBuilder()

    iiwa_instance, ee_frame = AddIiwa7Model(
        plant=robot_builder.plant(),
        end_effector_z_offset=end_effector_z_offset,
        joint_position_margin=joint_position_margin,
        max_joint_velocity=max_joint_velocity,
    )

    robot_diagram = robot_builder.Build()
    plant = robot_diagram.plant()

    collision_checker = SceneGraphCollisionChecker(
        CollisionCheckerParams(
            model=robot_diagram,
            robot_model_instances=[iiwa_instance],
            edge_step_size=1.0,
        )
    )

    ee_frame = ee_frame.scoped_name().get_full()
    cartesian_axis_masks = { ee_frame: np.ones(6) }

    assert plant.num_positions() == 7
    active_dof = DofMask(plant.num_positions(), True)

    recipe = DifferentialInverseKinematicsSystem.Recipe()
    recipe.AddIngredient(
        DifferentialInverseKinematicsSystem.LeastSquaresCost(
            DifferentialInverseKinematicsSystem.LeastSquaresCost.Config(
                cartesian_qp_weight=1.0,
                cartesian_axis_masks=cartesian_axis_masks,
            )
        )
    )
    recipe.AddIngredient(
        DifferentialInverseKinematicsSystem.JointCenteringCost(
            DifferentialInverseKinematicsSystem.JointCenteringCost.Config(
                posture_gain=1.0,
                cartesian_axis_masks=cartesian_axis_masks,
            )
        )
    )
    recipe.AddIngredient(
        DifferentialInverseKinematicsSystem.CartesianVelocityLimitConstraint(
            DifferentialInverseKinematicsSystem.CartesianVelocityLimitConstraint.Config(
                V_next_TG_limit=max_cartesian_velocity
            )
        )
    )
    recipe.AddIngredient(
        DifferentialInverseKinematicsSystem.JointVelocityLimitConstraint(
            DifferentialInverseKinematicsSystem.JointVelocityLimitConstraint.Config(),
            JointLimits(plant, active_dof)
        )
    )

    diff_ik_system = DifferentialInverseKinematicsSystem(
        recipe=recipe,
        task_frame=plant.world_frame().scoped_name().get_full(),
        collision_checker=collision_checker,
        active_dof=active_dof,
        time_step=time_step,
        K_VX=K_VX,
        Vd_TG_limit=SpatialVelocity(max_cartesian_velocity),
    )

    return diff_ik_system, ee_frame


class PassThroughSelector(LeafSystem):
    """
    Outputs zero velocity if the input of type RigidTransform or SpatialVelocity has
    nan. Otherwise input velocity passes throgh to the output velocity.
    """

    def __init__(
            self,
            type: type[SpatialVelocity] | type[RigidTransform],
            num_joints: int = 7,
        ):
        super().__init__()

        self._type = type

        self._selector_input_port = self.DeclareAbstractInputPort(
            "selector",
            AbstractValue.Make(type()),
        )

        self._velocity_input_port = self.DeclareVectorInputPort(
            "velocity", num_joints
        )

        self.DeclareVectorOutputPort(
            "velocity", num_joints, self._CalcOutput
        )

    def _CalcOutput(self, context, output):
        if self._type is SpatialVelocity:
            vel = self._selector_input_port.Eval(context)
            vel = np.concatenate([vel.rotational(), vel.translational()])
            output_zero = np.isnan(vel).any() or (vel == 0).all()
        elif self._type is RigidTransform:
            pose = self._selector_input_port.Eval(context)
            output_zero = (
                np.isnan(pose.translation()).any()
                or np.isnan(pose.rotation().matrix()).any()
            )

        if output_zero:
            output.set_value(np.zeros(self._velocity_input_port.size()))
        else:
            output.set_value(self._velocity_input_port.Eval(context))


class CartesianVelocityController(Diagram):
    """
    The CartesianVelocityController converts desired end-effector velocity into joint
    velocity commands using Differential Inverse Kinematics.
    """

    def __init__(
        self,
        end_effector_z_offset: float,
        time_step: float,
        max_linear_velocity: np.ndarray,
        max_angular_velocity: np.ndarray,
        max_joint_velocity: np.ndarray | None = None,
        joint_position_margin: np.ndarray | None = None,
    ):
        super().__init__()

        builder = DiagramBuilder()

        diff_ik, ee_frame = IiwaDifferentialInverseKinematics(
            end_effector_z_offset=end_effector_z_offset,
            time_step=time_step,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_joint_velocity=max_joint_velocity,
            joint_position_margin=joint_position_margin,
        )
        diff_ik = builder.AddSystem(diff_ik)

        vel_bus = BusCreator()
        vel_bus.DeclareAbstractInputPort(
            ee_frame,
            AbstractValue.Make(SpatialVelocity())
        )
        vel_bus = builder.AddSystem(vel_bus)

        zero_position = builder.AddSystem(ConstantVectorSource(np.zeros(7)))

        builder.Connect(
            vel_bus.get_output_port(),
            diff_ik.GetInputPort("desired_cartesian_velocities"),
        )

        builder.Connect(
            zero_position.get_output_port(),
            diff_ik.GetInputPort("nominal_posture"),
        )

        builder.ExportInput(
            diff_ik.GetInputPort("position"),
            "position",
        )

        builder.ExportInput(
            vel_bus.GetInputPort(ee_frame),
            "desired_cartesian_velocity",
        )

        selector = builder.AddSystem(
            PassThroughSelector(type=SpatialVelocity)
        )
        builder.ConnectToSame(
            vel_bus.GetInputPort(ee_frame),
            selector.GetInputPort("selector"),
        )
        builder.Connect(
            diff_ik.GetOutputPort("commanded_velocity"),
            selector.GetInputPort("velocity"),
        )
        builder.ExportOutput(
            selector.get_output_port(),
            "commanded_velocity",
        )

        builder.BuildInto(self)


class CartesianPoseController(Diagram):
    """
    The CartesianPoseController converts desired end-effector poses into joint
    velocity commands using Differential Inverse Kinematics. If the desired pose
    is invalid (contains NaN values), the controller outputs zero joint
    velocities.
    """

    def __init__(
        self,
        end_effector_z_offset: float,
        time_step: float,
        max_linear_velocity: np.ndarray,
        max_angular_velocity: np.ndarray,
        max_joint_velocity: np.ndarray | None = None,
        joint_position_margin: np.ndarray | None = None,
    ):
        super().__init__()

        builder = DiagramBuilder()

        diff_ik, ee_frame = IiwaDifferentialInverseKinematics(
            end_effector_z_offset=end_effector_z_offset,
            time_step=time_step,
            max_linear_velocity=max_linear_velocity,
            max_angular_velocity=max_angular_velocity,
            max_joint_velocity=max_joint_velocity,
            joint_position_margin=joint_position_margin,
        )
        diff_ik = builder.AddSystem(diff_ik)

        pose_bus = BusCreator()
        pose_bus.DeclareAbstractInputPort(
            ee_frame,
            AbstractValue.Make(RigidTransform())
        )
        pose_bus = builder.AddSystem(pose_bus)

        zero_position = builder.AddSystem(ConstantVectorSource(np.zeros(7)))

        builder.Connect(
            pose_bus.get_output_port(),
            diff_ik.GetInputPort("desired_cartesian_poses"),
        )

        builder.Connect(
            zero_position.get_output_port(),
            diff_ik.GetInputPort("nominal_posture"),
        )

        builder.ExportInput(
            diff_ik.GetInputPort("position"),
            "position",
        )

        builder.ExportInput(
            pose_bus.GetInputPort(ee_frame),
            "desired_cartesian_pose",
        )

        selector = builder.AddSystem(
            PassThroughSelector(type=RigidTransform)
        )
        builder.ConnectToSame(
            pose_bus.GetInputPort(ee_frame),
            selector.GetInputPort("selector"),
        )
        builder.Connect(
            diff_ik.GetOutputPort("commanded_velocity"),
            selector.GetInputPort("velocity"),
        )
        builder.ExportOutput(
            selector.get_output_port(),
            "commanded_velocity",
        )

        builder.BuildInto(self)


class CommandMode(Enum):
    JOINT_VEL = "joint_vel"
    JOINT_POS = "joint_pos"
    CARTESIAN_VEL = "cartesian_vel"
    CARTESIAN_POS = "cartesian_pos"


class IiwaRosInterface(Node):
    def __init__(self, namespace: str):
        super().__init__(namespace)

        self._lock = threading.Lock()

        self._command_mode = None
        self._command_value = None
        self._new_command = False

        self.create_subscription(
            JointState,
            f"/{namespace}/cmd_joint_vel",
            self._CmdJointVelCallback,
            1,
        )
        self.create_subscription(
            JointState,
            f"/{namespace}/cmd_joint_pos",
            self._CmdJointPosCallback,
            1,
        )
        self.create_subscription(
            TwistStamped,
            f"/{namespace}/cmd_cartesian_vel",
            self._CmdCartesianVelCallback,
            1,
        )
        self.create_subscription(
            PoseStamped,
            f"/{namespace}/cmd_cartesian_pos",
            self._CmdCartesianPosCallback,
            1,
        )

        self._joint_position_commanded_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_position_commanded",
            10,
        )
        self._joint_position_measured_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_position_measured",
            10,
        )
        self._joint_velocity_estimated_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_velocity_estimated",
            10,
        )
        self._joint_torque_commanded_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_torque_commanded",
            10,
        )
        self._joint_torque_measured_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_torque_measured",
            10,
        )
        self._joint_torque_external_pub = self.create_publisher(
            JointState,
            f"/{namespace}/joint_torque_external",
            10,
        )
        self._cartesian_pose_commanded_pub = self.create_publisher(
            PoseStamped,
            f"/{namespace}/cartesian_pose_commanded",
            10,
        )
        self._cartesian_pose_measured_pub = self.create_publisher(
            PoseStamped,
            f"/{namespace}/cartesian_pose_measured",
            10,
        )
        self._cartesian_velocity_estimated_pub = self.create_publisher(
            TwistStamped,
            f"/{namespace}/cartesian_velocity_estimated",
            10,
        )
        self._cartesian_wrench_estimated_pub = self.create_publisher(
            WrenchStamped,
            f"/{namespace}/cartesian_wrench_estimated",
            10,
        )

        self.get_logger().info(f"Using namespace /{namespace} for ROS communication")

    def _CmdJointVelCallback(self, msg: JointState):
        if len(msg.velocity) != 7:
            self.get_logger().warn(f"Expected 7 joint velocities, got {np.array(msg.velocity)}")
            return

        with self._lock:
            self._command_mode = CommandMode.JOINT_VEL
            self._command_value = np.array(msg.velocity)
            self._new_command = True
            self.get_logger().info(f"Received joint velocity command {self._command_value}")

    def _CmdJointPosCallback(self, msg: JointState):
        if len(msg.position) != 7:
            self.get_logger().warn(f"Expected 7 joint positions, got {np.array(msg.position)}")
            return

        with self._lock:
            self._command_mode = CommandMode.JOINT_POS
            self._command_value = np.array(msg.position)
            self._new_command = True
            self.get_logger().info(f"Received joint position command {self._command_value}")

    def _CmdCartesianVelCallback(self, msg: TwistStamped):
        twist = msg.twist
        linear = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        angular = np.array([twist.angular.x, twist.angular.y, twist.angular.z])

        V_WT =  SpatialVelocity(angular, linear)

        with self._lock:
            self._command_mode = CommandMode.CARTESIAN_VEL
            self._command_value = V_WT
            self._new_command = True
            self.get_logger().info(f"Received cartesian velocity command {self._command_value}")

    def _CmdCartesianPosCallback(self, msg: PoseStamped):
        p = msg.pose
        translation = np.array([p.position.x, p.position.y, p.position.z])
        quaternion = Quaternion(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z)

        X_WT = RigidTransform(RotationMatrix(quaternion), translation)

        with self._lock:
            self._command_mode = CommandMode.CARTESIAN_POS
            self._command_value = X_WT
            self._new_command = True
            self.get_logger().info(f"Received cartesian pose command {self._command_value}")

    def ConsumeCommand(self):
        with self._lock:
            command_mode = self._command_mode
            command_value = copy.copy(self._command_value)
            new_command = self._new_command
            self._new_command = False

        return command_mode, command_value, new_command

    def PublishJointStatus(
        self,
        position_commanded: np.ndarray,
        position_measured: np.ndarray,
        velocity_estimated: np.ndarray,
        torque_commanded: np.ndarray,
        torque_measured: np.ndarray,
        torque_external: np.ndarray,
    ):
        stamp = self.get_clock().now().to_msg()
        names = [f"joint_{i}" for i in range(1, 8)]

        def MakeMsg(position=None, velocity=None, effort=None):
            msg = JointState()
            msg.header.stamp = stamp
            msg.name = names
            if position is not None:
                msg.position = list(position)
            if velocity is not None:
                msg.velocity = list(velocity)
            if effort is not None:
                msg.effort = list(effort)
            return msg

        self._joint_position_commanded_pub.publish(MakeMsg(position=position_commanded))
        self._joint_position_measured_pub.publish(MakeMsg(position=position_measured))
        self._joint_velocity_estimated_pub.publish(MakeMsg(velocity=velocity_estimated))
        self._joint_torque_commanded_pub.publish(MakeMsg(effort=torque_commanded))
        self._joint_torque_measured_pub.publish(MakeMsg(effort=torque_measured))
        self._joint_torque_external_pub.publish(MakeMsg(effort=torque_external))

    def PublishCartesianStatus(
        self,
        pose_commanded: RigidTransform,
        pose_measured: RigidTransform,
        velocity_estimated: SpatialVelocity,
        wrench_estimated: SpatialForce,
    ):
        stamp = self.get_clock().now().to_msg()

        def MakePoseMsg(X_WT: RigidTransform) -> PoseStamped:
            msg = PoseStamped()
            msg.header.stamp = stamp

            translation = X_WT.translation()
            quaternion = X_WT.rotation().ToQuaternion()

            msg.pose.position.x = translation[0]
            msg.pose.position.y = translation[1]
            msg.pose.position.z = translation[2]

            msg.pose.orientation.w = quaternion.w()
            msg.pose.orientation.x = quaternion.x()
            msg.pose.orientation.y = quaternion.y()
            msg.pose.orientation.z = quaternion.z()

            return msg

        def MakeTwistMsg(V_WT: SpatialVelocity) -> TwistStamped:
            msg = TwistStamped()
            msg.header.stamp = stamp

            angular = V_WT.rotational()
            linear = V_WT.translational()

            msg.twist.linear.x = linear[0]
            msg.twist.linear.y = linear[1]
            msg.twist.linear.z = linear[2]

            msg.twist.angular.x = angular[0]
            msg.twist.angular.y = angular[1]
            msg.twist.angular.z = angular[2]

            return msg

        def MakeWrenchMsg(F_WT: SpatialForce) -> TwistStamped:
            msg = WrenchStamped()
            msg.header.stamp = stamp

            torque = F_WT.rotational()
            force = F_WT.translational()

            msg.wrench.force.x = force[0]
            msg.wrench.force.y = force[1]
            msg.wrench.force.z = force[2]

            msg.wrench.torque.x = torque[0]
            msg.wrench.torque.y = torque[1]
            msg.wrench.torque.z = torque[2]

            return msg

        self._cartesian_pose_commanded_pub.publish(MakePoseMsg(pose_commanded))
        self._cartesian_pose_measured_pub.publish(MakePoseMsg(pose_measured))
        self._cartesian_velocity_estimated_pub.publish(MakeTwistMsg(velocity_estimated))
        self._cartesian_wrench_estimated_pub.publish(MakeWrenchMsg(wrench_estimated))


class RosCommandSource(LeafSystem):
    """
    Bridges ROS command messages received by an IiwaRosInterface into Drake
    output ports.

    Whenever a new command arrives on the IiwaRosInterface (joint velocity,
    joint position, cartesian velocity, or cartesian pose), it is latched
    into internal state and exposed on the corresponding output port. The
    "active_port" output identifies which command mode is currently active.
    Inactive joint output ports default to zero (velocity) or NaN
    (position), and inactive cartesian output ports default to zero
    (velocity) or NaN translation (pose), so downstream consumers can detect
    an unset command.
    """

    def __init__(self, ros_interface: IiwaRosInterface):
        super().__init__()

        self._ros_interface = ros_interface

        self._mode_state = self.DeclareAbstractState(
            AbstractValue.Make(None)
        )
        self._value_state = self.DeclareAbstractState(
            AbstractValue.Make(None)
        )

        self.DeclarePerStepUnrestrictedUpdateEvent(self._Update)

        self.DeclareAbstractOutputPort(
            "active_port",
            lambda: AbstractValue.Make(InputPortIndex(0)),
            self._CalcActivePort,
        )

        self.DeclareVectorOutputPort(
            "desired_joint_velocity",
            7,
            self._CalcJointVelocity,
        )

        self.DeclareVectorOutputPort(
            "desired_joint_position",
            7,
            self._CalcJointPosition,
        )

        self.DeclareAbstractOutputPort(
            "desired_cartesian_velocity",
            lambda: AbstractValue.Make(SpatialVelocity.Zero()),
            self._CalcCartesianVelocity,
        )

        self.DeclareAbstractOutputPort(
            "desired_cartesian_pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._CalcCartesianPose,
        )

    def _Update(self, context, state):
        mode, value, new = self._ros_interface.ConsumeCommand()
        if not new:
            return

        state.get_mutable_abstract_state(int(self._mode_state)).set_value(mode)
        state.get_mutable_abstract_state(int(self._value_state)).set_value(value)

    def _GetCommand(self, context):
        mode = context.get_abstract_state(int(self._mode_state)).get_value()
        value = context.get_abstract_state(int(self._value_state)).get_value()
        return mode, value

    def _CalcActivePort(self, context, output):
        mode, _ = self._GetCommand(context)

        mapping = {
            None: 1,
            CommandMode.JOINT_VEL: 1,
            CommandMode.JOINT_POS: 2,
            CommandMode.CARTESIAN_VEL: 3,
            CommandMode.CARTESIAN_POS: 4,
        }

        output.set_value(InputPortIndex(mapping[mode]))

    def _CalcJointVelocity(self, context, output):
        mode, value = self._GetCommand(context)

        if mode == CommandMode.JOINT_VEL and value is not None:
            output.SetFromVector(value)
        else:
            output.SetFromVector(np.zeros(7))

    def _CalcJointPosition(self, context, output):
        mode, value = self._GetCommand(context)

        if mode == CommandMode.JOINT_POS and value is not None:
            output.SetFromVector(value)
        else:
            output.SetFromVector(np.full(7, np.nan))

    def _CalcCartesianVelocity(self, context, output):
        mode, value = self._GetCommand(context)

        if mode == CommandMode.CARTESIAN_VEL and value is not None:
            output.set_value(value)
        else:
            output.set_value(SpatialVelocity.Zero())

    def _CalcCartesianPose(self, context, output):
        mode, value = self._GetCommand(context)

        if mode == CommandMode.CARTESIAN_POS and value is not None:
            output.set_value(value)
        else:
            output.set_value(RigidTransform(np.full(3, np.nan)))


class RosReportSink(LeafSystem):
    """
    Publishes IIWA status signals to ROS through the given IiwaRosInterface.
    """

    def __init__(
        self,
        ros_interface: IiwaRosInterface,
        publish_period: float,
        num_joints: int = 7,
    ):
        super().__init__()

        self._ros_interface = ros_interface

        self.position_commanded_input_port = self.DeclareVectorInputPort(
            "position_commanded", num_joints
        )
        self.position_measured_input_port = self.DeclareVectorInputPort(
            "position_measured", num_joints
        )
        self.velocity_estimated_input_port = self.DeclareVectorInputPort(
            "velocity_estimated", num_joints
        )
        self.torque_commanded_input_port = self.DeclareVectorInputPort(
            "torque_commanded", num_joints
        )
        self.torque_measured_input_port = self.DeclareVectorInputPort(
            "torque_measured", num_joints
        )
        self.torque_external_input_port = self.DeclareVectorInputPort(
            "torque_external", num_joints
        )
        self.cartesian_pose_commanded_input_port = self.DeclareAbstractInputPort(
            "cartesian_pose_commanded", AbstractValue.Make(RigidTransform())
        )
        self.cartesian_pose_measured_input_port = self.DeclareAbstractInputPort(
            "cartesian_pose_measured", AbstractValue.Make(RigidTransform())
        )
        self.cartesian_velocity_estimated_input_port = self.DeclareAbstractInputPort(
            "cartesian_velocity_estimated", AbstractValue.Make(SpatialVelocity())
        )
        self.cartesian_wrench_estimated_input_port = self.DeclareAbstractInputPort(
            "cartesian_wrench_estimated", AbstractValue.Make(SpatialForce())
        )

        self.DeclarePeriodicPublishEvent(
            period_sec=publish_period,
            offset_sec=0.0,
            publish=self._Publish,
        )

    def _Publish(self, context):
        self._ros_interface.PublishJointStatus(
            position_commanded=self.position_commanded_input_port.Eval(context),
            position_measured=self.position_measured_input_port.Eval(context),
            velocity_estimated=self.velocity_estimated_input_port.Eval(context),
            torque_commanded=self.torque_commanded_input_port.Eval(context),
            torque_measured=self.torque_measured_input_port.Eval(context),
            torque_external=self.torque_external_input_port.Eval(context),
        )
        self._ros_interface.PublishCartesianStatus(
            pose_commanded=self.cartesian_pose_commanded_input_port.Eval(context),
            pose_measured=self.cartesian_pose_measured_input_port.Eval(context),
            velocity_estimated=self.cartesian_velocity_estimated_input_port.Eval(context),
            wrench_estimated=self.cartesian_wrench_estimated_input_port.Eval(context),
        )


class IiwaSystem(Diagram):
    def __init__(
        self,
        lcm: DrakeLcmInterface,
        lcm_channel_suffix: str,
        ros_interface: IiwaRosInterface,
        max_joint_velocity: np.ndarray,
        max_linear_velocity: np.ndarray,
        max_angular_velocity: np.ndarray,
        joint_position_margin: np.ndarray,
        time_step: float,
        end_effector_z_offset: float,
    ):
        super().__init__()

        builder = DiagramBuilder()

        robot = builder.AddSystem(
            IiwaRobot(
                lcm=lcm,
                lcm_channel_suffix=lcm_channel_suffix,
                control_mode=IiwaControlMode.kPositionOnly,
            )
        )
        integrated_velocity = builder.AddSystem(
            IntegratedVelocitySwitch(
                num_velocity_inputs=4,
                max_velocity=max_joint_velocity,
                position_margin=joint_position_margin,
                time_step=time_step,
            )
        )
        builder.Connect(
            integrated_velocity.GetOutputPort("position"),
            robot.GetInputPort("position"),
        )
        builder.Connect(
            robot.GetOutputPort("position_commanded"),
            integrated_velocity.GetInputPort("position_reset"),
        )

        command_source = builder.AddSystem(
            RosCommandSource(ros_interface)
        )
        builder.Connect(
            command_source.GetOutputPort("active_port"),
            integrated_velocity.GetInputPort("active_input"),
        )

        # Joint velocity control
        builder.Connect(
            command_source.GetOutputPort("desired_joint_velocity"),
            integrated_velocity.get_input_port(1),
        )

        # Joint position controller
        joint_pos_controller = builder.AddSystem(
            JointPositionController(time_step=time_step)
        )
        builder.Connect(
            integrated_velocity.GetOutputPort("position"),
            joint_pos_controller.GetInputPort("position"),
        )
        builder.Connect(
            command_source.GetOutputPort("desired_joint_position"),
            joint_pos_controller.GetInputPort("desired_position"),
        )
        builder.Connect(
            joint_pos_controller.GetOutputPort("commanded_velocity"),
            integrated_velocity.get_input_port(2),
        )

        # Cartesian velocity controller
        cartesian_vel_controller = builder.AddSystem(
            CartesianVelocityController(
                end_effector_z_offset=end_effector_z_offset,
                time_step=time_step,
                max_linear_velocity=max_linear_velocity,
                max_angular_velocity=max_angular_velocity,
                max_joint_velocity=max_joint_velocity,
                joint_position_margin=joint_position_margin,
            )
        )
        builder.Connect(
            integrated_velocity.GetOutputPort("position"),
            cartesian_vel_controller.GetInputPort("position"),
        )
        builder.Connect(
            command_source.GetOutputPort("desired_cartesian_velocity"),
            cartesian_vel_controller.GetInputPort("desired_cartesian_velocity"),
        )
        builder.Connect(
            cartesian_vel_controller.GetOutputPort("commanded_velocity"),
            integrated_velocity.get_input_port(3),
        )

        # Cartesian pose controller
        cartesian_pos_controller = builder.AddSystem(
            CartesianPoseController(
                end_effector_z_offset=end_effector_z_offset,
                time_step=time_step,
                max_linear_velocity=max_linear_velocity,
                max_angular_velocity=max_angular_velocity,
                max_joint_velocity=max_joint_velocity,
                joint_position_margin=joint_position_margin,
            )
        )
        builder.Connect(
            integrated_velocity.GetOutputPort("position"),
            cartesian_pos_controller.GetInputPort("position"),
        )
        builder.Connect(
            command_source.GetOutputPort("desired_cartesian_pose"),
            cartesian_pos_controller.GetInputPort("desired_cartesian_pose"),
        )
        builder.Connect(
            cartesian_pos_controller.GetOutputPort("commanded_velocity"),
            integrated_velocity.get_input_port(4),
        )

        # Report joint state
        report_sink = builder.AddSystem(
            RosReportSink(
                ros_interface=ros_interface,
                publish_period=time_step,
            )
        )
        for port_name in (
            "position_commanded",
            "position_measured",
            "velocity_estimated",
            "torque_commanded",
            "torque_measured",
            "torque_external",
        ):
            builder.Connect(
                robot.GetOutputPort(port_name),
                report_sink.GetInputPort(port_name),
            )

        # Report cartesian state
        forward_kinamatics = builder.AddSystem(
            IiwaForwardKinematics(end_effector_z_offset=end_effector_z_offset)
        )
        builder.Connect(
            robot.GetOutputPort("position_measured"),
            forward_kinamatics.GetInputPort("position"),
        )
        builder.Connect(
            robot.GetOutputPort("velocity_estimated"),
            forward_kinamatics.GetInputPort("velocity"),
        )
        builder.Connect(
            robot.GetOutputPort("torque_external"),
            forward_kinamatics.GetInputPort("torque"),
        )
        builder.Connect(
            forward_kinamatics.GetOutputPort("cartesian_pose"),
            report_sink.GetInputPort("cartesian_pose_measured"),
        )
        builder.Connect(
            forward_kinamatics.GetOutputPort("cartesian_velocity"),
            report_sink.GetInputPort("cartesian_velocity_estimated"),
        )
        builder.Connect(
            forward_kinamatics.GetOutputPort("cartesian_wrench"),
            report_sink.GetInputPort("cartesian_wrench_estimated"),
        )

        forward_kinamatics_2 = builder.AddSystem(
            IiwaForwardKinematics(end_effector_z_offset=end_effector_z_offset)
        )
        builder.Connect(
            robot.GetOutputPort("position_commanded"),
            forward_kinamatics_2.GetInputPort("position"),
        )
        builder.Connect(
            forward_kinamatics_2.GetOutputPort("cartesian_pose"),
            report_sink.GetInputPort("cartesian_pose_commanded"),
        )

        builder.BuildInto(self)


def CheckLcmActive(lcm_channel_suffix: str):
    lcm = DrakeLcm()
    lcm.Subscribe("IIWA_STATUS" + lcm_channel_suffix, lambda s: s)
    count = lcm.HandleSubscriptions(100)
    return count > 0


def AddLcm(builder: DiagramBuilder):
    lcm = DrakeLcm()
    builder.AddSystem(LcmInterfaceSystem(lcm))
    return lcm


def AddIiwaSystems(
    diagram_builder: DiagramBuilder,
    rclpy_executor: rclpy.executors.Executor,
    tool_z_offset: float = 0.0,
    max_joint_velocity: float | list[float] = 0.5,
    max_linear_velocity: float | list[float] = 0.1,
    max_angular_velocity: float | list[float] = 0.5,
) -> None:

    ros_namespaces = ["left_iiwa", "right_iiwa"]
    lcm_channel_suffixs = ["", "_2"]
    end_effector_z_offsets = [
        tool_z_offset + 0.045,  # old flange face is 45mm from link7 frame
        tool_z_offset + 0.071,  # new flange face is 71mm from link7 frame
    ]
    enabled = [CheckLcmActive(s) for s in lcm_channel_suffixs]

    if not any(enabled):
        print("No IIWA robot found.")
        return

    if np.isscalar(max_joint_velocity):
        max_joint_velocity = np.full(7, max_joint_velocity)
    max_joint_velocity = np.array(max_joint_velocity)
    assert max_joint_velocity.shape == (7,), "max_joint_velocity must have 7 values"
    assert (max_joint_velocity > 0).all(), "max_joint_velocity must be positive"

    if np.isscalar(max_linear_velocity):
        max_linear_velocity = np.full(3, max_linear_velocity / np.sqrt(3))
    max_linear_velocity = np.array(max_linear_velocity)
    assert max_linear_velocity.shape == (3,), "max_linear_velocity must have 3 values"
    assert (max_linear_velocity > 0).all(), "max_linear_velocity must be positive"

    if np.isscalar(max_angular_velocity):
        max_angular_velocity = np.full(3, max_angular_velocity / np.sqrt(3))
    max_angular_velocity = np.array(max_angular_velocity)
    assert max_angular_velocity.shape == (3,), "max_angular_velocity must have 3 values"
    assert (max_angular_velocity > 0).all(), "max_angular_velocity must be positive"

    joint_position_margin = np.full(7, np.deg2rad(10))

    lcm = AddLcm(diagram_builder)

    for k in range(len(lcm_channel_suffixs)):
        if not enabled[k]:
            continue

        ros_interface = IiwaRosInterface(namespace=ros_namespaces[k])
        rclpy_executor.add_node(ros_interface)

        diagram_builder.AddSystem(
            IiwaSystem(
                lcm=lcm,
                lcm_channel_suffix=lcm_channel_suffixs[k],
                ros_interface=ros_interface,
                end_effector_z_offset=end_effector_z_offsets[k],
                max_joint_velocity=max_joint_velocity,
                max_linear_velocity=max_linear_velocity,
                max_angular_velocity=max_angular_velocity,
                joint_position_margin=joint_position_margin,
                time_step=0.005,
            )
        )
