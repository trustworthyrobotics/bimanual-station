import numpy as np

import threading
import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

from pydrake.all import (
    DiagramBuilder,
    LeafSystem,
    Simulator,
)

from hardware.iiwa7 import Iiwa7System


class LeftArmRosInterface(Node):
    def __init__(self):
        super().__init__("left_arm_interface")

        self._lock = threading.Lock()

        self._desired_position = np.full(7, np.nan)
        self._new_command = False

        self.create_subscription(
            JointState,
            "/left_arm/command_joint_position",
            self._command_callback,
            1,
        )

        self._joint_state_pub = self.create_publisher(
            JointState,
            "/left_arm/joint_states",
            10,
        )

    def _command_callback(self, msg: JointState):
        if len(msg.position) != 7:
            self.get_logger().warn(
                "Expected 7 joint positions."
            )
            return

        with self._lock:
            self._desired_position[:] = msg.position
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
        msg.name = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
            "joint_7",
        ]

        msg.position = list(position)
        msg.velocity = list(velocity)
        msg.effort = list(effort)

        self._joint_state_pub.publish(msg)


class RosCommandSource(LeafSystem):
    def __init__(self, ros_interface):
        super().__init__()

        self._ros = ros_interface
        self._command = np.zeros(7)
        self._new_command = False

        self.DeclareVectorOutputPort(
            "desired_position",
            7,
            self._calc_output,
        )

    def _calc_output(self, context, output):
        rclpy.spin_once(self._ros, timeout_sec=0.0)

        q, new = self._ros.consume_command()

        self._command = q
        self._new_command = new

        output.SetFromVector(q)

    @property
    def new_command(self):
        return self._new_command


def main():
    builder = DiagramBuilder()

    outer = builder.AddSystem(Iiwa7System())

    rclpy.init()

    ros = LeftArmRosInterface()

    desired_source = builder.AddSystem(
        RosCommandSource(ros)
    )

    builder.Connect(
        desired_source.get_output_port(),
        outer.GetInputPort("desired_position"),
    )

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(np.inf)


if __name__ == "__main__":
    main()
