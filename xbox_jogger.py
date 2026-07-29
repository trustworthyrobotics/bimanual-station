import subprocess
from enum import Enum

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped


class ActiveArm(Enum):
    LEFT = 0
    RIGHT = 1


class XboxCartesianVel(Node):
    def __init__(self):
        super().__init__("xbox_jogger")

        self.declare_parameter("linear_scale", 0.08)
        self.declare_parameter("angular_scale", 0.2)

        self.left_iiwa_pub = self.create_publisher(
            TwistStamped,
            "/left_iiwa/cmd_cartesian_vel",
            10,
        )

        self.right_iiwa_pub = self.create_publisher(
            TwistStamped,
            "/right_iiwa/cmd_cartesian_vel",
            10,
        )

        self.joy_sub = self.create_subscription(
            Joy,
            "/joy",
            self.joy_callback,
            10,
        )

        self.active_arm = ActiveArm.LEFT
        self.switching_enabled = False

        self.prev_a = False
        self.prev_b = False

        # Detect connected robots after publishers have had time to match.
        self.detect_timer = self.create_timer(1.0, self.detect_arms)

        self.get_logger().info("Xbox jogger started")

    def detect_arms(self):
        left = self.left_iiwa_pub.get_subscription_count() > 0
        right = self.right_iiwa_pub.get_subscription_count() > 0

        if left and right:
            self.switching_enabled = True
            self.active_arm = ActiveArm.LEFT
            self.get_logger().info("Detected both arms. A=left, B=right.")
        elif left:
            self.switching_enabled = False
            self.active_arm = ActiveArm.LEFT
            self.get_logger().info("Detected left arm only.")
        elif right:
            self.switching_enabled = False
            self.active_arm = ActiveArm.RIGHT
            self.get_logger().info("Detected right arm only.")
        else:
            # Keep waiting.
            return

        # Detection complete.
        self.destroy_timer(self.detect_timer)

    def joy_callback(self, msg: Joy):
        linear_scale = self.get_parameter("linear_scale").value
        angular_scale = self.get_parameter("angular_scale").value

        a_pressed = msg.buttons[0]
        b_pressed = msg.buttons[1]

        if self.switching_enabled:
            if a_pressed and not self.prev_a:
                self.active_arm = ActiveArm.LEFT
                self.get_logger().info("Controlling left arm")

            if b_pressed and not self.prev_b:
                self.active_arm = ActiveArm.RIGHT
                self.get_logger().info("Controlling right arm")

        self.prev_a = a_pressed
        self.prev_b = b_pressed

        cmd = TwistStamped()
        twist = cmd.twist

        # Translation
        twist.linear.x = linear_scale * msg.axes[1]
        twist.linear.y = linear_scale * msg.axes[0]
        twist.linear.z = linear_scale * msg.axes[4]

        # Rotation
        twist.angular.z = angular_scale * msg.axes[3]
        twist.angular.x = angular_scale * msg.axes[6]
        twist.angular.y = angular_scale * msg.axes[7]

        if self.active_arm == ActiveArm.LEFT:
            self.left_iiwa_pub.publish(cmd)
        else:
            self.right_iiwa_pub.publish(cmd)


def main():
    rclpy.init()

    node = XboxCartesianVel()

    joy_process = subprocess.Popen([
        "ros2", "run", "joy", "joy_node"
    ])

    try:
        rclpy.spin(node)
    finally:
        joy_process.terminate()
        joy_process.wait()

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
