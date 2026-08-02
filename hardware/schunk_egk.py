import threading
import uuid

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, FloatingPointRange
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from control_msgs.action import GripperCommand


from bkstools.bks_lib.bks_base import BKSBase
from pyschunk.generated.generated_enums import eCmdCode


class SchunkGripperNode(Node):

    def __init__(self, namespace: str, host: str):
        super().__init__(namespace)

        bksb = BKSBase(host)
        bksb.SetAttributes()
        bksb.command_code = eCmdCode.CMD_ACK

        self._goal_id = None
        self._goal_id_lock = threading.Lock()

        self._bksb = bksb
        self._bksb_lock = threading.Lock()

        self._min_pos = bksb.min_pos * 1e-3
        self._max_pos = bksb.max_pos * 1e-3
        self._min_force = bksb.min_grp_force
        self._max_force = bksb.max_grp_force

        self.declare_parameter(
            "grip_velocity",
            bksb.min_vel * 1e-3,
            ParameterDescriptor(
                description="Grip velocity (m/s)",
                floating_point_range=[FloatingPointRange(
                    from_value=bksb.min_vel * 1e-3,
                    to_value=bksb.max_grp_vel / 2 * 1e-3,
                )]
            )
        )

        self._action_server = ActionServer(
            self,
            GripperCommand,
            f"/{namespace}/cmd_gripper",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=ReentrantCallbackGroup(),
        )

        self.get_logger().info(f"Using namespace /{namespace} for ROS communication")

    def goal_callback(self, goal_request):
        cmd = goal_request.command
        if not (self._min_pos <= cmd.position and cmd.position <= self._max_pos):
            self.get_logger().warn(f"Goal position {cmd.position} not within [{self._min_pos}, {self._max_pos}] (m)")
            return GoalResponse.REJECT
        if not (self._min_force <= cmd.max_effort and cmd.max_effort <= self._max_force):
            self.get_logger().warn(f"Goal max_effort {cmd.max_effort} not within [{self._min_force}, {self._max_force}] (N)")
            return GoalResponse.REJECT

        self.get_logger().info(f"Received goal position {cmd.position} (m), max_effort {cmd.max_effort} (N)")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        goal_id = uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)).hex
        self.get_logger().info(f"Received goal (ID: {goal_id}) cancel request")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        cmd = goal_handle.request.command

        cmd_pos = cmd.position * 1e3
        cmd_force = cmd.max_effort / self._max_force * 100
        cmd_vel = self.get_parameter("grip_velocity").value * 1e3

        goal_id = uuid.UUID(bytes=bytes(goal_handle.goal_id.uuid)).hex

        with self._goal_id_lock:
            self._goal_id = goal_id

        with self._bksb_lock:
            self.get_logger().info(f"Executing goal (ID: {goal_id}) position {cmd.position} [m], max_effort {cmd.max_effort} [N]")

            bksb = self._bksb

            cmd_dir = (cmd_pos > bksb.actual_pos)

            bksb.command_code = eCmdCode.CMD_ACK
            bksb.set_force = cmd_force
            bksb.set_vel = cmd_vel
            bksb.grp_dir = cmd_dir  # False: grip from outside, True: grip from inside
            bksb.command_code = eCmdCode.MOVE_FORCE

            def gripped_or_error():
                return (bksb.plc_sync_input[0] &
                    (
                        bksb.sw_error
                        | bksb.sw_not_feasible
                        | bksb.sw_gripped
                        | bksb.sw_no_workpiece_detected
                        | bksb.sw_workpiece_lost
                        | bksb.sw_wrong_workpiece_detected
                    )
                ) != 0

            def gripped():
                return (bksb.plc_sync_input[0] & bksb.sw_gripped) != 0

            def reached_goal():
                if not cmd_dir:
                    return bksb.actual_pos <= cmd_pos + 0.1
                else:
                    return bksb.actual_pos >= cmd_pos - 0.1

            canceled = False
            while not (gripped_or_error() or reached_goal()):
                if goal_handle.is_cancel_requested:
                    self.get_logger().info(f"Canceling goal (ID: {goal_id}) as requested")
                    goal_handle.canceled()
                    canceled = True
                    break

                if self._goal_id != goal_id:
                    self.get_logger().info(f"Canceling goal (ID: {goal_id}) due to new goal request")
                    goal_handle.abort()
                    canceled = True
                    break

                feedback = GripperCommand.Feedback()
                feedback.position = bksb.actual_pos * 1e-3
                feedback.stalled = gripped()
                feedback.reached_goal = reached_goal()
                feedback.effort = 0.0
                goal_handle.publish_feedback(feedback)

            bksb.command_code = eCmdCode.CMD_STOP

            if not canceled:
                goal_handle.succeed()
                self.get_logger().info(f"Finished goal (ID: {goal_id})")

            result = GripperCommand.Result()
            result.position = bksb.actual_pos * 1e-3
            result.stalled = gripped()
            result.reached_goal = reached_goal()
            result.effort = cmd.max_effort if (not canceled and result.stalled) else 0.0
            return result


def main(args=None):
    rclpy.init(args=args)

    node = SchunkGripperNode(
        namespace="left_schunk",
        host="192.170.10.4",
    )

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()


if __name__ == "__main__":
    main()
