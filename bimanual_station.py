import argparse
import threading

import numpy as np

import rclpy
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from hardware.iiwa7 import AddIiwa7Systems, Iiwa7SystemsConfig


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("true", "1", "yes", "y"):
            return True
        if v.lower() in ("false", "0", "no", "n"):
            return False
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--enable_left",
        type=str2bool,
        default=True,
        help="Enable left robot.",
    )

    parser.add_argument(
        "--enable_right",
        type=str2bool,
        default=True,
        help="Enable right robot.",
    )

    parser.add_argument(
        "--max_joint_vel",
        type=float,
        nargs="+",
        default=[1.0],
        metavar="VEL",
        help="IIWA robot maximum joint velocity (rad/s). Specify one value or seven values.",
    )

    parser.add_argument(
        "--max_linear_vel",
        type=float,
        nargs="+",
        default=[0.5],
        metavar="VEL",
        help="IIWA robot maximum certesian linear velocity (m/s). Specify one value or three values.",
    )

    parser.add_argument(
        "--max_angular_vel",
        type=float,
        nargs="+",
        default=[1.8],
        metavar="VEL",
        help="IIWA robot maximum certesian angular velocity (rad/s). Specify one value or three values.",
    )

    args = parser.parse_args()

    if len(args.max_joint_vel) == 1:
        args.max_joint_vel = args.max_joint_vel[0]
    elif len(args.max_joint_vel) != 7:
        parser.error("--max_joint_vel must be specified as one value or seven values.")

    if len(args.max_linear_vel) == 1:
        args.max_linear_vel = args.max_linear_vel[0]
    elif len(args.max_linear_vel) != 3:
        parser.error("--max_linear_vel must be specified as one value or three values.")

    if len(args.max_angular_vel) == 1:
        args.max_angular_vel = args.max_angular_vel[0]
    elif len(args.max_angular_vel) != 3:
        parser.error("--max_angular_vel must be specified as one value or three values.")

    return args


def main():
    args = parse_args()

    rclpy.init()

    rclpy_executor = rclpy.executors.SingleThreadedExecutor()

    diagram_builder = DiagramBuilder()

    config = Iiwa7SystemsConfig(
        enable_left_arm=args.enable_left,
        enable_right_arm=args.enable_right,
        max_joint_velocity=args.max_joint_vel,
        max_linear_velocity=args.max_linear_vel,
        max_angular_velocity=args.max_angular_vel,
    )

    AddIiwa7Systems(
        diagram_builder=diagram_builder,
        rclpy_executor=rclpy_executor,
        config=config,
    )

    ros_thread = threading.Thread(
        target=rclpy_executor.spin,
        daemon=True,
    )
    ros_thread.start()

    diagram = diagram_builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(np.inf)


if __name__ == "__main__":
    main()
