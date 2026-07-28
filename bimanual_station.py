import argparse
import threading

import numpy as np

import rclpy
from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

from hardware.iiwa7 import AddIiwa7Systems, Iiwa7SystemsConfig, IiwaOperatingMode


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
        "--operating_mode",
        choices=[m.value for m in IiwaOperatingMode],
        default=IiwaOperatingMode.JOINT_POS.value,
        help="IIWA robot operating mode.",
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
        help="IIWA robot maximum joint velocity. Specify one value or seven values.",
    )

    args = parser.parse_args()

    args.operating_mode = IiwaOperatingMode(args.operating_mode)

    if len(args.max_joint_vel) == 1:
        args.max_joint_vel = args.max_joint_vel[0]
    elif len(args.max_joint_vel) != 7:
        parser.error("--max_joint_vel must be specified as either a single value or 7 values.")

    return args


def main():
    args = parse_args()

    rclpy.init()

    rclpy_executor = rclpy.executors.SingleThreadedExecutor()

    diagram_builder = DiagramBuilder()

    config = Iiwa7SystemsConfig(
        operating_mode=args.operating_mode,
        enable_left_arm=args.enable_left,
        enable_right_arm=args.enable_right,
        max_joint_velocity=args.max_joint_vel,
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
