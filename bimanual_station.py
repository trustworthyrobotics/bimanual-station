import threading

import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Simulator,
)

import rclpy
from rclpy.node import Node

from hardware.iiwa7 import AddIiwa7Systems


def main():
    rclpy.init()

    rclpy_executor = rclpy.executors.SingleThreadedExecutor()

    diagram_builder = DiagramBuilder()

    AddIiwa7Systems(
        diagram_builder=diagram_builder,
        rclpy_executor=rclpy_executor,
    )

    ros_thread = threading.Thread(
        target=rclpy_executor.spin,
        daemon=True
    )
    ros_thread.start()

    diagram = diagram_builder.Build()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()
    simulator.AdvanceTo(np.inf)


if __name__ == "__main__":
    main()
