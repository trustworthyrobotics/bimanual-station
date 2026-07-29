## Dependencies

This package requires **ROS 2** and the Python dependencies listed in `requirements.txt`.

Install the Python dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Before using the package, make sure the required environments are sourced:

```bash
source /opt/ros/<distro>/setup.sh
source .venv/bin/activate
```

## Usage

### `iiwa_driver`

The `iiwa_driver` executable communicates with a KUKA IIWA robot through the Fast Robot Interface (FRI).

To connect to the **left IIWA**, run:

```bash
./iiwa_driver \
    --fri_ip 192.170.10.2 \
    --fri_port 30200 \
    --lcm_command_channel IIWA_COMMAND \
    --lcm_status_channel IIWA_STATUS
```

These are the default arguments, so the same connection can be started with:

```bash
./iiwa_driver
```

To connect to the **right IIWA**, run:

```bash
./iiwa_driver \
    --fri_ip 192.170.10.3 \
    --fri_port 30201 \
    --lcm_command_channel IIWA_COMMAND_2 \
    --lcm_status_channel IIWA_STATUS_2
```

### `bimanual_station.py`

The `bimanual_station.py` script provides ROS 2 interfaces for controlling the IIWA robots in four control modes:

| Control mode       | ROS 2 topic                      |
| ------------------ | -------------------------------- |
| Joint velocity     | `/<namespace>/cmd_joint_vel`     |
| Joint position     | `/<namespace>/cmd_joint_pos`     |
| Cartesian velocity | `/<namespace>/cmd_cartesian_vel` |
| Cartesian pose     | `/<namespace>/cmd_cartesian_pos` |

The maximum joint velocity, cartesian linear velocity, cartesian angular velocity, and tool offset can be configured through command-line arguments:

```bash
bimanual_station.py \
    [--max_joint_vel VEL [VEL ...]] \
    [--max_linear_vel VEL [VEL ...]] \
    [--max_angular_vel VEL [VEL ...]] \
    [--tool_z_offset OFFSET]
```

Each velocity argument accepts either a single value or a list of values for per-joint/per-axis limits.
