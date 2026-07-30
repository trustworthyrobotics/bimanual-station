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

The `iiwa_driver` executable connects a KUKA IIWA robot to the host computer using the KUKA Fast Robot Interface (FRI).

Before launching the driver, on the KUKA smartPAD, start the appropriate FRI application:

| Application                         | Description                   |
| ----------------------------------- | ----------------------------- |
| `FRIPositionDriver`                 | No compliance                 |
| `FRIJointImpedanceDriver(Tool)`     | Compliance in joint space     |
| `FRICartesianImpedanceDriver(Tool)` | Compliance in Cartesian space |

On the host computer, launch the driver for the desired robot:

**Left IIWA**
```bash
./iiwa_driver \
    --fri_ip 192.170.10.2 \
    --fri_port 30200 \
    --lcm_command_channel IIWA_COMMAND \
    --lcm_status_channel IIWA_STATUS
```

**Right IIWA**
```bash
./iiwa_driver \
    --fri_ip 192.170.10.3 \
    --fri_port 30201 \
    --lcm_command_channel IIWA_COMMAND_2 \
    --lcm_status_channel IIWA_STATUS_2
```


### `bimanual_station.py`

The `bimanual_station.py` script provides ROS 2 topics for commanding the IIWA robots in four modes:

| Command mode       | ROS 2 topic                      | ROS 2 message type               |
| ------------------ | -------------------------------- | -------------------------------- |
| Joint velocity     | `/<namespace>/cmd_joint_vel`     | `sensor_msgs/msg/JointState`     |
| Joint position     | `/<namespace>/cmd_joint_pos`     | `sensor_msgs/msg/JointState`     |
| Cartesian velocity | `/<namespace>/cmd_cartesian_vel` | `geometry_msgs/msg/TwistStamped` |
| Cartesian pose     | `/<namespace>/cmd_cartesian_pos` | `geometry_msgs/msg/PoseStamped`  |

where `<namespace>` is either `left_iiwa` or `right_iiwa`. There are also topics for reporting the joint state and Cartesian state.

The joint velocity limit, Cartesian linear/angular velocity limit, and the tool offset can be configured through command-line arguments:

```bash
bimanual_station.py \
    [--max_joint_vel VEL [VEL ...]] \
    [--max_linear_vel VEL [VEL ...]] \
    [--max_angular_vel VEL [VEL ...]] \
    [--tool_z_offset OFFSET]
```

### `xbox_jogger.py`

The `xbox_jogger.py` script provides interactive Cartesian velocity control of one or two arms using an Xbox controller.
