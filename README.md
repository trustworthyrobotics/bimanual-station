# Usage

Start the left robot and run
```sh
./iiwa_driver
```

Start the right robot and run
```sh
./iiwa_driver --fri_ip 192.170.10.3 --fri_port 30201 --lcm_command_channel IIWA_COMMAND_2 --lcm_status_channel IIWA_STATUS_2
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

The tool z-offset, which defines the point where Cartesian pose and velocity commands are applied, can be configured using the following command-line argument:

```bash
python bimanual_station.py [--tool_z_offset OFFSET]
```

### `xbox_jogger.py`

The `xbox_jogger.py` script provides interactive Cartesian velocity control of one or two arms using an Xbox controller.

c
