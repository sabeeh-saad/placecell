# Gazebo office environment

This first simulation milestone runs an office world and a differential-drive robot
with rendered RGB-D, lidar, wheel odometry and TF. Drive it from the keyboard or run
the automated smoke test. No physical robot, API key or embedding-model download is
required. All world geometry is bundled in the repository.

![RGB camera view of the printer and desk in the bundled Gazebo office](assets/gazebo-camera.png)

The image above is an actual 320 × 240 camera capture from the headless smoke test.

This environment currently validates the robot and sensor interface. Mapping,
localization, Nav2 goals and Placecell memory ingestion are subsequent milestones.
It does not yet navigate to objects from spoken commands. In particular, it does not
publish a fabricated `map` transform or localization confidence to bypass Placecell's
navigation checks.

## Start on Linux with Docker

Install Docker Engine with the Compose v2 plugin. From the repository root:

```bash
./simulation/sim build
./simulation/sim start
./simulation/sim check
```

The first build downloads ROS 2 Jazzy, Gazebo Harmonic and their dependencies and needs
several gigabytes of disk space. Later builds reuse Docker's layers. ROS/Gazebo run
inside the container; no host ROS installation is necessary. This uses the official
[Jazzy/Harmonic pairing](https://gazebosim.org/docs/harmonic/ros_installation/).
The image uses software rendering and a virtual X display by default, so the headless
test does not require a passed-through GPU or desktop session. Performance depends on
CPU resources; sensor rates are specified in simulation time.

The smoke test **moves the simulated robot** a short distance and turns it. Run it on
a freshly started world with no concurrent keyboard controller. It checks:

- An advancing simulation clock and wheel odometry.
- Nonblank 320 × 240 RGB and aligned floating-point depth from one RGB-D sensor.
- Matching capture stamps, optical frame and valid CameraInfo through Placecell's
  actual `aligned_snapshot` validator.
- A 360-sample laser scan and the sensor TF chain, including the optical-axis rotation.
- Forward motion, rotation and the velocity watchdog's stop after command silence.

It writes `simulation/artifacts/camera.png` and `simulation/artifacts/report.json`.
These files are ignored by Git. To inspect sensors without moving:

```bash
./simulation/sim check --sensors-only
```

Use `./simulation/sim logs` to follow startup or failure logs, and
`./simulation/sim stop` to stop and remove this environment's container. Starting it
again resets the office. This does not stop other Docker projects.

## Open the 3D view and drive

On a Linux X11 or XWayland desktop with `DISPLAY` set and the host `xauth` utility:

```bash
./simulation/sim gui
```

This recreates the simulation with the Gazebo desktop window and stays in the foreground.
It passes a temporary display cookie and mounts the X socket read-only; it does not
disable X access control. Ctrl+C closes the environment and removes the temporary cookie.
Native Wayland without XWayland and remote desktops without an X display should use
headless mode. Docker Desktop on macOS/Windows is not covered by the GUI launcher.

In a second terminal:

```bash
./simulation/sim teleop
```

Focus that terminal. Use `i` for forward, `,` for reverse, `j`/`l` for turning and `k`
to stop. The keyboard publisher sends a command for each key event; hold a key to use
your terminal's key repeat. An independent watchdog stops after 0.5 seconds without
a command and limits forward speed to 0.35 m/s and turning to 0.8 rad/s. Its
timer uses wall time, including while Gazebo is paused. This is manual driving: lidar
is published, but there is no autonomous obstacle avoidance in this milestone.

## World and interfaces

The office is 10 × 8 m with an open passage, a printer on a desk, a workstation, chair
and bookshelf. The robot starts at the origin facing the printer. The objects use
simple authored geometry; recognition accuracy with a vision model has not been measured.
Object model names are never supplied to perception as detection results.

ROS topics inside the container are:

- `/clock`: `rosgraph_msgs/msg/Clock`.
- `/cmd_vel`: manual `geometry_msgs/msg/Twist` input to the watchdog.
- `/sim/cmd_vel`: bounded watchdog output bridged to the simulator.
- `/odom`: `nav_msgs/msg/Odometry`, from `odom` to `base_footprint`.
- `/tf`, `/tf_static`, `/joint_states`: the robot and sensor transforms.
- `/scan`: `sensor_msgs/msg/LaserScan`, frame `laser_frame`, 10 Hz target.
- `/camera/color/image_raw`: `sensor_msgs/msg/Image`, `rgb8`, 5 Hz target.
- `/camera/aligned_depth_to_color/image_raw`: `sensor_msgs/msg/Image`, `32FC1` metres.
- `/camera/color/camera_info`: `sensor_msgs/msg/CameraInfo` for that same optical image.

RGB, depth and calibration use `camera_optical_frame`. The sensor's local viewing axis
and ROS optical convention are connected by the robot description. The camera is
0.53 m above the base footprint and points forward. The bridge configuration follows
the official [ROS/Gazebo bridge](https://gazebosim.org/docs/harmonic/ros2_integration/).

For inspection, open a sourced terminal in the container:

```bash
./simulation/sim shell
ros2 topic list
ros2 topic hz /camera/color/image_raw
ros2 run tf2_ros tf2_echo odom camera_optical_frame
```

The default network is isolated from host ROS discovery, uses ROS domain 77 and exposes
no ports. It does not mount robot devices, host credentials, the Docker socket or the
working tree. The GUI mode adds only display access. Applications that join this
simulation should run inside its container or an explicitly configured simulation network.

## Development and regression checks

World geometry is in `simulation/worlds/office.sdf`; the robot and both sensor definitions
are in `simulation/models/robot/robot.urdf.xacro`. The Xacro is the single source for
Gazebo spawning and `robot_state_publisher`. Edit these files, rebuild and restart:

```bash
./simulation/sim build
./simulation/sim stop
./simulation/sim start
./simulation/sim check
```

The `simulation` GitHub workflow runs the same headless checks when simulator files or
the depth interface change, and supports manual runs. It uploads the camera image,
report and simulator logs for inspection. Python tests still run separately in the
existing CI workflow. Container dependency versions can change when rebuilding against
updated package repositories; retain the tested image digest for an exact deployment.

The next milestone adds a saved map, AMCL and Nav2. Then Placecell can ingest simulated
observations under checked map localization, resolve object requests, and test arrival
and changing-world scenarios using this same environment.
