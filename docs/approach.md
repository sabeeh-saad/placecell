# Stopping near a remembered object

Optional approach planning turns a localized object position into a stopping pose that
faces it. It runs after visual destination selection and before the existing Nav2
navigation action. Named places, numeric coordinates and whole-scene destinations keep
their existing behavior.

## Enable on ROS 2

First configure [object memory](objects.md), checked localization, and the
[navigation verifier](navigation.md). Add these parameters to that launch:

```bash
-p approach_enabled:=true \
-p approach_costmap_topic:=/global_costmap/costmap_raw \
-p approach_footprint_topic:=/local_costmap/published_footprint \
-p approach_planner_action:=/compute_path_to_pose \
-p approach_planner_id:=GridBased \
-p approach_clearance_m:=0.5 \
-p approach_max_uncertainty_m:=0.35 \
-p approach_max_position_age_s:=300.0 \
-p approach_max_sensor_age_s:=2.0 \
-p approach_camera_yaw_offset_rad:=0.0
```

Use your actual topic names and planner ID. An empty planner ID lets Nav2 use its default
when unambiguous. `approach_enabled` defaults to false and requires `objects_enabled`.
The costmap must be a full `nav2_msgs/Costmap` in the configured map frame, published
often enough to satisfy the age limit. The footprint must be `PolygonStamped`, with a
TF transform from the robot base into its published frame at the polygon timestamp.
The bridge subtracts that robot origin before measuring footprint size; global polygon
coordinates are not mistaken for distances from the robot.

For example, a one-second costmap publication interval fits the default two-second limit.
The [Humble raw costmap publisher](https://github.com/ros-navigation/navigation2/blob/humble/nav2_costmap_2d/src/costmap_2d_publisher.cpp)
sends a full raw grid to subscribers each publication cycle. Its header records publication
time. That check detects stopped publication, but does not independently establish that
every obstacle sensor is current; Nav2's sensor freshness and obstacle-layer configuration
still apply.

Set the camera yaw offset to its calibrated horizontal viewing direction relative to
the robot's forward axis. Zero assumes a forward-facing camera. Keep the depth camera's
intrinsics and map transform calibrated, including mounting translation and tilt.

## Selection and checks

The planner samples nine poses within 60 degrees of the previously observed side of the
object. Distance from the estimated visible surface is the sum of the robot footprint's
enclosing-circle radius, requested clearance, object extent and position uncertainty.
The base heading points the camera toward the object. These are surface estimates;
unseen object geometry is not reconstructed.

Candidate endpoints and paths must fit the enclosing circle inside the costmap. Inscribed
obstacle, lethal and unknown cells are rejected. The check covers cell interiors and
interpolates path segments; checking a clear endpoint alone is insufficient. This circle
is conservative for rectangular or articulated robots and may reject narrow passages
that their exact footprint could traverse.

At most three candidates are queried through Nav2's native
[ComputePathToPose action](https://api.nav2.org/actions/humble/computepathtopose.html).
Partial paths, foreign frames, paths starting away from the current robot pose, paths
longer than 30 m and paths with collisions are rejected. Candidates closest to the learned
viewing side are checked first; among valid paths, viewing-angle consistency takes priority
over path length. This preserves identifying details rather than choosing a shorter route
to an object's unfamiliar side. The best checked path selects
the goal. Defaults allow two seconds per action request and eight seconds overall,
controlled by `approach_request_timeout_s` and `approach_planning_timeout_s`. These planning
queries do not move the robot. Cancellation and timeouts also cancel late accepted queries.

The selected path is checked again against the latest costmap, footprint and robot pose.
The object revision and localization are checked before the navigation goal is sent.
A blocked path, stale planning input, planner failure or changing object aborts the
command. It does not silently switch to an unchecked approach.

If depth position is absent, has unknown age, is older than five minutes or exceeds
the configured uncertainty limit, the already verified observation viewpoint remains
the fallback. An RGB-only revisit updates the sighting time but preserves the original
`position_timestamp`. Upgraded records lacking this timestamp need a new depth observation
before they can produce an approach pose.

Navigation status includes `object_id` and `goal_kind: "object_approach"` for these goals.
After Nav2 reports arrival, the fresh-image check must come from the new stopping pose.
It compares the selected object's saved crops with fresh detections, geometry and a
paired-image check, then verifies the original request. Optional [nearby viewpoint
search](object-arrival.md) reuses this planner with tighter distance and time limits.

## Library use and validation

Supply an `ApproachPlanner` to `DestinationResolver(approach=...)`. Its
`PlanningEnvironment` supplies immutable timestamped `PlanningSnapshot` data and bounded,
cancelable path requests. `Costmap` uses Nav2's raw 0–255 cost values, not occupancy
probabilities. Pose frames and map IDs must agree. Direct library callers are responsible
for trustworthy localization and sensor timestamps.

Tests use synthetic maps, geometry and a mocked native action client, including blocked
paths, stale geometry, cancellation, object changes and arrival at the selected pose.
No physical robot commissioning has been performed for this feature. Nav2 still performs
runtime planning, obstacle handling and motion control, and may replan after dispatch;
the earlier path check is not a guarantee against later environmental changes.
