"""ROS 2 wrapper: a node that feeds camera images and TF into the pipeline and answers questions.

Only `placecell_node` imports rclpy when it loads; everything else is tested without ROS.
`config` reads the ROS parameters into typed settings, `components` builds the pipeline and
the navigation stack from them, `capture` turns camera frames into observations and
`housekeeping` runs corrections, maintenance and diagnostics, with `workers` for the
background workers and `answers` for the `~/answer` payloads. `bridge` converts messages and
writes keyframes. `placecell_node` wires these to topics, timers and TF; `node` is the entry point.
"""
