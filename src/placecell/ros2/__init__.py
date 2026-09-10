"""ROS 2 wrapper: a node that feeds camera images and TF into the pipeline and answers questions.

`bridge` holds everything that can be tested without ROS: message-to-observation conversion,
pose extraction from transforms, keyframe writing. `node` is the thin rclpy shell around it.
"""
