ARG ROS_IMAGE=ros:jazzy-ros-base-noble@sha256:c3706ef0a0aa45413c07803cf433602f543b22e45b4855f6fca955c2d8ecc4e8
FROM ${ROS_IMAGE}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-ros-gz \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-xacro \
    python3-numpy python3-pil python3-yaml \
    libgl1-mesa-dri mesa-utils xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 2000 simulator && useradd --uid 2000 --gid 2000 --create-home simulator
COPY --chown=simulator:simulator simulation /opt/placecell-sim
COPY --chown=simulator:simulator src/placecell /opt/placecell-src/placecell
ENV PYTHONPATH=/opt/placecell-src \
    ROS_DOMAIN_ID=77 \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    GZ_PARTITION=placecell-office \
    GZ_SIM_RESOURCE_PATH=/opt/placecell-sim/models \
    LIBGL_ALWAYS_SOFTWARE=1
USER simulator
WORKDIR /opt/placecell-sim
ENTRYPOINT ["/bin/bash", "/opt/placecell-sim/entrypoint.sh"]
CMD ["ros2", "launch", "/opt/placecell-sim/launch/office.launch.py"]
