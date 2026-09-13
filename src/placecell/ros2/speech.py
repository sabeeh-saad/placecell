"""Microphone-to-command entry point. Speech dependencies are optional and loaded on use."""

from __future__ import annotations

import argparse
from typing import Any

from placecell.speech import SpeechGate, SpeechWorker, load_vosk


def main(args: list[str] | None = None) -> None:  # pragma: no cover - requires a microphone and ROS
    parser = argparse.ArgumentParser(description="Send spoken movement commands to the robot.")
    parser.add_argument("--model", required=True, help="Path to an extracted local Vosk model")
    parser.add_argument("--device", help="Microphone device index or name; default uses the system input")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--wake-word", default="robot")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--command-topic", default="/placecell/command")
    options, ros_args = parser.parse_known_args(args)
    import rclpy
    import sounddevice as sd
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from std_msgs.msg import String

    gate = SpeechGate(options.wake_word, options.min_confidence)
    recognizer = load_vosk(options.model, options.sample_rate)
    device: str | int | None = options.device
    if isinstance(device, str) and device.isdecimal():
        device = int(device)
    rclpy.init(args=ros_args)
    node = Node("placecell_speech")
    publisher = node.create_publisher(String, options.command_topic, QoSProfile(depth=1, lifespan=Duration(seconds=2)))
    worker = SpeechWorker(
        recognizer,
        gate,
        lambda text: publisher.publish(String(data=text)),
        node.get_logger(),
        sample_rate=options.sample_rate,
    )

    def audio(data: Any, frames: int, timing: Any, status: Any) -> None:
        worker.feed(bytes(data), overflow=bool(status))

    try:
        with sd.RawInputStream(
            samplerate=options.sample_rate,
            blocksize=options.sample_rate // 10,
            device=device,
            dtype="int16",
            channels=1,
            callback=audio,
        ):
            worker.start()
            node.get_logger().info(f"Listening for '{options.wake_word} go to <place>' or '{options.wake_word} stop'.")
            while rclpy.ok() and worker.running:
                rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        worker.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":  # pragma: no cover
    main()
