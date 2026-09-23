"""Deterministic providers for Gazebo integration, never model-quality evidence.

Only rendered pixels reach these fixtures. The blue display is a deliberately narrow
office-specific marker; no simulator poses or entity labels are supplied to perception.
"""

import base64
import io
import json
import threading

import cv2
import numpy as np
from PIL import Image

from placecell import ChatReply, ToolCall
from placecell.depth import Box
from placecell.object_types import ArrivalComparison, Detection
from placecell.providers.base import Capabilities
from placecell.verification import SceneVerdict


def pixels(source):
    if isinstance(source, bytes):
        stream = io.BytesIO(source)
    elif isinstance(source, str) and source.startswith("data:"):
        stream = io.BytesIO(base64.b64decode(source.split(",", 1)[1]))
    else:
        stream = source.uri
    with Image.open(stream) as image:
        return np.asarray(image.convert("RGB"), dtype=np.int16)


def displays(array):
    r, g, b = array[:, :, 0], array[:, :, 1], array[:, :, 2]
    mask = ((b > r + 18) & (b > g + 7) & (g > r + 10)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    # JPEG chroma artifacts can form tiny blue components beside the desk. They are
    # not resolvable displays. Keep the same threshold for detection and comparison.
    return [tuple(map(int, row[:4])) for row in stats[1:count] if 24 <= row[4] <= 2200 and row[2] >= 6 and row[3] >= 3]


class PixelFixture:
    model_name = "checkpoint-blue-display-fixture-v1"
    dimension = 8
    capabilities = Capabilities(text=True, image=True)

    @staticmethod
    def vectors(flags):
        result = np.zeros((len(flags), 8), dtype=np.float32)
        for index, flag in enumerate(flags):
            result[index, 0 if flag else 1] = 1
        return result

    def embed_text(self, texts):
        return self.vectors(["printer" in text.lower() for text in texts])

    def embed_media(self, items):
        return self.vectors([bool(displays(pixels(item))) for item in items])


class DetectorFixture:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, evidence):
        array = pixels(evidence)
        height, width = array.shape[:2]
        result = []
        for x, y, w, h in displays(array):
            # The display occupies a fixed proportion of this bundled printer mesh.
            # This fixture intentionally cannot distinguish two identical printers.
            left, top = max(0, x - 4.3 * w), max(0, y - 1.5 * h)
            right, bottom = min(width, x + 1.7 * w), min(height, y + 5.5 * h)
            if w > 50 or h > 30 or right - left > width * 0.8:
                continue
            result.append(
                Detection(
                    "printer",
                    "office fixture with blue display",
                    Box(left / width, top / height, right / width, bottom / height),
                )
            )
        return result

    def absent(self, reference_png, evidence, region):
        return not any(region.overlap(d.box) > 0.1 for d in self.detect(evidence))

    def compare(self, references, candidate):
        matched = bool(displays(pixels(candidate))) and any(displays(pixels(reference)) for reference in references)
        return SceneVerdict("matched" if matched else "uncertain", "Deterministic blue-display fixture comparison")

    def compare_arrival(self, references, candidates, image, target):
        """Exercise the combined comparison without hosted calls or simulator labels."""
        return ArrivalComparison(
            0, self.compare(references, candidates[0]), VisionFixture().verify(target, candidates[0])
        )


class CaptionFixture:
    def __init__(self, *args, **kwargs):
        pass

    def caption(self, items):
        return [
            "printer with blue display" if DetectorFixture().detect(item) else "office background" for item in items
        ]


class VisionFixture:
    def __init__(self, *args, **kwargs):
        pass

    def verify(self, target, image):
        matched = "printer" in target.lower() and bool(displays(pixels(image)))
        return SceneVerdict("matched" if matched else "not_matched", "Deterministic fixture visual decision")


class PlanFixture:
    def __init__(self, reviewer=False):
        self.reviewer = reviewer
        self.calls = 0
        self.blocked = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.malformed = False

    def complete(self, messages, tools):
        self.calls += 1
        self.blocked.set()
        if not self.release.wait(30):
            raise TimeoutError("Fixture planning delayed beyond bound")
        if self.reviewer:
            return ChatReply(
                None,
                (ToolCall("review", "review_navigation_plan", {"decision": "approve", "message": "Scripted review"}),),
            )
        data = json.loads(messages[-1].content)
        text = data["instruction"].lower()
        targets = [part.removeprefix("go to ").removeprefix("return to ").strip(" .") for part in text.split(" then ")]
        args = {"decision": "ready", "destinations": targets, "message": "Scripted fixture plan"}
        if self.malformed:
            args["action"] = "unsupported"
        return ChatReply(None, (ToolCall("plan", "propose_navigation_plan", args),))
