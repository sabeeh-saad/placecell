"""Rasterize the bundled static box collisions at the robot's lidar height."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image


def pose(element):
    values = [float(v) for v in element.findtext("pose", "0 0 0 0 0 0").split()]
    if values[3] or values[4]:
        raise ValueError("Map generation supports planar poses only")
    return values[0], values[1], values[2], values[5]


def compose(parent, child):
    x, y, z, angle = parent
    cx, cy, cz, ca = child
    return (
        x + math.cos(angle) * cx - math.sin(angle) * cy,
        y + math.sin(angle) * cx + math.cos(angle) * cy,
        z + cz,
        angle + ca,
    )


def rasterize(world, resolution=0.025, laser_height=0.38):
    width, height = round(12 / resolution), round(10 / resolution)
    x, y = np.meshgrid(-6 + (np.arange(width) + 0.5) * resolution, 5 - (np.arange(height) + 0.5) * resolution)
    pixels = np.full((height, width), 205, dtype=np.uint8)
    pixels[(abs(x) < 5) & (abs(y) < 4)] = 254
    for model in ET.parse(world).getroot().findall("world/model"):  # noqa: S314 - bundled authored SDF
        if model.findtext("static") != "true":
            continue
        for link in model.findall("link"):
            base = compose(pose(model), pose(link))
            for collision in link.findall("collision"):
                size = collision.findtext("geometry/box/size")
                if size is None:
                    raise ValueError("Map generation supports box collisions only")
                sx, sy, sz = map(float, size.split())
                cx, cy, cz, angle = compose(base, pose(collision))
                if not cz - sz / 2 <= laser_height <= cz + sz / 2:
                    continue
                dx, dy = x - cx, y - cy
                local_x = math.cos(angle) * dx + math.sin(angle) * dy
                local_y = -math.sin(angle) * dx + math.cos(angle) * dy
                pixels[(abs(local_x) <= sx / 2) & (abs(local_y) <= sy / 2)] = 0
    return pixels


def write_map(world, output):
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rasterize(world)).save(output / "office.pgm")
    path = output / "office.yaml"
    path.write_text(
        "image: office.pgm\nmode: trinary\nresolution: 0.025\norigin: [-6.0, -5.0, 0.0]\n"
        "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n"
    )
    return path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write_map(root / "worlds/office.sdf", Path.home() / "maps")
