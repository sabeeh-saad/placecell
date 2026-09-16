from pathlib import Path

import pytest
from PIL import Image
from simulation.scripts.make_map import compose, rasterize, write_map

WORLD = Path(__file__).resolve().parents[1] / "simulation/worlds/office.sdf"


def test_navigation_map_matches_collision_geometry_at_lidar_height():
    pixels = rasterize(WORLD)

    def cell(x, y):
        return pixels[int((5 - y) / 0.025), int((x + 6) / 0.025)]

    assert pixels.shape == (400, 480)
    assert cell(0, 0) == 254  # Open connecting passage.
    assert cell(0, 2.0) == cell(4.98, 0) == 0  # Divider and perimeter wall.
    assert cell(3.3, 0.53) == 0  # Leg transformed with the printer desk's rotation.
    assert cell(3, 0) == 254  # Tabletop and printer are above the laser plane.
    assert cell(5.5, 0) == 205  # Unknown outside the office, not navigable free space.


def test_map_writer_preserves_resolution_origin_and_image_orientation(tmp_path):
    path = write_map(WORLD, tmp_path)
    text = path.read_text()
    assert "resolution: 0.025" in text and "origin: [-6.0, -5.0, 0.0]" in text
    assert "image: office.pgm" in text
    with Image.open(tmp_path / "office.pgm") as image:
        assert image.size == (480, 400)
    assert compose((1, 2, 3, 0), (4, 5, 6, 0)) == (5, 7, 9, 0)


def test_unsupported_collision_geometry_fails_instead_of_silently_disappearing(tmp_path):
    world = tmp_path / "world.sdf"
    world.write_text(
        "<sdf><world><model><static>true</static><link><collision><geometry><sphere/></geometry></collision></link></model></world></sdf>"
    )
    with pytest.raises(ValueError, match="box collisions"):
        rasterize(world)
