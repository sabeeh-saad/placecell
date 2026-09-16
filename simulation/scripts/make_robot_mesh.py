"""Generate the robot's authored, unit-sized rounded shell mesh without extra tools."""

import math
from pathlib import Path


def rounded_box():
    positions, normals, triangles = [], [], []
    steps, radius = 12, 0.18
    for axis in range(3):
        for sign in (-1, 1):
            start = len(positions)
            for row in range(steps + 1):
                for col in range(steps + 1):
                    point = [0.0, 0.0, 0.0]
                    point[axis] = sign * 0.5
                    point[(axis + 1) % 3] = col / steps - 0.5
                    point[(axis + 2) % 3] = row / steps - 0.5
                    inner = [max(-0.5 + radius, min(0.5 - radius, v)) for v in point]
                    delta = [p - q for p, q in zip(point, inner, strict=True)]
                    length = math.sqrt(sum(v * v for v in delta))
                    normal = [v / length for v in delta]
                    positions.append([p + radius * n for p, n in zip(inner, normal, strict=True)])
                    normals.append(normal)
            for row in range(steps):
                for col in range(steps):
                    a = start + row * (steps + 1) + col
                    b, c, d = a + 1, a + steps + 1, a + steps + 2
                    faces = [(a, b, d), (a, d, c)] if sign > 0 else [(a, d, b), (a, c, d)]
                    triangles.extend(faces)

    def source(name, values):
        data = " ".join(f"{v:.7f}" for row in values for v in row)
        return (
            f'<source id="{name}"><float_array id="{name}-data" count="{len(values) * 3}">{data}</float_array>'
            f'<technique_common><accessor source="#{name}-data" count="{len(values)}" stride="3">'
            '<param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>'
            "</accessor></technique_common></source>"
        )

    indices = " ".join(f"{i} {i}" for triangle in triangles for i in triangle)
    return (
        '<?xml version="1.0"?>\n<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">'
        "<asset><created>2026-09-16T00:00:00Z</created><modified>2026-09-16T00:00:00Z</modified>"
        '<unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>'
        '<library_geometries><geometry id="shell"><mesh>'
        + source("positions", positions)
        + source("normals", normals)
        + '<vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>'
        + f'<triangles count="{len(triangles)}"><input semantic="VERTEX" source="#vertices" offset="0"/>'
        + '<input semantic="NORMAL" source="#normals" offset="1"/>'
        + f"<p>{indices}</p></triangles></mesh></geometry></library_geometries>"
        + '<library_visual_scenes><visual_scene id="scene"><node id="shell-node">'
        + '<instance_geometry url="#shell"/></node></visual_scene></library_visual_scenes>'
        + '<scene><instance_visual_scene url="#scene"/></scene></COLLADA>\n'
    )


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "models/robot/meshes/rounded_box.dae"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rounded_box())
