"""Authored products and labelled evaluation routes; never input to perception/planning."""

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAP_ID = "office-products-v1"
PRODUCTS = {
    "printer": {"aliases": ("printer",), "position": (3.0, 0.0, 1.0), "view": (1.0, 0.0, 0.0)},
    "microwave": {"aliases": ("microwave",), "position": (3.0, -2.5, 0.8), "view": (1.15, -2.5, 0.0)},
    "fire extinguisher": {
        "aliases": ("extinguisher",),
        "position": (-3.0, -2.5, 0.75),
        "view": (-1.2, -2.5, math.pi),
    },
}
HOME = (-1.0, 0.3, math.pi)
CASES = (
    {
        "id": "three_products",
        "instruction": "Go to the printer, then the microwave, then the fire extinguisher, and finally return home.",
        "expected": ("printer", "microwave", "fire extinguisher", "home"),
    },
    {
        "id": "purpose_reasoning",
        "instruction": (
            "First go to the appliance where I can heat my lunch, "
            "then to where I can print documents, and finally return home."
        ),
        "expected": ("microwave", "printer", "home"),
    },
    {
        "id": "repeat_visit",
        "instruction": (
            "Go to the fire extinguisher, then the microwave, "
            "then go back to the fire extinguisher, and finish at home."
        ),
        "expected": ("fire extinguisher", "microwave", "fire extinguisher", "home"),
    },
    {
        "id": "microwave_first",
        "instruction": "Go to the microwave, then the printer, then the fire extinguisher, and finally return home.",
        "expected": ("microwave", "printer", "fire extinguisher", "home"),
    },
    {
        "id": "missing_second_target",
        "instruction": "Go to the printer, then the refrigerator, then return home.",
        "expected": ("printer",),
        "must_stop_before": "refrigerator",
    },
)


def values(numbers):
    return " ".join(str(v) for v in numbers)


def part(link, name, position, size, color, *, cylinder=None, collide=True):
    visual = ET.SubElement(link, "visual", name=name)
    ET.SubElement(visual, "pose").text = values((*position, 0, 0, 0))
    geometry = ET.SubElement(visual, "geometry")
    if cylinder is None:
        ET.SubElement(ET.SubElement(geometry, "box"), "size").text = values(size)
    else:
        shape = ET.SubElement(geometry, "cylinder")
        ET.SubElement(shape, "radius").text = str(cylinder[0])
        ET.SubElement(shape, "length").text = str(cylinder[1])
    material = ET.SubElement(visual, "material")
    for key in ("ambient", "diffuse"):
        ET.SubElement(material, key).text = values((*color, 1))
    if collide:
        collision = ET.SubElement(link, "collision", name=name)
        ET.SubElement(collision, "pose").text = values((*position, 0, 0, 0))
        # Conservative physical boxes also keep the authored 2D map reproducible.
        ET.SubElement(ET.SubElement(ET.SubElement(collision, "geometry"), "box"), "size").text = values(size)


def model(world, name, pose):
    item = ET.SubElement(world, "model", name=name)
    ET.SubElement(item, "static").text = "true"
    ET.SubElement(item, "pose").text = values(pose)
    return ET.SubElement(item, "link", name="body")


def write_world(path):
    tree = ET.parse(ROOT / "worlds/office.sdf")  # noqa: S314 - authored local SDF only
    world = tree.getroot().find("world")
    desk = copy.deepcopy(world.find("model[@name='printer_desk']"))
    desk.set("name", "microwave_desk")
    desk.find("pose").text = "3 -2.5 0 0 0 1.5707963268"
    # A low appliance table keeps the whole front face within the fixed camera FOV.
    for element in [*desk.findall("link/visual"), *desk.findall("link/collision")]:
        pose = [float(value) for value in element.findtext("pose").split()]
        pose[2] -= 0.2 if element.attrib["name"] == "top" else 0.1
        element.find("pose").text = values(pose)
        if element.attrib["name"] != "top":
            size = [float(value) for value in element.findtext("geometry/box/size").split()]
            size[2] -= 0.2
            element.find("geometry/box/size").text = values(size)
    world.append(desk)
    dark, grey, white, red = (0.045, 0.05, 0.06), (0.65, 0.67, 0.7), (0.92, 0.92, 0.9), (0.78, 0.025, 0.02)
    link = model(world, "microwave", (3, -2.5, 0.58, 0, 0, -math.pi / 2))
    part(link, "case", (0, 0, 0.19), (0.68, 0.43, 0.38), grey)
    part(link, "door_frame", (-0.075, -0.223, 0.2), (0.48, 0.018, 0.31), dark)
    part(link, "door_glass", (-0.095, -0.235, 0.2), (0.36, 0.012, 0.22), (0.10, 0.15, 0.17))
    part(link, "handle", (0.14, -0.255, 0.20), (0.022, 0.04, 0.25), white)
    part(link, "control_panel", (0.257, -0.225, 0.2), (0.125, 0.014, 0.31), dark)
    part(link, "display", (0.257, -0.237, 0.3), (0.095, 0.008, 0.05), (0.03, 0.45, 0.17))
    for row in range(3):
        for col in range(3):
            part(
                link,
                f"key_{row}_{col}",
                (0.227 + col * 0.03, -0.24, 0.235 - row * 0.04),
                (0.018, 0.01, 0.018),
                white,
                collide=False,
            )
    part(link, "start_button", (0.257, -0.24, 0.075), (0.075, 0.012, 0.028), white, collide=False)

    link = model(world, "fire_extinguisher", (-3, -2.5, 0, 0, 0, math.pi / 2))
    part(link, "pedestal", (0, 0, 0.2), (0.42, 0.38, 0.4), (0.3, 0.3, 0.31))
    part(link, "red_tank", (0, 0, 0.70), (0.28, 0.28, 0.6), red, cylinder=(0.14, 0.6))
    part(link, "neck", (0, 0, 1.015), (0.095, 0.095, 0.06), dark, cylinder=(0.0475, 0.06))
    part(link, "handle", (0, 0, 1.075), (0.27, 0.08, 0.035), dark)
    part(link, "lever", (0.03, 0, 1.12), (0.22, 0.04, 0.025), dark)
    part(link, "label", (0, -0.137, 0.7), (0.14, 0.008, 0.25), white, collide=False)
    for row in range(4):
        part(link, f"label_line_{row}", (0, -0.143, 0.77 - row * 0.045), (0.1, 0.004, 0.009), dark, collide=False)
    part(link, "hose_upper", (0.18, 0.015, 0.97), (0.04, 0.05, 0.15), dark)
    part(link, "hose", (0.18, -0.01, 0.72), (0.04, 0.05, 0.42), dark)
    part(link, "nozzle", (0.18, -0.04, 0.5), (0.065, 0.1, 0.13), dark)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def score_mission(case, completions, statuses, identities):
    observed = []
    for completion in completions:
        status = completion["status"]
        destination = status.get("destination") or {}
        if destination.get("source") == "named_place" and destination.get("label") == "home":
            observed.append("home")
        else:
            name = next(
                (name for name, identity in identities.items() if identity == destination.get("object_id")), None
            )
            observed.append(name if status.get("object_result") == "matched" else None)
    terminal = statuses[-1] if statuses else {}
    dispatches = [status for status in statuses if status["state"] == "submitting"]
    passed = observed == list(case["expected"])
    if case.get("must_stop_before"):
        passed = passed and terminal.get("state") in {
            "not_found",
            "destination_unverified",
            "ambiguous",
            "destination_ambiguous",
        }
        passed = passed and len(dispatches) == len(case["expected"])
    else:
        passed = passed and terminal.get("state") == "succeeded" and len(dispatches) == len(case["expected"])
    return {
        "passed": passed,
        "observed_order": observed,
        "expected_order": list(case["expected"]),
        "terminal": terminal,
        "submissions": len(dispatches),
        "completions": completions,
    }
