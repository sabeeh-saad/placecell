import xml.etree.ElementTree as ET

from simulation.scripts.make_map import rasterize
from simulation.scripts.product_scene import CASES, HOME, PRODUCTS, score_mission, write_world


def completion(name, step, *, final=False, matched=True):
    destination = {"source": "named_place", "label": "home"} if name == "home" else {"object_id": name}
    return {
        "status": {
            "state": "succeeded" if final else "step_succeeded",
            "mission_step": step,
            "destination": destination,
            "object_result": "matched" if matched else "unobserved",
        }
    }


def evaluate(case, names):
    done = [completion(name, i + 1, final=i == len(names) - 1) for i, name in enumerate(names)]
    statuses = [{"state": "submitting"} for _ in names] + [done[-1]["status"]]
    return score_mission(case, done, statuses, {name: name for name in PRODUCTS})


def test_product_mission_scores_verified_order_including_repeated_visits():
    for case in CASES[:-1]:
        assert evaluate(case, case["expected"])["passed"]
        wrong = list(case["expected"])
        wrong[0], wrong[1] = wrong[1], wrong[0]
        assert not evaluate(case, wrong)["passed"]
        assert not evaluate(case, case["expected"][:-1])["passed"]


def test_arrival_oracle_rejects_unverified_identity_and_skipped_missing_target():
    case = CASES[0]
    done = [completion(name, i + 1, final=i == 3, matched=i != 1) for i, name in enumerate(case["expected"])]
    statuses = [{"state": "submitting"}] * 4 + [done[-1]["status"]]
    assert not score_mission(case, done, statuses, {name: name for name in PRODUCTS})["passed"]
    missing = CASES[-1]
    done = [completion("printer", 1)]
    statuses = [{"state": "submitting"}, {"state": "not_found"}]
    assert score_mission(missing, done, statuses, {"printer": "printer"})["passed"]
    statuses.insert(1, {"state": "submitting"})  # An unrequested skip to home must fail.
    assert not score_mission(missing, done, statuses, {"printer": "printer"})["passed"]


def test_product_world_is_mappable_and_its_tour_stops_have_clear_footprints(tmp_path):
    path = write_world(tmp_path / "products.sdf")
    world = ET.parse(path).getroot().find("world")  # noqa: S314 - generated authored fixture
    names = [item.attrib["name"] for item in world.findall("model")]
    assert len(names) == len(set(names))
    assert {"printer", "microwave", "fire_extinguisher", "microwave_desk"} <= set(names)
    grid = rasterize(path)
    for x, y, _ in [HOME, *(product["view"] for product in PRODUCTS.values())]:
        col, row = round((x + 6) / 0.025), round((5 - y) / 0.025)
        assert (grid[row - 12 : row + 13, col - 12 : col + 13] == 254).all()
