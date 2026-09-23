"""Start a fresh product scene, run its bounded evaluation, then stop the simulator."""

import argparse
import json
import os
import signal
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from product_scene import write_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--explicit-home", action="store_true")
    parser.add_argument("--max-requests", type=int, default=400)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--reported-cost-stop", type=float, default=1.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.preflight:
        if not args.key_file:
            parser.error("Live evaluation needs a private credential file")
        key = args.key_file.read_text().strip()
        request = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + key})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310 - fixed HTTPS endpoint
                print(json.dumps({"credential_http_status": response.status}), flush=True)  # noqa: T201
        except urllib.error.HTTPError as error:
            print(json.dumps({"credential_http_status": error.code}), flush=True)  # noqa: T201
            return 2
        del key, request
    os.environ["PLACECELL_SIM_WORLD"] = str(write_world(args.output / "products.sdf"))
    root = Path(__file__).resolve().parents[1]
    with (args.output / "simulator.log").open("w") as sim_log, (args.output / "evaluation.log").open("w") as run_log:
        simulator = subprocess.Popen(  # noqa: S603 - fixed authored launch
            ["ros2", "launch", str(root / "launch/office.launch.py"), "navigation:=true"],  # noqa: S607
            stdout=sim_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        command = ["python3", str(root / "scripts/product_missions.py"), "--output", str(args.output / "results")]
        if args.preflight:
            command.append("--preflight")
        else:
            command.extend(["--key-file", str(args.key_file)])
        for case in args.case:
            command.extend(["--case", case])
        if args.explicit_home:
            command.append("--explicit-home")
        command.extend(
            [
                "--max-requests",
                str(args.max_requests),
                "--max-seconds",
                str(args.max_seconds),
                "--reported-cost-stop",
                str(args.reported_cost_stop),
            ]
        )
        try:
            result = subprocess.run(  # noqa: S603 - fixed authored evaluator and parsed arguments
                command, stdout=run_log, stderr=subprocess.STDOUT, timeout=args.max_seconds + 100, check=False
            )
            print(json.dumps({"evaluation_exit_code": result.returncode}), flush=True)  # noqa: T201
        finally:
            if simulator.poll() is None:
                os.killpg(simulator.pid, signal.SIGINT)
                try:
                    simulator.wait(25)
                except subprocess.TimeoutExpired:
                    os.killpg(simulator.pid, signal.SIGKILL)
                    simulator.wait()
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
