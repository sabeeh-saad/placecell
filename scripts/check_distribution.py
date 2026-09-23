"""Install the built wheel and sdist separately, then run offline checks outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def run(command: list[str], *, cwd: Path, env: dict[str, str], log: Path) -> None:
    with log.open("a") as output:
        output.write(json.dumps(command) + "\n")
        output.flush()
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT, check=True, timeout=180)  # noqa: S603


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True, help="Pre-downloaded NumPy/setuptools/wheel wheels")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifacts = args.dist_dir.resolve()
    wheelhouse = args.wheelhouse.resolve()
    if not wheelhouse.is_dir():
        parser.error("wheelhouse must exist; stage dependencies before the offline check")
    distributions = {
        kind: list(artifacts.glob(pattern))
        for kind, pattern in (("wheel", "placecell-*.whl"), ("sdist", "placecell-*.tar.gz"))
    }
    if any(len(paths) != 1 for paths in distributions.values()):
        parser.error("dist-dir must contain exactly one PlaceCell wheel and one source archive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.update(
        PIP_NO_INDEX="1", PIP_FIND_LINKS=str(wheelhouse), PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONNOUSERSITE="1"
    )
    results = []
    for kind, paths in distributions.items():
        case = output / kind
        case.mkdir()
        result = {"kind": kind, "artifact": paths[0].name, "passed": False}
        try:
            with tempfile.TemporaryDirectory(prefix=f"placecell-{kind}-") as directory:
                work = Path(directory)
                environment = work / "venv"
                venv.EnvBuilder(with_pip=True).create(environment)
                python = str(environment / "bin/python")
                run(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--no-index",
                        "--find-links",
                        str(wheelhouse),
                        "--no-cache-dir",
                        str(paths[0]),
                    ],
                    cwd=work,
                    env=env,
                    log=case / "install.log",
                )
                run([python, "-m", "pip", "check"], cwd=work, env=env, log=case / "install.log")
                run(
                    [
                        python,
                        "-I",
                        str(root / "scripts/smoke_installed.py"),
                        "--output",
                        str(case),
                        "--fixtures",
                        str(root / "evaluation/missions"),
                    ],
                    cwd=work,
                    env=env,
                    log=case / "smoke.log",
                )
                result["passed"] = True
        except (OSError, subprocess.SubprocessError) as exc:
            result["error"] = str(exc)
        results.append(result)
    report = {
        "schema_version": 1,
        "python": sys.version,
        "passed": all(r["passed"] for r in results),
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))  # noqa: T201 - validation result
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
