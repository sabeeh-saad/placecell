"""Export the editable HTML architecture to a 2x PNG with a local Chrome/Chromium.

Run from any directory: python3 docs/assets/render_mission_architecture.py
No Python packages, network fonts, JavaScript libraries, or running web server required.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", help="Path to a local Chrome or Chromium executable")
    args = parser.parse_args()
    browser = args.browser or next(
        (path for name in ("google-chrome", "chromium", "chromium-browser") if (path := shutil.which(name))), None
    )
    if not browser:
        parser.error("Install Chrome/Chromium, or provide --browser /path/to/browser")
    assets = Path(__file__).resolve().parent
    source = assets / "mission-architecture.html"
    output = assets / "mission-architecture.png"
    with tempfile.TemporaryDirectory(prefix="placecell-diagram-") as temporary:
        command = [
            browser,
            "--headless=new",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-component-update",
            "--disable-sync",
            "--hide-scrollbars",
            f"--user-data-dir={temporary}/profile",
        ]
        dom = subprocess.run(  # noqa: S603 - explicitly selected local rendering executable
            [*command, "--window-size=1080,1600", "--dump-dom", source.as_uri()],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        match = re.search(r'<script id="render-report" type="application/json">(.*?)</script>', dom, re.S)
        if match is None:
            raise RuntimeError("The browser did not return a layout report")
        report = json.loads(match[1])
        if report["overflow"]:
            raise RuntimeError(f"Clipped text: {report['overflow']}")
        width, height = report["width"], report["height"]
        subprocess.run(  # noqa: S603 - local HTML screenshot, with a temporary browser profile
            [
                *command,
                f"--window-size={width},{height}",
                "--force-device-scale-factor=2",
                f"--screenshot={output}",
                source.as_uri(),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    print(f"Rendered {output.name} at {width * 2} x {height * 2}; no text overflow.")  # noqa: T201


if __name__ == "__main__":
    main()
