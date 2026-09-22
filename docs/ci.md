# Continuous integration and installed-package checks

Both workflows run for every pull request, every push to `main`, merge-queue groups and
manual dispatch. There are no path filters: changes to planners, memory, controllers,
providers, ROS adapters, evaluation data, packaging or newly added modules all reach the
same gates. Documentation-only changes run them too. This trades some runner time for
avoiding incomplete dependency lists and missing checks. GitHub's
[workflow trigger reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#on)
defines these events.

These jobs validate software behavior on their stated environments. They do not qualify
physical robots or real-model perception quality. Repository branch-protection settings
are managed separately; editing workflow files does not make checks mandatory for merging
unless repository rules require them.

## Python checks

`.github/workflows/ci.yml` runs on Ubuntu 24.04 for Python 3.10, 3.11 and 3.12:

- Install the development, video and LanceDB extras.
- Check lint, formatting and strict types.
- Run the full pytest suite with the existing 90% coverage gate.
- Run the scripted offline fault campaign and retain its report.

JUnit results, coverage JSON and fault reports are uploaded even after failures. The
matrix keeps running its other versions when one version fails. Jobs have bounded
timeouts; superseded runs of the same workflow/ref are canceled.

The suite includes regression checks that prohibit workflow/job filters from silently
excluding core changes, verify the supported Python matrix and ROS smoke gates, and
ensure a failing package installation yields a failed result with diagnostics.

## Clean installation checks

The separate `package` matrix runs for the same three Python versions. It uses
[PyPA build](https://build.pypa.io/en/stable/) to create a source distribution and then
a wheel from that source distribution. Build and dependency preparation need access to
the package index. After preparation, each distribution is installed offline:

1. Stage NumPy and the setuptools/wheel build dependencies in a local wheel directory.
2. Create two independent temporary virtual environments without system site packages.
3. Install the wheel in one and the source archive in the other. Use `--no-index`, the
   staged wheels and no pip cache, including for the source archive's isolated build.
4. Run `pip check`, then run the smoke script with Python isolated mode from a temporary
   working directory outside the checkout. Confirm PlaceCell was imported from that
   environment, so an editable install or `PYTHONPATH` cannot hide missing package files.

Only core runtime dependencies are installed. The smoke checks exercise the typing marker,
all seven entry points, CLI help for the offline-capable tools, memory retrieval, the
24-case scripted mission baseline, the offline fault campaign and persistent trace export.
ROS and microphone entry points are loaded without starting devices; actual ROS startup
is covered separately below. Each distribution must pass independently.

To reproduce on Linux with the checkout's development environment:

```bash
python -m pip install build==1.6.1
python -m build
python -m pip download --only-binary=:all: --dest /tmp/placecell-package-deps \
  'numpy>=1.24' 'setuptools>=68' wheel
python scripts/check_distribution.py --dist-dir dist \
  --wheelhouse /tmp/placecell-package-deps --output /tmp/placecell-package-checks
```

Use a new output directory for each run and exactly one wheel/source archive in `dist`.
The report records per-format success/failure. Installation logs, smoke logs, mission
results, fault reports and trace exports are preserved. CI uploads the distributions and
reports even when a check fails. A nonzero subprocess exit or timeout fails the job.

## ROS and Gazebo checks

`.github/workflows/simulation.yml` builds the bundled Jazzy/Gazebo image and runs:

```bash
simulation/sim build
simulation/sim check-operator
simulation/sim check-cancel
simulation/sim check-sensors
simulation/sim start
simulation/sim check
simulation/sim start-nav
simulation/sim check-nav
simulation/sim stop
```

`check-operator` runs in a disposable container with networking disabled, without needing
the office to be running. It checks real ROS command/status topics, snapshot services,
late subscribers, paused simulated time, cancellation, durable ID retries/conflicts/restart,
scope/expiry refusal and node startup. Multi-goal model
and navigation results are scripted. Its artifacts are saved under a unique
`simulation/artifacts/check-operator-*` directory.
Day 10 adds malformed provider replies through the actual chat decoder, their visible
ROS refusal state, and a valid two-goal control. See [model/input contracts](model-input-contracts.md).
Day 12 adds delivery checks for retrieval, identity, geometry and execution attribution
through status, snapshot services and retained history. Target regressions and nine new
fault scenarios run in the existing Python suite. See [target freshness](target-freshness.md).

`check-sensors` sends RGB-D, localization, TF and `/clock` through the production node
in a disposable container with networking disabled. It checks loss, skew, malformed and
repeated timestamps, delayed TF, recovery and a backward clock reset. Destination selection
and action results are scripted; provider calls are blocked. Reports are retained under
`simulation/artifacts/check-sensors-*`. See [sensor/clock contracts](sensor-clock-contracts.md).

`check-cancel` uses real command topics and a controlled `NavigateToPose` action server,
with network access disabled, eight CPUs and 16 GiB RAM. It checks cancellation with
blocked scripted providers, delayed acceptance and missing/rejected acknowledgements,
then measures 100 cancellation requests under a synthetic slow observation callback.
It includes a successful two-goal control. See [ownership and measurement scope](cancellation-ownership.md).
Every trial is retained under `simulation/artifacts/check-cancel-*`; failure is nonzero.

`check` validates RGB/depth/lidar, advancing simulation time, PlaceCell RGB-D admission,
simulated forward motion and turning, and stopping after command silence. `check-nav`
starts from a fresh world with AMCL/Nav2 and checks localization and a planned route.
The office is stopped and simulator logs/artifacts are collected through `always()`
steps, including after test failures.

These checks exercise the container build and middleware/navigation wiring; the Python
suite supplies detailed planner, memory, execution, trace and operator regressions. CI
does not substitute scripted replies for a live-model qualification result.

## Paid evaluation remains explicit

Ordinary CI does not invoke `simulation/sim missions` or `simulation/sim check-pipeline`,
and does not receive provider credentials. Those opt-in workflows retain their existing
funded-key requirement and need an agreed model budget. Downloading software dependencies
or images is distinct from making model-provider calls.

Local validation records identify which interpreter and image were exercised. The hosted
GitHub matrix becomes evidence for a particular commit only after that commit is pushed
and its jobs finish; a local pass is not a hosted CI result.
