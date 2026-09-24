# Backup, restore and upgrades

`placecell-backup` (also `python -m placecell.backup`) creates an offline snapshot of
the reference deployment. It preserves the collection metadata, SQLite memory and
object records, crop bytes, sightings, ingestion/refinement jobs, cleanup intents,
Lance projection files, keyframe files, corrections, and configured command, context,
navigation and trace journals. Provider credentials and launch configuration are
not collected. Keep the matching map, model configuration, dependency versions and
software revision separately with the deployment record.

The supported profile is Linux local storage, one controller and one collection per
database directory, with all evidence under one keyframe directory. These commands
do not contact models, start the controller or send robot commands.

## Create and verify

Stop the controller and all standalone ingestion, correction, maintenance and command
writers. A graceful shutdown is preferable. The ROS node now holds maintenance leases
for its configured storage paths from startup until its writers finish. Backup refuses
those leases while the controller is running, and controller startup refuses active
maintenance. A stuck writer retains the lease until process exit. Earlier binaries
and direct library users do not acquire these new leases: stopping them is still an
operator responsibility. Use the same canonical paths in every process.

Create a JSON storage profile matching the actual ROS parameters. All fields are
required; use an empty string only for disabled optional journals. Do not omit an
enabled journal. The memory, keyframe and corrections paths are mandatory and must
be distinct, nonoverlapping locations. Example for the mission profile:

```json
{
  "collection": "office_missions",
  "db_path": "/home/simulator/placecell-missions/db",
  "keyframe_dir": "/home/simulator/placecell-missions/keyframes",
  "corrections_path": "/home/simulator/placecell-missions/corrections.jsonl",
  "command_journal_path": "/home/simulator/placecell-missions/commands.sqlite3",
  "mission_context_path": "/home/simulator/placecell-missions/missions.sqlite3",
  "mission_trace_path": "/home/simulator/placecell-missions/traces.sqlite3",
  "navigation_ownership_path": "/home/simulator/placecell-missions/navigation.sqlite3"
}
```

Run inside the storage namespace that owns these paths, with a **new** backup directory:

```bash
python -m placecell.backup create --profile /path/storage-profile.json \
  --destination /backups/office-before-upgrade --confirm-stopped
python -m placecell.backup verify --backup /backups/office-before-upgrade
```

The confirmation records that external writers are stopped; it does not stop them.
SQLite's backup API includes committed WAL data and excludes uncommitted writes.
With writers stopped, images and separate journals belong to the same quiescent
recovery point. The manifest records that time, completion time, source paths, schema
versions, counts, sizes and SHA-256 checksums. Verification checks the exact file
inventory, SQLite integrity and foreign keys, supported versions, vector dimensions
and finite values, and image references/digests from memories, objects and all jobs,
including failed jobs. A cleanup intent may refer to an already deleted image.

A missing configured journal, missing referenced image, unexpected file, symbolic
link, invalid payload or checksum mismatch refuses publication. Empty correction
logs are now created at normal initialization so their presence is explicit. For an
older deployment that never created its correction log, first establish whether it
ever contained feedback; a missing log is not automatically treated as empty.

Snapshots are staged privately, flushed and published with Linux `renameat2` without
replacing an existing destination. Interrupted unpublished work may leave a hidden
`.NAME.incomplete-*` directory. It is not a backup; after confirming its process has
exited, remove that specific staging directory and retry with a new destination.
If interruption occurs after publication, verify the completed destination before
using it. A directory-sync failure after publication also requires verification.

Checksums detect accidental corruption; they are not authentication or encryption.
Store backups securely: they contain the robot's images and conversation history.

## Restore into a fresh instance

Stop the old controller and all other command sources before switching deployments.
Keep the source backup intact. Restore always uses a **new** destination:

```bash
python -m placecell.backup restore --backup /backups/office-before-upgrade \
  --destination /home/simulator/restored-office
```

Restore verifies the backup, copies and verifies it again in a private staging
directory, relocates the known image references in SQLite, and marks memory vectors
for projection refresh on store open. It does not recaption or re-embed images.
Pending jobs and failure counters survive. Existing destinations and destinations
inside the backup are refused. Files after the recorded recovery point are **not**
recovered; the restore report makes that boundary explicit.

The destination contains:

- `restore-report.json`: recovery point, component counts and required startup actions.
- `storage-profile.json`: paths for backing up the restored deployment.
- `restore-parameters.yaml`: ROS parameter overlay, expressed as valid JSON/YAML.
- `restore-parameters.json`: the same flat parameter values for inspection.

Load the overlay **after** the saved deployment configuration:

```bash
python -m placecell.ros2.node --ros-args \
  --params-file /path/deployment.yaml \
  --params-file /home/simulator/restored-office/restore-parameters.yaml
```

Retain the original robot/map/model identity and scoped action configuration. The
overlay supplies storage paths and a fresh `mission_conversation_id`; it does not
supply provider settings. Identified command publishers must adopt the new session.
Old command history remains stored, but cannot deduplicate commands accepted after
the backup. A restored journal therefore refuses its old conversation ID. Mission
context starts a new scoped conversation; historical events are not executable work.
Legacy unversioned text publishers must be stopped/drained before reconnecting.

**Restored navigation ownership is always unknown**, even if the backup said clean
or pending. An old UUID cannot prove the current action server's state. Startup stays
uncertain/busy and does not replay missions. Before admitting new movement, perform
the [independent Nav2 reset and explicit attestation](crash-recovery.md#first-use-or-unresolved-ownership)
against the restored journal, with the controller and other command sources stopped.
The restore tool does not stop/reset Nav2 or establish physical safety.

## Upgrade and rollback

The qualified transition is the unversioned SQLite state layout used through Day 15
(`user_version=0`, collection schema 9) to the explicitly admitted layout (`user_version=1`).
The existing additive table/column migrations are idempotent; the completion marker is
written last. A process killed during initialization can retry. Unsupported future
SQLite versions are rejected before schema or Lance projection changes. SQLite may
still create its normal empty WAL/shared-memory coordination files on a read-only open.

1. Stop writers, save the current software/configuration identity, create and verify a
   backup, and retain an independent copy.
2. Restore a trial instance with the new software. Check memory retrieval, images,
   queued/failed work and the restore report while motion remains blocked.
3. For rollback, stop the trial and restore the **pre-upgrade backup** into another new
   directory. Use its matching old software/configuration and the fresh session overlay.
   Do not point an older binary at upgraded live files or manually downgrade schema numbers.
4. Re-establish the Nav2 baseline before enabling a controller qualified for that deployment.

The Day 16 drill seeds data with actual main revision
`76900778084da4f18606b4f51bfc1d404fcb3f92`, interrupts the new upgrade, then verifies
retrieval and accepted-job evidence with that previous revision on a restored backup.
This qualifies **offline data rollback** to that revision. That older controller lacks
Day 15's startup reconciliation; robot navigation after a runtime downgrade is not
qualified by this drill. Collection schemas older than 9 must use their matching
backup tooling before a separately qualified migration.

## Reproduce and limits

```bash
.venv/bin/pytest tests/test_backup.py
.venv/bin/python scripts/check_backup.py --repeat 3 --output /tmp/backup-checks
# Add --legacy-source /path/to/exported-previous-source for the prior-revision drill.
simulation/sim build
simulation/sim check-backup
```

The process harness uses real SIGKILL at copy, pre-publication, post-publication and
upgrade boundaries. The ROS check uses the production node and its parameter loader
in a network-disabled container. Neither requires Gazebo motion or paid model access.
CI repeats the current-version process checks; the prior-revision drill is recorded
separately in [Day 16 validation](validation/day-16.json).

The recorded checkpoint passes 1,359 tests at 95.42% coverage, 18 repeated
SIGKILL/upgrade trials, and six production ROS startup/restore checks. These are
local results; the new CI job is configured but has not run remotely for this work.

This work detects corruption and provides recovery from a verified backup; it does not
repair damaged internal tables, promise zero data loss after the recovery point, or
qualify power loss, hardware/media failure, network filesystems, encrypted backups,
full-disk runtime behavior, arbitrary historic migrations, or physical-robot stopping.
