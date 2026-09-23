# Multi-product mission repairs — 22–23 September 2026

This follows the [multi-product evaluation](product-missions.md); its original failed
reports remain unchanged. The repair runs are stored under
`simulation/artifacts/product-repairs-20260922/`.

**Status: live qualification remains incomplete.** The current code passes 1,242 unit
tests with 95.55% coverage and 246/246 offline fault runs. All 11 no-provider
Gazebo cases and 14 isolated DDS sensor/clock checks also pass on the repaired image.
No live repair attempt has passed all five product missions. Saved-image checks found that Gemini
2.5 Flash still split the extinguisher into separately labeled parts. Gemini 3.1
Flash Lite returned accurate whole-object bounds for the saved printer, microwave
and extinguisher with the same production prompt. The simulation profile now uses
it for object learning. The arrival profile uses Gemini 2.5 Flash Lite: sequential
detection and comparison with 3.1 Flash Lite exceeded the existing freshness limit
in a full run. Full live qualification awaits a fresh credential: the latest temporary key expired
with HTTP 401 during the learning tour of the twelfth attempt. That attempt used
42 requests and $0.0102471 reported cost; cleanup released ownership and flushed traces.
The earlier padding experiment is removed. See the [validation manifest](validation/product-mission-repairs-2026-09-23.json)
for preserved failed attempts, current source/image hashes and provider accounting.

The planner and independent reviewer now receive configured place **names** from the
current map. They receive no coordinates, and the catalog does not authorize extra
visits. Thus a configured `home` can be interpreted without adding clarification text
to the operator's instruction. The current instruction and proposed destinations are
placed after the historical events, with an explicit review rule distinguishing those
root fields from past plans. A live replay of all five instructions against the saved
20-event history initially passed both planning and review. Longer live runs exposed
further contradictory review decisions. The simulation now uses Gemini 3.8 Flash for
review: a saved-context diagnostic approved the valid functional and repeated-visit
plans and rejected both a reordered plan and a plan missing the repeated visit. The
lighter 3.1 reviewer failed the latter negative case and was not adopted.

When object retrieval has no candidate above its existing threshold, a bounded model
call can propose one search category from categories actually observed in that scope.
The expanded query still has to pass the original similarity threshold. Candidate
images and arrival evidence are checked against the **original user description**,
including its distinguishing attributes. Invented categories, unsupported purposes
and ambiguous mappings cannot select a destination.

Approach planning prefers the learned viewing side among collision-checked paths,
then minimizes route length. Previously, the shortest route could approach a product
from a substantially different angle and hide identifying details. Path, footprint,
clearance, localization and cancellation checks remain unchanged.

Detection asks for coordinates before descriptions, explicitly normalizes each axis
against the original image dimensions, and treats attached handles, legs and panels as
parts of the whole object. This is intended to address a live failure where a box labeled pedestal
actually cropped the lower half of an extinguisher, creating a false competing appearance.
A square-padding experiment improved that saved image but misclassified the printer in
live learning, so it was removed. The original camera pixels and all identity margins
remain unchanged.

Hosted object providers detect boxes in a single image. Crop embeddings, rival margins
and geometry first select one unique candidate. Visual verification receives only
that exact crop, the frozen saved references and the full scene. It must confirm both
identity and the original destination. Explicit image labels alone did not prevent a
live model from describing the correct appliance while returning a table's index;
restricting the supplied candidate removes that selection error structurally. All
rivals still participate in the embedding checks before and after visual comparison.
The existing similarity, rival-margin, depth, motion, evidence-version and five-second
freshness checks still apply. Legacy providers retain the separate comparison path,
and additional caller-supplied request checks must also agree.

A passive camera diagnostic found missing RGB and depth samples despite continuous
calibration messages. The simulation now uses reliable delivery for all three camera
streams, with a matching optional application QoS setting. The sensor-data QoS default
for hardware remains unchanged. Pairing still requires capture times within 80 ms;
neither transport retries nor delayed callbacks extend the freshness limit. A 12-second
reliable-stream sample contained 48 complete RGB/depth/calibration pairs with no interior
capture gaps (0.2-second cadence); the earlier best-effort sample contained 26 complete
pairs. This sample supports the transport repair, not long-duration reliability.

Arrival captures now request an object refresh through normal ingestion. Previously,
the global object-scan interval could skip the arrival image even though scene memory
accepted it; a later visit then used an expired position. The durable observation carries
this refresh request. It bypasses only the routine scan interval: old/replayed timestamps,
localization, depth, object association and identity checks still apply. Verification
continues to use its frozen pre-departure reference and never writes object memory.

An earlier repair attempted detection and comparison together in one multi-image
call. Live evidence showed inaccurate oversized boxes, so that approach was removed.
Detection now receives only the current scene and cannot alter boxes during comparison.

The third live run verified each product, but inter-product routes failed the existing
circular footprint check at obstacle corners. A separate no-provider path diagnostic
reproduced those rejections. The simulation's global costmap now extends its inflation
cost gradient from 0.5 m to 1.2 m to encourage wider routes. Occupancy, footprint and
path validation thresholds remain unchanged.

Arrival traces now include the target score, live and known rival scores, and required
versus observed margin. These explain an identity rejection without exposing hidden
model reasoning. The evaluator also saves a terminal image and pose for failed cases.

The full regression run passed 1,242 tests with 95.55% coverage, and the offline fault
suite passed 246/246 runs after the single-candidate and reliable RGB-D repairs. Regression
coverage includes the request-check veto, the exact selected crop bytes, rejection of
lookalikes before visual comparison and unchanged arrival protections. All 14 isolated DDS sensor/clock checks also passed,
including the default hardware QoS with a reliable camera publisher. Live and Gazebo integration outcomes are recorded
separately; unit success alone does not qualify model behavior.

Earlier repair attempts and saved-image diagnostics used 786 provider requests and
$0.2490547 in reported cost before the fresh-credential rerun on 23 September. This
includes the expiry response and excludes the original pre-repair evaluation.
The new rerun and saved-image checks initially had a separate aggregate limit of 800
additional requests and a $0.50 provider-reported-cost stop. The user raised the request
limit to 1,600 while retaining that total cost stop; final accounting is in the manifest.

The sixth attempt was interrupted after confirming the crop-index defect. Its
per-case evidence and provider accounting remain available, but the default ROS
interrupt handler shut down the context before cleanup could write the aggregate
report. The evaluator now retains the ROS context until cancellation and report
writing finish, including when interrupted.

Two later attempts spent no model requests because localization remained inactive:
AMCL completed configuration but Fast DDS dropped its service reply during startup.
This matches the upstream [service discovery race](https://github.com/ros2/rmw_fastrtps/issues/842).
The simulator now waits briefly before asking the two lifecycle managers to perform
normal startup, and bounds discovery/activation at 90 seconds. It does not force
individual lifecycle states or bypass the navigation readiness checks.

The final no-provider Gazebo run passed **11/11 cases** on the single-crop and reliable
RGB-D image: ordered
visits with duplicate-command suppression, single-object arrival, cancellation during
motion and planning, invalid model output, depth loss, camera loss, removed objects,
moved objects, occlusion and lookalikes. Every case released navigation ownership and
retained complete critical trace events. The suite used real Gazebo/ROS/Nav2 with
scripted language, vision and embedding providers; it does not establish live model
accuracy. The moved object was verified only after confirming its previous location
was empty, and the lookalike case correctly ended as ambiguous.

The eighth live attempt completed the full printer → microwave → fire extinguisher →
home chain. Its other cases exposed reviewer and position-refresh defects repaired
subsequently; that success does not qualify the latest code for all five missions.
The twelfth attempt stopped during learning after its credential expired. Current
checkpoint accounting is **980 of 1,600 requests** and **$0.29307992 of the $0.50
reported-cost stop**, leaving 620 requests and about $0.20692. The expired credential
file was removed. A fresh credential is needed to complete live qualification.
