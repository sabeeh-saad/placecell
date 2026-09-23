# Recognizing a remembered object on arrival

Object navigation compares the selected object's saved crops with a new observation
after Nav2 reaches its goal. This check runs automatically in ROS when both
`navigation_enabled` and `objects_enabled` are enabled. It uses the configured
multimodal embedder and a Gemini vision model; it does not download model weights.

## What must agree

Before departure, the resolver snapshots up to four localized crop views, their image
vectors, the object record and vectors for known alternatives across category labels. The saved
reference survives concurrent ingestion, keyframe cleanup and updates to the object's
latest view. Deleting the object, making its identity ambiguous or rejecting its memory
through operator feedback prevents successful verification.

After arrival, the camera must supply a newly captured, localized frame from the same
robot, camera and versioned map, within 0.35 m and 0.35 rad of the requested pose.
The controller waits for a frame with aligned depth before starting object verification;
an RGB-only frame leaves it waiting within the original arrival deadline. Waiting does
not refresh sensor trust or extend the capture-age limit. Scene arrival retains its
RGB-only path. Object verification starts with a capture no older than five seconds. It then:

1. Detects fresh objects and compares their crop embeddings with the saved image vectors.
   The best score must reach 0.85 and exceed competing live detections and saved
   alternatives by at least 0.08. Close alternatives produce an ambiguous result.
2. Checks RGB-D geometry. By default, the saved surface position must be no older than
   300 seconds and both positions must have uncertainty at most 0.35 m. A nearby
   candidate must agree within the larger of 0.35 m and the summed position uncertainty.
3. For a larger move, requires similarity of at least 0.95, movement within 3 m, and
   confirmation that the old region is empty in the same observation. Every relevant
   depth ray through the retained observed surface must show background behind it, and a separate visual
   absence check must agree. Occlusion cannot establish a move.
4. Requires a comparison of the saved views and selected fresh object using visible
   distinguishing details; matching category, colour or shape alone is insufficient.
   Indistinguishable objects must produce `uncertain`.
5. Checks that the same selected object satisfies the user's original destination and
   attributes, using the fresh scene for context. Hosted providers first detect boxes
   in the current image alone. Embeddings, rival margins and depth then select one
   unique crop. Only that crop is sent with the saved references and scene for the
   visual identity and request checks; the model cannot select a different crop by
   an incorrect image index. Providers without the combined comparison contract use
   the previous comparator plus parallel full-scene request check. All results must
   finish while the image is fresh. Arrival does not request unused caption embeddings.

`arrival.identity_scores` traces expose the target, live-rival and known-rival cosine
scores, the required margin and the observed margin when deciding uniqueness.

Depth positions retain at most 25 map-frame samples from the observed central surface.
Absence projects these samples into the new view and requires valid background across
the projected patch, beyond the surface plus position/angular uncertainty and a margin.
This avoids treating the supporting table as part of the object that must disappear.
Invalid depth, foreground occlusion, out-of-view support or inconsistent support points
cannot establish absence. Older records without samples retain the enclosing-sphere check.

When using the verifier directly without reliable depth, verification is allowed only
from within 0.1 m and 0.1 rad of a saved viewpoint, with crop-box overlap of at least 0.6.
This cannot confirm a larger move. Appearance scores and spatial margins are starting thresholds, not probabilities
or proof of physical identity. Identical products, changed appearance and detector errors
remain reasons to stop and ask the user to identify the destination more precisely.

## Results and memory updates

Navigation status adds `object_result` and `search_attempt`. Object results are:

- `matched`: appearance, spatial checks, paired comparison and the destination request agree.
- `ambiguous`: competing identities or uncertain evidence; terminal state is `destination_ambiguous`.
- `missing`: the old occupied region is visible and confirmed empty in this observation.
- `unobserved`: the selected object was not established in this view.
- `unavailable`: required evidence or verification could not be obtained.

Only an accepted match produces `succeeded`. Other terminal results use
`destination_unverified`, except for ambiguity. Transport failures and user cancellation
retain their existing states. An empty `object_result` means no object verdict was made.

Verification is read-only: it does not strengthen memory, rewrite identity or count
absence evidence. Ordinary camera ingestion continues independently, applying its usual
association, replay and independent-visit rules. A single arrival `missing` verdict does
not mark a persistent object missing. Confirmed movement updates memory when normal
ingestion processes that observation and its association checks pass.
Scan scheduling metadata does not invalidate a checked verdict. Changes to object
evidence still do: identity alternatives and the evidence generation are checked again
after both provider checks finish and before success is accepted.

## Optional nearby viewpoint search

Search defaults to disabled. After a `missing` or `unobserved` result, it can try another
checked viewpoint around the remembered position. Configure the costmap, footprint,
planner action and calibrated camera yaw described in [approach planning](approach.md),
then add:

```bash
-p object_search_enabled:=true \
-p object_search_max_viewpoints:=3 \
-p object_search_timeout_s:=60.0 \
-p object_search_radius_m:=1.5 \
-p object_search_max_path_m:=4.0
```

The limit allows three additional viewpoints. The 60-second budget starts when the
first search is requested and includes planning, navigation, waiting for images and
verification. Each new viewpoint must be at least 0.35 m from every previously attempted
viewpoint. Goals and the complete checked path must stay within 1.5 m of the original
arrival goal, with at most 4 m of planned travel per leg. The search centre never advances
with the robot. An original observation pose far from the object may leave no candidate
inside this area.

Search uses the approach planner's nine candidates within 60 degrees of the previously
observed side. It requires a fresh saved depth position, current costmap, footprint,
localization and a collision-checked Nav2 path. There is no unchecked viewpoint fallback,
full-room exploration or spin action. Search can be enabled independently of
`approach_enabled`; initial navigation then still uses the recorded observation pose.

Ambiguity, unavailable depth, provider errors, exhausted limits or unavailable planning
inputs stop search. Every reached viewpoint requires another capture taken after that
arrival. Stop cancels outstanding work. A timeout while moving requests Nav2 cancellation
and keeps the trip busy until its terminal result confirms completion. The public request
ID stays the same; each transport goal has a separate ID so earlier callbacks cannot
complete a later leg.

The distance limits apply to the selected goals and checked paths. Nav2 may replan during
execution; these checks are not a runtime geofence. Its obstacle handling, recovery
configuration and any deployment geofence remain responsible for actual motion.

## Models, latency and library use

`object_arrival_model` defaults to `object_model` and uses `object_base_url` plus
`object_api_key_env`. The native Gemini and compatible chat providers support combined
arrival comparison; absence remains a separate check when required. Each vision request has an eight-second timeout with no
retries, configurable through `object_arrival_request_timeout_s`. Crop embedding uses
the collection's embedder and its existing request settings. The legacy full-scene
request check uses the separate `verification_model` settings in the
[navigation guide](navigation.md).

The arrival deadline remains `navigation_arrival_timeout_s` (default 30 seconds), capped
by the search deadline during recovery. Hosted calls run on the command worker; an
in-flight HTTP request may finish after cancellation or the deadline, but its late
answer cannot authorize success or another goal. ROS permits up to three fresh-capture
attempts (`navigation_arrival_max_attempts`, range 1–5) if verification outlives the
five-second image limit. Each expired result is discarded; the original arrival deadline
is unchanged, and another attempt starts only after the previous worker finishes.
Cancellation, sensor loss and substantive negative verdicts do not trigger this retry.
The library default remains one attempt. Arrival captures bypass a full background
ingestion queue, while ordinary ingestion retains its bounded queue and cleanup path.

ROS similarity and geometry controls are `object_arrival_min_similarity`,
`object_arrival_moved_similarity`, `object_arrival_similarity_margin`,
`object_arrival_max_uncertainty_m`, `object_arrival_max_position_age_s`, and
`object_arrival_max_move_m`. Their defaults are the values above. They are independent
of ingestion's association policy; evaluate both when changing models.

Library callers supply `ObjectArrivalVerifier(tracker, comparator)` to
`DestinationResolver(object_arrival=...)`. `ObjectComparator.compare` accepts one to
four saved PNG byte strings and one fresh PNG, returning a `SceneVerdict` with
`matched`, `not_matched` or `uncertain` and a reason. `GeminiObjectDetector` implements
this interface, and other providers can replace it. Supply `ObjectSearch(planner)` to
`NavigationCommands(search=...)` to enable search. Callers must feed fresh observations,
poll deadlines, and provide trusted localization and clocks. Object goals without an
instance verifier finish unverified rather than reporting scene-only arrival success.

## Validation before enabling motion

Tests use synthetic RGB-D, deterministic embeddings, injected model responses and mock
Nav2 clients. They exercise lookalikes, movement, occlusion, stale geometry, provider
failures, cancellation, late callbacks, path bounds and search limits. They do not
measure real model accuracy or physical robot behavior.

On held-out robot recordings, label the intended physical object and whether it is
visible in each arrival frame. Include identical nearby objects, removed targets,
lighting changes, occlusions and moved targets with the old location visible. Record
false instance matches, correct matches, ambiguous/unverified outcomes, provider latency
and cost. The existing [tracking evaluator](object-evaluation.md) measures ingestion;
arrival identity decisions and search success need separate validation. Commission
search in simulation and supervised robot trials, checking blocked paths, stale sensor
input, cancellation and every configured search limit before unattended operation.
