# PlaceCell development roadmap

Planning window: 18 September–18 October 2026. Status: proposed; no milestones below
are claimed complete. Review priorities together at the start of each working session.

## Direction

Build an open-source spoken/written instruction-to-navigation system that robotics
developers can reproduce, inspect and connect to their robots. The user states a
single destination or a chain of visits. A planning agent interprets the request, a
separate review agent checks the proposed sequence, and the system grounds each goal
in current visual memory, resolves ambiguity, navigates, verifies the destination and
reports each goal's progress over ROS. User prompts and mission outcomes form persistent
conversation context. Memory continues to update
as the environment changes. State-of-the-art robustness is a research objective to
measure against suitable baselines, not a claim about the current implementation.

Success means useful deployments, measured reliability, reproducible examples and
contributions from other developers. Repository size and daily commit count are not
quality measures. The first month targets a credible first release; broad community
adoption is a longer-term objective.

Keep the core robot-independent, with the existing ROS 2/Nav2 integration as the first
reference implementation. Its initial supported configuration is one robot/camera stream
with aligned RGB-D and trusted localization in a known map. Hardware availability is
still to be arranged. Camera-only localization, navigation without a prior map and
learned motion policies need separate milestones.

The long-term input interface should accept instructions in any form through suitable
adapters. Text and completed speech transcripts share the first implementation; images,
gestures and other modalities need explicit grounding adapters and their own evaluations.
Model-based intent interpretation must preserve the controller's explicit capabilities,
ordering, cancellation and completion checks.

## Starting point

Already implemented: image/caption retrieval, persistent scene and object memory,
reinforcement and retention, operator corrections, caption refinement, RGB-D object
association, optional checked approach poses, arrival verification, bounded nearby
viewpoint search, ROS 2 integration, and a recorded Gazebo printer-navigation demo.

Existing automated tests establish software behavior using controlled inputs. They do
not establish perception accuracy. Retrieval and object-recording evaluators exist;
arrival identity and end-to-end search evaluation still need work. The current system
requires a known map and external localization; Nav2 provides motion planning/control.

## Week 1: Establish a repeatable baseline

**Core changes**

- Add scenario labels and paired wins/regressions to retrieval reports, so aggregate
  improvements cannot hide failures on lookalikes or details omitted from captions.
- Record evaluation configuration, model identifiers, dataset identity and reference
  time. Keep human labels independent of model output and separate tuning/test sessions.
- Add labelled evaluation of fresh arrival outcomes: correct instance, wrong instance,
  ambiguous/unverified outcome and unavailable evidence. Report denominators explicitly.
- Reproduce the documented setup from a clean environment and fix demonstrated failures.

**Deliverable:** a versioned evaluation protocol and a reproducible baseline report,
with real recordings when available. Synthetic fixtures must be labelled as such.

**Acceptance:** commands reproduce the report; all failures remain visible; reviewers
can identify which evidence led to a selected destination and arrival decision.

**Stretch:** rosbag2-to-recording import, scoped to one documented RGB-D/pose layout.

## Week 2: Understand spoken and written destinations reliably

**Core changes**

- Extend the existing explicit-command parser with a bounded structured interpretation
  path for natural phrasing. Spoken final transcripts and typed text use the same goal
  interface. Distinguish questions, movement, clarification and cancellation explicitly.
- Use planning and plan-review agents for both single goals and ordered multi-goal
  missions. Advance only after the current step's required completion checks; never
  silently skip a failed or unverified target. Keep stop independent of model responses.
- Store scoped prompt/mission context for follow-ups and publish mission ID, step index,
  destination list, current goal and outcomes over the existing ROS status topic.
  Recover conversation history after restart without replaying movement.
- Resolve model-proposed destinations to retrieved object/place IDs and checked poses;
  model text must not invent coordinates or bypass localization/verification checks.
- Preserve multiple plausible candidates and expose their saved views for selection.
  Extend the existing numbered-choice flow instead of hiding ambiguity in one score.
- Keep the destination context through follow-up clarification and stop/cancel requests.
  Test negation, hypothetical language, repeated transcripts and provider failures.
- Compare current ranking with bounded candidate reranking on held-out recordings.
  Adopt a change only when its measured benefit justifies latency and provider cost.
- Add regressions for visually similar objects, changed viewpoints and missing targets.

**Deliverable:** equivalent spoken and typed requests reach the same grounded destination;
two similar objects prompt a useful clarification; cancellation interrupts the trip.

**Acceptance:** existing explicit commands retain their behavior; questions and negations
do not start trips; uncertain identity is reported; the selected object and fresh arrival
evidence agree under the evaluation protocol. Evaluate microphone input separately from
the current published-text simulation demo.

**Stretch:** room-scoped text queries. Relational queries such as "the printer next to
the window" require explicit spatial/context evidence and remain a later feature if
the core language-to-navigation flow is not yet reliable.

## Week 3: Improve behavior when the world changes

**Core changes**

- Evaluate the existing movement, disappearance and occlusion rules on repeated visits.
  Fix demonstrated identity switches, false disappearance decisions and stale-goal use.
- Evaluate the existing nearby search: success, false matches, travel, attempts and
  timeout behavior. Tune or improve viewpoint selection using these results.
- Expose an inspectable object history: sightings, measured positions, evidence for
  updates and reasons a destination is currently unavailable.
- Add a compact replay inspector for saved frames/crops, selected candidates and trip
  outcomes, reusing the same data as the evaluators.

**Deliverable:** repeatable moved/removed/occluded/lookalike demonstrations, with failures
and uncertain outcomes included in the report.

**Acceptance:** improvements hold on held-out sessions; fresh arrival checks are never
replaced by similarity alone; search keeps its configured distance/time/attempt bounds.

**Stretch:** design room-level search for never-observed targets. Do not silently expand
the current bounded search into unrestricted exploration.

## Week 4: Make the first release usable by other teams

**Core changes**

- Run repeatable complete navigation trials and longer replay/operation sessions;
  measure restart recovery, queue growth, storage growth and processing latency.
- Profile existing hosted and local embedding paths. If compute and suitable models
  are available, add one local object-detection/comparison adapter and document measured
  limitations; full offline operation is conditional on validation of every model path.
- Add a diagnostic command for configuration, map/frame compatibility, RGB-D inputs,
  provider capabilities and database health, with actionable errors.
- Validate package installation, document one tested robot configuration and one
  reproducible simulation route, and prepare a versioned release candidate.
- Publish reproducible comparisons and known limitations; prepare small contributor
  issues with acceptance criteria after confirming the underlying needs.

**Deliverable:** a v0.1 release candidate with reproducible setup, benchmark commands,
recorded evidence, a compatibility statement and clear supported/experimental features.

**Acceptance:** another developer can reproduce the documented example; critical checks
pass; performance claims trace to measurements. Review package publication and the final
release together. Reserve the final days for integration and failures, not new features.

## Later expansion

Advance these after the first release is reproducible; they are not first-month promises.

1. Room/topological memory and relational queries supported by explicit geometry.
2. Active exploration for unknown targets, with search budgets and coverage tracking.
3. Reference-image goals ("go to the object in this photo"), with explicit uncertainty
   when a photograph cannot distinguish physical instances.
4. Cross-camera and cross-session identity, including map-version/relocalization handling.
5. Visual localization and SLAM adapters for deployments without a prebuilt map.
6. Short-clip/event memory for motion and temporal questions, alongside retained images.
7. Additional robot and simulator adapters, followed by shared multi-robot memory.
8. Optional learned visual navigation backends, evaluated through a common goal/outcome
   interface rather than replacing the complete stack at once.

Broader visual navigation already includes substantial projects such as
[Nav2](https://github.com/ros-navigation/navigation2),
[HomeRobot](https://github.com/facebookresearch/home-robot), and
[ViNT/NoMaD](https://github.com/robodhruv/visualnav-transformer).
Interoperability and reproducible comparisons are part of PlaceCell's growth strategy.
Research claims also need comparison with dynamic memory systems such as
[DynaMem](https://dynamem.github.io/) and
[OpenBelief-Nav](https://arxiv.org/abs/2608.13923).

## Daily working agreement

No scheduled unattended changes or pushes. Work together in this task:

1. Choose one useful outcome and state its observable acceptance criteria.
2. Explain what will change and which evidence motivates it.
3. Implement a bounded change and run the relevant checks.
4. Review the behavior, measured results, limitations and diff together.
5. Commit and push a working increment during the session; carry unfinished work on a
   branch. A difficult item may take multiple sessions. Do not create cosmetic churn
   merely to maintain a daily streak.

At each weekly checkpoint, reprioritize using failures, recording availability and
feedback. Hardware motion, access to paid model services and public dataset availability
are dependencies to arrange explicitly; they are not assumed by this roadmap.

## First working increment

The user's clarified direction prioritizes [agent-planned missions](missions.md): planning
and review, sequential verified execution, persistent conversation context and goal status.
The current local increment implements this as an opt-in path; real-model and robot
validation remain necessary before making reliability claims.

Follow with the proposed retrieval evaluator extension: scenario-level summaries and
explicit caption-versus-combined recoveries/regressions. Keep it backward-compatible with
current labels, add deterministic offline fixtures, and distinguish software checks from
model/perception measurements.
