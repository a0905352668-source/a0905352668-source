# Screen-capture possibility review: offline design

Date: 2026-09-16

Status: approved for offline validation; not approved for production filtering

## Objective

Change the alarm meaning from "non-call phone use" to "the phone camera may be
able to capture a protected screen".

The intended policy is:

- keep a candidate when a raised phone can plausibly see a protected screen;
- keep an ambiguous raised-phone candidate as `SUSPECTED_CAPTURE` when the
  pixels do not reveal the front/back orientation;
- filter ordinary phone viewing, a flat or downward-pointing phone, a phone
  clearly pointing away from every protected screen, and a phone separated
  from the screen by a verified partition or other opaque obstacle;
- fail open to a suspected event when geometry and visual evidence conflict.

This design covers an offline experiment only. It must not change the live
selector, live calibration, event visibility, services, cameras, recording,
DeepStream, or the Mage-VL service on `192.168.104.54`.

## Current-system findings

The current pipeline already has screen polygons and several two-dimensional
zones (`near_zone`, `danger_zones`, and `ignore_zones`). It associates a phone
with a person and estimates aim from arm/wrist vectors toward a screen center.
This is not a phone optical-axis estimate, and the current allowed angle is
broad enough to admit many unrelated poses.

The calibration schema contains `occluded_enable`, but the current generated
calibrations set it to false and the active scoring path does not model a
partition between a phone and a screen.

The current Mage-VL review receives chronological tight crops of the target
person. It does not receive enough full-scene context to see the protected
screen or intervening partitions. Its prompt asks whether non-call phone use is
visible, so looking at or tapping a real phone is currently a valid keep result
even when the phone cannot photograph a screen.

## Considered approaches

### Prompt-only review

Change only the Mage-VL prompt. This is small but cannot reliably solve the
problem because the existing evidence omits screens and partitions. It is not
selected.

### Full metric 3D reconstruction

Calibrate camera intrinsics/extrinsics and reconstruct screens, partitions,
people, and phone optical axes in metric 3D. This has the strongest theoretical
model but is fragile for tiny phones in monocular surveillance video and adds
substantial measurement and maintenance work. It is not selected for the first
iteration.

### Fixed-scene 2.5D visibility plus temporal visual review

Use manually verified, camera-specific visibility relationships for the fixed
office scene and combine them with temporal phone-pose review. This directly
represents the facts that remain constant (screen locations and partitions)
while asking the visual model only about the changing evidence (phone pose and
action). This is the selected approach.

## Proposed architecture

The offline evaluator has five stages:

1. Load an immutable historical event clip and its existing detector timeline.
2. Resolve the target person, phone, protected screen, and camera-specific
   visibility relationship.
3. Apply only manually verified spatial impossibility rules, recording rather
   than hiding every intermediate decision.
4. Build chronological evidence panels containing the full scene, the target
   person at native detail, and a phone/hand detail view, then run an offline
   Mage-VL prompt concerned only with screen-capture possibility.
5. Combine geometry and temporal review into a final offline label and a
   machine-readable reason trace.

The evaluator is a separate offline entry point. It reads copied or read-only
historical inputs and writes to a new experiment directory. It does not import
or overwrite live event review state.

## Fixed-scene visibility model

Existing protected-screen polygons remain the source of screen identity. Each
camera gains an experimental sidecar with manually reviewed relationships:

- `capture_position_zones`: person/phone anchor regions from which a particular
  screen may be photographable;
- `blocked_position_zones`: regions from which that screen is definitely
  hidden by an opaque partition, wall, cabinet, or equivalent obstacle;
- `occluders`: annotated polygons with a stable identifier and the screens
  whose visibility they block;
- `unknown_position_zones`: unverified areas that must not be hard-filtered.

This is a coarse 2.5D scene model rather than image overlap alone. A partition
polygon does not suppress every phone drawn over it in the CCTV image; the
explicit position-to-screen relationship expresses which side of the
partition the person occupies and which screens are physically visible from
there.

Only reviewed `blocked_position_zones` are eligible for a hard offline
`IMPOSSIBLE_BLOCKED` result. Unlabelled or conflicting space remains unknown.

## Temporal phone-orientation review

The existing phone box is treated as a candidate location, not proof of
capture. Multiple frames are used to distinguish:

- a phone raised or held in a plausible filming posture;
- a phone being viewed or tapped with its camera directed away from the target;
- a phone resting flat or directed downward;
- a phone clearly directed away from every visible protected screen;
- insufficient detail to decide the device orientation.

The evidence panel preserves a full-scene view for geometry and obstruction
context. It also includes the native-detail person crop and a phone/hand detail
crop. Any enlarged phone tile is supplemental; the original native pixels must
remain visible so interpolation artifacts cannot become evidence.

The offline Mage-VL contract returns one decision and one constrained reason:

- `CAPTURE_POSSIBLE`
- `IMPOSSIBLE_FLAT_OR_DOWN`
- `IMPOSSIBLE_AWAY_FROM_SCREEN`
- `IMPOSSIBLE_BLOCKED`
- `NOT_PHONE_OR_NO_CAPTURE_ACTION`
- `UNCERTAIN`

The model may identify a reason, but a visual `IMPOSSIBLE_BLOCKED` result alone
is not a hard filter unless the fixed-scene sidecar independently confirms the
blockage.

## Decision policy

The offline aggregate produces one of three operator-facing outcomes:

- `KEEP_CAPTURE_POSSIBLE`: geometry is possible and temporal evidence supports
  a capture posture;
- `KEEP_SUSPECTED_CAPTURE`: a raised phone is spatially capable of seeing a
  screen but orientation is unclear, or the evidence sources conflict;
- `FILTER_CAPTURE_IMPOSSIBLE`: a verified spatial block or consistent temporal
  evidence proves the phone is flat/down, facing away, or only being viewed.

Filtering is conservative:

- uncertain evidence is never automatically filtered;
- one ambiguous frame cannot override a plausible sequence;
- a static block must be manually verified for that camera and screen;
- disagreement between geometry and Mage-VL becomes
  `KEEP_SUSPECTED_CAPTURE`, with the conflict recorded for inspection.

## Offline dataset

Build a frozen manifest from historical clips and existing review artifacts.
Sampling is stratified by camera and includes:

- confirmed or strongly suspected screen photography;
- raised phones with unclear front/back orientation;
- ordinary phone viewing or tapping;
- flat/downward phones;
- phones behind partitions or outside screen visibility;
- phones facing away from protected screens;
- non-phone detector false positives.

Every selected event receives a human ground-truth label from the policy above
and a short reason. Train/tuning and final evaluation manifests are separated
by event and source clip so adjacent frames from one incident cannot leak into
both sets. Results are reported per camera as well as in aggregate.

If the archive has too few confirmed capture-positive examples, the report
must state that limitation. Such a set can prove removal of known false
positives but cannot claim general capture recall.

## Experiment outputs

Each run writes a new, versioned directory containing:

- the immutable input manifest hash;
- the experimental calibration and prompt revisions;
- one JSONL decision trace per event;
- confusion matrices and per-camera metrics;
- a small HTML or file-based review index with representative before/after
  examples;
- runtime and queue-capacity measurements;
- no credentials, private keys, or copied production secrets.

The baseline and candidate run against the same frozen manifest. A comparison
must show which individual events changed and why, not only aggregate counts.

## Acceptance criteria for leaving offline mode

The experiment may proceed to a separate shadow-mode proposal only if:

1. Every available ground-truth capture-positive event is retained as
   `KEEP_CAPTURE_POSSIBLE` or `KEEP_SUSPECTED_CAPTURE`.
2. No `UNCERTAIN` case is converted to an automatic filter.
3. Known flat/down, blocked, away-facing, and ordinary-viewing false positives
   are materially reduced; the initial target is at least a 50% aggregate
   reduction without hiding a per-camera regression.
4. Every filter decision has an auditable reason and supporting frames.
5. The candidate is tested on every enabled camera represented in the archive.
6. Measured review throughput is sufficient for the observed event arrival
   rate on the dedicated `.54` reviewer, with the measurement and headroom
   reported rather than assumed.
7. The live release, selector, runtime configuration, event records, and public
   endpoint remain byte-for-byte or state-equivalent to their pre-experiment
   baseline.

Passing these criteria does not itself authorize deployment. Shadow mode and
production filtering require separate review and approval.

## Failure handling and auditability

Malformed sidecars, missing screen relationships, undecodable clips, Mage-VL
errors, and timeouts produce explicit experiment errors. They never become
filter decisions. Partial runs remain resumable by immutable event identifier
and experiment revision.

The comparison report retains both the current-system result and the candidate
result. This allows incorrect new decisions to be traced to the spatial model,
evidence construction, visual review, or aggregation policy without modifying
the historical source event.

## Production invariants

During this design's offline phase:

- the public web service remains `http://192.168.50.2:8767/`;
- the dedicated Mage-VL host remains `192.168.104.54`;
- no listener, route, process, watchdog, camera, recording, or live event
  record is changed;
- no live calibration or prompt revision is published;
- experiment output is kept outside the live run and dashboard directories.
