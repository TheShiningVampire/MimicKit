# Whole-body motion imitation vs manipulation imitation: what actually changes

This note collects, in one place, the differences that matter when the MimicKit recipe
(reference clip + physics simulator + state-action policy + tracking reward + PPO) is applied to
hand-object manipulation instead of locomotion and acrobatics. It is a synthesis of reports 01 to 04;
citations are in those reports.

## 1. The objective moves from the body to the object

In whole-body imitation the reference *is* the task: if the simulated pose matches the clip the
behaviour is correct. In manipulation the reference is only a *witness* of the task; the task is the
object's trajectory. Two consequences:

- The reward must be dominated by object terms (pose, velocity) and hand-object relative terms, with
  body-pose terms acting as a style regularizer. PhysHOI, SkillMimic and InterMimic use a
  **product** of factors so that a perfect body with a dropped object scores near zero. Weighted sums
  let the policy ignore the object; every ablation that compared them found this.
- Evaluation changes: report object position/rotation error and success rate, not pose error.
  Pose error is secondary and can legitimately be larger than in locomotion because the embodiment
  differs from the human hand.

## 2. Contact is many, small, staggered, and decisive

Locomotion: one or two flat feet on a plane, periodic, with well-understood timing. A centimetre of
foot placement error does not matter. Manipulation: five to ten fingertip contacts on curved
surfaces, onsets separated by tens of milliseconds, normal forces of a few newtons, friction cones
that decide whether an object slips. A millimetre of fingertip error can change a stable grasp into
a drop.

What this forces:

- A **contact graph** in the reference (which body parts touch which object when) and a reward on
  matching it. This is the single most important addition in PhysHOI/InterMimic.
- Finer simulation: 120 to 240 Hz physics (vs 60 to 120), smaller contact offset, more solver
  iterations, convex-decomposed meshes for objects, friction coefficients of about 1 on finger pads.
- Termination on **object** failure (dropped, deviated) rather than only on the character falling.
- Isaac Gym's "ground contact" test in MimicKit uses a height mask; objects resting on a table or the
  floor will trigger it unless the threshold is lowered.

## 3. The observation must include the object and the future

A locomotion policy can be nearly blind: proprioception plus a heading suffices. A manipulation
policy must know where the object is and how it moves, expressed relative to the hand and root, and
must see a short window of the *future* reference (object and hand) so it can pre-shape the fingers
before contact. InterMimic looks at t+1 and t+16; MimicKit's `tar_obs_steps` already provides this
mechanism for the body, so the object channel is appended to it.

## 4. The data are scarcer, noisier, and mostly gated

AMASS-class body mocap is clean, large, and shipped with MimicKit. Hand-object capture suffers from
occlusion (the object hides the fingers, the fingers hide each other) and produces interpenetration
and floating contacts. Datasets with full body + hands + 6D object pose are small (GRAB: 1.3 M
frames, 10 subjects, 51 objects) and mostly behind registration forms. Consequences:

- A data-cleaning step is part of the pipeline: teacher policies that correct references
  (InterMimic), physics-based retargeting (SPIDER, OmniRetarget), or soft "scope" tracking (Dexplore).
- Augmentation tricks matter: epsilon-neighbourhood state initialization, stitched trajectory graphs,
  adaptive clip sampling (SkillMimic-V2).
- Egocentric video datasets (EgoDex 829 h, HOI4D, Ego-Exo4D) are large but give hand pose from
  vision and often no object pose, so they feed Track B only after a pose-estimation stage.

## 5. Retargeting must preserve the hand-object relation

Body retargeting maps joint angles or keypoints between two similar skeletons. Hand retargeting maps
a 20-DoF five-finger hand to a 7-DoF three-finger G1 hand or a 16-DoF Allegro. Matching joint angles
is meaningless; what must be preserved is **where the fingertips are relative to the object** and
which surfaces are in contact. Hence: keep the object fixed, express fingertip targets in the object
frame, solve for robot joints (dex-retargeting, DexPilot-style), and for whole-body clips use an
interaction mesh that ties body, object and terrain (OmniRetarget). Functional, object-centric
tracking consistently beats hand-pose tracking when the embodiments differ.

## 6. Reference state initialization and curricula need the object

RSI must set the object pose and velocity together with the character. Because grasp phases are the
hard part, several methods bias the start distribution toward pre-grasp frames or sample from a buffer
of good simulated states. Curricula that do not exist in locomotion: a virtual controller that drags
the object along its reference with decaying strength (DexMachina, ResMimic), or starting with a
kinematic object and releasing it.

## 7. The embodiment gap is larger, so "style" and "function" separate

For locomotion the humanoid is close enough to a human that tracking the pose is tracking the
function. For hands, the robot cannot reproduce the human finger configuration, so the body-pose term
is only a prior on how the arm moves while the object term carries the task. This is why adversarial
or latent priors (AMP, ASE, PULSE in Omnigrasp) remain useful: they keep the motion human-like
without forcing an impossible pose match.

## 8. What stays the same

The outer loop is unchanged: reference clip, parallel simulation, PPO, early termination, reference
state initialization, future-target observations, multiplicative or additive tracking terms. MimicKit's
engine abstraction already supports rigid objects per environment (used for static obstacles and a
dodgeball), so the extension is a new motion library with an object channel, a new environment
class, and reward/observation code, not a new framework. The design document (report 02) lists the
exact files and estimated effort.

## 9. One-table summary

| | Whole body | Manipulation |
|---|---|---|
| Success criterion | pose match | object trajectory match |
| Reward composition | weighted sum | product of body, object, relative, contact factors |
| Contacts | 1 to 2, planar, periodic | 5 to 10, curved, staggered, force-sensitive |
| Sim rate / solver | 60 to 120 Hz, default | 120 to 240 Hz, more iterations, finer meshes |
| Observation | proprioception + future body targets | plus object state, hand-object vectors, future object targets, contact flags |
| Data | AMASS, clean, shipped | GRAB/ARCTIC/OakInk2/OMOMO, noisy, gated; cleaning stage needed |
| Retargeting | skeleton to skeleton | object-anchored fingertip matching; interaction mesh for whole body |
| RSI | character state | plus object state; pre-grasp bias; buffer of good states |
| Termination | fall | fall, object deviation, object drop, contact mismatch |
| Metrics | pose error, return | object error, success rate, contact F1, then pose error |
