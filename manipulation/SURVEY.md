# MimicKit for manipulation learned from human demonstrations: full survey

Single-file compilation of the research package in `manipulation/` (branch `feat/manipulation-from-human-research`, 2026-10-12). Each part is also available as its own file; this document only concatenates them, demotes their headings one level, and turns cross-references into plain names.

## Contents

1. [Working plan](#1-working-plan)
2. [How manipulation differs from whole-body control](#2-how-manipulation-differs-from-whole-body-control)
3. [Literature survey: methods](#3-literature-survey-methods)
4. [Deep research with adversarial claim verification](#4-deep-research-with-adversarial-claim-verification)
5. [Open-source codebase survey](#5-open-source-codebase-survey)
6. [Dataset survey](#6-dataset-survey)
7. [MimicKit gap analysis and design](#7-mimickit-gap-analysis-and-design)

Supporting material: the explainer video `video/manipulation_explained.mp4`, the tools in `tools/`, and the sample clips in `datasets/samples/`.


---

## 1. Working plan

Status: v1, written 2026-10-12 (overnight research run). Companion documents:

- the methods survey: methods landscape with citations.
- the gap analysis and design: file-level gap analysis of this repo and the proposed `DeepMimicHOIEnv` design.
- the codebase survey: the open-source codebases closest to "MimicKit for manipulation", with reward/obs details.
- the dataset survey: human manipulation datasets, licences, access, schemas, and what was downloaded.
- the differences note: the conceptual differences, in one place.
- the deep-research report: fan-out deep research with 3-vote adversarial verification of every claim; confirms the reward and data-repair choices, flags unverifiable numbers.

### 0. Decisions made overnight (you can overrule any of these)

1. **Same learning setup as MimicKit.** State-action RL in a GPU physics simulator, PPO, a human reference clip per episode, reference state initialization, tracking reward. No behaviour cloning, no vision-language-action model. Those appear in the survey only as context.
2. **Track A first: whole-body humanoid + rigid object, on the Unitree G1 that MimicKit already has.** Reason: the HOI-Retarget (CC BY-NC-SA, ungated, 13.9k clips) and OmniRetarget (MIT, ungated, 3 h) datasets ship G1 + object clips whose 29-joint order matches MimicKit's existing `g1` character, so the first result needs no retargeting work and no SMPL-X licence.
3. **Track B second: dexterous hand (floating wrist + fingers) tracking hand+object clips**, following ManipTrans / Dexplore. This is where "manipulation" in the fine-motor sense lives, but it needs a hand asset and a retargeting step first.
4. **Reward follows PhysHOI / SkillMimic (multiplicative body x object x relative x contact)** rather than a weighted sum: SkillMimic's ablation drops success from 82.4% to 7.5% with a sum. The newer InterMimic / HumanX / InterReal line uses sums but adds interaction-specific termination and learned reward weights; phase 1 ablates both and implements object-deviation and contact-mismatch termination from the start (see report 06).
5. **Reference implementation to study: InterMimic** (MIT, Isaac Gym + Isaac Lab, SMPL-X and G1 variants). Reference for the G1 / Isaac Lab path: HDMI and ResMimic.
6. **Sample data was downloaded to** `/data/user_data/vawale/MimicKit/manip_data/` (large) and a small curated subset is committed under `manipulation/datasets/samples/`.

### 1. Goal

Extend MimicKit so that a simulated character (G1 humanoid, later a dexterous hand) learns to reproduce a *human-object interaction* clip: the body or hand motion **and** the object trajectory. Success is measured on the object, not on the pose.

Out of scope for the first three months: deformable objects, articulated objects (drawers, scissors), vision-based policies, real hardware.

### 2. How this differs from whole-body control

The full comparison is in report 05. The short version, which also drives the design:

| Aspect | Whole-body tracking (MimicKit today) | Manipulation tracking (what we add) |
|---|---|---|
| What defines success | pose matches the reference | the **object** ends up where the reference put it |
| Reward | sum of pose, velocity, root, key-body terms | product of body, object pose, hand-object relative, and contact-graph terms |
| Contacts | 1 to 2 feet on a flat floor, periodic, known timing | 5 to 10 small fingertip contacts on curved surfaces, staggered onsets, millimetre sensitivity |
| Observation | own joint state, root frame, future reference poses | plus object pose/velocity relative to root and hand, hand-to-object vectors, future object targets, contact phase |
| Reference data | AMASS-style mocap, hundreds of clean clips shipped | hand+object mocap is noisy (occlusion, penetration), smaller, mostly gated; needs cleaning or teacher correction |
| Retargeting | joint-angle or keypoint IK to the robot body | must preserve the **hand-object relation**: object is the anchor, fingertips are matched in the object frame |
| Reset (RSI) | sample a frame, set body state | also set object pose and velocity; optionally reset from a buffer of good simulated states |
| Termination | fall detection by non-foot ground contact | plus object deviation from reference (> 0.3 to 0.5 m), object dropped, contact mismatch |
| Simulation settings | 60 to 120 Hz, default contact offset, 4 solver iterations | 120 to 240 Hz, smaller contact offset, more iterations, friction on finger geoms, convex-decomposed object meshes |
| Embodiment gap | human and humanoid have the same topology | a 20-DoF human hand maps to a 7-DoF G1 Dex3 or 16-DoF Allegro; functional (object-centric) tracking beats pose tracking |
| Evaluation | pose error, episode length, return | object position/rotation error over the clip, success rate (object within threshold at the end), contact F1, pose error, energy |

### 3. Tracks

#### Track A: G1 + rigid object (start here)

- Character: existing `data/assets/g1/g1.xml` (29 DoF). Hands stay fixed stubs in phase 1; carrying uses forearms and palms, as in OmniRetarget and ResMimic.
- Data: HOI-Retarget G1 clips (parquet: root pos, root quat wxyz, 29 joints in g1.xml order, object pos/quat, palm and ankle contact flags; 30 fps; CC BY-NC-SA) converted with `manipulation/tools/convert_hoiretarget_to_mimickit.py` (done, four samples committed), and OmniRetarget `robot-object/` clips (`qpos[T, 43]` = base pos 3, base quat 4, 29 joints, object quat 4, object pos 3; 30 fps; MIT) with a sibling converter still to write.
- Env: `DeepMimicHOIEnv` from the design document, one dynamic rigid object per env.
- First milestone: one box-carry clip tracked for its full length with object error < 10 cm in > 90% of episodes.

#### Track B: dexterous hand + object

- Character: a floating-wrist hand (Inspire 12-DoF, Allegro 16-DoF, or Shadow 22-DoF; URDFs from `dex-urdf`). MimicKit's MJCF parser needs one hinge per nested body, so the hand asset is written in that style.
- Data: GRAB (registration), OakInk2 (registration), ManipTrans DexManipNet (robot-hand episodes), ARCTIC (articulated). Retarget with `dex-retargeting`.
- Env: same `DeepMimicHOIEnv` with the hand as the character; the wrist is the "root".
- First milestone: a GRAB pick-and-lift clip with the object lifted and held for the clip's duration.

#### Track C: whole body + dexterous hands (later)

- G1 + Dex3 (7 DoF per hand) or SMPL-X with 30 hand joints (needs the SMPL-X licence). InterMimic's G1 variant is the reference.

### 4. Phases

Each phase has a test that can fail early. Compute figures assume one RTX 6000 Ada / A6000 on `katef` with 4096 envs, which is what the five whole-body runs used.

#### Phase 0: ground truth and plumbing (week 1)

Tasks
- Run InterMimic's released Isaac Gym inference on one OMOMO clip to see a working HOI tracker and its numbers.
- Convert 3 OmniRetarget clips with the converter and replay them kinematically in MimicKit's `view_motion` env (character only, object drawn as a ghost).
- Write motion format v2 (`objects` block) and `HOIMotionLib`; unit tests on a synthetic clip.

Test: converted clips replay without foot skating or joint-order errors; the object ghost moves with the hands.

#### Phase 1: single clip, G1 + box (weeks 2 to 4)

Tasks
- `DeepMimicHOIEnv`: object spawn per env, object state in reset, object and relative observations, four new reward factors, object termination.
- Engine config overrides: 240 Hz sim, 4 to 8 solver iterations, `ground_contact_height` lowered so object contacts near the floor are not read as falls.
- Train DeepMimic on one carry clip. Log object error, success rate, contact F1 next to the usual diagnostics.
- Ablation: additive vs multiplicative reward; with and without hand-object relative term.

Test: > 90% success on the training clip (object within 10 cm for the whole clip), pose error no worse than 2x the body-only run.

Compute: about 6 runs x 8 to 12 h.

#### Phase 2: dataset scale and priors (weeks 5 to 8)

Tasks
- All OmniRetarget robot-object clips (about 3 h) in one dataset yaml; adaptive clip sampling (SkillMimic-V2 ATS) and epsilon-neighbourhood state init (STF) to cope with sparse clips.
- `AMPHOIEnv`: discriminator over body + object + relative features; compare DeepMimic vs AMP vs ADD on the same data.
- Track B start: hand asset, `dex-retargeting` pipeline, one GRAB clip.
- Decaying virtual object controller (DexMachina / ResMimic trick) as a curriculum for hard clips.

Test: one policy over 100+ clips with > 70% success on held-out clips of seen objects; Track B lifts and holds one object.

Compute: about 10 runs x 12 to 24 h.

#### Phase 3: robustness and transfer (weeks 9 to 12)

Tasks
- Domain randomization of object mass, friction, scale; sim-to-sim check in Newton or MuJoCo.
- Distill many clips into one goal-conditioned policy (future object trajectory as the goal), as in Omnigrasp and MaskedManipulator.
- If a G1 is available: deploy a carry policy, following HDMI's and ResMimic's deployment stacks.

Test: policy survives 20% mass and friction changes with < 15% drop in success; sim-to-sim success within 20% of Isaac Gym.

### 5. Reward and observation specification (summary)

Full equations and suggested weights are in report 02, section 4. The reward in phase 1 is

```
r = r_body * r_obj * r_rel * r_contact
r_body    = exp(-s_p * pose_err) * exp(-s_v * vel_err) * exp(-s_r * root_err)
r_obj     = exp(-s_op * ||p_o - p̂_o||²) * exp(-s_or * angle(q_o, q̂_o)²) * exp(-s_ov * ||v_o - v̂_o||²)
r_rel     = exp(-s_rel * Σ_h || R̂_oᵀ(p_h - p_o) - R̂_oᵀ(p̂_h - p̂_o) ||²)     # hand positions in the object frame
r_contact = exp(-s_c * Σ_h | c_h - ĉ_h |)                                     # contact graph match, c from contact forces
```

Observations append to MimicKit's current vector: object position and rotation relative to the root (heading frame), object linear and angular velocity, per-hand vector to the object, future object targets at the same `tar_obs_steps` as the body targets, and the reference contact flags for the next step.

### 6. Data plan

Ranked for Track A (whole body + object), from report 04: OmniRetarget (ready for G1, MIT), OMOMO (source of the above, MIT), InterMimic's processed OMOMO (ungated Google Drive), BEHAVE and GRAB (registration), ParaHome and HUMOTO for household tasks.

Ranked for Track B (hand + object): GRAB, OakInk2, ARCTIC (all registration), DexManipNet (robot-hand episodes from ManipTrans), GigaHands and EgoDex for scale (egocentric, hand pose from video).

Committed samples under `manipulation/datasets/samples/` are small (a few MB): schema examples and one or two short clips per source where the licence allows redistribution. Everything else stays in `/data/user_data/vawale/MimicKit/manip_data/` and is listed in its `MANIFEST.md`.

### 7. Compute plan on Babel

- Training: `sbatch` on `katef`, `--gres=gpu:1 --cpus-per-task=8 --mem=60G --time=1-00:00:00`, 4096 envs, logs under `logs/`. Same template as `experiments/<skill>/train.sbatch`.
- Phase 1 total: about 60 GPU hours. Phase 2: about 200 GPU hours. Checkpoint every hour because `preempt` can kill jobs.
- Rendering and conversion: CPU, `katef-cpu`.

### 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Isaac Gym contact model too coarse for fingertip grasps | high for Track B | start Track A (palms/forearms) where contact is forgiving; for Track B use 240 Hz, higher friction, and the virtual object controller curriculum |
| Reference clips physically inconsistent (penetration, float) | high | use OmniRetarget clips (already penetration-free), add InterMimic-style teacher correction in phase 2 |
| Multiplicative reward too sparse early in training | medium | warm up with additive reward for the first N iterations, or start the object kinematic (fixed to reference) and release it |
| Every engine requires identical object slots per env | certain | fixed object-slot list per dataset, park unused objects far away with an `active` flag |
| MJCF parser fuses three hinges in one body into a spherical joint | certain for hands | write finger joints as nested single-hinge bodies, validate with `char_dof_test` |
| Gated datasets block Track B | medium | request GRAB, ARCTIC, OakInk2 access now; use DexManipNet and OmniRetarget meanwhile |
| Isaac Gym is deprecated | medium | develop against MimicKit's engine abstraction so Isaac Lab and Newton work; InterMimic shows the Isaac Lab path |

### 9. Evaluation protocol

For every run, log per episode: object position error (mean and final), object rotation error, success (object within 10 cm and 30 degrees at the clip end), pose error (MPJPE on key bodies), contact F1 against reference contact labels, episode length, and energy (sum of squared torques). Report over 1024 episodes with random reference starts. Keep the whole-body runs' diagnostics so the two settings can be compared on the same plots.

### 10. Questions for you

1. Which robot hand do you want for Track B? The choice fixes the asset and the retargeting target. Default if no answer: Inspire (12 DoF, used by ManipTrans, OmniH2O and Weave).
2. Do you have or expect access to a physical G1? It decides whether phase 3 is hardware or sim-to-sim only.
3. Can you register for GRAB, ARCTIC and OakInk2 under the lab's name? They are the standard hand+object benchmarks and the forms take a day or two.


---

## 2. How manipulation differs from whole-body control

This note collects, in one place, the differences that matter when the MimicKit recipe
(reference clip + physics simulator + state-action policy + tracking reward + PPO) is applied to
hand-object manipulation instead of locomotion and acrobatics. It is a synthesis of reports 01 to 04;
citations are in those reports.

### 1. The objective moves from the body to the object

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

### 2. Contact is many, small, staggered, and decisive

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

### 3. The observation must include the object and the future

A locomotion policy can be nearly blind: proprioception plus a heading suffices. A manipulation
policy must know where the object is and how it moves, expressed relative to the hand and root, and
must see a short window of the *future* reference (object and hand) so it can pre-shape the fingers
before contact. InterMimic looks at t+1 and t+16; MimicKit's `tar_obs_steps` already provides this
mechanism for the body, so the object channel is appended to it.

### 4. The data are scarcer, noisier, and mostly gated

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

### 5. Retargeting must preserve the hand-object relation

Body retargeting maps joint angles or keypoints between two similar skeletons. Hand retargeting maps
a 20-DoF five-finger hand to a 7-DoF three-finger G1 hand or a 16-DoF Allegro. Matching joint angles
is meaningless; what must be preserved is **where the fingertips are relative to the object** and
which surfaces are in contact. Hence: keep the object fixed, express fingertip targets in the object
frame, solve for robot joints (dex-retargeting, DexPilot-style), and for whole-body clips use an
interaction mesh that ties body, object and terrain (OmniRetarget). Functional, object-centric
tracking consistently beats hand-pose tracking when the embodiments differ.

### 6. Reference state initialization and curricula need the object

RSI must set the object pose and velocity together with the character. Because grasp phases are the
hard part, several methods bias the start distribution toward pre-grasp frames or sample from a buffer
of good simulated states. Curricula that do not exist in locomotion: a virtual controller that drags
the object along its reference with decaying strength (DexMachina, ResMimic), or starting with a
kinematic object and releasing it.

### 7. The embodiment gap is larger, so "style" and "function" separate

For locomotion the humanoid is close enough to a human that tracking the pose is tracking the
function. For hands, the robot cannot reproduce the human finger configuration, so the body-pose term
is only a prior on how the arm moves while the object term carries the task. This is why adversarial
or latent priors (AMP, ASE, PULSE in Omnigrasp) remain useful: they keep the motion human-like
without forcing an impossible pose match.

### 8. What stays the same

The outer loop is unchanged: reference clip, parallel simulation, PPO, early termination, reference
state initialization, future-target observations, multiplicative or additive tracking terms. MimicKit's
engine abstraction already supports rigid objects per environment (used for static obstacles and a
dodgeball), so the extension is a new motion library with an object channel, a new environment
class, and reward/observation code, not a new framework. The design document (report 02) lists the
exact files and estimated effort.

### 9. One-table summary

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


---

## 3. Literature survey: methods

**Scope.** MimicKit (Peng et al., [arXiv:2510.13794](https://arxiv.org/abs/2510.13794), [github.com/xbpeng/MimicKit](https://github.com/xbpeng/MimicKit)) is a tracking-RL framework: reference motion → physics sim (Isaac Gym / Isaac Lab / Newton) → DeepMimic/AMP/ASE/ADD/SMP objectives. The question is which work does the same thing when the reference contains **hands + objects**. I grouped the literature into five families and, for every entry, give: method, human data, embodiment, sim, release status, and relevance. Facts I verified by fetching the paper/repo are stated plainly; anything taken from a search snippet only or from memory is marked **[unverified]**.

**Baseline fact about the local repo** (checked on disk): MimicKit already has `StaticObjectsEnv` (fixed-base rigid objects via `engine.create_obj`), a dynamic projectile in `task_dodgeball_env.py`, vault/climbing boxes in `data/assets/objects/`, and SMPL/humanoid/G1/Go2/Pi+ characters. There is no hand-articulated character, no object-motion reference, and no object/contact reward term. The `manipulation/` directory exists but is empty apart from a narration script. So the gap to close is: (1) a character with fingers, (2) an object track in the motion file, (3) object + contact reward terms, (4) RSI that resets object state alongside the character.

---

### 1. Physics-based human-object interaction (HOI) imitation for simulated characters

This is the family closest to MimicKit: a simulated SMPL-X-like humanoid (with fingers) tracks both body and object kinematics from mocap.

#### PhysHOI (2023, arXiv; Wang et al., HKUST)
[arXiv:2312.04393](https://arxiv.org/abs/2312.04393) · [github.com/wyhuai/PhysHOI](https://github.com/wyhuai/PhysHOI)
Whole-body HOI imitation without task-specific rewards. The humanoid follows the SMPL-X tree with 51×3 actuated DoF including full fingers. The key idea is the **contact graph (CG)**: a binary graph of body-part/object contact relations, and a **contact graph reward** that is *multiplied* with the kinematic rewards (body, object, relative body-object positions). The multiplicative form removes the local optima of additive rewards (e.g., tracking the body perfectly while dropping the ball). Data: the **BallPlay** dataset (8 basketball clips, SMPL-X + ball pose + contact labels, 25 fps). Sim: Isaac Gym; code builds on ASE. Code, data and checkpoints are released. **Relevance: the most direct template for a MimicKit "deepmimic_hoi" env** — it is literally DeepMimic + object track + CG reward, on an ASE codebase MimicKit's authors wrote.

#### SkillMimic (CVPR 2025 Highlight)
[arXiv:2408.15270](https://arxiv.org/abs/2408.15270) · [github.com/wyhuai/SkillMimic](https://github.com/wyhuai/SkillMimic)
Scales PhysHOI to a single policy over many basketball skills (dribble variants, layup, shoot, …), and adds a high-level controller that composes skills into long-horizon tasks (consecutive scoring). Reward is the unified multiplicative `r = r_body · r_object · r_rel · r_cg`. Datasets: **BallPlay-M** (35 min optical mocap) and **BallPlay-V** (8 skills estimated from RGB video). Sim: Isaac Gym. Released: code, pretrained low-level + high-level controllers, a subset of BallPlay-M (README says full raw release is still TODO). RSI from random reference frames (`--state_init`). **Relevance: high** — the multi-skill, skill-conditioned variant of the PhysHOI recipe, structurally identical to MimicKit's DeepMimic + skill-conditioning.

#### SkillMimic-V2 (SIGGRAPH 2025)
[arXiv:2505.02094](https://arxiv.org/abs/2505.02094) · [github.com/Ingrid789/SkillMimic-V2](https://github.com/Ingrid789/SkillMimic-V2)
Same reward as V1 (confirmed in the paper HTML: "no fundamental changes"); the contribution is robustness to **sparse and noisy demonstrations** via data augmentation of the reference set:
- **STG (Stitched Trajectory Graph)**: offline graph that discovers transitions between different skill clips, with masked intermediate states bridging large gaps — gives skill transitions from fragment clips.
- **STF (State Transition Field)**: ε-neighborhood State Initialization (ε-NSI) — reset into a noisy ball around reference states and connect each such state to a unique target, so the policy learns recovery into the manifold.
- **ATS (Adaptive Trajectory Sampling)**: softmax sampling weights `p_i ∝ exp(-λ r̄_i)` so harder clips are sampled more (fixes "chain break").
- **History encoder**: frozen, pre-trained net compressing the last k states for memory-dependent behaviours.
Sim: Isaac Gym, 2048 envs, 120 Hz PD / 60 Hz policy. Data: BallPlay-M (5 skills) + **ParaHome** (7 household manipulation tasks: book grasping, cube reorientation, etc.). Released: pretrained models; training code is still listed as TODO. **Relevance: very high** — STG/STF/ATS are exactly the augmentations you need when your manipulation reference set is a handful of mocap clips, and they are cheap to add to MimicKit's motion-sampling and RSI code.

#### InterMimic (CVPR 2025 Highlight; Xu et al., UIUC + EA)
[arXiv:2502.20390](https://arxiv.org/abs/2502.20390) · [github.com/Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic)
Universal HOI controller from "hours of imperfect mocap". Curriculum is "perfect first, then scale up": subject-specific **teacher** policies are trained to correct mocap artifacts (penetration, floating), then a **student** is distilled with the teachers acting as online experts, then RL-finetuned so it surpasses replay. Data: OMOMO (authors recommend reprocessing via their **InterAct** pipeline, [arXiv:2509.09555](https://arxiv.org/abs/2509.09555)), plus other HOI datasets. Embodiments: SMPL-X **and Unitree G1 with three-fingered hands, "directly learned from mocap without any external retargeting."** Sim: **both Isaac Gym and Isaac Lab** (Isaac Sim 5.1 / IsaacLab 2.3.1). MIT license; code, teacher checkpoints, processed mocap, multi-GPU training released. **Relevance: very high** — the only released codebase that does physics HOI tracking in Isaac Lab for both SMPL-X and G1, i.e. the same engine pair MimicKit supports.

#### Omnigrasp (NeurIPS 2024; Luo et al., CMU/Meta)
[arXiv:2407.11385](https://arxiv.org/abs/2407.11385) · [github.com/ZhengyiLuo/Omnigrasp](https://github.com/ZhengyiLuo/Omnigrasp)
Full-body SMPL-X humanoid with dexterous hands grasps >1200 objects and follows object trajectories. Key trick: act in the **PULSE** latent motion space (learned from PHC), which gives human-like motor priors and makes the grasp RL tractable with simplistic rewards (object pose tracking) and no paired body+object dataset — it only needs pre-grasp hand poses (from GRAB/OakInk/OMOMO) and object meshes. Sim: Isaac Gym. Inference code + models (Dec 2024) and training code (Aug 2025) released; data-processing scripts still pending. **Relevance: high** as the "object-centric reward + motion prior" alternative to full kinematic tracking; ASE/SMP-style priors in MimicKit could play PULSE's role.

#### MaskedMimic → MaskedManipulator (NVIDIA; SIGGRAPH Asia 2024 / 2025)
MaskedMimic [arXiv:2409.14393](https://arxiv.org/abs/2409.14393) frames control as masked motion inpainting; it lives in **ProtoMotions** ([github.com/NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)). **MaskedManipulator** [arXiv:2505.19086](https://arxiv.org/abs/2505.19086) extends it: a tracking controller trained on large-scale HOI mocap is distilled into a generative policy conditioned on sparse spatio-temporal goals for *both* body parts and the object (e.g. "object here at t=3s"). Sim: Isaac Gym/Lab via ProtoMotions [unverified which]. I could not confirm a public MaskedManipulator code release. **ProtoMotions object support (checked):** ProtoMotions3 ships **SceneLib** — scenes with one or more static or *movable* objects, each with an optional **object motion file used during motion-tracking tasks** — and supports IsaacGym, IsaacLab 3.0, Newton 1.0, MuJoCo 3 and Genesis. So ProtoMotions already has the object-track plumbing MimicKit lacks; worth reading `protomotions/components/scene_lib` when designing MimicKit's version.

#### HumanVLA (NeurIPS 2024), UniHSI (ICLR 2024), CHOIS (ECCV 2024), InterAct (2025), HOIDiNi (2025)
HumanVLA ([NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/215aeb07b5996c969c0123c3c6ee8f54-Paper-Conference.pdf)) does vision+language-directed physics-based object rearrangement (teacher-student, Isaac Gym). UniHSI does sequential contact-reaching with *static* scene objects in physics sim (no object dynamics). CHOIS and HOIDiNi ([arXiv:2506.15625](https://arxiv.org/abs/2506.15625)) are **kinematic** HOI generators (diffusion), i.e. reference-motion *sources*, not trackers. InterAct ([arXiv:2509.09555](https://arxiv.org/abs/2509.09555)) consolidates/cleans HOI mocap (OMOMO, BEHAVE, …) and is InterMimic's recommended preprocessing. **Relevance:** moderate; useful as data sources/front-ends, not as the RL core.

#### Dexterous-hand tracking (floating or arm-mounted robot hands, no body)

**DexTrack (ICLR 2025)** [arXiv:2502.09614](https://arxiv.org/abs/2502.09614) · [github.com/Meowuu7/DexTrack](https://github.com/Meowuu7/DexTrack). Generalizable neural tracking controller mapping kinematic human hand-object references to robot-hand actions; trained by an RL↔IL flywheel with homotopy optimization to grow a dataset of successful tracking demos; >10% success gain over baselines in sim and real. Hands/sim: Allegro/LEAP in Isaac Gym, GRAB/TACO references **[unverified — from memory]**.

**ManipTrans (CVPR 2025)** [arXiv:2503.21860](https://arxiv.org/abs/2503.21860) · [github.com/ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans). Two stages: a **generalist hand-trajectory imitator** (hand-only tracking, no object) pre-trained once, then a per-task **residual module** fine-tuned under interaction constraints. Produces **DexManipNet** (3.3K bimanual robot episodes, e.g. pen capping, bottle unscrewing) from OakInk-V2, FAVOR, GRAB. Hands: Inspire, Shadow, Allegro, Arti-Mano, XHand, Inspire-FTP. Sim: Isaac Gym Preview 4. Released: full code, HF checkpoints, DexManipNet; GPL-3. **Relevance: very high** — the cleanest "pretrained tracker + residual per-object" decomposition, and a ready dataset.

**DexMachina (2025)** [arXiv:2505.24853](https://arxiv.org/abs/2505.24853) · [github.com/MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina). *Functional* retargeting: track **object** state (articulated ARCTIC objects, bimanual, long-horizon) rather than hand pose. Curriculum via **virtual object controllers with decaying strength** — the object is first dragged to its reference pose, and assistance decays so the hands take over — plus contact and fingertip guidance rewards. Hands: Allegro, Inspire, Schunk, XHand and more; sim: a custom **Genesis** fork; code, hand assets, ARCTIC demos released. **Relevance: very high** for the curriculum idea (object "training wheels") — trivially expressible as an extra engine force in MimicKit.

**Dexplore (CoRL 2025, NVIDIA)** [arXiv:2509.09671](https://arxiv.org/abs/2509.09671) · [github.com/NVlabs/dexplore](https://github.com/NVlabs/dexplore). Collapses retarget-then-track into **one RL loop**: references define *adaptive spatial scopes* (soft guidance) instead of hard targets; reward keeps the policy in-scope while minimizing effort, so robot-specific strategies emerge and noise is tolerated. Then distills to a vision-based, skill-conditioned generative controller. Sim: Isaac Gym Preview 4; hands: Inspire(18), LEAP(22), Allegro(22), Shadow(30) with 6-DoF floating wrist; data: GRAB; checkpoint for Inspire released.

**SPIDER (2025, Meta FAIR)** [arXiv:2511.09484](https://arxiv.org/abs/2511.09484) · [github.com/facebookresearch/spider](https://github.com/facebookresearch/spider). Training-free **sampling-based physics retargeting** with contact guidance: IK gives kinematics, a sampling solver finds dynamically feasible trajectories; 9 embodiments (Allegro, XHand, Inspire, Ability, Schunk; G1, H1-2, Fourier N1, Booster T1), 6 datasets, 3 simulators; ~10× faster than RL; released 6,523 demos over 4 hands/191 objects. Output can be replayed open-loop or used as cleaned references for RL — i.e., a **reference-cleaning front-end for MimicKit**.

**DexterityGen (2025)** [arXiv:2502.04307](https://arxiv.org/abs/2502.04307): a generative "foundation controller" trained on sim-generated dexterous motion that turns coarse teleop commands into safe fine actions. **DexPBT (2023)**: population-based training for Allegro-Kuka regrasp/throw/reorient in Isaac Gym (lives in [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)). **DexMV (ECCV 2022)** [github.com/yzqin/dexmv-sim](https://github.com/yzqin/dexmv-sim): the original video→hand/object pose→retarget→DAPG pipeline (Adroit hand, MuJoCo/SAPIEN). These are reward-engineered or IL baselines, not tracking frameworks, but DexPBT's envs are the standard Isaac dexterous testbed.

#### 2025–2026 follow-ups (physics HOI, whole-body + hands)
- **HumanX (CoRL 2026)** [arXiv:2602.02473](https://arxiv.org/abs/2602.02473) · [github.com/wyhuai/HumanX](https://github.com/wyhuai/HumanX). Video → GVHMR → UMR retarget to G1 → **XGen** (anchor-centric HOI editor/augmenter) → **XMimic** (SkillMimic-style interaction-prioritized imitation). Nine skills (basketball, football, badminton, cargo pickup, fighting) transferred zero-shot to a real G1. Apache-2.0 + Commons Clause. Same lab as SkillMimic; this is the "SkillMimic on a real robot from video" paper.
- **InterReal (Mar 2026)** [arXiv:2603.07516](https://arxiv.org/abs/2603.07516): unified physics HOI imitation for G1 with hand-object-contact-constrained data augmentation and an **automatic reward learner** (meta-policy allocates reward weights from tracking errors); real G1. Code not confirmed.
- **Weave (Sep 2026)** [arXiv:2609.16683](https://arxiv.org/abs/2609.16683) · [HF dataset appolyn/Weave](https://huggingface.co/datasets/appolyn/Weave): contact-aware retargeting + approach-motion completion; one policy over 29 body + 12 finger joints; 92.5% / 65% success on seen/unseen; ~9k rollouts (23 h) with contact annotations released. **DexWeave** [arXiv:2609.34724](https://arxiv.org/abs/2609.34724) is a sibling.
- **DeVI (Apr 2026)** [arXiv:2604.20841](https://arxiv.org/abs/2604.20841) · [github.com/snuvclab/devi](https://github.com/snuvclab/devi): imitates **text-conditioned synthetic videos** with a hybrid reward (3D human tracking + 2D object tracking) for unseen objects.
- **DexMan (Oct 2025)** [arXiv:2510.08475](https://arxiv.org/abs/2510.08475): bimanual humanoid-hand skills from third-person real/generated video, contact-based rewards against noisy hand-object poses; TACO & OakInk-v2.
- **ResMimic (Oct 2025)** [arXiv:2510.05070](https://arxiv.org/abs/2510.05070): residual policy on top of a General Motion Tracker (GMT) for G1 loco-manipulation, sim + real.
- **MaskedManipulator**, **HAIC (RSS 2026)** [github.com/ldt29/HAIC](https://github.com/ldt29/HAIC) (underactuated objects — carts, skateboard — via proprioceptive dynamics predictor), **Sugar** [arXiv:2605.20373](https://arxiv.org/abs/2605.20373), **GenHOI** [arXiv:2606.12995](https://arxiv.org/abs/2606.12995), **Progressive Kinematic-Dynamic Alignment** [arXiv:2511.10987](https://arxiv.org/abs/2511.10987), **DexForge** [arXiv:2610.06331](https://arxiv.org/abs/2610.06331), **AdvDex** [arXiv:2608.14028](https://arxiv.org/abs/2608.14028) (AMP-style adversarial learning for dexterous hands — closest to "AMP for manipulation"), **DemoBot** [arXiv:2601.01651](https://arxiv.org/abs/2601.01651). I only read abstracts/snippets of these.

---

### 2. Humanoid / robot manipulation from human data (real robots, mostly IL)

These are less "MimicKit-like" (most are behaviour cloning on real hardware) but define the data sources and retargeting conventions.

| Work | Year/venue | Human data | Embodiment | Learning | Code |
|---|---|---|---|---|---|
| **OmniH2O** [arXiv:2406.08858](https://arxiv.org/abs/2406.08858) | CoRL 2024 | AMASS + VR/RGB teleop; OmniH2O-6 dataset | Unitree H1 + Inspire hands | RL tracker (PHC-style, Isaac Gym) + IL on teleop | [LeCAR-Lab/human2humanoid](https://github.com/LeCAR-Lab/human2humanoid) |
| **HumanPlus** [arXiv:2406.10454](https://arxiv.org/abs/2406.10454) | CoRL 2024 | 40 h human motion; RGB shadowing | H1 + dexterous hands | RL low-level in sim + IL (HIT) | released [unverified URL] |
| **OKAMI** [arXiv:2410.11792](https://arxiv.org/abs/2410.11792) | CoRL 2024 | single RGB-D video | Fourier GR1 | **object-aware retargeting** (body & hand separately, warped to object location) + IL | [ut-austin-rpl.github.io/OKAMI](https://ut-austin-rpl.github.io/OKAMI/) |
| **HOMIE** [arXiv:2502.13013](https://arxiv.org/abs/2502.13013) | 2025 | exoskeleton cockpit teleop | G1 | RL lower body + teleop upper | released [unverified] |
| **Human2Sim2Robot** [arXiv:2504.12609](https://arxiv.org/abs/2504.12609) | 2025 | **one** RGB-D human video (SAM2 + FoundationPose) | arm + Allegro [unverified] | sim RL with **object-centric reward** + pre-manipulation hand pose init; sim-to-real | [unverified] |
| **EgoMimic** [arXiv:2410.24221](https://arxiv.org/abs/2410.24221) | CoRL 2024 | Aria glasses egocentric | bimanual arms | co-trained IL | open source |
| **MimicPlay** [arXiv:2302.12422](https://arxiv.org/abs/2302.12422) | CoRL 2023 | human play video | Franka | hierarchical IL | [j96w/MimicPlay](https://github.com/j96w/MimicPlay) |
| **DexCap** [arXiv:2403.07788](https://arxiv.org/abs/2403.07788) | 2024 | glove mocap + SLAM wrist | LEAP hands | DexIL | [dex-cap.github.io](https://dex-cap.github.io/) |
| **DexWild** [arXiv:2505.07813](https://arxiv.org/abs/2505.07813) | 2025 | 9,290 in-the-wild human demos, 93 envs | LEAP/xArm | co-trained IL; 68.5% unseen-env | [dexwild/dexwild-training](https://github.com/dexwild/dexwild-training) |
| **EgoDex (Apple)** [arXiv:2505.11709](https://arxiv.org/abs/2505.11709) | 2025 | 829 h Vision Pro ego video, 3D hand/upper-body pose, 194 tasks | — (dataset) | — | [apple/ml-egodex](https://github.com/apple/ml-egodex) |
| **Being-H0** [arXiv:2507.15597](https://arxiv.org/abs/2507.15597) | 2025 | large-scale human video, part-level hand-motion tokens | dexterous VLA | "physical instruction tuning" | [BeingBeyond/Being-H0](https://github.com/BeingBeyond/Being-H0) |
| **GR00T N1/N1.7** [arXiv:2503.14734](https://arxiv.org/abs/2503.14734) | 2025 | data pyramid: ego human video → synthetic → robot; N1.7 reportedly 20K h EgoScale | cross-embodiment VLA | flow-matching VLA | [NVIDIA/Isaac-GR00T](https://github.com/Nvidia/Isaac-GR00T) |
| **LAPA** [arXiv:2410.11758](https://arxiv.org/abs/2410.11758) | ICLR 2025 | action-free video (VQ latent actions) | VLA | latent-action pretrain → finetune | [latentactionpretraining.github.io](https://latentactionpretraining.github.io/) |
| **UniVLA** (two papers share the name): task-centric latent actions [arXiv:2505.06111](https://arxiv.org/abs/2505.06111); unified token VLA [arXiv:2506.19850](https://arxiv.org/abs/2506.19850) (ICLR 2026) | 2025/26 | video | VLA | — | — |
| **Gen2Act** [arXiv:2409.16283](https://arxiv.org/abs/2409.16283) | CoRL 2025 | zero-shot generated human video | arm | video-conditioned policy | — |
| **UniDex** [arXiv:2603.22264](https://arxiv.org/abs/2603.22264) | CVPR 2026 | ego video → human-in-the-loop contact-preserving retarget, 50k+ trajectories | **8 hands (6–24 DoF)**, FAAS unified action space | VLA IL | [unverified] |
| **Humanoid Policy ~ Human Policy (HAT/PH²D)** [arXiv:2503.13441](https://arxiv.org/abs/2503.13441) | 2025 | VR-captured human hand/finger data aligned to robot tasks | G1-class humanoid | unified human/robot state-action space, differentiable retarget | [RogerQi/human-policy](https://github.com/RogerQi/human-policy) |
| **GigaHands** [arXiv:2412.04244](https://arxiv.org/abs/2412.04244) | CVPR 2025 | 34 h, 56 subjects, 417 objects, 14k clips, full hand+object 3D | dataset | — | [brown-ivl/gigahands](https://github.com/brown-ivl/gigahands) |
| **Humanoid Everyday** [arXiv:2510.08807](https://arxiv.org/abs/2510.08807) | 2025 | robot teleop (G1/H1), 260 tasks | dataset | — | [HF](https://huggingface.co/datasets/USC-GVL/humanoid-everyday) |
| Phantom (CoRL 2025), Masquerade (ICRA 2026, [arXiv:2508.09976](https://arxiv.org/abs/2508.09976)), EgoZero ([arXiv:2505.20290](https://arxiv.org/abs/2505.20290)), R+X, Real2Render2Real, ZeroMimic (ICRA 2025), VidBot (CVPR 2025), Motion Tracks, Vid2Sim, Video2Policy | 2024–26 | in-the-wild video, edited/inpainted into robot embodiment or distilled to keypoint tracks | arms | IL / zero-shot | partial [titles verified; details from memory] |

**"HOP"**: I could not find a paper by this acronym in the humanoid-object space; the closest matches were HAT/PH²D above and HAIC. Treat "HOP" as **unresolved**.

**Relevance of family 2 to MimicKit:** low for the algorithms (BC/VLA), high for *data* — EgoDex, GigaHands, OakInk2, ARCTIC, TACO, ParaHome, OMOMO are the manipulation analogue of AMASS, and OKAMI/UniDex/HAT define object-aware retargeting conventions you can reuse.

---

### 3. Retargeting

#### Human hand → robot hand
- **dex-retargeting / AnyTeleop (RSS 2023)** [github.com/dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting): optimization-based position / vector / **DexPilot-style** retargeting to many hands (uses [dex-urdf](https://github.com/dexsuite/dex-urdf) assets). The de-facto standard; both ManipTrans and Dexplore start from it. **DexPilot (ICRA 2020)**: task-space fingertip-vector objectives with a pinch-closing prior. **GeoRT** [arXiv:2503.07541](https://arxiv.org/abs/2503.07541): ultrafast neural geometric retargeting. **dexi** ([yuxiang-gao/dexi](https://github.com/yuxiang-gao/dexi)) is a Rust reimplementation.
- **Contact-preserving kinematic retargeting**: Lakshmipathy, Hodgins, Pollard, *Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations*, ACM TOG 44(2), 2025, [arXiv:2402.04820](https://arxiv.org/abs/2402.04820) — shape-matching transfers **contact areas** across pre-grasp/pickup/in-hand/release; **TopoRetarget (2026)** [arXiv:2606.16272](https://arxiv.org/abs/2606.16272) — sparse hand-object interaction graph + Laplacian deformation, +40.6 pp on pen-spin RL; **REGRIND (2026)** [arXiv:2607.11874](https://arxiv.org/abs/2607.11874), [yunhaifeng.com/REGRIND](https://yunhaifeng.com/REGRIND) — interaction-mesh retargeting that preserves hand-object spatial/contact relations, then **residual RL tracking object-centric keypoints**, zero-shot to two real hands (scissors, screwdriver); **GenHand** (Nature Rob. 2026 [unverified]). Physics-informed: **SPIDER**, **DexForge**, **C2Dex** [arXiv:2608.07045](https://arxiv.org/abs/2608.07045).

#### Human body → humanoid (used by tracking frameworks)
- **PHC** (Luo et al., ICCV 2023): SMPL→humanoid via shape-fitted joint IK; MimicKit's own `smpl`/G1 pipelines and InterMimic/Omnigrasp build on it.
- **GMR** [arXiv:2510.02252](https://arxiv.org/abs/2510.02252), [github.com/YanjieZe/GMR](https://github.com/YanjieZe/GMR), ICRA 2026: real-time CPU IK retargeting to many humanoids; "Retargeting Matters" shows it beats PHC and ProtoMotions retargeters when tracked with BeyondMimic. Retargeter for **TWIST/TWIST2** ([amazon-far/TWIST2](https://github.com/amazon-far/TWIST2)); exports CSV for BeyondMimic.
- **OmniRetarget** [arXiv:2509.26633](https://arxiv.org/abs/2509.26633), [omniretarget.github.io](https://omniretarget.github.io/), [HF dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset): **interaction mesh** (Delaunay tetrahedralization over human/robot keypoints + object/terrain samples) with constrained optimization so human-object/terrain relations are preserved and penetration/foot-skate removed; G1 policies trained with 5 reward terms, no curriculum, up to 30 s loco-manipulation. Reported ICRA 2026 best manipulation & locomotion paper [from snippet]. This is the whole-body analogue of contact-preserving hand retargeting and the right tool to produce G1+object reference motions for MimicKit.
- Trackers that consume these: **BeyondMimic** [arXiv:2508.08241](https://arxiv.org/abs/2508.08241), [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) (Isaac Lab; mjlab port), **KungfuBot/PBHC** [arXiv:2506.12851](https://arxiv.org/abs/2506.12851), [TeleHuman/PBHC](https://github.com/TeleHuman/PBHC), **ASAP** [arXiv:2502.01143 — id from memory], **GMT** [arXiv:2506.14770 — id from memory], **TWIST** [arXiv:2505.02833 — id from memory]. None of these track objects; ResMimic, OmniRetarget-RL, HumanX, Weave, InterMimic-G1 are the ones that do.

---

### 4. Key technical ingredients for physics HOI tracking (what to put in the reward/env)

1. **Object-pose tracking reward.** Every physics HOI paper has `r_obj = exp(-k·‖p_obj − p̂_obj‖) · exp(-k·rot_err)` (plus object velocity in PhysHOI/SkillMimic). Human2Sim2Robot and DexMachina argue the object trajectory is the *embodiment-agnostic* task spec, so weight it higher than body pose when the hand morphology differs.
2. **Relative (hand-to-object) terms.** PhysHOI/SkillMimic `r_rel` tracks body-keypoint-minus-object vectors; Omnigrasp/REGRIND use object-centric keypoints. This is what fixes "arm is right but ball is 5 cm off".
3. **Contact graph / contact rewards.** PhysHOI's binary CG with multiplicative reward; InterMimic's contact-aware terms; DexMachina/DexMan contact guidance; InterReal's contact-constrained augmentation. Multiplicative composition (`Π r_i`) rather than weighted sum is a consistent finding (PhysHOI, SkillMimic, V2).
4. **Kinematic vs dynamic tracking.** Pure kinematic references are physically inconsistent (penetration, float). Fixes: teacher-correction (InterMimic), sampling-based physics retargeting (SPIDER), single-loop soft-scope tracking (Dexplore), residual on a pretrained tracker (ManipTrans, ResMimic, REGRIND, Video2Sim2Real).
5. **Curricula.** DexMachina's decaying virtual object controller; ManipTrans' hand-only pretrain → residual; InterMimic's "perfect first then scale"; SkillMimic-V2's ATS difficulty-weighted clip sampling.
6. **Noisy/sparse data.** SkillMimic-V2 STG (stitch clips) + STF/ε-NSI (reset in a noise ball, learn recovery) + history encoder; HumanX XGen augmentation; InterReal augmentation + learned reward weights.
7. **Reference state initialization with objects.** All tracking methods reset *both* character and object to a reference frame (SkillMimic `--state_init`). Human2Sim2Robot resets to the *pre-manipulation* hand pose; Omnigrasp samples pre-grasp poses. MimicKit's RSI needs an object-state channel in the motion file.
8. **Sim-to-real for contacts.** REGRIND/HumanX/Weave rely on system ID + domain randomization of object mass/friction/size (OmniRetarget: "simple DR shared by all tasks"); SPIDER augments with external payloads/perturbations. Finger PD gains and contact stiffness are the usual culprits; Dexplore's "minimize control effort" regularizer helps.

---

### 5. Simulators / benchmarks with objects + dexterous hands + large-scale RL

| Platform | Objects + hands | GPU-parallel RL | Notes |
|---|---|---|---|
| **Isaac Gym Preview 4** | yes (IsaacGymEnvs: ShadowHand, AllegroKuka/DexPBT) | yes | Deprecated but still the engine of PhysHOI/SkillMimic/InterMimic/ManipTrans/Dexplore/Omnigrasp. |
| **Isaac Lab** ([arXiv:2511.04831](https://arxiv.org/abs/2511.04831), [envs](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)) | yes: Shadow in-hand, **Dexsuite Kuka-Allegro Lift/Reorient** ([PR #3378](https://github.com/isaac-sim/IsaacLab/pull/3378)), G1 loco-manipulation Mimic env (v2.3) | yes | **3.0 beta integrates Newton** alongside PhysX. InterMimic already runs here. |
| **Newton** (Linux Foundation; Warp + MuJoCo-Warp) | engine only | yes | MimicKit and ProtoMotions3 support it; contact fidelity for fingers in Newton is **unverified** by me. |
| **ProtoMotions3** | SceneLib: movable objects with motion files | yes | Multi-engine; has MaskedMimic, AMP, ADD; closest "sibling" framework to MimicKit with object tracking already wired. |
| **ManiSkill3** ([arXiv:2410.00425](https://arxiv.org/abs/2410.00425)) | 12 domains incl. dexterous hands and humanoids | yes (SAPIEN/PhysX, 30k+ FPS) | Good for task benchmarks; not a tracking framework. |
| **MuJoCo Playground / MJX** ([arXiv:2502.08844](https://arxiv.org/abs/2502.08844)) | LEAP-hand reorientation | yes | Sim-to-real oriented. |
| **Genesis** | custom fork used by DexMachina | yes | Articulated-object support good; ecosystem smaller. |
| **HumanoidBench** | H1 + Shadow hands, MuJoCo | CPU | Task suite, no human data. |
| **RoboCasa / RoboCasa365**, **BiGym** | kitchen / bimanual mobile | **no** reliable GPU RL (per ManiSkill3 paper) | Demo-driven IL benchmarks. |
| Newer: **DexJoCo** [arXiv:2605.16257](https://arxiv.org/abs/2605.16257), **DexVerse** [arXiv:2607.08751](https://arxiv.org/abs/2607.08751) | multi-hand MuJoCo benchmarks | varies | Only abstracts read. |

---

### 6. Ranked table: 15 works closest to "MimicKit for manipulation"

| # | Work | Why it ranks here | Sim | Code/data |
|---|---|---|---|---|
| 1 | **InterMimic** | Whole-body + fingers + object tracking; SMPL-X *and* G1; Isaac Gym **and Isaac Lab**; MIT; data pipeline (InterAct) | IG + IL | full |
| 2 | **SkillMimic / SkillMimic-V2** | DeepMimic-shaped HOI reward (`r_b·r_o·r_rel·r_cg`), multi-skill, STG/STF/ATS for sparse data | IG | code (V1), models (V2), data partial |
| 3 | **PhysHOI** | Simplest complete HOI tracker; contact-graph reward; ASE codebase | IG | full |
| 4 | **ManipTrans** | Pretrained hand tracker + residual; 6 hands; DexManipNet | IG | full (GPL-3) |
| 5 | **Dexplore** | Single-loop soft tracking; 4 hands; NVIDIA code | IG | full |
| 6 | **DexMachina** | Object-state tracking with decaying virtual controller; articulated bimanual | Genesis | full |
| 7 | **Omnigrasp** | Object-centric reward on top of a motion prior (PULSE ≈ ASE/SMP role) | IG | mostly |
| 8 | **HumanX** | Video→G1 interaction skills, SkillMimic lineage, real robot | [unverified] | code+examples |
| 9 | **ProtoMotions3 (SceneLib / MaskedManipulator)** | Only multi-engine framework with object motion files; same engine set as MimicKit | IG/IL/Newton/MuJoCo | framework yes; MaskedManipulator unclear |
| 10 | **DexTrack** | Generalizable hand-object tracking controller, RL↔IL flywheel, real | IG [unv.] | full |
| 11 | **SPIDER** | Physics-feasible reference generation, 9 embodiments, 6.5k demos | 3 sims | full |
| 12 | **OmniRetarget** | Interaction-preserving whole-body retargeting; G1 + objects dataset | IL [unv.] | retarget code + data |
| 13 | **REGRIND** | Contact-preserving retarget + residual RL + real transfer, single demo | [unv.] | code |
| 14 | **Weave / InterReal / ResMimic** | 2025–26 G1 loco-manipulation trackers with hands | [unv.] | data (Weave) |
| 15 | **Human2Sim2Robot** | Object-centric reward + pre-manip init from one RGB-D video | [unv.] | [unv.] |

---

### 7. Open problems

1. **Reference quality is the bottleneck, not the RL.** Mocap hand-object data is noisy (penetration, missing contacts); every strong 2025–26 method is mostly a data-cleaning/curriculum story (InterMimic teachers, SPIDER, Dexplore, V2's STF). No agreed-upon "AMASS-for-HOI" with physically consistent references exists; InterAct, OmniRetarget-Dataset, DexManipNet, SPIDER's 6.5k and Weave's 9k rollouts are partial answers.
2. **Embodiment gap for hands** is far larger than for bodies: 5-finger 20-DoF human → 3-finger G1 hand or 16-DoF Allegro. Functional (object-centric) tracking beats hand-pose tracking in every comparison (DexMachina, Human2Sim2Robot, REGRIND), but object-only rewards lose style/naturalness — the body-prior trick (Omnigrasp/PULSE) has no mature hand equivalent.
3. **Contact-rich sim-to-real**: GPU-sim contact models (PhysX, Newton/MuJoCo-Warp) still disagree with reality on friction/compliance; tactile sensing is nearly absent from tracking frameworks (Dex-X [arXiv:2609.07747](https://arxiv.org/abs/2609.07747) is an early exception).
4. **Articulated and deformable objects**: only DexMachina (ARCTIC) and HAIC (underactuated carts) go beyond rigid bodies.
5. **Long-horizon composition**: SkillMimic's HLC and MaskedManipulator's goal conditioning are the only released multi-skill HOI controllers; nothing comparable to ASE/MaskedMimic latent spaces exists for hands.
6. **Adversarial (AMP-style) objectives for manipulation** are barely explored — AdvDex is the only 2026 example — so MimicKit's AMP/ADD/SMP code has an open niche.
7. **No multi-engine HOI tracking codebase**: InterMimic covers Isaac Gym + Lab; nobody has Newton/MuJoCo-Warp HOI tracking released. That is exactly the slot a MimicKit extension would fill.

---

### 8. Recommendations: what to replicate first

**(1) PhysHOI → SkillMimic reward inside MimicKit (first milestone, 2–4 weeks).**
Rationale: it is DeepMimic plus three reward factors and an object channel, on a codebase (ASE) whose structure MimicKit already mirrors, with released SMPL-X-with-fingers assets and the BallPlay data. Concretely: add a `DeepMimicHOIEnv` deriving from `DeepMimicEnv` + `StaticObjectsEnv` logic but with `fix_root=False` and an object track in the motion file; implement `r = r_body · r_obj · r_rel · r_cg`; extend RSI to reset object pose/velocity. Validate on one BallPlay clip, then ParaHome household clips (book grasp, cube reorient) which are the actual "manipulation" targets. Add SkillMimic-V2's ε-NSI and ATS next — both are ~50-line changes to motion sampling. Cite: PhysHOI [arXiv:2312.04393](https://arxiv.org/abs/2312.04393), SkillMimic [arXiv:2408.15270](https://arxiv.org/abs/2408.15270), V2 [arXiv:2505.02094](https://arxiv.org/abs/2505.02094).

**(2) InterMimic as the reference implementation for Isaac Lab + G1-with-hands.**
Rationale: it is the only released system doing exactly your target (whole-body HOI tracking, SMPL-X and G1, Isaac Lab 2.3), MIT-licensed, with processed OMOMO data and the InterAct cleaning pipeline. Run it as-is first to get a working G1+object tracking baseline and reference numbers; then port its teacher-student + contact reward into MimicKit's Isaac Lab engine so you can compare DeepMimic/AMP/ADD objectives on the same data. Its retargeting-free G1 result ("directly learned from mocap") is worth reproducing before investing in OmniRetarget. Cite: [arXiv:2502.20390](https://arxiv.org/abs/2502.20390), [github.com/Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic).

**(3) ManipTrans (or Dexplore) for the hand-only / robot-hand branch.**
Rationale: if the goal includes real dexterous hands rather than SMPL-X fingers, ManipTrans gives a pretrained hand-trajectory imitator, six hand URDFs, DexManipNet (3.3k bimanual episodes from OakInk-V2/FAVOR/GRAB) and the "residual per task" recipe; Dexplore is the NVIDIA alternative with cleaner soft-scope tracking and four hands on GRAB. Either can be re-implemented as a MimicKit "character" that is a floating-wrist hand (Dexplore's 6-DoF floating base is the trick) tracking a hand+object reference — which also makes DexMachina's decaying virtual-object-controller curriculum a one-line engine force. Cite: ManipTrans [arXiv:2503.21860](https://arxiv.org/abs/2503.21860), Dexplore [arXiv:2509.09671](https://arxiv.org/abs/2509.09671), DexMachina [arXiv:2505.24853](https://arxiv.org/abs/2505.24853).

Supporting tools to adopt rather than build: `dex-retargeting` for hand IK, GMR/OmniRetarget for body+object retargeting to G1, SPIDER for physics-cleaning references, and ProtoMotions' SceneLib as the design reference for object motion files.

**Verification note.** Items marked [unverified] were taken from search snippets or memory and should be checked before citing; everything else was confirmed from the arXiv abstract, project page or GitHub README during this survey. I did not run any of the released code.


---

## 4. Deep research with adversarial claim verification

Generated 2026-10-12 by a fan-out research workflow: 5 search angles, top sources fetched, every extracted claim put to a 3-vote adversarial verification (2 of 3 refutes kill a claim), then synthesized. Source search and claim extraction ran on Claude Fable 5.1, verification votes on Sonnet, synthesis on Opus. Confidence labels come from vote unanimity and source quality. Companion to reports 01 to 05; see the comparison note at the end.

### Question

Physics-based human-object interaction (HOI) imitation with reinforcement learning: simulated humanoids or robot hands tracking human mocap/video clips that include objects (PhysHOI, SkillMimic, InterMimic, Omnigrasp, ManipTrans, DexMachina, Dexplore, HDMI, ResMimic, 2023-2026). Focus on: (1) reward and observation design (object-centric terms, contact-graph rewards, multiplicative vs additive composition, future reference windows); (2) handling noisy or sparse reference data (teacher correction, physics-based retargeting, state-initialization and trajectory-sampling tricks); (3) curricula such as virtual object controllers and residual policies; (4) simulators, training cost, and reported success rates; (5) sim-to-real results on humanoids (Unitree G1) and dexterous hands, and the main open problems. Context: the findings will guide extending the MimicKit motion-imitation framework (DeepMimic/AMP in Isaac Gym / Isaac Lab / Newton) from whole-body control to manipulation.

### Executive summary

Across 2023-2026, physics-based HOI imitation has converged on a small, reusable recipe: (1) an object-centric observation (humanoid proprioception + contact forces + full object state + joint-to-object vectors) and a reward built from exp(-lambda*error) terms that includes explicit object-tracking and contact-graph terms, with the PhysHOI/SkillMimic line composing those terms multiplicatively (SkillMimic's own ablation shows additive composition collapsing BallPlay-V success from 82.4% to 7.5%) while the newer InterMimic/HumanX/InterReal line keeps an additive sum and buys back interaction fidelity with other machinery (interaction-specific early termination, or a learned meta-policy that multiplicatively reweights the reward from tracking-error trends); (2) noisy or sparse references are not trusted as ground truth but repaired - subject-specific RL teachers mimic-retarget-refine imperfect MoCap and are distilled into one student (InterMimic's "perfect first, then scale up"), monocular video is replaced by physics-synthesized, force-closure-refined object trajectories (HumanX), MoCap is reduced to adaptive spatial scopes used as soft guidance inside a single retarget+track RL loop (Dexplore), and the reference set is augmented (stitched masked bridging trajectories and epsilon-neighborhood initialization in SkillMimic-V2, contact-constrained object-offset grids in InterReal, object-geometry/trajectory variation in HumanX); (3) the curricula that matter in practice are teacher-student distillation with a DAgger-to-PPO schedule, reward-weight meta-learning, reward-driven adaptive clip sampling, and initialization-buffer tricks (PSI) rather than the virtual-object-controller or residual-correction curricula implied by the question - Dexplore explicitly deletes the residual-correction stage; and (4) everything still trains in Isaac Gym with PPO, with sim-to-real demonstrated on a Unitree G1 only by self-reported preprint results (InterReal: 96.41% box-picking, 87.45% box-pushing, sim-to-sim in MuJoCo, FoundationPose for object pose) and no independently replicated numbers, no wall-clock/compute costs, and no surviving evidence at all on Omnigrasp, ManipTrans, DexMachina, HDMI or ResMimic.

### Findings

#### 1. Multiplicative reward composition is the defining design choice of the PhysHOI/SkillMimic line, and its removal is catastrophic rather than cosmetic: total reward is a product of body-motion, object-motion, relative/interaction-graph and contact-graph factors, each an exp(-lambda*error) term.

**Confidence:** high · **Votes:** 3-0 on each of the four constituent claims

PhysHOI: r_t = r^b * r^o * r^ig * r^cg, with r^b itself r^p*r^r*r^pv*r^rv and per-dataset lambdas (BallPlay lambda_p=50, lambda_r=20, lambda_op=1, lambda_ig=20, contact-graph 5/5/5). SkillMimic: r_t = r^b * r^o * r^rel * r^reg * r^cg, with the authors justifying multiplication over summation as forcing balanced learning and avoiding local optima. SkillMimic-V2 inherits the same product (r^b*r^o*r^rel*r^cg) and explicitly contrasts it with an additive DeepMimic-style baseline. The ablation is the load-bearing evidence: on BallPlay-V, replacing the product with a sum drops success 82.4% -> 7.5% and explodes object position error 82.9mm -> 1662.5mm, while removing just the contact-graph reward drops success to 13.6% and raises contact error 0.087 -> 0.412. For MimicKit, this is the single cheapest high-value change: the existing DeepMimic-style additive tracking reward should be refactored into a product of exp-kernel factors before any object term is added, since an added object factor in a sum can be ignored by the policy. Merged from claims 0, 2, 3, 8.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/abs/2505.02094>

#### 2. Additive composition is still used by the newest systems, but only when paired with extra machinery that forces the policy to care about the interaction - interaction-specific early termination, or a learned meta-policy that multiplicatively reallocates reward weights.

**Confidence:** high · **Votes:** 3-0 on claims 14 and 11

HumanX's privileged teacher uses r_t = r^body + r^obj + r^rel + r^c + r^reg with sub-rewards following a shared gamma*exp(-lambda*e) kernel template and a per-key-body contact-graph term exp(-sum_j lambda_j * e^cg[j]); the paper's own motivation for its Interaction Termination trick is that additive rewards let the policy ignore interaction-specific learning in favour of body-motion imitation. InterReal keeps a per-term exponential reward but replaces hand-tuned weighting with a meta-policy, Theta' = Theta_0 * sigma(t) * mu_psi(Theta_t | u_t), driven by the change in joint-position, object-position and link-position tracking error over a window Delta_t, on top of an interaction-graph contact reward r^ig = exp(-theta^ig * e^ig). InterMimic likewise maps each cost to exp(-lambda*E) and sums. Practical reading for MimicKit: additive is viable but then you owe the system either an interaction-gated termination condition or automated weight adaptation; multiplicative is the lower-engineering path. Merged from claims 14, 11, and the reward parts of 6/18.

Sources: <https://arxiv.org/html/2602.02473v1>, <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2502.20390>

#### 3. Observations are object-centric in a consistent way across the field: humanoid proprioception plus contact/force signals plus the object's full 6-DoF state and velocities, plus explicit relational features (joint-to-object vectors, interaction graph) and a binary contact graph that is also what the contact reward supervises.

**Confidence:** high · **Votes:** 3-0 on each constituent claim

PhysHOI: s_t = {g_t, h_hat_{t+1}} where g_t is global root height, local body position/rotation/linear+angular velocity, net contact forces on contact bodies, and object local position/rotation/velocities. SkillMimic: s_t = {o^prop, o^f, o^obj} = proprioception + fingertip contact forces + object position/rotation/velocity/angular velocity, with r^cg = exp(-sum_j lambda^cg[j] * e^cg[j]) penalizing simulated-vs-reference contact mismatch. InterMimic adds vectors from joints to nearest object points and contact markers to the observation, distance-weighted joint error terms (weights inversely proportional to joint-object distance), object position/rotation costs E^o_p = ||p_hat^o - p^o|| and E^o_theta, and a three-level contact reward (promotion / penalty / neutral) whose levels are discretized by distance thresholds specifically to tolerate MoCap contact error. HumanX supervises a binary per-key-body contact vector. Concretely for MimicKit: the observation extension is proprioception + object state + joint-to-object vectors + per-contact-body force/contact bits, and the contact term should be tri-level rather than binary if the reference contacts come from noisy retargeting. Merged from claims 1, 4, 6, 18, and the contact-graph part of 14.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/html/2602.02473v1>

#### 4. Practice on future reference windows is split and the split tracks task difficulty: single-frame lookahead suffices for a tightly tracked single skill, while multi-frame sparse keyframe sets are used once one policy must cover many interactions, and the student gets a denser set than the teacher.

**Confidence:** high · **Votes:** 3-0 on each constituent claim

PhysHOI feeds only the single next reference frame h_hat_{t+1} (human motion, object motion, interaction graph, contact graph) - explicitly not a multi-frame window. InterMimic instead sets keyframe offsets K = {1,16} for the 17 subject-specific teachers and K = {1,2,4,16} for the distilled student, i.e. a sparse, log-spaced horizon rather than a dense window, with the denser set reserved for the policy that must generalize across subjects and objects. Design implication: start MimicKit's HOI observation with next-frame-only parity with PhysHOI, and move to {1,2,4,16}-style sparse keyframes when moving from per-clip policies to one multi-clip policy. Merged from claims 1, 6, 18.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/abs/2502.20390>

#### 5. The dominant answer to noisy MoCap is to let the simulator repair the data: train per-subject RL teachers that mimic, retarget and refine imperfect references, then distill them into one student that is supervised both by online teacher actions and by the teachers' physics-corrected rollouts, and finally RL-fine-tune the student.

**Confidence:** high · **Votes:** 3-0 on claims 5, 17, 12

InterMimic states the strategy as 'perfect first, then scale up': 17 subject-specific MLP teachers each imitate and refine a small subset of interactions (the physics simulator inherently corrects MoCap inaccuracies), subject 14 held out as test; the teachers are distilled into a single Transformer student by Reference Distillation (teacher rollouts become the student's goal states and reward references) and Policy Distillation (behavior cloning on actions), with training beginning as DAgger and transitioning to PPO via w = min(t/beta, 1), followed by RL fine-tuning to surpass demonstration replication. The pattern is reused downstream rather than merely proposed: InterReal adopts InterMimic as its physics-based HOI tracker to verify SMPL->G1 retargeted motions by enforcing physical constraints in Isaac Gym. For MimicKit this maps onto a two-stage pipeline - per-clip (or per-source) teacher tracking policies whose rollouts become the cleaned motion dataset, then one distilled multi-skill policy. Merged from claims 5, 17, and the verification half of 12.

Sources: <https://arxiv.org/abs/2502.20390>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/html/2603.07516>

#### 6. Reference data is treated as something to augment, not just to clean; four distinct augmentation mechanisms are in use, all aimed at robustness to perturbation and at bridging sparse clips.

**Confidence:** high · **Votes:** 3-0 on claims 9, 12, 15

SkillMimic-V2 builds a Stitched Trajectory Graph that inserts masked intermediate states between demonstration clips when state similarity crosses a threshold (N = min(-floor(log10 beta), N_max) masked frames), and a State Transition Field via epsilon-Neighborhood State Initialization: sample s_new uniformly from the epsilon-neighborhood of a reference state s^i and connect it back with masked frames, {s_new, s_empty x N, s^j, ..., s^T}. InterReal applies IK-preserved hand-object contact constraints while sweeping object offsets Delta_x, Delta_y over linspace(-eps, eps, sqrt(c3)) to generate augmented multi-motions that improve policy stability under object perturbation. HumanX augments by scaling or substituting object geometry at mesh-acquisition time and by applying translation/scaling to the contact-phase object trajectory plus non-contact-phase velocity randomization. These are cheap to port: object-offset grids and epsilon-neighborhood initialization need no new policy machinery, only a dataset/initialization layer in MimicKit's motion loader. Merged from claims 9, 12, 15.

Sources: <https://arxiv.org/abs/2505.02094>, <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2602.02473v1>

#### 7. Initialization and clip-sampling are used as the main implicit curriculum: a FIFO buffer mixing MoCap references with high-return simulated states replaces plain Reference State Initialization, and clips are sampled inversely to how well they are currently tracked.

**Confidence:** medium · **Votes:** 3-0 on claims 7, 19, 10 (with PSI's advantage over RSI explicitly scoped as the authors' claim)

InterMimic's Physical State Initialization (PSI) creates an initialization buffer holding MoCap reference states and simulation states from prior rollouts; each rollout starts from a state drawn at random from the buffer, and completed trajectories whose expected discounted return exceeds a threshold are admitted FIFO while older or lower-quality ones are discarded. The paper's own framing is that PSI collects trajectories for policy updates that RSI does not effectively utilize - an authors' claim, not an independently replicated result, which is why this finding is medium rather than high. SkillMimic-V2 supplies the complementary curriculum: clip sampling probability p_i = exp(-lambda_s * r_bar_i) / sum_j exp(-lambda_s * r_bar_j) with r_bar_i the average per-frame reward from initializing at s^i, so badly-tracked segments are visited more, plus a behavior-cloning-pretrained History Encoder h_t = theta(s_{t-k},...,s_{t-1}) frozen during RL and appended to the policy input. Both are small, self-contained additions to an existing DeepMimic-style trainer. Merged from claims 7, 19, 10.

Sources: <https://arxiv.org/abs/2502.20390>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/abs/2505.02094>

#### 8. Interaction-specific early termination plus disturbed initialization and object-property domain randomization are what make an additive-reward HOI policy actually attend to the object.

**Confidence:** medium · **Votes:** 3-0

HumanX monitors, on reference frames that involve contact, the relative position error between the object and predefined key bodies, and terminates the episode probabilistically when that error exceeds a threshold - deliberately prioritizing interaction success over body-motion matching, which the paper motivates by the failure mode of additive rewards. It also applies disturbed initialization (random perturbations to root rotation/displacement, joint angles and object pose) and randomizes object size, mass, coefficient of restitution and robot friction. Single-source (CoRL 2026, public repo at github.com/wyhuai/HumanX), hence medium, but it is the most directly portable trick in the set: MimicKit already has early-termination machinery, and gating it on object-to-key-body error is a few lines. Claim 16.

Sources: <https://arxiv.org/html/2602.02473v1>

#### 9. For dexterous hands, the newest work deletes the retarget -> track -> residual-correct pipeline entirely, treating MoCap as soft guidance via adaptive spatial scopes inside one RL loop, and modulates the reward by hand-object distance so tracking relaxes near contact while effort is penalized more there.

**Confidence:** medium · **Votes:** 3-0 on claims 20 and 21

Dexplore (CoRL 2025) argues the standard three-stage workflow leaves demonstrations underused and compounds errors across stages, and replaces it with a unified single-loop optimization that jointly retargets and tracks, deriving adaptive spatial scopes from raw trajectories and training with RL to stay in-scope while minimizing control effort and accomplishing the task. Its per-timestep reward is a state-reference matching term plus an energy regularization term, where the kinematic matching weights lambda_R^h and lambda_J^h are proportional to hand-object surface distance (via w(D) = min(1, D/0.20m)) while the energy penalty is inversely proportional - so as D -> 0 the matching weight shrinks and the effort penalty grows, which the paper describes as encouraging smoother, lower-energy motions when tracking constraints are relaxed. Two independently verified claims from one primary peer-reviewed source, hence medium. This directly contradicts the residual-policy framing in the research question: the trend is away from residual correction, not toward it. Merged from claims 20, 21.

Sources: <https://arxiv.org/abs/2509.09671>

#### 10. Simulator and optimizer choices are uniform and boring - Isaac Gym with PPO - and the reported success rates that survived verification come from two sources only, with the most-cited success-rate table (InterMimic's) failing verification.

**Confidence:** medium · **Votes:** 3-0 on claims 13 and 3; the competing InterMimic success-rate claim was refuted 1-2

InterReal trains in Isaac Gym with PPO for 5e4 epochs and validates sim-to-sim in MuJoCo, reporting 96.41% box-picking success (MPJPE 0.1076 rad vs 0.1634 for the authors' ASAP* reproduction and 0.1984 for InterMimic*) and 87.45% box-pushing. SkillMimic reports 82.4% on BallPlay-V for the full method (E_b-mpjpe 56.8mm, E_o-mpjpe 82.9mm, E_cg 0.087). InterMimic's own OMOMO/BEHAVE numbers (90.7%/95.5% success, 42.6s vs 12.2s sustained tracking) did NOT survive verification (1-2 vote) and are excluded. No claim in the surviving set reports wall-clock time, GPU count, or sample count, so training cost is effectively unknown from this evidence. Confidence is medium because every number is a self-reported, mostly preprint figure with no independent replication, and the baselines in InterReal's table are the authors' own adaptations (marked ASAP*, InterMimic*). Merged from claims 13, 3; refuted InterMimic numbers noted.

Sources: <https://arxiv.org/html/2603.07516>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/html/2502.20390>

#### 11. Sim-to-real HOI on a Unitree G1 is demonstrated but thin: one preprint deploys on hardware with FoundationPose for object pose after a SMPL->G1 retarget + physics-verify + augment pipeline, and one CoRL paper drives the same class of skills from monocular human video, with no third-party replication of either.

**Confidence:** medium · **Votes:** 3-0 on claims 12, 13, 15

InterReal retargets SMPL HOI data to G1 shape parameters, verifies it with InterMimic as a physics-based HOI tracker, augments with contact-constrained object-offset variants, trains in Isaac Gym, validates sim-to-sim in MuJoCo, and deploys on a real Unitree G1 using FoundationPose to extract object-related observations; notably the authors defer the SMPL->G1 retargeting algorithm itself to a later publication, so that step is not reproducible from the paper. HumanX (CoRL 2026) handles monocular video by physics-based synthesis of object trajectories with frame-by-frame force-closure optimization yielding refined robot and object poses, explicitly arguing that physically plausible interaction matters more than photometrically faithful reconstruction. Together these define the realistic target for a MimicKit manipulation extension - object pose from a 6-DoF pose estimator at deployment, references repaired in simulation beforehand - but the evidence base is two self-reported systems, hence medium. Merged from claims 12, 13, 15.

Sources: <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2602.02473v1>

### Caveats

- Evidence quality: every surviving claim is sourced to the primary paper describing its own method, which is appropriate for design descriptions but means all quantitative results are self-reported, and the only head-to-head baselines (ASAP*, InterMimic* in InterReal) are the authors' own reproductions rather than the original authors' numbers. Three papers are preprints without peer review (InterReal arXiv 2603.07516, PhysHOI, SkillMimic-V2 was later SIGGRAPH 2025); InterMimic (CVPR 2025), SkillMimic (CVPR 2025), Dexplore (CoRL 2025) and HumanX (CoRL 2026) are venue-published. One notable refutation: InterMimic's widely quoted OMOMO/BEHAVE success rates and tracking durations failed verification (1-2), as did an asymmetric actor-critic claim about InterReal (0-3) and a contact-graph topology claim about PhysHOI (0-3) - do not reuse those numbers from secondary summaries. Coverage gap: of the systems named in the research question, only PhysHOI, SkillMimic, SkillMimic-V2, InterMimic, Dexplore, HumanX and InterReal produced surviving claims; nothing survived on Omnigrasp, ManipTrans, DexMachina, HDMI or ResMimic, so any statement about those (including about virtual object controllers and residual policies, which this evidence set only touches via Dexplore's rejection of residual correction) is unsupported here. Training cost is entirely unanswered - no wall-clock, GPU-count, or sample-count figure survived. Time sensitivity is real but mild: the two most recent sources are dated Feb and Mar 2026 relative to today (2026-10-12), so later work may already have superseded the InterReal/HumanX G1 results, and Isaac Gym is deprecated upstream even though every paper here still uses it - a MimicKit port to Isaac Lab or Newton will not find published HOI baselines on those backends.

### Open questions

- Multiplicative vs additive composition is unresolved across generations: SkillMimic's ablation shows additive collapsing to 7.5% success, yet InterMimic, HumanX and InterReal all ship additive rewards with extra machinery. Is the multiplicative advantage specific to single-skill, dense-contact tracking (basketball), and does it survive when one policy must cover many objects and subjects?
- What is the actual training cost of these systems (GPU-hours, env steps, number of teacher policies) - none of the surviving claims reports it, and that number decides whether the teacher-distillation recipe is affordable on a shared SLURM cluster before MimicKit adopts it.
- How is reference contact state obtained for a robot embodiment rather than SMPL? InterReal defers its SMPL->G1 retargeting algorithm to a later paper, and the three-level contact discretization in InterMimic assumes MoCap-derived contact markers - an open question for building MimicKit's G1 HOI data layer.
- Do any of these methods report sim-to-real success for dexterous hands (as opposed to a G1's grippers/links), and does the Dexplore single-loop, soft-guidance formulation transfer to a whole-body humanoid where the "hand-object surface distance" that modulates its reward is far less informative?

### Claims refuted by verification

- (0-3) The contact graph is a complete graph whose nodes are objects and (optionally aggregated) humanoid body parts, with binary edge labels for contact; for basketball the two hands are aggregated as one node, the rest of the body as another, and the ball as the third. The contact-graph reward is reported as critical: removing it drops success from 84.8% to 20.7% on the Toss skill and to roughly 13.6-38.6% across ablations. — source https://arxiv.org/abs/2312.04393
- (1-2) In Isaac Gym at 30 Hz control, the full InterMimic model reports 90.7% success on OMOMO-Train and 95.5% on OMOMO-Test (success = proportion of references imitated at least once), with human/object tracking errors of roughly 5.5/9.7 cm (train) and 5.4/11.9 cm (test); on BEHAVE it sustains 42.6 s average tracking vs 12.2 s for SkillMimic, with object error 9.2 cm vs 13.4 cm, and PhysHOI yields clearly shorter tracking durations. — source https://arxiv.org/html/2502.20390
- (0-3) The policy uses an asymmetric actor-critic: the actor sees only deployable observations (proprioception, object position, interaction-graph features) while the critic receives privileged object velocity and rotation. — source https://arxiv.org/html/2603.07516

### How this compares with the overnight survey (reports 01 to 05)

- **Confirms** the central design choice in the plan: object-centric observations, contact-graph rewards, and
  a product of exp-kernel factors in the PhysHOI/SkillMimic line, with SkillMimic's ablation (82.4% to 7.5%
  success when the product becomes a sum) as the strongest single piece of evidence.
- **Adds a nuance the plan should carry**: InterMimic, HumanX and InterReal ship *additive* rewards and get
  interaction fidelity from other machinery (interaction-specific early termination, learned reward-weight
  meta-policy, distance-weighted joint terms). So phase 1 should run the additive-vs-multiplicative ablation
  as planned, but also implement object-deviation and contact-mismatch termination from day one, since that is
  what makes the additive variant work at all.
- **Confirms** the data-repair stance: teacher correction (InterMimic), physics-synthesized references
  (HumanX), soft-scope tracking (Dexplore), and augmentation (SkillMimic-V2 STG/STF, InterReal offset grids).
- **Weakens** two items the overnight reports took from READMEs and secondary summaries: InterMimic's quoted
  OMOMO/BEHAVE success rates failed verification (1 to 2), and the PhysHOI contact-graph topology description
  failed (0 to 3). Do not reuse those numbers; cite the papers' own tables after reading them.
- **Coverage gap, not contradiction**: the verified claim set contains nothing on Omnigrasp, ManipTrans,
  DexMachina, HDMI or ResMimic, so the plan's virtual-object-controller and residual-policy curricula rest on
  the overnight README-level survey only. Dexplore argues for deleting the residual stage. Treat those two
  curricula as experiments, not defaults.
- **Still missing everywhere**: training cost in GPU-hours. The plan's compute estimates remain extrapolations
  from MimicKit's own whole-body runs.


---

## 5. Open-source codebase survey

### Open-source physics-based imitation of human–object interaction / dexterous manipulation — codebase survey (as of 2026-10-12)

Scope note: "MimicKit for manipulation" means a GPU-parallel RL tracker (DeepMimic/AMP-style) whose reference clips contain a human **and** an object, with object-aware rewards/observations. I grouped the ~45 codebases into (A) whole-body humanoid HOI imitation (the direct analogues), (B) dexterous-hand-only imitation, (C) real-robot humanoid HOI from video/mocap, (D) retargeting/data engines, and (E) adjacent/not-really-imitation. Stars/license/last-push come from the GitHub API on 2026-10-12. Reward/obs details come from the paper HTML plus, where noted, the actual source files. Anything I could not confirm is marked **[unverified]**.

For reference, MimicKit's motion format (checked locally in `/home/vawale/my_things/codes/MimicKit/mimickit/anim/motion.py`, `motion_lib.py`) is a pickle `{loop_mode, fps, frames}` where each frame is `[root_pos(3), root_rot_expmap(3), joint_dof(...)]`; there is no object channel, and `motion_lib.py` stores only `_frame_root_pos/_frame_root_rot/_frame_joint_rot`. Engines: Isaac Gym, Isaac Lab, Newton. The `manipulation/plan` and `manipulation/report` dirs in the repo are currently empty.

---

### A. Whole-body humanoid HOI imitation (closest analogues)

#### 1. InterMimic (CVPR 2025 Highlight) — the primary reference
- Repo: https://github.com/Sirui-Xu/InterMimic — 538 stars, MIT, last push 2026-04-21, 1 open issue. Paper: https://arxiv.org/abs/2502.20390. Project: https://sirui-xu.github.io/InterMimic/
- Simulator: Isaac Gym (training, `isaacgym/src/intermimic/`) and Isaac Lab v2.3.1 / Isaac Sim 5.1 (inference from Isaac Gym checkpoints, `isaaclab/src/intermimic_lab/`; Dec 2025). Multi-GPU training added Feb 2026.
- Embodiment: SMPL-X humanoid, 52 bodies / 51 actuated joints / 153 DoF (22 body + 30 hand joints); also **Unitree G1 29-DoF + three-finger dex hands** (`omomo_g1_29dof_with_hand.yaml`, `intermimic_g1.py`), "learned directly from MoCap without external retargeting" (the distance-weighted reward does the retargeting).
- Data: OMOMO (15 objects, ~10 h, 17 subject-specific teachers), with BEHAVE/HODome/IMHD/HIMO via InterAct for evaluation. Ships a preprocessed OMOMO (`InterAct/OMOMO_new`, Google Drive, ungated), teacher-corrected references, teacher + student checkpoints. The InterAct repo provides the conversion pipeline from raw datasets to this format (Dec 2025).
- Motion file format (read from `intermimic.py`): one `.pt` tensor per clip, 30 fps, per-frame row = `root_pos(3) root_rot_quat(4) [pad] dof_pos(153) dof_vel(153) body_pos(52x3) body_rot(52x4) body_lin_vel(52x3) body_ang_vel(52x3) obj_pos(3) obj_rot_quat(4) obj_lin_vel(3) obj_ang_vel(3) ig(52x3 = vector from each body to nearest object surface point) contact_human(52) contact_obj(1)`. Objects are URDFs with convex decomposition under `data/assets/objects/{dataset}`.
- Reward (code): `r = rb * ro * rig * rcg`, every term `exp(-w * err)`. Config `omomo_train_new.yaml` weights: `p=30, r=1.5, op=5, or=0.1, cg_hand=5, cg_other=5, cg_all=3, eg1=eg2=2e-5, eg3=1e-11`. Body position error is multiplied by a contact/distance weight and rotation error by `(1 - weight)` (track rotation when far from object, position when close); `rig` is a weighted L2 on the human–object distance vectors; contact graph reward splits hand contacts, other-body contacts, and a "no spurious contact" term; energy penalties on dof accel, object accel, and contact-force magnitude.
- Observation: humanoid proprio (root height, local key-body positions, tan-norm rotations, velocities, contact flags) + 21-D object-relative task obs (object vel/ang-vel in root frame, pos/rot/vel/ang-vel errors vs reference) + interaction-graph obs (`ig / |ig| * exp(-5|ig|)`), computed for **two future frames (t+1 and t+16)**; student uses K={1,2,4,16}. `numObs=3198`.
- RL: PPO (rl_games), 4096 envs, MLP teachers (1024-1024-512); student = 3-layer Transformer, trained with DAgger (BC from teachers) then mixed PPO+DAgger. Reference distillation: teacher rollouts replace noisy mocap as the student's reference.
- RSI: `StateInit.Hybrid` (p=0.1 start-of-clip, else random frame) plus **Physical State Initialization**: a buffer of high-reward simulated states is sampled as initial states; reference trajectories sampled by a reward-weighted CDF.
- Termination: mean key-body error > 0.5 m, object point-cloud error > 0.5 m, interaction-graph ratio > 2x, hand-contact mismatch, root height < 0.15.
- Training cost: not stated in README/paper; `numEnvs=4096` at 30 Hz. Community reports of ~1–2 days per teacher on one GPU **[unverified]**.
- Sim-to-real: none in this repo (G1 is sim only); the follow-up InterPrior (https://sirui-xu.github.io/InterPrior/, arXiv 2602.06035) scales to a generative controller; code not yet public **[unverified]**.
- Port difficulty into MimicKit: **moderate**. The env is an ASE-style Isaac Gym task so the structure matches MimicKit's `envs/`; what you'd port is (a) the `.pt` layout above into a MimicKit `Motion` with extra `obj_pos/obj_rot/contact/ig` channels, (b) `compute_humanoid_reward` (multiplicative terms + the contact graph), (c) the 21-D object task obs + future-frame stacking, (d) hybrid/physical-state init.

#### 2. PhysHOI (Tech report 2023) and SkillMimic (CVPR 2025 Highlight) / SkillMimic-V2 (SIGGRAPH 2025)
- PhysHOI: https://github.com/wyhuai/PhysHOI — 249 stars, custom license (NOASSERTION; check `LICENSE.txt` for non-commercial terms **[unverified]**), last push 2024-09. Paper https://arxiv.org/abs/2312.04393.
- SkillMimic: https://github.com/wyhuai/SkillMimic — 445 stars, Apache-2.0, last push 2025-06-30. Paper https://arxiv.org/abs/2408.15270.
- SkillMimic-V2: https://github.com/Ingrid789/SkillMimic-V2 — 160 stars, Apache-2.0, last push 2025-07-24. Paper https://arxiv.org/abs/2505.02094.
- Simulator: Isaac Gym Preview 4 for all three (60–120 Hz sim, 30–60 Hz policy). Embodiment: SMPL-X with full hands, 51x3 DoF.
- Data: PhysHOI ships BallPlay (8 basketball clips, 25 fps, SMPL-X + ball pose + contact labels from video). SkillMimic ships a subset of BallPlay-M (mocap, ~35 min total at 120 fps, full set "coming soon" as of mid-2025). SkillMimic-V2 adds ParaHome household clips (place pan/kettle, drink, pour, sit) and locomotion; data in-repo, ungated.
- Motion format (from `physhoi.py` / `skillmimic.py`): `.pt` tensor, 30 fps, row = `root_pos(3) root_rot(3 or 4) dof_pos(153) dof_vel(153) obj_pos(3) obj_rot(4) obj_vel(3) key_body_pos(10x3) contact(1)`; SkillMimic appends a per-clip skill label (64-D one-hot in obs). Indices differ slightly between the two readers I saw (323 vs 328 offset) — verify before writing a converter.
- Reward: PhysHOI `r = r_b * r_o * r_ig * r_cg` where `r_b = r_p r_r r_pv r_rv`, `r_o = r_op r_or r_opv r_orv`, `r_ig = exp(-λ_ig e_ig)` on relative body–object positions, `r_cg = exp(-Σ λ_cg[j] e_cg[j])` on binary contact-graph edges (3 nodes for basketball: hands, rest-of-body, ball). SkillMimic adds a velocity-regularization term (`exp(-0.1 e_vel_diff)`) and zeroes object-rotation terms for the ball. Coefficients come from the `w` dict in `skillmimic.yaml` / `physhoi.yaml` (not reproduced in paper).
- Observation: humanoid proprio + fingertip contact forces + object local pose/vel (+ skill one-hot). No future-frame stack in PhysHOI; SkillMimic-V2 adds a learned history encoder.
- RL: PPO, 2048 envs on a single RTX 3090/4090. PhysHOI: ~9 h on a 4090 Ti for some tasks, 5k (GRAB) / 15k (BallPlay) epochs. SkillMimic-V2: ~24 h for BallPlay-M on one 4090, ~1e9 samples per ParaHome task. SkillMimic recommends `num_envs=16384` for large datasets.
- RSI: random frame of a random clip (RRSI). SkillMimic-V2 data augmentation: **Stitched Trajectory Graph** (offline connects similar states across clips), **State Transition Field** (ε-neighborhood init + nearest reference state), **Adaptive Trajectory Sampling** (softmax over per-frame reward difficulty). These are exactly the "augmentation tricks" you asked about and are the most portable idea here.
- Sim-to-real: none. Hierarchical high-level controller (discrete skill tokens) in SkillMimic.
- Port difficulty: **easy–moderate**. Same ASE lineage as MimicKit; the reward and `MotionDataHandler` are self-contained files.

#### 3. Omnigrasp (NeurIPS 2024, Luo et al.; PHC/PULSE family)
- Repo: https://github.com/ZhengyiLuo/Omnigrasp — 159 stars, no license file (paper page says CC BY-NC-SA 4.0 **[unverified for code]**), last push 2025-11-17, 8 open issues. Paper https://arxiv.org/abs/2407.11385.
- Simulator: Isaac Gym (`sim=hand_sim`), SMPL-X v1.1 humanoid 52 joints / 153 DoF. Hydra configs (`learning=omnigrasp_rnn robot=smplx_humanoid`).
- Data: GRAB (50 objects, pre-grasp + initial pose provider), OakInk (1700 objects), OMOMO (7 objects). Ships `download_data.sh` (models + `grab_sample.pkl`); SMPL-X body models require MPI registration. Training code released Aug 2025; **data-processing code still pending** (you must write the GRAB → `.pkl` converter yourself, based on PHC's format).
- Method: PULSE-X 48-D full-body+hand latent prior (distilled from PHC-X trained on dexterous AMASS); task policy outputs latents, GRU (512), PPO (batch 3072, lr 5e-4). Reward is **not full-motion imitation**: approach reward (hand → pre-grasp distance), pre-grasp hand-pose imitation, then object-trajectory tracking (exp terms on pos/rot/vel/ang-vel) + contact bonus. Obs: proprio + 20 future object-trajectory frames at 15 Hz + 512-D BPS object shape + hand-to-object offsets. Curriculum: hard-negative mining over objects by failure rate. Cost: ~3 days, ~1e9 samples on a single A100.
- Port difficulty: **hard** (depends on PHC/PULSE latent prior and SMPL-X asset pipeline); but the pre-grasp + object-trajectory reward and BPS obs are portable concepts.

#### 4. HumanVLA (NeurIPS 2024) and HumanoidVerse (vision-language rearrangement, 2025)
- HumanVLA: https://github.com/AllenXuuu/HumanVLA — 136 stars, MIT, last push 2024-12. Isaac Gym, SMPL-like humanoid, Human-in-the-Room (HITR) dataset on Google Drive (ungated), HSSD scenes. Teacher = goal-conditioned PPO + AMP (carry curriculum → rearrangement), student = VLA via BC; 3-stage `torchrun` on 4 GPUs. Not a motion tracker: objects enter via goal rewards + AMP style prior. Relevant mostly as an AMP-with-objects example (MimicKit already has AMP).
- HumanoidVerse (Zhang et al., arXiv 2508.16943, Isaac Gym, multi-object rearrangement, dual-teacher distillation): project page https://haozhuo-zhang.github.io/HumanoidVerse-project-page/; code status **[unverified]**. Not to be confused with LeCAR's HumanoidVerse framework (below).

#### 5. UniHSI (ICLR 2024 Spotlight)
- https://github.com/InternRobotics/UniHSI (moved from OpenRobotLab) — ~248 stars, CC BY-NC-SA 4.0, last update 2024. Isaac Gym, ASE/AMP humanoid, PartNet/ShapeNet/ScanNet scenes, "Chain of Contacts" (joint–object-part pairs) as the task spec and reward, LLM planner. Static scenes, no object dynamics tracking. Useful only for its contact-pair reward and scene loading.

#### 6. ProtoMotions / MaskedMimic (NVIDIA)
- https://github.com/NVlabs/ProtoMotions — 2.4k stars, Apache-2.0, last push 2026-09-22. Backends: Newton 1.0, Isaac Lab 3.0, Isaac Gym, MuJoCo, Genesis (community). Embodiments: SMPL/SMPL-H, H1-2, G1. Datasets: AMASS, BONES-SEED (142k motions), PHUMA, Kimodo.
- Object support: "scene and object spawning" and MaskedMimic's object-conditioned inpainting (sit/reach on SAMP-style static objects). **No dynamic object tracking or contact rewards**; Genesis backend cannot spawn per-env unique objects. Treat as the closest architectural cousin to MimicKit (same Newton/Isaac Lab/Isaac Gym trio) and a source for the scene-spawning code, not for HOI rewards.

#### 7. Newer physics-based HOI works (2025–2026)
- **InterPrior** (Xu et al., arXiv 2602.06035): masked conditional VAE policy distilled from InterMimic experts + RL post-training; code not public **[unverified]**.
- **LessMimic** (arXiv 2602.21723): https://github.com/Yutang-Lin/LessMimic — 57 stars; distance-field conditioning, VAE + "Adversarial Interaction Prior" RL, no motion references at test time. README: "Code will be released" — **not yet usable**.
- **HOI-Dyn** (NeurIPS 2025, https://github.com/wulin97/HOI-Dyn) and **SimGenHOI** (arXiv 2508.14120) are generative + RL hybrids; code maturity **[unverified]**.

---

### B. Dexterous-hand (hand-only / floating-hand) imitation from human hand–object data

#### 8. ManipTrans (CVPR 2025)
- https://github.com/ManipTrans/ManipTrans — 342 stars, **GPL-3.0**, last push 2025-10-10. Paper https://arxiv.org/abs/2503.21860.
- Isaac Gym Preview 4, Hydra. Hands: Inspire (12-DoF), Shadow (22), Allegro (16), XHand, Arti-Mano, Inspire-FTP; single and bimanual; floating wrist (6-DoF) + fingers.
- Data: GRAB demo in repo; OakInk-V2, FAVOR, ARCTIC via their own licenses; **DexManipNet** (3.3k episodes, 1.34M frames, 61 tasks, 1.2k objects; download from project site / HF mirrors, gating **[unverified]**). Format: 60 fps MANO joints + wrist kinematics + object mesh/URDF + object pose.
- Two-stage: generalist hand imitator (wrist SE(3) + finger keypoint rewards, joint-power penalty) pretrained on FAVOR/OakInk-V2/GRAB/ARCTIC + interpolated synthetic data, then per-skill **residual** policy with object pose/vel reward and contact-force reward (`exp(-λ_c Σ f · 1[d<thr])`). Obs: joint q/qd, wrist pose, object 6D pose/vel/CoM/gravity, BPS shape, contact forces, fingertip–object distances. PPO, 4096 envs, single RTX 4090; RSI + early termination with curriculum threshold relaxation. Sim-to-real: Realman arms + Inspire hands via fingertip IK.
- Port: **moderate**; GPL is the main caveat if MimicKit stays MIT/BSD.

#### 9. DexMachina (ICML 2026)
- https://github.com/MandiZhao/dexmachina — 269 stars, MIT, last push 2026-03-23. Paper https://arxiv.org/abs/2505.24853.
- **Genesis** (custom fork), rl_games PPO, 12k envs/GPU, single L40S/H100 per run. Six hands (Allegro, Inspire, Schunk, XHand, Ability, DexRobot), ARCTIC bimanual articulated objects (box, mixer, notebook, ketchup, waffle iron). Reward: object pose/joint-angle tracking + hand keypoint imitation + BC-to-retargeted-joints + contact-position matching. Key trick: **virtual object controller curriculum** (spring-damper drives object along reference; gains decay with policy success). Actions: residual wrist + absolute normalized fingers. Ships hand assets + ARCTIC assets (ARCTIC itself needs registration).
- Port: **moderate**; the decaying virtual-object-controller is a very portable curriculum for any MimicKit HOI env.

#### 10. DexTrack (ICLR 2025)
- https://github.com/Meowuu7/DexTrack — 135 stars, MIT, last push 2026-04-14. Isaac Gym, Allegro (floating) and LEAP-on-Franka, GRAB + TACO retargeted via MANO→robot keypoints; 22k envs single GPU. Per-trajectory RL tracking released; the homotopy-optimization + IL/RL generalist code is explicitly **not released** ("too messy").

#### 11. ObjDex (Chen, Wang, Yang, Liu)
- https://github.com/cypypccpy/ObjDexEnvs — 63 stars, MIT, last push 2025-02. Isaac Gym, Shadow hand, ARCTIC clips by `--object --traj_index`, 2048 envs, object-centric reward (hand data only initializes). Small but clean.

#### 12. Human2Sim2Robot and SimToolReal / DexToolBench (Lum et al.)
- https://github.com/tylerlum/human2sim2robot — 160 stars, MIT, last push 2026-08-27. Isaac Gym, Kuka iiwa + Allegro, one RGB-D human video → FoundationPose object trajectory + HaMeR pre-manipulation hand pose; **object-centric reward only** (no finger imitation), PPO-LSTM, DR, zero-shot real.
- https://github.com/tylerlum/simtoolreal — 409 stars, MIT, last push 2026-09-21. Isaac Sim/Isaac Lab (Isaac Gym legacy), PPO + SAPG, up to 24,576 envs. **DexToolBench**: 24 tool-use tasks (6 categories x 2 objects x 2 tasks), RGB-D human videos + processed data + sim envs; download script provided (terms **[unverified]**).

#### 13. Grasp-synthesis / benchmark repos (not imitation, listed for completeness)
- GraspXL https://github.com/zdchan/GraspXL — 272 stars, CC BY-NC 4.0, RaiSim (license needed), MANO/Allegro/Shadow/LEAP, 500k Objaverse objects on HF (ungated).
- DexGraspNet 2.0 https://github.com/PKU-EPIC/DexGraspNet2 (162 stars, no license, CoRL 2024, synthetic cluttered grasps, HF dataset) and UniDexGrasp++ https://github.com/PKU-EPIC/UniDexGrasp2 (219 stars, MIT, Isaac Gym, geometry-aware curriculum + generalist-specialist). No human data.
- Bi-DexHands https://github.com/PKU-MARL/DexterousHands — 1.1k stars, Apache-2.0, Isaac Gym bimanual Shadow tasks; last push 2025-02.
- DexPBT: lives in IsaacGymEnvs (`AllegroKukaLSTM`, PBT docs https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/pbt.md); Isaac Lab now has a PBT feature page. DexterityGen (https://zhaohengyin.github.io/dexteritygen/): **no code released**.
- DexMimicGen https://github.com/NVlabs/dexmimicgen (288 stars, NVIDIA Source Code License; datasets CC-BY 4.0 on HF `MimicGen/dexmimicgen_datasets`) and MimicGen (652 stars): robosuite/MuJoCo, GR1 + bimanual arms, **imitation learning (BC-RNN) not RL**; object-centric transform of a few teleop demos. MimicLabs https://github.com/Gatech-RL2/mimiclabs (MuJoCo, MimicGen-based), RoboCasa (MuJoCo kitchens, arm-centric). HumanoidGen https://github.com/TeleHuman/HumanoidGen (157 stars, NeurIPS 2025, LLM-generated bimanual dex tasks). These are data-generation for IL, not physics imitation of human data.

---

### C. Real-robot humanoid HOI tracking from human mocap/video (G1-centric, 2025–2026)

#### 14. HDMI (LeCAR, arXiv 2509.16757)
- https://github.com/LeCAR-Lab/HDMI — 672 stars, **no license file**, last push 2026-01-17. Isaac Lab v2.2 / Isaac Sim; Unitree G1 (no dex hands in released tasks). Sim2real stack: https://github.com/EGalahad/sim2real.
- Data pipeline: monocular RGB → GVHMR human pose → GMR/LocoMujoco retargeting; object pose from video; reference = `(robot state, object pose [+ articulation], binary contact c_t with contact points in object frame)`. Motion files: `.npz` with body pos/quat/lin-vel/ang-vel `[T,B,·]` and joint pos/vel `[T,J]`; 13 task configs (`move_suitcase`, `traverse_door`, `carry_box_over_shoulder`, `open_foldchair-sit`, `roll_ball-hand`, ...), object randomization (mass, friction, scale 0.9–1.1).
- Reward (paper table + `rewards.py`): body local pose 2.0, root global pose 1.0, body global vel 1.0, joint tracking 1.0, **object pose 2.0**, **contact 5.0**, action-rate -0.1, joint-limit 10, joint-vel 5e-4; tracking terms `exp(-err/σ)` with σ=0.03 (keypoints) and σ=0.25 (object); contact reward `exp(-eef_pos_err/0.1) * exp(f_contact/10)` gated by "in range". Obs: proprio + object pose in robot root frame + phase φ∈[0,1] + reference contact points in robot frame. PPO with **residual action space** (offsets from reference joint targets), teacher (`ppo_roa_train`) → student (`ppo_roa_finetune`), RSI with perturbation, DR. Real: 67 consecutive door traversals on G1.
- Port: **moderate**; the Isaac Lab MDP terms map cleanly to MimicKit's Isaac Lab engine; the main missing piece in MimicKit is the object asset per env.

#### 15. ResMimic (Amazon FAR, arXiv 2510.05070)
- https://github.com/amazon-far/ResMimic — 74 stars, Apache-2.0. legged_gym/rsl_rl fork on Isaac Gym (MuJoCo sim2sim), built on TWIST2. G1 29-DoF. Stage 1 general motion tracker (GMT) on AMASS + OMOMO via GMR (~15k clips / 42 h); stage 2 residual policy `a = a_gmt + Δa`. Motion data: humanoid `.pkl` (PHC/TWIST style) + object `.npz` with `trans`, `rot` (`motion_lib_hoi.py`); examples `kneel`, `squat`, `carry`, `chair`, suitcase URDF (mass 4 kg). Reward scales in `g1_hoi_config.py`: `tracking_keybody_pos_global 10`, `tracking_joint_dof 2`, `tracking_object_point_cloud 2`, `feet_air_time 5`, `feet_slip -0.1`, `dof_torque_limits -1` (16 terms total); contact reward `Σ ĉ_i exp(-λ/f_i)` from oracle contacts; object point-cloud tracking `exp(-λ Σ‖P_i − P̂_i‖)`. Obs: proprio history (10) + reference window (±10 frames) + object `[p, θ, v, ω]` (7-D "nonblind"). **Virtual object force controller with decaying PD gains** (same trick as DexMachina). Termination: object > 0.3 m from reference. Real G1 carries 4.5 kg.
- Port: **moderate**; the object-point-cloud reward and decaying virtual controller are small, self-contained.

#### 16. OmniRetarget (Amazon FAR / project-instinct) — data engine + dataset
- Code https://github.com/project-instinct/omniretargeting — 82 stars, MIT, last push 2026-10-10. Paper https://arxiv.org/abs/2509.26633. Dataset https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset — MIT, **ungated**, 4.0 h total: `robot-object/` (3.0 h from OMOMO), `robot-terrain/` (0.5 h), `robot-object-terrain/` (0.5 h), `models/` (G1 URDF/SDF, object OBJs). Format: `.npz` with `fps` and `qpos [T, 36 or 43]` = base pos(3)+quat(4)+29 joints (+ object quat(4)+pos(3)). Retargeter: interaction-mesh (Laplacian) preserving body–object–terrain relations, hard kinematic constraints, augmentation by object/terrain scaling and relocation; input adapters for SMPL-X/OMOMO/LAFAN1/Nokov. Tracking policies in the paper use a BeyondMimic-style Isaac Lab tracker with 5 reward terms and show zero-shot G1 parkour/carry **[the tracker code is not in this repo; BeyondMimic's is public, below]**.
- This is the single most "ready-to-use" G1 + object dataset.

#### 17. BeyondMimic (Science Robotics 2025) and SONIC / GR00T-WBC (NVIDIA)
- https://github.com/HybridRobotics/whole_body_tracking — 2.4k stars, MIT, Isaac Lab 2.1, G1, LAFAN1 CSV → npz via WandB registry, DeepMimic-style rewards; **no object support** but the reference Isaac Lab tracker that OmniRetarget data is designed for.
- https://github.com/NVlabs/GR00T-WholeBodyControl — 3.7k stars, Apache-2.0 code / NVIDIA Open Model License weights, last push 2026-08-31; SONIC PPO training stack, BONES-SEED 142k motions, MuJoCo retargeting; manipulation only via teleop, no HOI imitation.

#### 18. HumanX (CoRL 2026, Wang et al.)
- https://github.com/wyhuai/HumanX — 58 stars, Apache-2.0 **with Commons Clause**, last push 2026-10-09. G1; **XGen** (video → HOI editor exporting `.npz` robot+object motions) + **XMimic** (`.pkl` training, SkillMimic-lineage rewards). Example data: basketball catch/throw, pump-fake, football kick, bin pick-and-place; zero-shot real G1 for 10 skills across 5 domains. Very new; simulator not stated in README **[unverified, likely Isaac Gym given SkillMimic lineage]**.

#### 19. Others in this bucket
- **Weave** (arXiv 2609.16683) https://github.com/xiaohu-art/Weave — 93 stars, Isaac Lab + RSL-RL, G1 with dex hands, 9 objects, contact-aware retargeting; HF data `appolyn/Weave`. **DexWeave** (arXiv 2609.34724) https://github.com/WU-CVGL/DexWeave — G1 + Inspire hands. Both too new for maturity claims.
- **PhysHSI** https://github.com/InternRobotics/PhysHSI — 321 stars, CC BY-NC-SA 4.0, Isaac Gym legged_gym + HIMLoco, G1, AMP from AMASS/SAMP/100STYLE, tasks CarryBox/SitDown/LieDown/StandUp; LiDAR+camera object localization for real. AMP-with-goals, not tracking.
- **VisualMimic** https://github.com/visualmimic/VisualMimic — 300 stars; only sim2sim + checkpoints released; training code pending. **VIRAL** (arXiv 2511.15200): no code found. **AMO** https://github.com/OpenTeleVision/AMO — 386 stars, inference-only JIT checkpoints, MuJoCo. **FALCON** (CoRL 2025): paper only, no repo found **[unverified]**. **CLONE** https://github.com/humanoid-clone/CLONE — 277 stars, CC BY-NC 4.0, checkpoints + deploy only; training code "planned via RoboVerse". **HOMIE** https://github.com/InternRobotics/OpenHomie and `/Homie` — teleop cockpit + locomotion RL, no HOI imitation. **OmniH2O/H2O** https://github.com/LeCAR-Lab/human2humanoid — 1.1k stars, CC BY-NC 4.0, Isaac Gym, H1 + Inspire, PHC-style AMASS `.pkl`, teacher-student DAgger; no objects. **HumanPlus** https://github.com/MarkFzp/humanplus — 853 stars, legged_gym/rsl_rl, H1, HST on AMASS + HIT; no objects. **OKAMI** https://github.com/UT-Austin-RPL/OKAMI — 37 stars, GR1, single RGB-D video → SLAHMR/HaMeR + object-aware kinematic retargeting (no RL tracking of dynamics). **TWIST** https://github.com/YanjieZe/TWIST — 821 stars, MIT, teleop via GMR; no objects. **VideoMimic** https://github.com/hongsukchoi/VideoMimic — 838 stars, MIT, CoRL 2025 best student paper, video → scene mesh + human motion, scene-conditioned tracking (stairs, sitting); static scenes only. **Workhorse** (arXiv 2610.09117), **CoorDex**, **Blind Dexterity**, **SUGAR**, **GRAIL**, **HumanoidMimicGen**: 2026 papers, code status **[unverified]**.

---

### D. Retargeting / data tooling

- **GMR** https://github.com/YanjieZe/GMR — 2.8k stars, MIT, ICRA 2026, 17 robots, inputs SMPL-X (AMASS/OMOMO), BVH (LAFAN1/Xsens/Nokov), FBX (OptiTrack), PICO, GVHMR video; output pickle `{root trans, root quat(wxyz), joint pos}` per frame; CPU real-time. **Objects are not retargeted** (used by HDMI/ResMimic for the body only).
- **OmniRetarget** (above) is the object-aware retargeter.
- **dex-retargeting** https://github.com/dexsuite/dex-retargeting — 1.3k stars, MIT, `pip install dex_retargeting`; vector/position/DexPilot optimizers from MANO keypoints → robot hand qpos (Allegro, Shadow, Leap, Inspire, Ability, etc. via dex-urdf). AnyTeleop (RSS 2023) is the origin; DexPilot itself (NVIDIA 2019) has no public code; the "DexPilot optimizer" here is a reimplementation.
- **Open-Teach** https://github.com/aadhithya14/Open-Teach — Quest-3 VR teleop for Franka/xArm/Allegro (data collection, no sim imitation).
- **InterAct** https://github.com/wzyabcas/InterAct — 211 stars, custom license, last push 2026-10-06; 21.8 h consolidated HOI (OMOMO, BEHAVE, GRAB, InterCap, IMHD, HODome, HIMO, CHAIRS, ...) corrected and augmented to 30.7 h, with text; **access via request form (non-commercial)**; ships converters to the InterMimic `.pt` format.
- **LocoMujoco** https://github.com/robfiras/loco-mujoco — 1.5k stars, MIT, MuJoCo/MjWarp, imitation-learning benchmark + retargeting used by HDMI; no objects. **Genesis** https://github.com/Genesis-Embodied-AI/Genesis (repo recently reorganized to `genesis-world` **[unverified]**): used only by DexMachina here; per-env unique objects remain a limitation per ProtoMotions.
- **CHOIS** https://github.com/lijiaman/chois_release (163 stars, MIT, ECCV 2024 oral) and **HOIDiNi** https://github.com/hoidini/HOIDiNi (80 stars, MIT, CVPRW 2026): kinematic diffusion HOI generators on OMOMO/GRAB producing SMPL-X + object poses. Not physics, but useful as **synthetic reference generators** feeding an InterMimic-style tracker (InterMimic's own pipeline does this with InterAct/text).

---

### Source datasets and gating (what each codebase actually needs)

| Dataset | License / gating | Used by |
|---|---|---|
| OMOMO (15 objects, ~10 h, SMPL-X + object pose) | MIT via https://github.com/lijiaman/omomo_release (Google Drive); HF mirror `snorfyang/omomo` | InterMimic, Omnigrasp, CHOIS, ResMimic, OmniRetarget, HDMI |
| GRAB | Registration + non-commercial MPI license | Omnigrasp, ManipTrans, DexTrack, HOIDiNi, PhysHOI |
| ARCTIC | Registration | DexMachina, ObjDex, ManipTrans |
| BEHAVE, InterCap, IMHD, HODome | Non-commercial MPI-style licenses | InterMimic (eval), InterAct |
| OakInk-V2, FAVOR, TACO | Per-dataset agreements | ManipTrans, DexTrack |
| SMPL-X body model | MPI registration | every SMPL-X repo |
| LAFAN1 | Ubisoft non-commercial | BeyondMimic, OmniRetarget |

---

### (1) Comparison table

| Codebase | Sim | Embodiment | Human data | Ref file has | Reward core | Obs (object) | RL | Envs / cost | Sim2real | Stars / license / last push |
|---|---|---|---|---|---|---|---|---|---|---|
| InterMimic | Isaac Gym (+Isaac Lab infer) | SMPL-X 153-DoF; G1+3-finger | OMOMO (+InterAct) | body, dof, obj pose/vel, 52 body→obj vectors, 53 contact bits | multiplicative body x obj x IG x contact-graph + energy | 21-D obj-rel + IG vectors, frames t+1,t+16 (student 1,2,4,16) | PPO teachers → Transformer student (DAgger+PPO) | 4096; not stated | no | 538 / MIT / 2026-04 |
| PhysHOI / SkillMimic / V2 | Isaac Gym | SMPL-X 153-DoF | BallPlay(-M), ParaHome | body, dof, obj pose/vel, 10 key bodies, contact bit | multiplicative body x obj x IG x CG (+vel reg) | obj local pose/vel + fingertip forces (+skill id) | PPO; V2 adds STG/STF/ATS + history enc | 2048; 9–24 h on 1x4090 | no | 249–445 / Apache (PhysHOI custom) / 2025-07 |
| Omnigrasp | Isaac Gym | SMPL-X | GRAB, OakInk, OMOMO | pkl (PHC fmt) + object mesh | pre-grasp + obj traj tracking + contact bonus | 20 future obj frames @15 Hz + 512 BPS | PPO over PULSE-X latent, GRU | ~3 d on 1xA100 | no | 159 / none / 2025-11 |
| ManipTrans | Isaac Gym | floating Inspire/Shadow/Allegro/XHand | GRAB, OakInk-V2, FAVOR, ARCTIC → DexManipNet | MANO 60 fps, wrist, obj pose, mesh | hand imitation + residual obj/contact-force | obj pose/vel + BPS + contact forces + tip dists | PPO imitator + residual | 4096; 1x4090 | Realman+Inspire | 342 / GPL-3 / 2025-10 |
| DexMachina | Genesis | 6 dex hands, bimanual | ARCTIC | hand kpts, obj articulated pose, contacts | obj tracking + kpt imitation + BC + contact | obj state, joint targets, contact forces | PPO; virtual-object curriculum | 12k/GPU; 1xL40S/H100 | no | 269 / MIT / 2026-03 |
| DexTrack | Isaac Gym | Allegro, LEAP | GRAB, TACO | retargeted kpts | tracking | — | PPO (+IL unreleased) | 22k; 1 GPU | partial | 135 / MIT / 2026-04 |
| HDMI | Isaac Lab | G1 29-DoF | video (GVHMR) | npz body states + obj pose + contact pts | summed: body 2/root 1/vel 1/joint 1/obj 2/contact 5 | obj pose in root frame + phase + contact pts | PPO residual, teacher→student | not stated | yes (G1) | 672 / none / 2026-01 |
| ResMimic | Isaac Gym (legged_gym) | G1 29-DoF | OMOMO via GMR + OptiTrack | pkl body + npz obj `trans/rot` | GMT tracking + obj point-cloud 2.0 + contact | obj [p,θ,v,ω] + ±10-frame ref window | PPO residual on GMT; decaying virtual obj PD | not stated | yes (G1, 4.5 kg) | 74 / Apache / 2025 |
| OmniRetarget (+BeyondMimic tracker) | — / Isaac Lab | G1 | OMOMO, LAFAN1, mocap | npz `qpos[T,43]` = base7+29j+obj7 | (tracker) DeepMimic 5 terms | — | — | — | yes | 82 / MIT / 2026-10 |
| HumanX | Isaac Gym? | G1 | video → XGen | npz/pkl robot+obj | SkillMimic-style | — | PPO | — | yes | 58 / Apache+CC / 2026-10 |
| ProtoMotions | Newton/Isaac Lab/IG/MuJoCo | SMPL, G1, H1 | AMASS etc. | npz/pt body only | DeepMimic/AMP/MaskedMimic | static scene objs | PPO | multi-GPU | yes (G1) | 2.4k / Apache / 2026-09 |
| HumanVLA / UniHSI / PhysHSI | Isaac Gym | humanoid / G1 | HITR / ScenePlan / AMASS+SAMP | goals + scenes | AMP + task goal / chain-of-contacts | goal/scene | PPO+AMP | 4 GPUs (HumanVLA) | PhysHSI yes | 136 / 248 / 321 |
| Human2Sim2Robot, SimToolReal | Isaac Gym / Isaac Lab | iiwa+Allegro | 1 RGB-D video; DexToolBench | obj pose traj + pre-manip hand pose | object-centric | obj pose | PPO-LSTM / SAPG | up to 24k envs | yes | 160 / 409 / MIT |
| DexMimicGen, MimicGen, RoboCasa, MimicLabs, HumanoidGen | MuJoCo | arms, GR1 | teleop demos | robomimic hdf5 | — (BC) | — | none | — | yes | 288–652 / NVIDIA SCL |

---

### (2) Ranked: which three to study/replicate first for MimicKit-manipulation

**Rank 1 — InterMimic (SMPL-X track) with SkillMimic-V2's augmentation.** It is the only mature, MIT-licensed, Isaac Gym *and* Isaac Lab codebase with dynamic-object tracking, contact-graph rewards, a documented `.pt` format, preprocessed OMOMO, and a G1+hands variant. Port these concretely:
- Motion format: extend MimicKit's `Motion`/`MotionLib` (`mimickit/anim/motion.py`, `motion_lib.py`) with per-frame `obj_pos(3) obj_rot(4)` (+ optional `contact_human(52)`, `contact_obj(1)`), and compute `obj_vel/obj_ang_vel` and the body→object distance vectors (`ig`) at load time from the object mesh, as `intermimic.py` does; write a one-off converter from `InterAct/OMOMO_new/*.pt` (slice indices listed in §A.1) into MimicKit pickles. Reuse their object URDFs in `isaacgym/src/intermimic/data/assets/objects/`.
- Reward: `compute_humanoid_reward` in `isaacgym/src/intermimic/env/tasks/intermimic.py` and the `w` dict in `data/cfg/omomo_train_new.yaml` (p=30, r=1.5, op=5, or=0.1, cg_hand=5, cg_other=5, cg_all=3, eg*). Implement as a new `DeepMimicHOIEnv` reward that multiplies MimicKit's existing pose/vel terms with `ro * rig * rcg`.
- Obs: `_compute_task_obs` (21-D object-relative block) and the `ig_obs` exp-weighted distance vectors, stacked for future frames {1,16}; MimicKit's tracker obs already includes future reference frames, so this is an append.
- Init/termination: `StateInit.Hybrid` + physical-state buffer; object-distance and contact-mismatch early termination.
- From SkillMimic-V2 (`Ingrid789/SkillMimic-V2`, `OfflineStateSearch` task and `--state_search_to_align_reward`): the STG/STF/ATS sampling to counter sparse clips.
- Isaac Lab side: `isaaclab/src/intermimic_lab/config/scene_cfg.py` + `smplx_humanoid_cfg.py` show how to spawn per-env rigid objects, which MimicKit's Isaac Lab engine lacks today.

**Rank 2 — HDMI (G1 / Isaac Lab track), with ResMimic's residual + virtual-object-controller tricks.** For the robot-embodiment path (your `g1_*` motions), HDMI's Isaac Lab MDP is the cleanest additive-reward HOI tracker and has real-world validation. Port:
- `active_adaptation/envs/mdp/commands/hdmi/{rewards,observations,terminations,randomizations}.py` (object_pos/ori tracking σ=0.25, `eef_contact_exp` with pos_sigma 0.1 / frc_sigma 10 / thres 2 N, weights in `cfg/task/base/hdmi-base.yaml`) and the residual action wrapper in `learning/ppo_roa.py`.
- Reference format: their `.npz` (body pos/quat/vel + joints + object pose + contact points in object frame) → MimicKit frames + object channel; note contact is specified as **points on the object**, which generalizes better than per-body bits.
- From ResMimic (`legged_gym/envs/g1/g1_hoi.py`, `g1_hoi_config.py`, `pose/utils/motion_lib_hoi.py`): the object **point-cloud tracking reward** (sample N surface points, `exp(-λ Σ‖P−P̂‖)`), oracle-contact reward `Σ ĉ_i exp(-λ/f_i)`, decaying virtual PD controller on the object for early training, and `termination_object_far_threshold=0.3`. Pair with OmniRetarget's ready-made G1 `qpos[T,43]` clips (MIT) as the first dataset, since MimicKit already has a G1 model.

**Rank 3 — ManipTrans (hand-only track) or DexMachina if you prefer Newton/Genesis-style engines.** If the goal includes fine finger control from GRAB/OakInk/ARCTIC, ManipTrans' two-stage design maps directly to MimicKit's AMP/DeepMimic structure: pretrain a hand imitator on kinematic hand data (MimicKit DeepMimic with a floating-base hand "character"), then a residual policy with object pose + contact-force rewards. Port `main/rl` env (obs: joint q/qd, wrist pose, obj 6D+vel, BPS 512, contact forces, tip distances), the residual action head, and the curriculum on termination thresholds; take hand URDFs from `dex-retargeting`/dex-urdf. GPL-3 means keep it as a reference, not a copy, if MimicKit's license matters. DexMachina's virtual-object-controller curriculum (decaying spring-damper on the object) is the single most reusable trick across all three tracks and is ~100 lines.

Why not the others first: Omnigrasp depends on PHC/PULSE latents and its data-processing code is unreleased; HumanX/Weave/DexWeave/LessMimic are weeks old or unreleased; DexTrack's key components are unreleased; UniHSI/PhysHSI/HumanVLA are goal/AMP style without dynamic-object tracking; DexMimicGen/RoboCasa are IL data generators.

---

### (3) Data that ships ready-to-use

- **InterMimic**: preprocessed OMOMO in `.pt` + object URDFs (Google Drive, ungated: https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view); teacher-corrected references and checkpoints (links in README). Full InterAct (30.7 h) requires the request form at https://github.com/wzyabcas/InterAct (non-commercial).
- **OmniRetarget_Dataset**: 4.0 h of G1 `.npz` (3 h object carrying from OMOMO, 0.5 h terrain, 0.5 h both) + URDF/OBJ assets, MIT, ungated: https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset. Best starting point for a G1 HOI tracker in MimicKit.
- **PhysHOI / SkillMimic / SkillMimic-V2**: BallPlay (8 clips) and a BallPlay-M subset plus ParaHome clips committed in-repo (`*/data/motions/`), ungated; full BallPlay-M not yet released.
- **HDMI**: task configs reference `data/motion/g1/omomo/sub1_suitcase_011` etc.; check the repo's data release notes for the processed `.npz` bundle **[availability unverified]**.
- **ResMimic**: example `kneel/squat/carry/chair` pkl+npz under `legged_gym/assets/motions` and suitcase URDF **[presence in public repo unverified; config references them]**.
- **ManipTrans / DexManipNet**: GRAB demo in repo; DexManipNet (3.3k episodes) from the project site/HF (`LiKailin/DexManipNet` mirror) **[gating unverified]**; OakInk-V2/FAVOR/ARCTIC under their own agreements.
- **DexMachina**: hand assets + processed ARCTIC assets released 2025-06-11 (ARCTIC raw needs registration).
- **DexToolBench** (SimToolReal): RGB-D human videos + processed data + sim envs via `download_dextoolbench_data.py` (MIT code; data terms unverified).
- **Omnigrasp**: `download_data.sh` gives models + one GRAB sample; GRAB/OakInk/OMOMO conversion not released.
- **HumanVLA HITR**: Google Drive link in README, ungated. **UniHSI ScenePlan**: released April 2024, CC BY-NC-SA.
- **GraspXL**: 500k+ object grasp motions on HF (CC BY-NC). **DexGraspNet 2.0**: HF `lhrlhr/DexGraspNet2.0`. **DexMimicGen**: HF `MimicGen/dexmimicgen_datasets` (CC-BY 4.0). **BONES-SEED** (GR00T-WBC): 142k motions, body only.
- **HumanX**: four example skills (npz/pkl) in repo. **Weave**: HF `appolyn/Weave` (9 objects) **[license unverified]**.
- Source mocap you will likely need regardless: OMOMO (MIT, Drive), GRAB/ARCTIC/BEHAVE (registration, non-commercial), SMPL-X (registration).

Sources: [InterMimic repo](https://github.com/Sirui-Xu/InterMimic), [InterMimic paper](https://arxiv.org/html/2502.20390), [SkillMimic](https://github.com/wyhuai/SkillMimic), [SkillMimic paper](https://arxiv.org/html/2408.15270), [SkillMimic-V2](https://github.com/Ingrid789/SkillMimic-V2), [SkillMimic-V2 paper](https://arxiv.org/html/2505.02094), [PhysHOI](https://github.com/wyhuai/PhysHOI), [PhysHOI paper](https://arxiv.org/html/2312.04393), [Omnigrasp](https://github.com/ZhengyiLuo/Omnigrasp), [Omnigrasp paper](https://arxiv.org/html/2407.11385v2), [ManipTrans](https://github.com/ManipTrans/ManipTrans), [ManipTrans paper](https://arxiv.org/html/2503.21860), [DexMachina](https://github.com/MandiZhao/dexmachina), [DexMachina paper](https://arxiv.org/html/2505.24853), [DexTrack](https://github.com/Meowuu7/DexTrack), [HDMI](https://github.com/LeCAR-Lab/HDMI), [HDMI paper](https://arxiv.org/html/2509.16757), [ResMimic](https://github.com/amazon-far/ResMimic), [ResMimic paper](https://arxiv.org/html/2510.05070), [OmniRetarget code](https://github.com/project-instinct/omniretargeting), [OmniRetarget dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset), [BeyondMimic tracker](https://github.com/HybridRobotics/whole_body_tracking), [GR00T-WBC/SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl), [ProtoMotions](https://github.com/NVlabs/ProtoMotions), [HumanVLA](https://github.com/AllenXuuu/HumanVLA), [UniHSI](https://github.com/InternRobotics/UniHSI), [PhysHSI](https://github.com/InternRobotics/PhysHSI), [CHOIS](https://github.com/lijiaman/chois_release), [HOIDiNi](https://github.com/hoidini/HOIDiNi), [DexMimicGen](https://github.com/NVlabs/dexmimicgen), [OKAMI](https://github.com/UT-Austin-RPL/OKAMI), [GMR](https://github.com/YanjieZe/GMR), [dex-retargeting](https://github.com/dexsuite/dex-retargeting), [Human2Sim2Robot](https://github.com/tylerlum/human2sim2robot), [SimToolReal/DexToolBench](https://github.com/tylerlum/simtoolreal), [ObjDexEnvs](https://github.com/cypypccpy/ObjDexEnvs), [GraspXL](https://github.com/zdchan/GraspXL), [DexGraspNet2](https://github.com/PKU-EPIC/DexGraspNet2), [UniDexGrasp++](https://github.com/PKU-EPIC/UniDexGrasp2), [Bi-DexHands](https://github.com/PKU-MARL/DexterousHands), [AMO](https://github.com/OpenTeleVision/AMO), [HumanX](https://github.com/wyhuai/HumanX), [Weave](https://github.com/xiaohu-art/Weave), [LessMimic](https://github.com/Yutang-Lin/LessMimic), [CLONE](https://github.com/humanoid-clone/CLONE), [human2humanoid/OmniH2O](https://github.com/LeCAR-Lab/human2humanoid), [HumanPlus](https://github.com/MarkFzp/humanplus), [VideoMimic](https://github.com/hongsukchoi/VideoMimic), [VisualMimic](https://github.com/visualmimic/VisualMimic), [OpenHomie](https://github.com/InternRobotics/OpenHomie), [InterAct](https://github.com/wzyabcas/InterAct), [InterPrior](https://sirui-xu.github.io/InterPrior/), [LocoMujoco](https://github.com/robfiras/loco-mujoco), [Open-Teach](https://github.com/aadhithya14/Open-Teach), [HumanoidGen](https://github.com/TeleHuman/HumanoidGen), [DexPBT docs](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/pbt.md), [DexterityGen](https://zhaohengyin.github.io/dexteritygen/), [OMOMO](https://github.com/lijiaman/omomo_release), [GRAB](https://github.com/otaheri/GRAB), [BEHAVE license](https://virtualhumans.mpi-inf.mpg.de/behave/license.html), [ARCTIC](https://github.com/zc-alexfan/arctic), [Workhorse](https://arxiv.org/html/2610.09117), [DexWeave](https://github.com/WU-CVGL/DexWeave).


---

## 6. Dataset survey

Verified on 2026-10-12 from a Babel compute node. "Open" = direct URL / ungated HF; "Reg" = account + license click-through; "Form" = Google/Office form with manual approval; "Email" = write to authors. Items marked **[unverified]** come from secondary sources only.

### 1.1 Comparison table

| Dataset (year) | Body / hand model | Object pose + mesh | Contact | Scale | fps | Access | License | Physics-ready? |
|---|---|---|---|---|---|---|---|---|
| **GRAB** (2020) | SMPL-X full body + hands + face | 6D + 51 ContactDB meshes | yes (vertex contact) | 10 subj, 51 obj, 1334 seq | 120 | Reg (grab.is.tue.mpg.de) + MANO/SMPL-X accounts | MPI non-commercial | via InterAct, OmniGrasp, HOIDiNi retarget |
| **ARCTIC** (2023) | SMPL-X + MANO (both hands) | 11 *articulated* objects (2 rigid parts + hinge), per-frame | dense hand–obj distance fields | 10 subj, 2.1M frames, 339 seq | 30 (mocap 60?) | Reg (arctic + smplx + mano accounts, scripted download) | MPI custom | partly (InterAct added support 2025-11; DexMachina) |
| **OakInk2** (2024) | SMPL-X (quat) + MANO both hands | 4×4 per object per mocap frame, scanned meshes (`object_raw/repair.tar`) | no explicit | 9 subj?, 627 seq, 4.01M frames | mocap 120 Hz / images 30 | **Open HF** `kelvin34501/OakInk-v2` (2.1 TB with images; ~37 GB anno) | CC BY-SA 4.0 (HF card) | yes — ManipTrans DexManipNet |
| **OakInk (v1)** (2022) | MANO | 100 scanned + 1800 virtual objects | yes (OakInk-Shape) | 792 clips, 230k imgs | 30 | per docs/datasets.md (form) **[unverified]** | MIT code | — |
| **HOI4D** (2022) | MANO (right hand) | category-level 6D, CAD per instance | no | 4 subj, 800 obj, 4000 seq, 2.4M RGB-D | 15 | OneDrive/Baidu direct (per site) | CC BY-NC 4.0 | no |
| **TACO** (2024) | MANO both hands (pkl) | tool+target 6D (npy), 206 OBJ | no | 2.3k seq, 151 triplets | 30 | Open Dropbox/OneDrive links (no form) | CC BY 4.0 | no |
| **DexYCB** (2021) / DexYCB-Mesh | MANO right (51-D PCA) | 21 YCB 6D | no | 10 subj, 1000 seq, 582k frames | 30 | Open (Drive 119 GB; image-free HF mirror 4.6 GB) | CC BY-NC 4.0 | no |
| **HO-3D v3 / H2O-3D** (2021) | MANO | 10 YCB 6D | no | 103k / 76k frames | 30 | OneDrive links (academic-only terms) | custom academic | no |
| **ContactPose** (2020) | MANO fits, 21 joints | 25 obj 6D, meshes | **thermal contact maps** | 50 subj, 2306 grasps, 2.9M RGB-D | static grasps + video | IEEE DataPort (login) or 2 GB Drive sample; Dropbox links dead | MIT (data) | no |
| **FreiHAND / HanCo** (2019/21) | MANO | none | no | 130k / 1.5M | — | Open direct zips (lmb.informatik.uni-freiburg.de) | research | no (hand only) |
| **H2O** (2021) | 21 kp both hands + 6D obj, meshes | yes | no | 4 subj, 571k frames | 30 | h2odataset.ethz.ch (form) | research | no |
| **FPHA** (2018) | 21 joints (magnetic) | 4 objects 6D | no | 100k frames | 30 | site form **[unverified]** | research | no |
| **GigaHands** (2025) | MANO both hands (48-D), 3D kp | 417 objects: scans + pose JSON (4 Globus zips) | no | 56 subj, 14k clips, 183M frames | 30 | **Open Globus HTTPS** (no account needed, verified) | CC BY-NC 4.0 | no (HandX uses it) |
| **EgoDex** (Apple, 2025) | 69-joint ARKit SE(3) skeleton (upper body + 25/hand) | **none** | no | 829 h, 338k demos, 194 tasks | 30 | **Open CDN zips** (2 TB; 17 GB test) | CC BY-NC-ND | no |
| **Ego-Exo4D** (2024) | 21-kp hands + 17-kp body (2D/3D), no MANO | no | no | 458 takes hand GT | 30 | Form (ego4d.dev, ~48 h) → AWS creds | Ego4D license | no |
| **EPIC-KITCHENS VISOR / EPIC-Fields** | masks + hand-object relations / camera poses | no 3D obj pose | 2D relation | 36 h / 671 videos | 60 | data.bris (open) | CC BY-NC 4.0 | no |
| **HOT3D / HOT3D-Clips** (Meta 2024) | MANO (15 thetas) + UmeTrack 22-DoF, both hands | 33 scanned GLB, 6D | no | 19 subj, 833 min, 3832 clips | 30 | **Open HF** `bop-benchmark/hot3d` (license click implied, not gated); full VRS at projectaria.com (email signup) | HOT3D license (CC BY-NC-SA for hands) | no (HandX uses it) |
| **Assembly101 / AssemblyHands** | 21-kp 3D hands (60 fps), no MANO | no | no | 3.0M images | 60 | HF `cvml-nus/assembly101` (auto-gated) | CC BY-NC 4.0 | no |
| **BEHAVE** (2022) | SMPL-H (156-D) 30 fps fits | 20 scanned objects, axis-angle+trans | annotated contacts | 8 subj, 321 seq | 30 (1 fps GT, 30 fps fits) | **Open direct** (datasets.d2.mpi-inf.mpg.de) | MPI non-commercial | via InterAct/InterMimic |
| **InterCap** (2022) | SMPL-X | 10 obj, meshes | contact | 10 subj, 223 seq, 67k frames | 30 | Reg (intercap.is.tue.mpg.de) | MPI | via InterAct |
| **OMOMO** (2023) | SMPL-H/-X | 15 large objects, Luma scans | no | 17 subj, 10 h | 30 | Drive link (README) ; InterMimic re-release `OMOMO_new.zip` open | MIT | **yes** (InterMimic, HOI-Retarget) |
| **HIMO** (2024) | SMPL-X + PNS gloves | multiple objects, OBJ | no | 3.3k seq, 4.08M frames | 30 | Form | custom | partly (InterMimic uses) |
| **CORE4D** (2025) | SMPL-X (PCA-12 hands), 2 persons | 4×4 obj, 38 OBJ categories | no | 1k real + 10k synth seq | 30 | **Open HF** `leolyliu/CORE4D` (255 GB) | CC BY 4.0 / MIT | no |
| **ParaHome** (2025) | 73-joint body+hand skeleton (gloves) + SMPL-X release | 22 articulated objects, per-part 4×4 + joint axes | no | 38 subj, 207 seq, 486 min | 30 | **Open Drive links** | CC BY-NC-SA 4.0 | via HOI-Retarget |
| **TRUMANS** (2024) | SMPL-X | scene objects, 4×4 transforms | no | 15 h, 100 scenes | 30 | Form → Drive | ? | no |
| **HUMOTO** (2025) | Mixamo rig (hands incl.) ; SMPL-X not public | 63 artist objects, 72 articulated parts | no | 735 seq, 7875 s | 30 | 10 % subset open on GitHub; full = Office form, CMU already licensed | Adobe Research License v1.1 | no |
| **IMHD²** (2024) | SMPL-H | 10 objects + IMU, scans | no | 295 seq, 892k frames | 60 | Form, but **HF `AfterJourney/IMHD-Dataset` ungated (986 GB)** | CC BY-NC-SA 4.0 | via InterAct/HOI-Retarget |
| **HODome (NeuralDome) / HOI-M3** | SMPL-X (2026 re-release), MHR, SMPL-H | 23 obj / multi-obj | no | 274 seq, 75M frames / 181M | 60 | Drive folder (gdown) ; HOI-M3 HF videos ungated | Apache-2.0 toolbox | via InterAct |
| **FORCE** (2025) | SMPL-H | 7 scanned objects + **resistance/force labels** | no | 450 seq | ? | **Open GitHub** | ? (MPI) | no |
| **CHAIRS** (2023) | SMPL-X | 81 sittable articulated objects (voxel/part) | no | 17.3 h, 46 subj | ? | Form (forms.gle/t4SjmJS4RPx7AFvFA) | ? | via InterAct |
| **PA-HOI** (2025) | SMPL-X + hands | 35 objects w/ **weights** | no | 562 seq | ? | Open Drive folder (listing blocked from CLI) | CC BY-SA 4.0 | no |
| **HOGraspNet** (2024) | MANO + HALO | 30 obj 6D (mocap) | contact maps + taxonomy | 99 subj, 1.5M frames | ? | Form (forms.gle/UqH15zN2PiBGQDUs7) | CC BY-SA 4.0 | no |
| **AffordPose** (2023) | MANO (synthetic grasps) | 641 objects, affordance | no | 26.7k poses | static | Open download (855 MB 7z) | ? | no |
| **ObMan** (2019) | MANO synthetic | ShapeNet | no | 150k | static | Email (yana.hasson) | ? | no |
| **DexGraspNet 2.0** (2024) | **robot hands** (LEAP) synthetic | cluttered scenes | n/a | 222 GB | static | Open HF `lhrlhr/DexGraspNet2.0` | CC BY-NC 4.0 | yes (grasp poses) |
| **UniDexGrasp** (2023) | ShadowHand synthetic | 3k objects | n/a | 1M grasps | static | GitHub/HF **[unverified]** | ? | yes |
| **RealDex** (2024) | ShadowHand real teleop | 52 obj | n/a | 2.6k seq | ? | Drive script | ? | robot data |
| **DexCanvas** (2025/26) | MANO 48-D + URDF 28-DoF MANO hand | 6D obj, per-frame **contact forces** | **yes, forces** | 112k traj, 55M frames (7k h, synthetic from 70 h real) | 200 | **Open HF** `Dex-GEM/DexCanvas` (958 GB) | ODbL | yes (URDF dof targets) |
| **DexterHand (DexterCap)** (2026) | MANO | rigid + Rubik's cube state | no | demo only | 20/60 | Open HF `pku-mocca/DexterHand` | Apache-2.0 | no |
| **HandX** (2026) | MANO both hands, 60-frame windows | no object pose | hand-hand | 54 h, 5.9M frames | ? | Open HF `alexzhang598/HandX` (8.9 GB) | Snap non-commercial | no |
| **DexWild / DexCap / EgoMimic / HumanPlus-1000 / OKAMI** | human-glove/robot teleop, hdf5 | mostly none | no | 33 h / 1.5 h / 3 tasks / 100 sessions | 10–30 | Open HF (+Drive for OKAMI) | MIT / CC BY / CC BY-NC | robot BC data |
| **Robot sets** (OXE, DROID, BridgeV2, LIBERO, RoboMimic, RH20T, AgiBot World, BEHAVIOR-1K) | robot joint/EEF only | no human | n/a | large | 10–30 | Open (RLDS / LeRobot / HF) | mostly CC BY | not human reference |
| **Motion-X++ / AMASS / InterHand2.6M** | SMPL-X 322-D / SMPL-H / 21-kp | **none** | no | 120k seq / 45 h / 2.6M | 30 / var / 5–30 | Form / Reg / HF-form | NC | body or hand only |
| **HiPHI** (Noitom, 2026) | 55-joint BVH optical mocap (gloves) | object trajectories + OBJ (39.8 % HOI) | no | 617 h, 132 performers | 90 | HF `noitomrobotics/HiPHI` **auto-gated** (accept license) | ModalityNet Open Research | no |

### 1.2 Already processed for physics imitation (most important for MimicKit)

1. **InterMimic** (CVPR'25) — `OMOMO_new.zip` (929 MB, open Drive) and `OMOMO_retarget.zip` (753 MB) are SMPL-X humanoid reference clips `Tensor[T,591]` @30 fps: root pos/quat, 51×3 DoF, 52 body positions + quats, object pos/quat, object-contact flag and per-body contact flags. Also ships Unitree G1 (29-DoF + 3-finger hands) configs. InterAct's `simulation/interact2mimic.py` converts any InterAct sub-dataset (OMOMO, BEHAVE, InterCap, GRAB, NeuralDome, IMHD², CHAIRS, ParaHome, ARCTIC) into this format + auto-generated object URDFs. Authorization form: https://docs.google.com/forms/d/e/1FAIpQLScMCfdd8BXzDBZ3iw0x5zA3KSTlD1F2GTaO8ylDG9Cj1upaPw/viewform. An HF mirror `guand1/interact-smplh` (2.6 GB, BEHAVE/GRAB/IMHD/NeuralDome SMPL-H) exists but is manually gated.
2. **HOI-Retarget** (ETH, 2026, HF `leggedrobotics/hoi-retarget`, open) — 13,904 robot-space clips (Unitree G1 + H2) from OMOMO/ParaHome/NeuralDome/CoRoleHOI/IMHD² with contact flags and object poses; `shinben0327/hoi-retarget-sapiens-k1` adds Robotis K1. 13 OMOMO object URDFs included.
3. **SkillMimic BallPlay-M** (subset on GitHub, Apache-2.0) and **PhysHOI BallPlay-V**: Isaac-Gym-ready `.pt` with 52-joint humanoid + ball; full BallPlay-M still "coming soon".
4. **ManipTrans DexManipNet** (HF `LiKailin/DexManipNet`, GPL-3.0, 8.3 GB) — 3.3k successful RL rollouts (Inspire hand, 12-DoF) on OakInk2 + FAVOR, with object URDFs; hdf5 with q/dq/state/contact tip forces at 60 fps.
5. **OmniGrasp** — no release of processed GRAB; processing scripts (`process_grab_raw.py`, `convert_grab_smplx.py`) require GRAB registration.
6. **HOIDiNi** (HF `Roey/hoidini`, open) — "GRAB_RETARGETED_compressed" (2.7 GB) + SMPL/MANO models redistributed (license caveat).
7. **DexCanvas** — 28-DoF MANO-URDF `urdf_dof_target` + contact forces: directly usable as dexterous-hand tracking targets.
8. **InterPrior** (CVPR'26) and **InterReal** (2026) build on InterMimic data; no new dataset released. **DreamMimic**, **DeVI** (2026) likewise **[unverified]**.

### 1.3 Notable 2025–2026 additions I added

GigaHands (2025, open), EgoDex (2025, open), HOT3D-Clips (open HF), CORE4D (open HF), ParaHome SMPL-X (2025), HUMOTO (2025), PA-HOI (2025), IMHD² on HF (2026), HODome SMPL-X re-release (2026-05), HOI-Retarget (2026), HandX (2026), DexCanvas (2025/26), DexterCap/DexterHand (2026), HiPHI (2026), OpenEgo (2025; aggregator of 6 ego sets, 21-joint MANO in camera frame; download via openegocentric.com, HDF5 **[unverified]**), HO-Cap (2024, open HF `JWRoboticsVision/HO-Cap-Dataset`, CC BY 4.0, 119 GB), MMHOI (WACV'26, Zenodo CC BY-SA), Glove2Hand/ART-Glove (2026, tactile gloves, release status **[unverified]**).

### 1.4 Access notes (registration-only sets)

- GRAB / ARCTIC / InterCap / AMASS / MANO / SMPL-X: create account at `*.is.tue.mpg.de`, accept license, then scripted download (ARCTIC: env vars `ARCTIC_USERNAME/PASSWORD`, `SMPLX_*`, `MANO_*`, `raw_seqs` 215 MB, `meta` 91 MB suffice for motion).
- InterAct: Google form above; HIMO: https://docs.google.com/forms/d/e/1FAIpQLSdl5adeyKxBSBFZpgs0A7-dAouRkMFAGUP5iz3zxGDj_PhB1w/viewform; IMHD²: https://forms.gle/3MDh3b4szhFwcYa26 (or HF); CHAIRS: https://forms.gle/t4SjmJS4RPx7AFvFA; HOGraspNet: https://forms.gle/UqH15zN2PiBGQDUs7; HUMOTO full: https://forms.office.com/r/3a7jgFi5UH (CMU is on the licensed-institution list — ask the lab's designated contact); Ego-Exo4D: ego4d.dev/request/ego-exo4d (AWS creds); Motion-X: Google form; HOT3D full: projectaria.com email signup; HiPHI / Assembly101 / interact-smplh: HF "accept conditions" button.

## 2. Download report

Location: `/data/user_data/vawale/MimicKit/manip_data/` (1.4 GB, 22 dataset dirs). Manifest: `MANIFEST.md`; full key/shape dump: `_SCHEMA_DUMP.txt`. All pulls were ungated. Tricks used: zip central-directory range requests (EgoDex 17 GB zip, GigaHands 1.7 GB scans, InterMimic/ParaHome Drive zips), partial `tar.gz` streaming (GigaHands hand_poses), sparse git clones, selective HF files. `h5py`, `gdown`, `pyarrow` were installed into `/data/user_data/vawale/envs/mimickit`.

Verified schemas (abridged; full version in MANIFEST.md):

- **InterMimic `.pt`** `[T,591]` 30 fps: `0:3` root_pos, `3:7` root quat, `9:162` dof (51×3), `162:318` body pos (52×3), `318:321` obj pos, `321:325` obj quat, `330` obj contact, `331:383` body contacts, `383:591` body quats (52×4). Saved with `requires_grad` — `.detach()`.
- **SkillMimic `.pt`** `[T,337]` 60 fps: `0:3` root_pos, `3:6` root exp-map, `9:165` dof (52×3), `165:324` body pos (53×3), `324:327` obj pos, `327:330` obj exp-map (negated in loader), `336` contact. **PhysHOI** `[T,331]`: same with `165:318` body (51×3), `318:321`/`321:324` object, `330` contact.
- **HOI-Retarget parquet** 30 fps: `root_pos[T,3]`, `root_rot[T,4]`, `dof_pos[T,29]` (G1, `dof_names`), `object_pos/rot/vel`, `per_link_contact_flags[T,4]`, `robot_foot_ground_contact_flags[T,2]`, `fixed_contact_points_per_frame_in_object_frame[T,4,3]`, `object_model_path` (URDF).
- **DexManipNet hdf5**: `rollouts/successful/rollout_k/{actions, q_rh[T,12], dq_rh, state_rh[T,13]=pos+quat+v+ω, state_manip_obj_rh[T,13], joint_state_rh[T,234]=18×13, tip_force_rh[T,15]}` + `seq_info.json` (hand side, primitive, object URDF path, start_frame, seq_len); 60 fps.
- **DexYCB-mesh**: `pose.npz {pose_y[N,n_obj,7] xyzw+t, pose_m[N,1,51] = 3 aa + 45 PCA + 3 t}` world frame, m; per-cam `labels_*.npz` with `joint_3d[1,21,3]`.
- **OakInk2 pkl**: `raw_smplx` (body_pose `[1,21,4]` quats, hand poses `[1,15,4]`, `world_rot[1,4]`, `world_tsl[1,3]`, `body_shape[1,300]`), `raw_mano` (`rh__pose_coeffs[1,16,4]`, `rh__tsl`, `rh__betas`), `obj_transf{oid:{frame:4×4}}`; 1147 mocap frames vs 287 image frames (≈120 Hz mocap).
- **DexCanvas parquet** 200 Hz: `hands[].{mano_pose_rotvec_rad[T,48], joint_positions_world_m[T,21,3], urdf_dof[T,28], urdf_dof_target[T,28], mano_shape[10]}`, `objects[].{position_world_m, rotation_world_rotvec_rad}`, `contacts[T][]` with force vectors.
- **DexterHand npz**: `hand_translations[N,3]`, `hand_orientations_axis_angle[N,3]`, `hand_poses[N,45]`, `hand_shapes[N,10]`, `object_translations[N,3]`, `object_orientations_quat_xyzw[N,4]`, `metadata{fps,...}`.
- **EgoDex hdf5**: `transforms/<69 joints>[N,4,4]` (ARKit world), `confidences/<joint>[N]`, `camera/intrinsic`, attrs `llm_description`; no object pose.
- **GigaHands**: `params/<seq>.json {left/right: poses[T,48], shapes[1,10], Rh[T,3], Th[T,3]}`, `keypoints_3d/<seq>/left.jsonl` (21×[x,y,z,conf]/frame), `keypoints_3d_mano/<seq>.json [T,126]`; object scans as textured OBJ.
- **BEHAVE 30 fps**: `smpl_fit_all.npz {poses[T,156] SMPL-H, betas, trans, gender}`, `object_fit_all.npz {angles[T,3], trans[T,3]}` (Kinect frame; `calibs.zip` 396 MB not pulled); 299 sequences, 20 OBJ/PLY objects.
- **HOT3D clip**: per-frame `hands.json` (MANO `thetas[15]`+`wrist_xform[6]`, UmeTrack `T_world_from_wrist` + `joint_angles[22]`), `objects.json` (`T_world_from_object` quat wxyz + t), `cameras.json`, `__hand_shapes.json__` (MANO betas[10]); objects as GLB with `models_info.json`.
- **ParaHome**: `joint_positions[T,73,3]`, body/hand orientations 6D `[T,23,6]`/`[T,40,6]`, `object_transformations{frame:{obj_part:4×4}}`, `joint_info.pkl` axes/pivots; SMPL-X `smplx_pose.pkl {body_pose[T,63], global_orient, transl, hand_pose[T,90]}`.
- **FORCE npz**: `smpl_pose[T,156]`, `smpl_trans`, `obj_pose[T,3]` aa, `obj_trans`; 7 PLY objects.
- **HUMOTO**: FBX/GLB Mixamo rigs (3 of 70 public seqs). **CORE4D**: object OBJs + labels only (motions are 8–12 GB zips). **ContactPose**: Dropbox links in `data/urls.json` are dead (HTML returned); data is on IEEE DataPort (login) or a 1.98 GB Drive sample — not pulled.

## 3. Recommendations

**Top 5 for body + hands + object, physics-ready (MimicKit manipulation)**
1. **InterMimic/InterAct-processed OMOMO (+BEHAVE/IMHD²/NeuralDome/CHAIRS/ParaHome/ARCTIC via `interact2mimic.py`)** — already a full-body SMPL-X humanoid tracking format with per-body contact flags and object URDFs; ~30 h unified at 30 fps; one form for the aggregate. Closest to MimicKit's motion format (root + dof + object state).
2. **GRAB** — gold-standard whole-body SMPL-X grasping at 120 fps with contact; small objects, finger detail; registration only; many retargeting pipelines exist (OmniGrasp, HOIDiNi, InterAct).
3. **ARCTIC** — the only mocap set with articulated objects + SMPL-X + MANO; `raw_seqs` (215 MB) is enough; needs 3 MPI accounts.
4. **HOI-Retarget (G1/H2/K1)** — if targeting a humanoid robot rather than SMPL-X, 13.9k contact-labelled robot-space clips are downloadable today with no form (OMOMO meshes included).
5. **OakInk2** (+ DexManipNet) — open, 4M frames of bimanual long-horizon tasks with SMPL-X body + MANO + object 4×4 at 120 Hz; DexManipNet gives sim-validated hand rollouts. Honorable mentions: ParaHome (articulated home objects, open), HUMOTO (CMU already licensed), BEHAVE (open, large objects).

**Top 3 for dexterous-hand-only**
1. **DexCanvas** — MANO + 28-DoF URDF targets + measured contact forces at 200 Hz, ODbL, open; directly a hand-tracking reference set.
2. **OakInk2 / DexManipNet** — MANO both hands + objects at 120 Hz with validated Inspire-hand rollouts and object URDFs.
3. **GigaHands** — 14k bimanual clips, MANO + 3D keypoints + 417 object scans and poses, fully open via Globus HTTPS (TACO is the CC BY runner-up with tool-use bimanual MANO + OBJ at 30 fps; HOT3D-Clips for egocentric with 6D objects).

Files: `/data/user_data/vawale/MimicKit/manip_data/MANIFEST.md`, `/data/user_data/vawale/MimicKit/manip_data/_SCHEMA_DUMP.txt`, scratch scripts in `/scratch/vawale_manip_tmp/` (`dl_zip_range.py` is reusable for any remote zip).


---

## 7. MimicKit gap analysis and design

Repo: `/home/vawale/my_things/codes/MimicKit` (fork of xbpeng/MimicKit, branch `main`). All paths below are relative to the repo root; assets/motions under `data/assets/*` and `data/motions/*` are symlinks into `/data/user_data/vawale/MimicKit/data/`.

### 1. Current architecture map

**Entry and config composition.** `mimickit/run.py:97-132` builds env (`env_builder.build_env`, `mimickit/envs/env_builder.py:8-53`, dispatch on `env_name`) then agent (`mimickit/learning/agent_builder.py:5-44`, dispatch on `agent_name`). Args come from `--key value` tokens; `mimickit/util/arg_parser.py:15-38` keeps the *first* occurrence, and `run.py:22-31` loads the command line before `--arg_file`, so CLI overrides the file. `args/*.txt` are just 4 lines (`--num_envs`, `--engine_config`, `--env_config`, `--agent_config`, `--out_dir`). An env yaml may carry an `engine:` sub-dict that overrides engine yaml keys (`env_builder.py:63-85`).

**Engine abstraction.** `mimickit/engines/engine.py:17-228` defines `Engine` with `create_env`, `create_obj(env_id, obj_type, asset_file, name, is_visual, enable_self_collisions, fix_root, start_pos, start_rot, color, disable_motors)` (L32-33), `ObjType.rigid|articulated` (L13-15), per-object getters/setters for root pos/rot/vel/ang_vel, dof pos/vel, body pos/rot/vel/ang_vel (L85-166), `get_contact_forces` (L129), `get_ground_contact_forces` (L133), `set_body_forces` (L169). Backends:

- Isaac Gym `mimickit/engines/isaac_gym_engine.py`: `create_obj` L136-211 (asserts `.xml`, L141-142; PD gains read from MJCF joint `stiffness`/`damping`, L179-181; torque limits from actuator `motor_effort`, L183-185); ground plane L548-556; sim tensors L755-857 (`objs_per_env = num_objs // num_envs`, L768-769, i.e. every env must contain the same object list in the same order); rigid objects with 0 DOF handled in reset L629-653; `get_ground_contact_forces` L311-320 is a *height mask* (net contact force on bodies with z < `ground_contact_height`=0.3 m), not a true ground filter.
- Isaac Lab `mimickit/engines/isaac_lab_engine.py`: `create_obj` L150-179 only records an `ObjCfg`; prims are built in `initialize_sim` L117-125 via `_build_objs` L815-863 with cloning that requires identical configs per obj slot (L835-843); `.xml` is mapped to a sibling `.usd` (L737-741); rigid prim L875-906 (`kinematic_enabled=fix_root`); ground-filtered `ContactSensor` per object L1079-1105; `get_contact_forces` L332-338 returns net force including object contacts.
- Newton `mimickit/engines/newton_engine.py`: `create_obj` L414-432 → `_create_obj_builder` L947-984 (`add_mjcf`/`add_urdf`); `_validate_envs` L791-799 enforces identical objects per env; `SensorContact` vs `ground*` with `include_total=True` L1012-1035, so both total and ground-only forces are available.

**Kinematic character model.** `mimickit/anim/kin_char_model.py`: `JointType {ROOT, HINGE, SPHERICAL, FIXED}` L9-13; DOF dims 0/1/3/0 L23-34; `dof_to_rot` L146-159, `rot_to_dof` L161-175, `forward_kinematics` L177-209, `compute_dof_vel` L221-250. Parsers: MJCF `mimickit/anim/mjcf_char_model.py:17-80` (DFS over `<body>`; `_parse_joint` L112-131: 0 joints→FIXED, 1→HINGE, exactly 3 hinges→SPHERICAL, any other count asserts "Series joints are not supported" L128-129; joint `pos` offsets assert L137-140/161-165), URDF `urdf_char_model.py:115-123` (revolute, fixed only), USD `usd_char_model.py`.

**Env hierarchy.** `BaseEnv` (`envs/base_env.py:18`) → `SimEnv` (`envs/sim_env.py:18`; construction order L21-40: engine → `_build_envs` → `initialize_sim` → action space → sim tensors → data buffers; `step` L77-89; `_post_physics_step` L171-178 = time → `_update_misc` → obs → info → reward → done) → `CharEnv` (`envs/char_env.py:17`; one articulated "character" per env L82-91; action bounds from DOF limits L122-205) → `DeepMimicEnv` (`envs/deepmimic_env.py:12`) → `AMPEnv` (`envs/amp_env.py:9`) → `ASEEnv`/`ADDEnv`/`SMPEnv` → `TaskLocationEnv`, `TaskSteeringEnv`, `TaskDodgeballEnv` (`envs/task_dodgeball_env.py:9`). `StaticObjectsEnv` (`envs/static_objects_env.py:5-38`) and `ViewMotionEnv` (`envs/view_motion_env.py:10`) hang off DeepMimic/Char.

**Observations.** `char_env.compute_char_obs` L411-446: `[root_h?] + root_rot(tan-norm,6) + root_vel(3) + root_ang_vel(3) + joint_rot(6 per non-root body, incl. fixed) + dof_vel(D) + key_pos(3 per key body, root-relative)`; `global_obs=False` rotates into the heading frame. `deepmimic_env.compute_deepmimic_obs` L680-721 appends phase (L585-600) and target obs for `tar_obs_steps` future frames (`_fetch_tar_obs_data` L563-581; `compute_tar_obs` L626-677: future root pos relative to current root, root rot, joint rots, key pos). The policy/critic are flat MLPs over the whole vector (`learning/ppo_model.py:37-59`, `learning/nets/fc_2layers_1024units.py`).

**Reward.** `deepmimic_env.compute_reward` L787-846:

```
r = w_pose·exp(-s_pose·Σ_j w_j θ_j²)                       # L795-796, 835
  + w_vel ·exp(-s_vel ·Σ_d w_d (q̇_d-q̂̇_d)²)                 # L798-799, 836
  + w_root·exp(-s_root·(‖p_r-p̂_r‖² + 0.1·θ_root²))          # L801-820, 837
  + w_rootv·exp(-s_rootv·(‖v_r-v̂_r‖² + 0.1‖ω_r-ω̂_r‖²))      # L822-826, 838
  + w_key ·exp(-s_key·Σ_k ‖(p_k-p_r)-(p̂_k-p̂_r)‖²)           # L811-813, 828-831, 839
```
Root xy error is zeroed unless `track_root = enable_tar_obs and global_obs` (L275-276, L803-804). Defaults (`data/envs/deepmimic_g1_env.yaml`): weights 0.5/0.1/0.15/0.1/0.15, scales 0.25/0.01/5/1/10.

**Termination.** `compute_done` L724-785: TIME at `episode_length`; SUCC at motion end for CLAMP clips (L737-738); FAIL if any body not in `contact_bodies` has `|ground_contact_force| > 0.1` (L744-750) or max body-position deviation > `pose_termination_dist` (L752-777). Fails are suppressed on the first step (L780-782).

**RSI.** `_reset_char` L151-157 → `_reset_ref_motion` L175-197 (sample clip by weight `motion_lib.sample_motions` L30-32, time uniformly in `[0, len]` if `rand_reset` L277-285, fetch blended frame, FK) → `_ref_state_init` L202-215 writes root/dof pos+vel to the engine. Every step, `_update_ref_motion` L232-252 refreshes the reference at `time_buf + motion_time_offset` (L217-222).

**Agents.** `BaseAgent` (`learning/base_agent.py:27`) only sees `obs`, `reward`, `done`, `info`; obs normalizer L312-314; nothing body-specific. `AMPAgent` (`learning/amp_agent.py`) reads `info["disc_obs"]` L66-74, pulls demo samples through `env.fetch_disc_obs_demo` L95 (`amp_env.py:32-35`), mixes `r = w_task·r_task + w_disc·r_disc` L117-132, and builds the discriminator from `env.get_disc_obs_space()` (`learning/amp_model.py:43-46`). Disc features: `amp_env.compute_disc_obs` L336-359 over a history of `num_disc_obs_steps` frames kept in circular buffers (L98-158, L165-184); demo history is sampled from the motion lib (L67-90).

**Existing object precedents.** (a) `StaticObjectsEnv` L17-37: `objects: [{pos, file, rot?}]` spawned as `ObjType.rigid`, `fix_root=True` (kinematic), no obs/reward coupling — used by `data/envs/vault_*_env.yaml`. (b) `TaskDodgeballEnv`: dynamic rigid projectiles L57-69, reset by root-state writes L151-167, object obs = heading-frame relative pos+vel L289-302, contact via `get_contact_forces` L120-129, hit-termination L304-326. (c) Held tools are *welded*: `sword`/`shield` are fixed child bodies of `right_hand`/`left_lower_arm` in `humanoid_sword_shield.xml:50-53,68-70`. (d) Visual "ghost" reference character: `is_visual=True`, `disable_motors=True` L318-328.

**Tools.** `tools/gmr_to_mimickit/gmr_to_mimickit.py:42-120` (GMR dict `{fps, root_pos[N,3], root_rot xyzw[N,4], dof_pos[N,D]}` → frames), `tools/smpl_to_mimickit/smpl_to_mimickit.py:74-203` (AMASS npz → SMPL MJCF order, Y-up→Z-up L47-48/136-139, hand DOFs zeroed L126, floor calibration L161-167). No HOI/object converter exists.

### 2. Motion format spec (exact, as implemented)

File: pickle (`mimickit/anim/motion.py:9-23`, `save` L32-43).

```
{
  "loop_mode": int,            # LoopMode.CLAMP=0, LoopMode.WRAP=1   (motion.py:5-7)
  "fps":       float|int,
  "frames":    list[list[float]] -> np.float32 [N, 6 + D]
}
frame[0:3] = root position (m, Z-up world)          # motion_lib.extract_pose_data L11-15
frame[3:6] = root rotation, 3D exponential map
frame[6:]  = joint DOFs in DFS order of the character file:
             spherical joint = 3D exp map, hinge = 1 angle (rad), fixed = 0 entries
length (s) = (N-1)/fps                                # motion.py:45-48
```
Verified: `humanoid_spinkick.pkl` → frames (78, 34) = 6 + 28 DOF, fps 60, WRAP; `g1_walk.pkl` → (125, 35) = 6 + 29, fps 120. `init_pose` in env yamls uses the same layout (`char_env.py:32-50`). No velocities are stored: `MotionLib._process_data` L161-196 finite-differences root vel (L176-178), root ang vel via `quat_diff → exp_map × fps` (L180-183), dof vel via `compute_frame_dof_vel` (L186); WRAP clips add a per-loop xy root offset `end − begin` (L189-195, used at L138-147). Sampling: `calc_motion_frame` L66-92 lerps root pos, slerps root/joint quats between `frame_idx0/1` (L123-136); velocities are taken un-blended from `frame_idx0` (L75-81). Datasets: yaml `{motions: [{file, weight}]}` (L198-231), sharded across workers (L219-229). All clips are concatenated into flat tensors `_frame_root_pos/_rot/_joint_rot` (L275-277), so a per-frame object channel slots in naturally alongside them.

### 3. Gaps for manipulation

| Component | Current | Needed | Effort |
|---|---|---|---|
| Motion file schema | char-only `frames` | optional `objects` block: per-object pose track, asset ref, contact labels | S (0.5 d) |
| `MotionLib` | char tensors only (`motion_lib.py:233-278`) | `HOIMotionLib`: object pos/rot/vel/ang_vel/contact tensors, blended sampling, loop offset applied to objects | M (2 d) |
| Hand DOF | none in any asset (no finger/thumb/palm in `data/assets/*/*.xml`); G1 wrists are terminal bodies (`g1.xml:204,252`); SMPL hands are 3-DOF stubs zeroed by converter | `g1_dex3.xml` (+`.usd` for Isaac Lab): hand subtrees under `*_wrist_yaw_link`, one hinge per nested body (MJCF parser forbids 2-joint bodies and joint offsets, `mjcf_char_model.py:128-140`), gains/limits per finger joint; `init_pose` length 6+D and `joint_err_w` length = bodies−1 updated (`deepmimic_env.py:125`) | M (2-3 d incl. sim tuning) |
| Object spawning | kinematic (`StaticObjectsEnv`) or scripted projectiles; identical per env enforced by all engines | dynamic rigid objects from env `objects:` list, validated against motion file; optional visual ref objects | S (1 d) |
| Reset | char only (`_ref_state_init` L202-215) | write object root pose/vel from reference at RSI; optional pose noise | S |
| Observations | char + future char targets; `TaskDodgeball` shows object obs pattern | object pose/vel (heading frame), hand↔object relative vectors, future object targets at `tar_obs_steps`, contact-phase flag | M (1-2 d) |
| Reward | 5 tracking terms, no object terms | + object pose, object vel, hand-object relative, contact-match terms | S (1 d) |
| Termination | ground-contact fall + pose deviation; Isaac Gym "ground" = any contact below 0.3 m (`isaac_gym_engine.py:311-320`) | + object deviation / drop; hands in `contact_bodies`; raise or disable height mask when objects rest low | S |
| Engine API | per-body *net* contact force only; no hand↔object pair query; no runtime mass/friction/scale, no object colour change | pair-filtered contact (Isaac Lab `ContactSensor(filter_prim_paths_expr=obj)`, Newton `SensorContact(counterpart_shapes=obj)`; Isaac Gym approximation = net force ∧ proximity); later: physical-property randomisation | M now, L for DR |
| AMP discriminator | char history only (`amp_env.py:336-359`) | object + hand-object features in both agent history and demo samples; agent code untouched | M (1-2 d) |
| Tools | GMR/SMPL converters | `convert_hoi_to_mimickit.py`: human+hand retarget output + object tracks + contact labels in one frame-aligned file | M-L (dataset-dependent) |
| Visualisation | ghost ref char only | ghost ref objects offset by `ref_char_offset`; `view_motion_hoi` env | S |
| Multi-clip datasets with different objects | n/a | fixed object-slot list per dataset; absent objects parked far away with `active=0` | M |
| Sim fidelity for grasping | `contact_offset 0.02`, 4 position iters, 120 Hz (`isaac_gym_engine.py:700-711`, `data/engines/*.yaml`) | per-env-yaml `engine:` overrides (240 Hz sim, more iterations), friction on hand geoms | S (config) |

### 4. Proposed minimal-change design

#### 4.1 Motion file v2 (backward compatible)

```python
{
  "loop_mode": 0,                 # CLAMP recommended; WRAP applies the root xy loop delta to objects too
  "fps": 30,
  "frames": [N, 6 + D],           # D now includes hand DOFs in DFS order of the target MJCF
  "objects": [                    # optional; absence == v1 file
    { "name": "box",
      "asset": "data/assets/objects/box_0.1.xml",   # default asset, overridable in env yaml
      "frames": [N, 6],           # world pos (3) + rotation exp-map (3), same convention as the root
      "contacts": [N, S],         # optional float/bool per contact site, S = len(contact_sites)
      "active": [N] }             # optional 0/1 (object exists / is relevant at this frame)
  ],
  "contact_sites": ["left_hand_palm_link", "right_hand_palm_link"]   # body names in the character
}
```
Objects are stored as 6-vectors so `extract_pose_data`-style slicing and `exp_map_to_quat` (`motion_lib.py:116`) are reused; quaternions are formed at load, so double-cover is irrelevant after `quat_pos`.

#### 4.2 `HOIMotionLib(MotionLib)` — new file `mimickit/anim/hoi_motion_lib.py`

- `_load_motion_pkl`: call super logic, additionally read `objects`; require every clip to expose the same ordered object-name list (or pad with `active=0` tracks); stack `_frame_obj_pos [F,O,3]`, `_frame_obj_rot [F,O,4]`, `_frame_contact [F,O,S]`, `_frame_obj_active [F,O]`.
- `_process_data`: after super, finite-difference `_frame_obj_vel`, `_frame_obj_ang_vel` exactly like L176-183, and build `_motion_wrap_delta` broadcasting to objects.
- `calc_obj_frame(motion_ids, times) -> (obj_pos, obj_rot, obj_vel, obj_ang_vel, contact, active)`: reuse `_calc_frame_blend` L123-136 and `_calc_loop_offset` L138-147; lerp/slerp pose, nearest-frame contact/active.
- `get_num_objects()`, `get_object_names()`, `get_object_assets()`, `get_num_contact_sites()`.

#### 4.3 `HOIMixin` + `DeepMimicHOIEnv` — new file `mimickit/envs/deepmimic_hoi_env.py`

A mixin keeps AMP/ASE variants to one-liners (`class AMPHOIEnv(HOIMixin, AMPEnv)`). Hooks, each extending the base method by `super()`:

- `_build_envs` (overrides `deepmimic_env.py:295-302`): build the kinematic model, load `HOIMotionLib` *before* creating envs so object slots are known, then `super()`.
- `_build_env` (after `deepmimic_env.py:304-316`): for each object slot create `ObjType.rigid`, `fix_root=False`, asset from env yaml override or the motion file, `start_pos` parked at `[10+i, 10, 0.1]` like `task_dodgeball_env.py:59`; when `_enable_ref_char()` also create a visual ghost object (`is_visual=True`). Record `self._obj_ids`, `self._ref_obj_ids`.
- `_build_sim_tensors`: `_ref_obj_pos/rot/vel/ang_vel`, `_ref_contact`, `_hand_body_ids` via `_build_body_ids_tensor` (`char_env.py:333-343`).
- `_reset_ref_motion` / `_update_ref_motion`: also call `calc_obj_frame`.
- `_ref_state_init`: after super, `set_root_pos/rot/vel/ang_vel(env_ids, obj_id, ref…)` per object (engine root state is authoritative for 0-DOF actors, `isaac_gym_engine.py:629-637`); optional `obj_reset_noise`.
- `_update_ref_char`: also move ghost objects by `_ref_char_offset`.
- `_compute_obs`: `torch.cat([super_obs, compute_hoi_obs(...)])` with, per object: object pos relative to root (heading frame unless `global_obs`), rot tan-norm (6), lin/ang vel (6), `active` flag; per (hand, object): `p_o − p_h` (3); per future step in `tar_obs_steps`: target object pos/rot relative to the current root (mirrors `compute_tar_obs` L626-677) and target contact labels.
- `_update_reward`: `r = r_track + r_hoi` where `r_track` is `compute_reward` with re-balanced weights and

```
r_obj     = w_obj   · exp(-s_obj   · Σ_o a_o (‖p_o − p̂_o‖² + 0.1·θ(q_o,q̂_o)²))
r_objv    = w_objv  · exp(-s_objv  · Σ_o a_o (‖v_o − v̂_o‖² + 0.1‖ω_o − ω̂_o‖²))
r_rel     = w_rel   · exp(-s_rel   · Σ_o Σ_h a_o ‖R̂_oᵀ(p_h − p_o) − R̂_oᵀ(p̂_h − p̂_o)‖²)   # hand-object relative, object frame
r_contact = w_cont  · exp(-s_cont  · Σ_o Σ_h |c_{h,o} − ĉ_{h,o}|)
```
with `c_{h,o} = 1[‖f_h‖ > f_min ∧ ‖p_h − p_o‖ < d_min]` from `get_contact_forces` (net force; pair-filtered once the engine API exists). Suggested start: `w_pose .3, w_vel .05, w_root .1, w_rootv .05, w_key .1, w_obj .2, w_objv .05, w_rel .1, w_cont .05`; `s_obj 10, s_objv 1, s_rel 20, s_cont 1`.
- `_update_done`: after super, FAIL when `Σ_o a_o ‖p_o − p̂_o‖ > obj_termination_dist` (0.5 m) or object z below `obj_drop_height` while `ĉ`=1; hands must be in `contact_bodies` (`compute_done` L744-750), and on Isaac Gym set `engine: {ground_contact_height: 0.05}` so object contacts near the floor are not mistaken for falls.

#### 4.4 Object asset / env config schema

```yaml
env_name: "deepmimic_hoi"
char_file: "data/assets/g1/g1_dex3.xml"
motion_file: "data/motions/g1_hoi/g1_pick_place_box.pkl"
objects:                               # ordered; must match motion-file object names
  - {name: "box", file: "data/assets/objects/box_0.1.xml", visual_ref: true}
hand_bodies: ["left_hand_palm_link", "right_hand_palm_link"]
key_bodies: [..., "left_hand_palm_link", "right_hand_palm_link"]
contact_bodies: [feet..., "left_hand_palm_link", "right_hand_palm_link", "left_hand_*_tip"...]
global_obs: True
enable_tar_obs: True
tar_obs_steps: [1, 2, 3, 10]
obj_termination_dist: 0.5
obj_reset_noise: 0.0
reward_obj_pos_w: 0.2 ... reward_contact_scale: 1.0
engine: {sim_freq: 240, control_freq: 30, ground_contact_height: 0.05}
```
`args/deepmimic_g1_pick_place_args.txt` is then the standard 5 lines pointing at this yaml and a copy of `data/agents/deepmimic_g1_ppo_agent.yaml` (unchanged; obs/action sizes are inferred at build time, `sim_env.py:42-52`, `char_env.py:122-153`). Adding `"deepmimic_hoi"`, `"amp_hoi"`, `"view_motion_hoi"` to `env_builder.py:14-51` is the only edit to existing code paths.

#### 4.5 Hand asset: `data/assets/g1/g1_dex3.xml`

Append Dex3-1 (7 DOF/hand: thumb 3, index 2, middle 2) subtrees under `left_wrist_yaw_link` (`g1.xml:204`) / `right_wrist_yaw_link` (L252), each finger segment as a nested `<body>` with exactly one `<joint type="hinge">`, `pos="0 0 0"`, `range`, `stiffness/damping` (hand gains ~1-2 N·m/rad, 0.05-0.1), `actuatorfrcrange`, plus a `<motor>` per joint. Fingertip collision geoms need `condim="3"` and friction ≥ 1 (the body default is `condim="1"`, `humanoid.xml:7`). D becomes 29+14 = 43; the frame is 49 wide; `init_pose` and `joint_err_w` (30 → 44 entries) must be regenerated; verify DFS order with `env_name: char_dof_test` before any training. Isaac Lab additionally needs `g1_dex3.usd` with the same body names (`_validate_envs`, `char_env.py:230-238`). Shadow/Allegro/Inspire work the same way, except coupled/mimic joints must be exposed as independent hinges (no mimic support in `kin_char_model.py`).

#### 4.6 AMP variant

`AMPHOIEnv(HOIMixin, AMPEnv)` extends `_build_disc_obs_buffers` (`amp_env.py:98-158`) with object circular buffers, `_update_disc_hist` (L165-184) pushes object pose/vel and hand-object vectors, `_fetch_disc_demo_data` (L67-90) calls `calc_obj_frame`, and `compute_disc_obs` (L336-359) concatenates object features expressed in the reference-root heading frame. `AMPAgent`/`AMPModel` need no change because the disc input size comes from `get_disc_obs_space()`.

#### 4.7 `tools/convert_hoi_to_mimickit.py`

Inputs: (i) retargeted body+hand pose for the target MJCF — from GMR (`{fps, root_pos, root_rot, dof_pos}`) or from SMPL-X via the SMPL path extended with dex-retargeting for fingers; (ii) object 6-DoF tracks per frame from the source dataset (GRAB/ARCTIC/OMOMO/BEHAVE/HOI-M3) in the *same* world frame; (iii) contact labels (dataset-provided or derived: fingertip-to-mesh distance < 5 mm). Steps: resample everything to one fps; apply the exact global transform used for the human (Y-up→Z-up as in `smpl_to_mimickit.py:47-48`, floor calibration L161-167, GMR scale) to object tracks; if the retarget rescaled limbs, re-anchor objects to the hand pose during labelled contact frames so the hand-object relative transform is preserved; convert object rotations to exp-maps; write the v2 pickle with `loop_mode=CLAMP`, and emit the matching `objects:` yaml snippet. Validate with the `view_motion_hoi` env.

### 5. Phased implementation checklist

**Phase 0 — Data plumbing (no RL).** ☐ Motion v2 loader/saver in `motion.py` (optional keys, old files unchanged). ☐ `HOIMotionLib` with unit tests on a synthetic clip (box translated along x, slerp sanity, velocity sanity, loop offset). ☐ `ViewMotionHOIEnv` (extend `view_motion_env.py:53-78` to also `set_root_pos/rot` objects) + `data/envs/view_motion_hoi_humanoid_env.yaml`. ☐ Register env names in `env_builder.py`.

**Phase 1 — Whole-body + object, no hand DOF.** ☐ `HOIMixin`, `DeepMimicHOIEnv`; use the existing humanoid (sphere hands) and a synthetic carry/push clip. ☐ Object obs, four reward terms, object termination, ghost objects. ☐ Config `engine:` overrides (240 Hz, `ground_contact_height`). ☐ Smoke-train 512 envs on a `katef` GPU via sbatch (see `experiments/*/train.sbatch`), check `Test_Return` rises and object deviation logs (add to `record_diagnostics`, cf. `deepmimic_env.py:63-75`).

**Phase 2 — Articulated hands.** ☐ `g1_dex3.xml`/`.usd`; validate joint order with `char_dof_test`. ☐ Regenerate `init_pose`, `joint_err_w`, `key_bodies`, `contact_bodies`. ☐ Hand-specific `action_std`/gain tuning; consider zeroing `joint_err_w` for fingers early (pose term) and relying on `r_rel`/`r_contact`.

**Phase 3 — Real demonstrations.** ☐ `convert_hoi_to_mimickit.py` for one dataset; dataset yaml with fixed object slots. ☐ Train DeepMimicHOI on 3-5 pick-and-place clips; RSI with `rand_reset` across the clip including post-grasp phases.

**Phase 4 — Priors and robustness.** ☐ `AMPHOIEnv` with object-conditioned discriminator. ☐ Engine API additions: pair-filtered contact sensors (Isaac Lab/Newton), object mass/friction/scale randomisation, `set_obj_color`. ☐ Curriculum: kinematic object (`fix_root=True`, driven from the reference) → dynamic object. ☐ Optional transformer/set encoder for variable object counts (replace `fc_*` net in `learning/nets/`).

**Risks to flag now.** Isaac Gym's height-masked ground contact (`isaac_gym_engine.py:311-320`) will misfire for floor-level manipulation; every engine requires identical object slots per env, so mixed-object datasets need placeholders; the MJCF parser's "3 hinges in one body = spherical" rule (`mjcf_char_model.py:150-180`) would silently fuse a 3-hinge finger root into an exp-map joint, so finger joints must live in separate nested bodies.
