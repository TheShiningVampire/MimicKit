# Working plan: MimicKit for manipulation learned from human demonstrations

Status: v1, written 2026-10-12 (overnight research run). Companion documents:

- [01_literature_survey_methods.md](../report/01_literature_survey_methods.md): methods landscape with citations.
- [02_mimickit_gap_analysis_and_design.md](../report/02_mimickit_gap_analysis_and_design.md): file-level gap analysis of this repo and the proposed `DeepMimicHOIEnv` design.
- [03_codebase_survey.md](../report/03_codebase_survey.md): the open-source codebases closest to "MimicKit for manipulation", with reward/obs details.
- [04_datasets_survey.md](../report/04_datasets_survey.md): human manipulation datasets, licences, access, schemas, and what was downloaded.
- [05_differences_whole_body_vs_manipulation.md](../report/05_differences_whole_body_vs_manipulation.md): the conceptual differences, in one place.
- [06_deep_research_hoi_rl.md](../report/06_deep_research_hoi_rl.md): fan-out deep research with 3-vote adversarial verification of every claim; confirms the reward and data-repair choices, flags unverifiable numbers.

## 0. Decisions made overnight (you can overrule any of these)

1. **Same learning setup as MimicKit.** State-action RL in a GPU physics simulator, PPO, a human reference clip per episode, reference state initialization, tracking reward. No behaviour cloning, no vision-language-action model. Those appear in the survey only as context.
2. **Track A first: whole-body humanoid + rigid object, on the Unitree G1 that MimicKit already has.** Reason: the OmniRetarget dataset (MIT, ungated, 3 h of G1 + object clips retargeted from OMOMO) matches MimicKit's existing `g1` character almost one-to-one, so the first result needs no retargeting work and no SMPL-X licence.
3. **Track B second: dexterous hand (floating wrist + fingers) tracking hand+object clips**, following ManipTrans / Dexplore. This is where "manipulation" in the fine-motor sense lives, but it needs a hand asset and a retargeting step first.
4. **Reward follows PhysHOI / SkillMimic (multiplicative body x object x relative x contact)** rather than a weighted sum: SkillMimic's ablation drops success from 82.4% to 7.5% with a sum. The newer InterMimic / HumanX / InterReal line uses sums but adds interaction-specific termination and learned reward weights; phase 1 ablates both and implements object-deviation and contact-mismatch termination from the start (see report 06).
5. **Reference implementation to study: InterMimic** (MIT, Isaac Gym + Isaac Lab, SMPL-X and G1 variants). Reference for the G1 / Isaac Lab path: HDMI and ResMimic.
6. **Sample data was downloaded to** `/data/user_data/vawale/MimicKit/manip_data/` (large) and a small curated subset is committed under `manipulation/datasets/samples/`.

## 1. Goal

Extend MimicKit so that a simulated character (G1 humanoid, later a dexterous hand) learns to reproduce a *human-object interaction* clip: the body or hand motion **and** the object trajectory. Success is measured on the object, not on the pose.

Out of scope for the first three months: deformable objects, articulated objects (drawers, scissors), vision-based policies, real hardware.

## 2. How this differs from whole-body control

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

## 3. Tracks

### Track A: G1 + rigid object (start here)

- Character: existing `data/assets/g1/g1.xml` (29 DoF). Hands stay fixed stubs in phase 1; carrying uses forearms and palms, as in OmniRetarget and ResMimic.
- Data: OmniRetarget `robot-object/` clips (`qpos[T, 43]` = base pos 3, base quat 4, 29 joints, object quat 4, object pos 3; 30 fps), objects as OBJ meshes. Converter: `manipulation/tools/convert_omniretarget_to_mimickit.py`.
- Env: `DeepMimicHOIEnv` from the design document, one dynamic rigid object per env.
- First milestone: one box-carry clip tracked for its full length with object error < 10 cm in > 90% of episodes.

### Track B: dexterous hand + object

- Character: a floating-wrist hand (Inspire 12-DoF, Allegro 16-DoF, or Shadow 22-DoF; URDFs from `dex-urdf`). MimicKit's MJCF parser needs one hinge per nested body, so the hand asset is written in that style.
- Data: GRAB (registration), OakInk2 (registration), ManipTrans DexManipNet (robot-hand episodes), ARCTIC (articulated). Retarget with `dex-retargeting`.
- Env: same `DeepMimicHOIEnv` with the hand as the character; the wrist is the "root".
- First milestone: a GRAB pick-and-lift clip with the object lifted and held for the clip's duration.

### Track C: whole body + dexterous hands (later)

- G1 + Dex3 (7 DoF per hand) or SMPL-X with 30 hand joints (needs the SMPL-X licence). InterMimic's G1 variant is the reference.

## 4. Phases

Each phase has a test that can fail early. Compute figures assume one RTX 6000 Ada / A6000 on `katef` with 4096 envs, which is what the five whole-body runs used.

### Phase 0: ground truth and plumbing (week 1)

Tasks
- Run InterMimic's released Isaac Gym inference on one OMOMO clip to see a working HOI tracker and its numbers.
- Convert 3 OmniRetarget clips with the converter and replay them kinematically in MimicKit's `view_motion` env (character only, object drawn as a ghost).
- Write motion format v2 (`objects` block) and `HOIMotionLib`; unit tests on a synthetic clip.

Test: converted clips replay without foot skating or joint-order errors; the object ghost moves with the hands.

### Phase 1: single clip, G1 + box (weeks 2 to 4)

Tasks
- `DeepMimicHOIEnv`: object spawn per env, object state in reset, object and relative observations, four new reward factors, object termination.
- Engine config overrides: 240 Hz sim, 4 to 8 solver iterations, `ground_contact_height` lowered so object contacts near the floor are not read as falls.
- Train DeepMimic on one carry clip. Log object error, success rate, contact F1 next to the usual diagnostics.
- Ablation: additive vs multiplicative reward; with and without hand-object relative term.

Test: > 90% success on the training clip (object within 10 cm for the whole clip), pose error no worse than 2x the body-only run.

Compute: about 6 runs x 8 to 12 h.

### Phase 2: dataset scale and priors (weeks 5 to 8)

Tasks
- All OmniRetarget robot-object clips (about 3 h) in one dataset yaml; adaptive clip sampling (SkillMimic-V2 ATS) and epsilon-neighbourhood state init (STF) to cope with sparse clips.
- `AMPHOIEnv`: discriminator over body + object + relative features; compare DeepMimic vs AMP vs ADD on the same data.
- Track B start: hand asset, `dex-retargeting` pipeline, one GRAB clip.
- Decaying virtual object controller (DexMachina / ResMimic trick) as a curriculum for hard clips.

Test: one policy over 100+ clips with > 70% success on held-out clips of seen objects; Track B lifts and holds one object.

Compute: about 10 runs x 12 to 24 h.

### Phase 3: robustness and transfer (weeks 9 to 12)

Tasks
- Domain randomization of object mass, friction, scale; sim-to-sim check in Newton or MuJoCo.
- Distill many clips into one goal-conditioned policy (future object trajectory as the goal), as in Omnigrasp and MaskedManipulator.
- If a G1 is available: deploy a carry policy, following HDMI's and ResMimic's deployment stacks.

Test: policy survives 20% mass and friction changes with < 15% drop in success; sim-to-sim success within 20% of Isaac Gym.

## 5. Reward and observation specification (summary)

Full equations and suggested weights are in report 02, section 4. The reward in phase 1 is

```
r = r_body * r_obj * r_rel * r_contact
r_body    = exp(-s_p * pose_err) * exp(-s_v * vel_err) * exp(-s_r * root_err)
r_obj     = exp(-s_op * ||p_o - p̂_o||²) * exp(-s_or * angle(q_o, q̂_o)²) * exp(-s_ov * ||v_o - v̂_o||²)
r_rel     = exp(-s_rel * Σ_h || R̂_oᵀ(p_h - p_o) - R̂_oᵀ(p̂_h - p̂_o) ||²)     # hand positions in the object frame
r_contact = exp(-s_c * Σ_h | c_h - ĉ_h |)                                     # contact graph match, c from contact forces
```

Observations append to MimicKit's current vector: object position and rotation relative to the root (heading frame), object linear and angular velocity, per-hand vector to the object, future object targets at the same `tar_obs_steps` as the body targets, and the reference contact flags for the next step.

## 6. Data plan

Ranked for Track A (whole body + object), from report 04: OmniRetarget (ready for G1, MIT), OMOMO (source of the above, MIT), InterMimic's processed OMOMO (ungated Google Drive), BEHAVE and GRAB (registration), ParaHome and HUMOTO for household tasks.

Ranked for Track B (hand + object): GRAB, OakInk2, ARCTIC (all registration), DexManipNet (robot-hand episodes from ManipTrans), GigaHands and EgoDex for scale (egocentric, hand pose from video).

Committed samples under `manipulation/datasets/samples/` are small (a few MB): schema examples and one or two short clips per source where the licence allows redistribution. Everything else stays in `/data/user_data/vawale/MimicKit/manip_data/` and is listed in its `MANIFEST.md`.

## 7. Compute plan on Babel

- Training: `sbatch` on `katef`, `--gres=gpu:1 --cpus-per-task=8 --mem=60G --time=1-00:00:00`, 4096 envs, logs under `logs/`. Same template as `experiments/<skill>/train.sbatch`.
- Phase 1 total: about 60 GPU hours. Phase 2: about 200 GPU hours. Checkpoint every hour because `preempt` can kill jobs.
- Rendering and conversion: CPU, `katef-cpu`.

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Isaac Gym contact model too coarse for fingertip grasps | high for Track B | start Track A (palms/forearms) where contact is forgiving; for Track B use 240 Hz, higher friction, and the virtual object controller curriculum |
| Reference clips physically inconsistent (penetration, float) | high | use OmniRetarget clips (already penetration-free), add InterMimic-style teacher correction in phase 2 |
| Multiplicative reward too sparse early in training | medium | warm up with additive reward for the first N iterations, or start the object kinematic (fixed to reference) and release it |
| Every engine requires identical object slots per env | certain | fixed object-slot list per dataset, park unused objects far away with an `active` flag |
| MJCF parser fuses three hinges in one body into a spherical joint | certain for hands | write finger joints as nested single-hinge bodies, validate with `char_dof_test` |
| Gated datasets block Track B | medium | request GRAB, ARCTIC, OakInk2 access now; use DexManipNet and OmniRetarget meanwhile |
| Isaac Gym is deprecated | medium | develop against MimicKit's engine abstraction so Isaac Lab and Newton work; InterMimic shows the Isaac Lab path |

## 9. Evaluation protocol

For every run, log per episode: object position error (mean and final), object rotation error, success (object within 10 cm and 30 degrees at the clip end), pose error (MPJPE on key bodies), contact F1 against reference contact labels, episode length, and energy (sum of squared torques). Report over 1024 episodes with random reference starts. Keep the whole-body runs' diagnostics so the two settings can be compared on the same plots.

## 10. Questions for you

1. Which robot hand do you want for Track B? The choice fixes the asset and the retargeting target. Default if no answer: Inspire (12 DoF, used by ManipTrans, OmniH2O and Weave).
2. Do you have or expect access to a physical G1? It decides whether phase 3 is hardware or sim-to-sim only.
3. Can you register for GRAB, ARCTIC and OakInk2 under the lab's name? They are the standard hand+object benchmarks and the forms take a day or two.
