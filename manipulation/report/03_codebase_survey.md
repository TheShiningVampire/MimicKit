## Open-source physics-based imitation of human–object interaction / dexterous manipulation — codebase survey (as of 2026-10-12)

Scope note: "MimicKit for manipulation" means a GPU-parallel RL tracker (DeepMimic/AMP-style) whose reference clips contain a human **and** an object, with object-aware rewards/observations. I grouped the ~45 codebases into (A) whole-body humanoid HOI imitation (the direct analogues), (B) dexterous-hand-only imitation, (C) real-robot humanoid HOI from video/mocap, (D) retargeting/data engines, and (E) adjacent/not-really-imitation. Stars/license/last-push come from the GitHub API on 2026-10-12. Reward/obs details come from the paper HTML plus, where noted, the actual source files. Anything I could not confirm is marked **[unverified]**.

For reference, MimicKit's motion format (checked locally in `/home/vawale/my_things/codes/MimicKit/mimickit/anim/motion.py`, `motion_lib.py`) is a pickle `{loop_mode, fps, frames}` where each frame is `[root_pos(3), root_rot_expmap(3), joint_dof(...)]`; there is no object channel, and `motion_lib.py` stores only `_frame_root_pos/_frame_root_rot/_frame_joint_rot`. Engines: Isaac Gym, Isaac Lab, Newton. The `manipulation/plan` and `manipulation/report` dirs in the repo are currently empty.

---

## A. Whole-body humanoid HOI imitation (closest analogues)

### 1. InterMimic (CVPR 2025 Highlight) — the primary reference
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

### 2. PhysHOI (Tech report 2023) and SkillMimic (CVPR 2025 Highlight) / SkillMimic-V2 (SIGGRAPH 2025)
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

### 3. Omnigrasp (NeurIPS 2024, Luo et al.; PHC/PULSE family)
- Repo: https://github.com/ZhengyiLuo/Omnigrasp — 159 stars, no license file (paper page says CC BY-NC-SA 4.0 **[unverified for code]**), last push 2025-11-17, 8 open issues. Paper https://arxiv.org/abs/2407.11385.
- Simulator: Isaac Gym (`sim=hand_sim`), SMPL-X v1.1 humanoid 52 joints / 153 DoF. Hydra configs (`learning=omnigrasp_rnn robot=smplx_humanoid`).
- Data: GRAB (50 objects, pre-grasp + initial pose provider), OakInk (1700 objects), OMOMO (7 objects). Ships `download_data.sh` (models + `grab_sample.pkl`); SMPL-X body models require MPI registration. Training code released Aug 2025; **data-processing code still pending** (you must write the GRAB → `.pkl` converter yourself, based on PHC's format).
- Method: PULSE-X 48-D full-body+hand latent prior (distilled from PHC-X trained on dexterous AMASS); task policy outputs latents, GRU (512), PPO (batch 3072, lr 5e-4). Reward is **not full-motion imitation**: approach reward (hand → pre-grasp distance), pre-grasp hand-pose imitation, then object-trajectory tracking (exp terms on pos/rot/vel/ang-vel) + contact bonus. Obs: proprio + 20 future object-trajectory frames at 15 Hz + 512-D BPS object shape + hand-to-object offsets. Curriculum: hard-negative mining over objects by failure rate. Cost: ~3 days, ~1e9 samples on a single A100.
- Port difficulty: **hard** (depends on PHC/PULSE latent prior and SMPL-X asset pipeline); but the pre-grasp + object-trajectory reward and BPS obs are portable concepts.

### 4. HumanVLA (NeurIPS 2024) and HumanoidVerse (vision-language rearrangement, 2025)
- HumanVLA: https://github.com/AllenXuuu/HumanVLA — 136 stars, MIT, last push 2024-12. Isaac Gym, SMPL-like humanoid, Human-in-the-Room (HITR) dataset on Google Drive (ungated), HSSD scenes. Teacher = goal-conditioned PPO + AMP (carry curriculum → rearrangement), student = VLA via BC; 3-stage `torchrun` on 4 GPUs. Not a motion tracker: objects enter via goal rewards + AMP style prior. Relevant mostly as an AMP-with-objects example (MimicKit already has AMP).
- HumanoidVerse (Zhang et al., arXiv 2508.16943, Isaac Gym, multi-object rearrangement, dual-teacher distillation): project page https://haozhuo-zhang.github.io/HumanoidVerse-project-page/; code status **[unverified]**. Not to be confused with LeCAR's HumanoidVerse framework (below).

### 5. UniHSI (ICLR 2024 Spotlight)
- https://github.com/InternRobotics/UniHSI (moved from OpenRobotLab) — ~248 stars, CC BY-NC-SA 4.0, last update 2024. Isaac Gym, ASE/AMP humanoid, PartNet/ShapeNet/ScanNet scenes, "Chain of Contacts" (joint–object-part pairs) as the task spec and reward, LLM planner. Static scenes, no object dynamics tracking. Useful only for its contact-pair reward and scene loading.

### 6. ProtoMotions / MaskedMimic (NVIDIA)
- https://github.com/NVlabs/ProtoMotions — 2.4k stars, Apache-2.0, last push 2026-09-22. Backends: Newton 1.0, Isaac Lab 3.0, Isaac Gym, MuJoCo, Genesis (community). Embodiments: SMPL/SMPL-H, H1-2, G1. Datasets: AMASS, BONES-SEED (142k motions), PHUMA, Kimodo.
- Object support: "scene and object spawning" and MaskedMimic's object-conditioned inpainting (sit/reach on SAMP-style static objects). **No dynamic object tracking or contact rewards**; Genesis backend cannot spawn per-env unique objects. Treat as the closest architectural cousin to MimicKit (same Newton/Isaac Lab/Isaac Gym trio) and a source for the scene-spawning code, not for HOI rewards.

### 7. Newer physics-based HOI works (2025–2026)
- **InterPrior** (Xu et al., arXiv 2602.06035): masked conditional VAE policy distilled from InterMimic experts + RL post-training; code not public **[unverified]**.
- **LessMimic** (arXiv 2602.21723): https://github.com/Yutang-Lin/LessMimic — 57 stars; distance-field conditioning, VAE + "Adversarial Interaction Prior" RL, no motion references at test time. README: "Code will be released" — **not yet usable**.
- **HOI-Dyn** (NeurIPS 2025, https://github.com/wulin97/HOI-Dyn) and **SimGenHOI** (arXiv 2508.14120) are generative + RL hybrids; code maturity **[unverified]**.

---

## B. Dexterous-hand (hand-only / floating-hand) imitation from human hand–object data

### 8. ManipTrans (CVPR 2025)
- https://github.com/ManipTrans/ManipTrans — 342 stars, **GPL-3.0**, last push 2025-10-10. Paper https://arxiv.org/abs/2503.21860.
- Isaac Gym Preview 4, Hydra. Hands: Inspire (12-DoF), Shadow (22), Allegro (16), XHand, Arti-Mano, Inspire-FTP; single and bimanual; floating wrist (6-DoF) + fingers.
- Data: GRAB demo in repo; OakInk-V2, FAVOR, ARCTIC via their own licenses; **DexManipNet** (3.3k episodes, 1.34M frames, 61 tasks, 1.2k objects; download from project site / HF mirrors, gating **[unverified]**). Format: 60 fps MANO joints + wrist kinematics + object mesh/URDF + object pose.
- Two-stage: generalist hand imitator (wrist SE(3) + finger keypoint rewards, joint-power penalty) pretrained on FAVOR/OakInk-V2/GRAB/ARCTIC + interpolated synthetic data, then per-skill **residual** policy with object pose/vel reward and contact-force reward (`exp(-λ_c Σ f · 1[d<thr])`). Obs: joint q/qd, wrist pose, object 6D pose/vel/CoM/gravity, BPS shape, contact forces, fingertip–object distances. PPO, 4096 envs, single RTX 4090; RSI + early termination with curriculum threshold relaxation. Sim-to-real: Realman arms + Inspire hands via fingertip IK.
- Port: **moderate**; GPL is the main caveat if MimicKit stays MIT/BSD.

### 9. DexMachina (ICML 2026)
- https://github.com/MandiZhao/dexmachina — 269 stars, MIT, last push 2026-03-23. Paper https://arxiv.org/abs/2505.24853.
- **Genesis** (custom fork), rl_games PPO, 12k envs/GPU, single L40S/H100 per run. Six hands (Allegro, Inspire, Schunk, XHand, Ability, DexRobot), ARCTIC bimanual articulated objects (box, mixer, notebook, ketchup, waffle iron). Reward: object pose/joint-angle tracking + hand keypoint imitation + BC-to-retargeted-joints + contact-position matching. Key trick: **virtual object controller curriculum** (spring-damper drives object along reference; gains decay with policy success). Actions: residual wrist + absolute normalized fingers. Ships hand assets + ARCTIC assets (ARCTIC itself needs registration).
- Port: **moderate**; the decaying virtual-object-controller is a very portable curriculum for any MimicKit HOI env.

### 10. DexTrack (ICLR 2025)
- https://github.com/Meowuu7/DexTrack — 135 stars, MIT, last push 2026-04-14. Isaac Gym, Allegro (floating) and LEAP-on-Franka, GRAB + TACO retargeted via MANO→robot keypoints; 22k envs single GPU. Per-trajectory RL tracking released; the homotopy-optimization + IL/RL generalist code is explicitly **not released** ("too messy").

### 11. ObjDex (Chen, Wang, Yang, Liu)
- https://github.com/cypypccpy/ObjDexEnvs — 63 stars, MIT, last push 2025-02. Isaac Gym, Shadow hand, ARCTIC clips by `--object --traj_index`, 2048 envs, object-centric reward (hand data only initializes). Small but clean.

### 12. Human2Sim2Robot and SimToolReal / DexToolBench (Lum et al.)
- https://github.com/tylerlum/human2sim2robot — 160 stars, MIT, last push 2026-08-27. Isaac Gym, Kuka iiwa + Allegro, one RGB-D human video → FoundationPose object trajectory + HaMeR pre-manipulation hand pose; **object-centric reward only** (no finger imitation), PPO-LSTM, DR, zero-shot real.
- https://github.com/tylerlum/simtoolreal — 409 stars, MIT, last push 2026-09-21. Isaac Sim/Isaac Lab (Isaac Gym legacy), PPO + SAPG, up to 24,576 envs. **DexToolBench**: 24 tool-use tasks (6 categories x 2 objects x 2 tasks), RGB-D human videos + processed data + sim envs; download script provided (terms **[unverified]**).

### 13. Grasp-synthesis / benchmark repos (not imitation, listed for completeness)
- GraspXL https://github.com/zdchan/GraspXL — 272 stars, CC BY-NC 4.0, RaiSim (license needed), MANO/Allegro/Shadow/LEAP, 500k Objaverse objects on HF (ungated).
- DexGraspNet 2.0 https://github.com/PKU-EPIC/DexGraspNet2 (162 stars, no license, CoRL 2024, synthetic cluttered grasps, HF dataset) and UniDexGrasp++ https://github.com/PKU-EPIC/UniDexGrasp2 (219 stars, MIT, Isaac Gym, geometry-aware curriculum + generalist-specialist). No human data.
- Bi-DexHands https://github.com/PKU-MARL/DexterousHands — 1.1k stars, Apache-2.0, Isaac Gym bimanual Shadow tasks; last push 2025-02.
- DexPBT: lives in IsaacGymEnvs (`AllegroKukaLSTM`, PBT docs https://github.com/isaac-sim/IsaacGymEnvs/blob/main/docs/pbt.md); Isaac Lab now has a PBT feature page. DexterityGen (https://zhaohengyin.github.io/dexteritygen/): **no code released**.
- DexMimicGen https://github.com/NVlabs/dexmimicgen (288 stars, NVIDIA Source Code License; datasets CC-BY 4.0 on HF `MimicGen/dexmimicgen_datasets`) and MimicGen (652 stars): robosuite/MuJoCo, GR1 + bimanual arms, **imitation learning (BC-RNN) not RL**; object-centric transform of a few teleop demos. MimicLabs https://github.com/Gatech-RL2/mimiclabs (MuJoCo, MimicGen-based), RoboCasa (MuJoCo kitchens, arm-centric). HumanoidGen https://github.com/TeleHuman/HumanoidGen (157 stars, NeurIPS 2025, LLM-generated bimanual dex tasks). These are data-generation for IL, not physics imitation of human data.

---

## C. Real-robot humanoid HOI tracking from human mocap/video (G1-centric, 2025–2026)

### 14. HDMI (LeCAR, arXiv 2509.16757)
- https://github.com/LeCAR-Lab/HDMI — 672 stars, **no license file**, last push 2026-01-17. Isaac Lab v2.2 / Isaac Sim; Unitree G1 (no dex hands in released tasks). Sim2real stack: https://github.com/EGalahad/sim2real.
- Data pipeline: monocular RGB → GVHMR human pose → GMR/LocoMujoco retargeting; object pose from video; reference = `(robot state, object pose [+ articulation], binary contact c_t with contact points in object frame)`. Motion files: `.npz` with body pos/quat/lin-vel/ang-vel `[T,B,·]` and joint pos/vel `[T,J]`; 13 task configs (`move_suitcase`, `traverse_door`, `carry_box_over_shoulder`, `open_foldchair-sit`, `roll_ball-hand`, ...), object randomization (mass, friction, scale 0.9–1.1).
- Reward (paper table + `rewards.py`): body local pose 2.0, root global pose 1.0, body global vel 1.0, joint tracking 1.0, **object pose 2.0**, **contact 5.0**, action-rate -0.1, joint-limit 10, joint-vel 5e-4; tracking terms `exp(-err/σ)` with σ=0.03 (keypoints) and σ=0.25 (object); contact reward `exp(-eef_pos_err/0.1) * exp(f_contact/10)` gated by "in range". Obs: proprio + object pose in robot root frame + phase φ∈[0,1] + reference contact points in robot frame. PPO with **residual action space** (offsets from reference joint targets), teacher (`ppo_roa_train`) → student (`ppo_roa_finetune`), RSI with perturbation, DR. Real: 67 consecutive door traversals on G1.
- Port: **moderate**; the Isaac Lab MDP terms map cleanly to MimicKit's Isaac Lab engine; the main missing piece in MimicKit is the object asset per env.

### 15. ResMimic (Amazon FAR, arXiv 2510.05070)
- https://github.com/amazon-far/ResMimic — 74 stars, Apache-2.0. legged_gym/rsl_rl fork on Isaac Gym (MuJoCo sim2sim), built on TWIST2. G1 29-DoF. Stage 1 general motion tracker (GMT) on AMASS + OMOMO via GMR (~15k clips / 42 h); stage 2 residual policy `a = a_gmt + Δa`. Motion data: humanoid `.pkl` (PHC/TWIST style) + object `.npz` with `trans`, `rot` (`motion_lib_hoi.py`); examples `kneel`, `squat`, `carry`, `chair`, suitcase URDF (mass 4 kg). Reward scales in `g1_hoi_config.py`: `tracking_keybody_pos_global 10`, `tracking_joint_dof 2`, `tracking_object_point_cloud 2`, `feet_air_time 5`, `feet_slip -0.1`, `dof_torque_limits -1` (16 terms total); contact reward `Σ ĉ_i exp(-λ/f_i)` from oracle contacts; object point-cloud tracking `exp(-λ Σ‖P_i − P̂_i‖)`. Obs: proprio history (10) + reference window (±10 frames) + object `[p, θ, v, ω]` (7-D "nonblind"). **Virtual object force controller with decaying PD gains** (same trick as DexMachina). Termination: object > 0.3 m from reference. Real G1 carries 4.5 kg.
- Port: **moderate**; the object-point-cloud reward and decaying virtual controller are small, self-contained.

### 16. OmniRetarget (Amazon FAR / project-instinct) — data engine + dataset
- Code https://github.com/project-instinct/omniretargeting — 82 stars, MIT, last push 2026-10-10. Paper https://arxiv.org/abs/2509.26633. Dataset https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset — MIT, **ungated**, 4.0 h total: `robot-object/` (3.0 h from OMOMO), `robot-terrain/` (0.5 h), `robot-object-terrain/` (0.5 h), `models/` (G1 URDF/SDF, object OBJs). Format: `.npz` with `fps` and `qpos [T, 36 or 43]` = base pos(3)+quat(4)+29 joints (+ object quat(4)+pos(3)). Retargeter: interaction-mesh (Laplacian) preserving body–object–terrain relations, hard kinematic constraints, augmentation by object/terrain scaling and relocation; input adapters for SMPL-X/OMOMO/LAFAN1/Nokov. Tracking policies in the paper use a BeyondMimic-style Isaac Lab tracker with 5 reward terms and show zero-shot G1 parkour/carry **[the tracker code is not in this repo; BeyondMimic's is public, below]**.
- This is the single most "ready-to-use" G1 + object dataset.

### 17. BeyondMimic (Science Robotics 2025) and SONIC / GR00T-WBC (NVIDIA)
- https://github.com/HybridRobotics/whole_body_tracking — 2.4k stars, MIT, Isaac Lab 2.1, G1, LAFAN1 CSV → npz via WandB registry, DeepMimic-style rewards; **no object support** but the reference Isaac Lab tracker that OmniRetarget data is designed for.
- https://github.com/NVlabs/GR00T-WholeBodyControl — 3.7k stars, Apache-2.0 code / NVIDIA Open Model License weights, last push 2026-08-31; SONIC PPO training stack, BONES-SEED 142k motions, MuJoCo retargeting; manipulation only via teleop, no HOI imitation.

### 18. HumanX (CoRL 2026, Wang et al.)
- https://github.com/wyhuai/HumanX — 58 stars, Apache-2.0 **with Commons Clause**, last push 2026-10-09. G1; **XGen** (video → HOI editor exporting `.npz` robot+object motions) + **XMimic** (`.pkl` training, SkillMimic-lineage rewards). Example data: basketball catch/throw, pump-fake, football kick, bin pick-and-place; zero-shot real G1 for 10 skills across 5 domains. Very new; simulator not stated in README **[unverified, likely Isaac Gym given SkillMimic lineage]**.

### 19. Others in this bucket
- **Weave** (arXiv 2609.16683) https://github.com/xiaohu-art/Weave — 93 stars, Isaac Lab + RSL-RL, G1 with dex hands, 9 objects, contact-aware retargeting; HF data `appolyn/Weave`. **DexWeave** (arXiv 2609.34724) https://github.com/WU-CVGL/DexWeave — G1 + Inspire hands. Both too new for maturity claims.
- **PhysHSI** https://github.com/InternRobotics/PhysHSI — 321 stars, CC BY-NC-SA 4.0, Isaac Gym legged_gym + HIMLoco, G1, AMP from AMASS/SAMP/100STYLE, tasks CarryBox/SitDown/LieDown/StandUp; LiDAR+camera object localization for real. AMP-with-goals, not tracking.
- **VisualMimic** https://github.com/visualmimic/VisualMimic — 300 stars; only sim2sim + checkpoints released; training code pending. **VIRAL** (arXiv 2511.15200): no code found. **AMO** https://github.com/OpenTeleVision/AMO — 386 stars, inference-only JIT checkpoints, MuJoCo. **FALCON** (CoRL 2025): paper only, no repo found **[unverified]**. **CLONE** https://github.com/humanoid-clone/CLONE — 277 stars, CC BY-NC 4.0, checkpoints + deploy only; training code "planned via RoboVerse". **HOMIE** https://github.com/InternRobotics/OpenHomie and `/Homie` — teleop cockpit + locomotion RL, no HOI imitation. **OmniH2O/H2O** https://github.com/LeCAR-Lab/human2humanoid — 1.1k stars, CC BY-NC 4.0, Isaac Gym, H1 + Inspire, PHC-style AMASS `.pkl`, teacher-student DAgger; no objects. **HumanPlus** https://github.com/MarkFzp/humanplus — 853 stars, legged_gym/rsl_rl, H1, HST on AMASS + HIT; no objects. **OKAMI** https://github.com/UT-Austin-RPL/OKAMI — 37 stars, GR1, single RGB-D video → SLAHMR/HaMeR + object-aware kinematic retargeting (no RL tracking of dynamics). **TWIST** https://github.com/YanjieZe/TWIST — 821 stars, MIT, teleop via GMR; no objects. **VideoMimic** https://github.com/hongsukchoi/VideoMimic — 838 stars, MIT, CoRL 2025 best student paper, video → scene mesh + human motion, scene-conditioned tracking (stairs, sitting); static scenes only. **Workhorse** (arXiv 2610.09117), **CoorDex**, **Blind Dexterity**, **SUGAR**, **GRAIL**, **HumanoidMimicGen**: 2026 papers, code status **[unverified]**.

---

## D. Retargeting / data tooling

- **GMR** https://github.com/YanjieZe/GMR — 2.8k stars, MIT, ICRA 2026, 17 robots, inputs SMPL-X (AMASS/OMOMO), BVH (LAFAN1/Xsens/Nokov), FBX (OptiTrack), PICO, GVHMR video; output pickle `{root trans, root quat(wxyz), joint pos}` per frame; CPU real-time. **Objects are not retargeted** (used by HDMI/ResMimic for the body only).
- **OmniRetarget** (above) is the object-aware retargeter.
- **dex-retargeting** https://github.com/dexsuite/dex-retargeting — 1.3k stars, MIT, `pip install dex_retargeting`; vector/position/DexPilot optimizers from MANO keypoints → robot hand qpos (Allegro, Shadow, Leap, Inspire, Ability, etc. via dex-urdf). AnyTeleop (RSS 2023) is the origin; DexPilot itself (NVIDIA 2019) has no public code; the "DexPilot optimizer" here is a reimplementation.
- **Open-Teach** https://github.com/aadhithya14/Open-Teach — Quest-3 VR teleop for Franka/xArm/Allegro (data collection, no sim imitation).
- **InterAct** https://github.com/wzyabcas/InterAct — 211 stars, custom license, last push 2026-10-06; 21.8 h consolidated HOI (OMOMO, BEHAVE, GRAB, InterCap, IMHD, HODome, HIMO, CHAIRS, ...) corrected and augmented to 30.7 h, with text; **access via request form (non-commercial)**; ships converters to the InterMimic `.pt` format.
- **LocoMujoco** https://github.com/robfiras/loco-mujoco — 1.5k stars, MIT, MuJoCo/MjWarp, imitation-learning benchmark + retargeting used by HDMI; no objects. **Genesis** https://github.com/Genesis-Embodied-AI/Genesis (repo recently reorganized to `genesis-world` **[unverified]**): used only by DexMachina here; per-env unique objects remain a limitation per ProtoMotions.
- **CHOIS** https://github.com/lijiaman/chois_release (163 stars, MIT, ECCV 2024 oral) and **HOIDiNi** https://github.com/hoidini/HOIDiNi (80 stars, MIT, CVPRW 2026): kinematic diffusion HOI generators on OMOMO/GRAB producing SMPL-X + object poses. Not physics, but useful as **synthetic reference generators** feeding an InterMimic-style tracker (InterMimic's own pipeline does this with InterAct/text).

---

## Source datasets and gating (what each codebase actually needs)

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

## (1) Comparison table

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

## (2) Ranked: which three to study/replicate first for MimicKit-manipulation

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

## (3) Data that ships ready-to-use

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
