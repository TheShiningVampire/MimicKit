# Learning Manipulation Skills from Human Data, MimicKit-Style: A Literature Survey (Oct 2026)

**Scope.** MimicKit (Peng et al., [arXiv:2510.13794](https://arxiv.org/abs/2510.13794), [github.com/xbpeng/MimicKit](https://github.com/xbpeng/MimicKit)) is a tracking-RL framework: reference motion → physics sim (Isaac Gym / Isaac Lab / Newton) → DeepMimic/AMP/ASE/ADD/SMP objectives. The question is which work does the same thing when the reference contains **hands + objects**. I grouped the literature into five families and, for every entry, give: method, human data, embodiment, sim, release status, and relevance. Facts I verified by fetching the paper/repo are stated plainly; anything taken from a search snippet only or from memory is marked **[unverified]**.

**Baseline fact about the local repo** (checked on disk): MimicKit already has `StaticObjectsEnv` (fixed-base rigid objects via `engine.create_obj`), a dynamic projectile in `task_dodgeball_env.py`, vault/climbing boxes in `data/assets/objects/`, and SMPL/humanoid/G1/Go2/Pi+ characters. There is no hand-articulated character, no object-motion reference, and no object/contact reward term. The `manipulation/` directory exists but is empty apart from a narration script. So the gap to close is: (1) a character with fingers, (2) an object track in the motion file, (3) object + contact reward terms, (4) RSI that resets object state alongside the character.

---

## 1. Physics-based human-object interaction (HOI) imitation for simulated characters

This is the family closest to MimicKit: a simulated SMPL-X-like humanoid (with fingers) tracks both body and object kinematics from mocap.

### PhysHOI (2023, arXiv; Wang et al., HKUST)
[arXiv:2312.04393](https://arxiv.org/abs/2312.04393) · [github.com/wyhuai/PhysHOI](https://github.com/wyhuai/PhysHOI)
Whole-body HOI imitation without task-specific rewards. The humanoid follows the SMPL-X tree with 51×3 actuated DoF including full fingers. The key idea is the **contact graph (CG)**: a binary graph of body-part/object contact relations, and a **contact graph reward** that is *multiplied* with the kinematic rewards (body, object, relative body-object positions). The multiplicative form removes the local optima of additive rewards (e.g., tracking the body perfectly while dropping the ball). Data: the **BallPlay** dataset (8 basketball clips, SMPL-X + ball pose + contact labels, 25 fps). Sim: Isaac Gym; code builds on ASE. Code, data and checkpoints are released. **Relevance: the most direct template for a MimicKit "deepmimic_hoi" env** — it is literally DeepMimic + object track + CG reward, on an ASE codebase MimicKit's authors wrote.

### SkillMimic (CVPR 2025 Highlight)
[arXiv:2408.15270](https://arxiv.org/abs/2408.15270) · [github.com/wyhuai/SkillMimic](https://github.com/wyhuai/SkillMimic)
Scales PhysHOI to a single policy over many basketball skills (dribble variants, layup, shoot, …), and adds a high-level controller that composes skills into long-horizon tasks (consecutive scoring). Reward is the unified multiplicative `r = r_body · r_object · r_rel · r_cg`. Datasets: **BallPlay-M** (35 min optical mocap) and **BallPlay-V** (8 skills estimated from RGB video). Sim: Isaac Gym. Released: code, pretrained low-level + high-level controllers, a subset of BallPlay-M (README says full raw release is still TODO). RSI from random reference frames (`--state_init`). **Relevance: high** — the multi-skill, skill-conditioned variant of the PhysHOI recipe, structurally identical to MimicKit's DeepMimic + skill-conditioning.

### SkillMimic-V2 (SIGGRAPH 2025)
[arXiv:2505.02094](https://arxiv.org/abs/2505.02094) · [github.com/Ingrid789/SkillMimic-V2](https://github.com/Ingrid789/SkillMimic-V2)
Same reward as V1 (confirmed in the paper HTML: "no fundamental changes"); the contribution is robustness to **sparse and noisy demonstrations** via data augmentation of the reference set:
- **STG (Stitched Trajectory Graph)**: offline graph that discovers transitions between different skill clips, with masked intermediate states bridging large gaps — gives skill transitions from fragment clips.
- **STF (State Transition Field)**: ε-neighborhood State Initialization (ε-NSI) — reset into a noisy ball around reference states and connect each such state to a unique target, so the policy learns recovery into the manifold.
- **ATS (Adaptive Trajectory Sampling)**: softmax sampling weights `p_i ∝ exp(-λ r̄_i)` so harder clips are sampled more (fixes "chain break").
- **History encoder**: frozen, pre-trained net compressing the last k states for memory-dependent behaviours.
Sim: Isaac Gym, 2048 envs, 120 Hz PD / 60 Hz policy. Data: BallPlay-M (5 skills) + **ParaHome** (7 household manipulation tasks: book grasping, cube reorientation, etc.). Released: pretrained models; training code is still listed as TODO. **Relevance: very high** — STG/STF/ATS are exactly the augmentations you need when your manipulation reference set is a handful of mocap clips, and they are cheap to add to MimicKit's motion-sampling and RSI code.

### InterMimic (CVPR 2025 Highlight; Xu et al., UIUC + EA)
[arXiv:2502.20390](https://arxiv.org/abs/2502.20390) · [github.com/Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic)
Universal HOI controller from "hours of imperfect mocap". Curriculum is "perfect first, then scale up": subject-specific **teacher** policies are trained to correct mocap artifacts (penetration, floating), then a **student** is distilled with the teachers acting as online experts, then RL-finetuned so it surpasses replay. Data: OMOMO (authors recommend reprocessing via their **InterAct** pipeline, [arXiv:2509.09555](https://arxiv.org/abs/2509.09555)), plus other HOI datasets. Embodiments: SMPL-X **and Unitree G1 with three-fingered hands, "directly learned from mocap without any external retargeting."** Sim: **both Isaac Gym and Isaac Lab** (Isaac Sim 5.1 / IsaacLab 2.3.1). MIT license; code, teacher checkpoints, processed mocap, multi-GPU training released. **Relevance: very high** — the only released codebase that does physics HOI tracking in Isaac Lab for both SMPL-X and G1, i.e. the same engine pair MimicKit supports.

### Omnigrasp (NeurIPS 2024; Luo et al., CMU/Meta)
[arXiv:2407.11385](https://arxiv.org/abs/2407.11385) · [github.com/ZhengyiLuo/Omnigrasp](https://github.com/ZhengyiLuo/Omnigrasp)
Full-body SMPL-X humanoid with dexterous hands grasps >1200 objects and follows object trajectories. Key trick: act in the **PULSE** latent motion space (learned from PHC), which gives human-like motor priors and makes the grasp RL tractable with simplistic rewards (object pose tracking) and no paired body+object dataset — it only needs pre-grasp hand poses (from GRAB/OakInk/OMOMO) and object meshes. Sim: Isaac Gym. Inference code + models (Dec 2024) and training code (Aug 2025) released; data-processing scripts still pending. **Relevance: high** as the "object-centric reward + motion prior" alternative to full kinematic tracking; ASE/SMP-style priors in MimicKit could play PULSE's role.

### MaskedMimic → MaskedManipulator (NVIDIA; SIGGRAPH Asia 2024 / 2025)
MaskedMimic [arXiv:2409.14393](https://arxiv.org/abs/2409.14393) frames control as masked motion inpainting; it lives in **ProtoMotions** ([github.com/NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions)). **MaskedManipulator** [arXiv:2505.19086](https://arxiv.org/abs/2505.19086) extends it: a tracking controller trained on large-scale HOI mocap is distilled into a generative policy conditioned on sparse spatio-temporal goals for *both* body parts and the object (e.g. "object here at t=3s"). Sim: Isaac Gym/Lab via ProtoMotions [unverified which]. I could not confirm a public MaskedManipulator code release. **ProtoMotions object support (checked):** ProtoMotions3 ships **SceneLib** — scenes with one or more static or *movable* objects, each with an optional **object motion file used during motion-tracking tasks** — and supports IsaacGym, IsaacLab 3.0, Newton 1.0, MuJoCo 3 and Genesis. So ProtoMotions already has the object-track plumbing MimicKit lacks; worth reading `protomotions/components/scene_lib` when designing MimicKit's version.

### HumanVLA (NeurIPS 2024), UniHSI (ICLR 2024), CHOIS (ECCV 2024), InterAct (2025), HOIDiNi (2025)
HumanVLA ([NeurIPS 2024 paper](https://papers.nips.cc/paper_files/paper/2024/file/215aeb07b5996c969c0123c3c6ee8f54-Paper-Conference.pdf)) does vision+language-directed physics-based object rearrangement (teacher-student, Isaac Gym). UniHSI does sequential contact-reaching with *static* scene objects in physics sim (no object dynamics). CHOIS and HOIDiNi ([arXiv:2506.15625](https://arxiv.org/abs/2506.15625)) are **kinematic** HOI generators (diffusion), i.e. reference-motion *sources*, not trackers. InterAct ([arXiv:2509.09555](https://arxiv.org/abs/2509.09555)) consolidates/cleans HOI mocap (OMOMO, BEHAVE, …) and is InterMimic's recommended preprocessing. **Relevance:** moderate; useful as data sources/front-ends, not as the RL core.

### Dexterous-hand tracking (floating or arm-mounted robot hands, no body)

**DexTrack (ICLR 2025)** [arXiv:2502.09614](https://arxiv.org/abs/2502.09614) · [github.com/Meowuu7/DexTrack](https://github.com/Meowuu7/DexTrack). Generalizable neural tracking controller mapping kinematic human hand-object references to robot-hand actions; trained by an RL↔IL flywheel with homotopy optimization to grow a dataset of successful tracking demos; >10% success gain over baselines in sim and real. Hands/sim: Allegro/LEAP in Isaac Gym, GRAB/TACO references **[unverified — from memory]**.

**ManipTrans (CVPR 2025)** [arXiv:2503.21860](https://arxiv.org/abs/2503.21860) · [github.com/ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans). Two stages: a **generalist hand-trajectory imitator** (hand-only tracking, no object) pre-trained once, then a per-task **residual module** fine-tuned under interaction constraints. Produces **DexManipNet** (3.3K bimanual robot episodes, e.g. pen capping, bottle unscrewing) from OakInk-V2, FAVOR, GRAB. Hands: Inspire, Shadow, Allegro, Arti-Mano, XHand, Inspire-FTP. Sim: Isaac Gym Preview 4. Released: full code, HF checkpoints, DexManipNet; GPL-3. **Relevance: very high** — the cleanest "pretrained tracker + residual per-object" decomposition, and a ready dataset.

**DexMachina (2025)** [arXiv:2505.24853](https://arxiv.org/abs/2505.24853) · [github.com/MandiZhao/dexmachina](https://github.com/MandiZhao/dexmachina). *Functional* retargeting: track **object** state (articulated ARCTIC objects, bimanual, long-horizon) rather than hand pose. Curriculum via **virtual object controllers with decaying strength** — the object is first dragged to its reference pose, and assistance decays so the hands take over — plus contact and fingertip guidance rewards. Hands: Allegro, Inspire, Schunk, XHand and more; sim: a custom **Genesis** fork; code, hand assets, ARCTIC demos released. **Relevance: very high** for the curriculum idea (object "training wheels") — trivially expressible as an extra engine force in MimicKit.

**Dexplore (CoRL 2025, NVIDIA)** [arXiv:2509.09671](https://arxiv.org/abs/2509.09671) · [github.com/NVlabs/dexplore](https://github.com/NVlabs/dexplore). Collapses retarget-then-track into **one RL loop**: references define *adaptive spatial scopes* (soft guidance) instead of hard targets; reward keeps the policy in-scope while minimizing effort, so robot-specific strategies emerge and noise is tolerated. Then distills to a vision-based, skill-conditioned generative controller. Sim: Isaac Gym Preview 4; hands: Inspire(18), LEAP(22), Allegro(22), Shadow(30) with 6-DoF floating wrist; data: GRAB; checkpoint for Inspire released.

**SPIDER (2025, Meta FAIR)** [arXiv:2511.09484](https://arxiv.org/abs/2511.09484) · [github.com/facebookresearch/spider](https://github.com/facebookresearch/spider). Training-free **sampling-based physics retargeting** with contact guidance: IK gives kinematics, a sampling solver finds dynamically feasible trajectories; 9 embodiments (Allegro, XHand, Inspire, Ability, Schunk; G1, H1-2, Fourier N1, Booster T1), 6 datasets, 3 simulators; ~10× faster than RL; released 6,523 demos over 4 hands/191 objects. Output can be replayed open-loop or used as cleaned references for RL — i.e., a **reference-cleaning front-end for MimicKit**.

**DexterityGen (2025)** [arXiv:2502.04307](https://arxiv.org/abs/2502.04307): a generative "foundation controller" trained on sim-generated dexterous motion that turns coarse teleop commands into safe fine actions. **DexPBT (2023)**: population-based training for Allegro-Kuka regrasp/throw/reorient in Isaac Gym (lives in [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs)). **DexMV (ECCV 2022)** [github.com/yzqin/dexmv-sim](https://github.com/yzqin/dexmv-sim): the original video→hand/object pose→retarget→DAPG pipeline (Adroit hand, MuJoCo/SAPIEN). These are reward-engineered or IL baselines, not tracking frameworks, but DexPBT's envs are the standard Isaac dexterous testbed.

### 2025–2026 follow-ups (physics HOI, whole-body + hands)
- **HumanX (CoRL 2026)** [arXiv:2602.02473](https://arxiv.org/abs/2602.02473) · [github.com/wyhuai/HumanX](https://github.com/wyhuai/HumanX). Video → GVHMR → UMR retarget to G1 → **XGen** (anchor-centric HOI editor/augmenter) → **XMimic** (SkillMimic-style interaction-prioritized imitation). Nine skills (basketball, football, badminton, cargo pickup, fighting) transferred zero-shot to a real G1. Apache-2.0 + Commons Clause. Same lab as SkillMimic; this is the "SkillMimic on a real robot from video" paper.
- **InterReal (Mar 2026)** [arXiv:2603.07516](https://arxiv.org/abs/2603.07516): unified physics HOI imitation for G1 with hand-object-contact-constrained data augmentation and an **automatic reward learner** (meta-policy allocates reward weights from tracking errors); real G1. Code not confirmed.
- **Weave (Sep 2026)** [arXiv:2609.16683](https://arxiv.org/abs/2609.16683) · [HF dataset appolyn/Weave](https://huggingface.co/datasets/appolyn/Weave): contact-aware retargeting + approach-motion completion; one policy over 29 body + 12 finger joints; 92.5% / 65% success on seen/unseen; ~9k rollouts (23 h) with contact annotations released. **DexWeave** [arXiv:2609.34724](https://arxiv.org/abs/2609.34724) is a sibling.
- **DeVI (Apr 2026)** [arXiv:2604.20841](https://arxiv.org/abs/2604.20841) · [github.com/snuvclab/devi](https://github.com/snuvclab/devi): imitates **text-conditioned synthetic videos** with a hybrid reward (3D human tracking + 2D object tracking) for unseen objects.
- **DexMan (Oct 2025)** [arXiv:2510.08475](https://arxiv.org/abs/2510.08475): bimanual humanoid-hand skills from third-person real/generated video, contact-based rewards against noisy hand-object poses; TACO & OakInk-v2.
- **ResMimic (Oct 2025)** [arXiv:2510.05070](https://arxiv.org/abs/2510.05070): residual policy on top of a General Motion Tracker (GMT) for G1 loco-manipulation, sim + real.
- **MaskedManipulator**, **HAIC (RSS 2026)** [github.com/ldt29/HAIC](https://github.com/ldt29/HAIC) (underactuated objects — carts, skateboard — via proprioceptive dynamics predictor), **Sugar** [arXiv:2605.20373](https://arxiv.org/abs/2605.20373), **GenHOI** [arXiv:2606.12995](https://arxiv.org/abs/2606.12995), **Progressive Kinematic-Dynamic Alignment** [arXiv:2511.10987](https://arxiv.org/abs/2511.10987), **DexForge** [arXiv:2610.06331](https://arxiv.org/abs/2610.06331), **AdvDex** [arXiv:2608.14028](https://arxiv.org/abs/2608.14028) (AMP-style adversarial learning for dexterous hands — closest to "AMP for manipulation"), **DemoBot** [arXiv:2601.01651](https://arxiv.org/abs/2601.01651). I only read abstracts/snippets of these.

---

## 2. Humanoid / robot manipulation from human data (real robots, mostly IL)

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

## 3. Retargeting

### Human hand → robot hand
- **dex-retargeting / AnyTeleop (RSS 2023)** [github.com/dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting): optimization-based position / vector / **DexPilot-style** retargeting to many hands (uses [dex-urdf](https://github.com/dexsuite/dex-urdf) assets). The de-facto standard; both ManipTrans and Dexplore start from it. **DexPilot (ICRA 2020)**: task-space fingertip-vector objectives with a pinch-closing prior. **GeoRT** [arXiv:2503.07541](https://arxiv.org/abs/2503.07541): ultrafast neural geometric retargeting. **dexi** ([yuxiang-gao/dexi](https://github.com/yuxiang-gao/dexi)) is a Rust reimplementation.
- **Contact-preserving kinematic retargeting**: Lakshmipathy, Hodgins, Pollard, *Kinematic Motion Retargeting for Contact-Rich Anthropomorphic Manipulations*, ACM TOG 44(2), 2025, [arXiv:2402.04820](https://arxiv.org/abs/2402.04820) — shape-matching transfers **contact areas** across pre-grasp/pickup/in-hand/release; **TopoRetarget (2026)** [arXiv:2606.16272](https://arxiv.org/abs/2606.16272) — sparse hand-object interaction graph + Laplacian deformation, +40.6 pp on pen-spin RL; **REGRIND (2026)** [arXiv:2607.11874](https://arxiv.org/abs/2607.11874), [yunhaifeng.com/REGRIND](https://yunhaifeng.com/REGRIND) — interaction-mesh retargeting that preserves hand-object spatial/contact relations, then **residual RL tracking object-centric keypoints**, zero-shot to two real hands (scissors, screwdriver); **GenHand** (Nature Rob. 2026 [unverified]). Physics-informed: **SPIDER**, **DexForge**, **C2Dex** [arXiv:2608.07045](https://arxiv.org/abs/2608.07045).

### Human body → humanoid (used by tracking frameworks)
- **PHC** (Luo et al., ICCV 2023): SMPL→humanoid via shape-fitted joint IK; MimicKit's own `smpl`/G1 pipelines and InterMimic/Omnigrasp build on it.
- **GMR** [arXiv:2510.02252](https://arxiv.org/abs/2510.02252), [github.com/YanjieZe/GMR](https://github.com/YanjieZe/GMR), ICRA 2026: real-time CPU IK retargeting to many humanoids; "Retargeting Matters" shows it beats PHC and ProtoMotions retargeters when tracked with BeyondMimic. Retargeter for **TWIST/TWIST2** ([amazon-far/TWIST2](https://github.com/amazon-far/TWIST2)); exports CSV for BeyondMimic.
- **OmniRetarget** [arXiv:2509.26633](https://arxiv.org/abs/2509.26633), [omniretarget.github.io](https://omniretarget.github.io/), [HF dataset](https://huggingface.co/datasets/omniretarget/OmniRetarget_Dataset): **interaction mesh** (Delaunay tetrahedralization over human/robot keypoints + object/terrain samples) with constrained optimization so human-object/terrain relations are preserved and penetration/foot-skate removed; G1 policies trained with 5 reward terms, no curriculum, up to 30 s loco-manipulation. Reported ICRA 2026 best manipulation & locomotion paper [from snippet]. This is the whole-body analogue of contact-preserving hand retargeting and the right tool to produce G1+object reference motions for MimicKit.
- Trackers that consume these: **BeyondMimic** [arXiv:2508.08241](https://arxiv.org/abs/2508.08241), [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) (Isaac Lab; mjlab port), **KungfuBot/PBHC** [arXiv:2506.12851](https://arxiv.org/abs/2506.12851), [TeleHuman/PBHC](https://github.com/TeleHuman/PBHC), **ASAP** [arXiv:2502.01143 — id from memory], **GMT** [arXiv:2506.14770 — id from memory], **TWIST** [arXiv:2505.02833 — id from memory]. None of these track objects; ResMimic, OmniRetarget-RL, HumanX, Weave, InterMimic-G1 are the ones that do.

---

## 4. Key technical ingredients for physics HOI tracking (what to put in the reward/env)

1. **Object-pose tracking reward.** Every physics HOI paper has `r_obj = exp(-k·‖p_obj − p̂_obj‖) · exp(-k·rot_err)` (plus object velocity in PhysHOI/SkillMimic). Human2Sim2Robot and DexMachina argue the object trajectory is the *embodiment-agnostic* task spec, so weight it higher than body pose when the hand morphology differs.
2. **Relative (hand-to-object) terms.** PhysHOI/SkillMimic `r_rel` tracks body-keypoint-minus-object vectors; Omnigrasp/REGRIND use object-centric keypoints. This is what fixes "arm is right but ball is 5 cm off".
3. **Contact graph / contact rewards.** PhysHOI's binary CG with multiplicative reward; InterMimic's contact-aware terms; DexMachina/DexMan contact guidance; InterReal's contact-constrained augmentation. Multiplicative composition (`Π r_i`) rather than weighted sum is a consistent finding (PhysHOI, SkillMimic, V2).
4. **Kinematic vs dynamic tracking.** Pure kinematic references are physically inconsistent (penetration, float). Fixes: teacher-correction (InterMimic), sampling-based physics retargeting (SPIDER), single-loop soft-scope tracking (Dexplore), residual on a pretrained tracker (ManipTrans, ResMimic, REGRIND, Video2Sim2Real).
5. **Curricula.** DexMachina's decaying virtual object controller; ManipTrans' hand-only pretrain → residual; InterMimic's "perfect first then scale"; SkillMimic-V2's ATS difficulty-weighted clip sampling.
6. **Noisy/sparse data.** SkillMimic-V2 STG (stitch clips) + STF/ε-NSI (reset in a noise ball, learn recovery) + history encoder; HumanX XGen augmentation; InterReal augmentation + learned reward weights.
7. **Reference state initialization with objects.** All tracking methods reset *both* character and object to a reference frame (SkillMimic `--state_init`). Human2Sim2Robot resets to the *pre-manipulation* hand pose; Omnigrasp samples pre-grasp poses. MimicKit's RSI needs an object-state channel in the motion file.
8. **Sim-to-real for contacts.** REGRIND/HumanX/Weave rely on system ID + domain randomization of object mass/friction/size (OmniRetarget: "simple DR shared by all tasks"); SPIDER augments with external payloads/perturbations. Finger PD gains and contact stiffness are the usual culprits; Dexplore's "minimize control effort" regularizer helps.

---

## 5. Simulators / benchmarks with objects + dexterous hands + large-scale RL

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

## 6. Ranked table: 15 works closest to "MimicKit for manipulation"

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

## 7. Open problems

1. **Reference quality is the bottleneck, not the RL.** Mocap hand-object data is noisy (penetration, missing contacts); every strong 2025–26 method is mostly a data-cleaning/curriculum story (InterMimic teachers, SPIDER, Dexplore, V2's STF). No agreed-upon "AMASS-for-HOI" with physically consistent references exists; InterAct, OmniRetarget-Dataset, DexManipNet, SPIDER's 6.5k and Weave's 9k rollouts are partial answers.
2. **Embodiment gap for hands** is far larger than for bodies: 5-finger 20-DoF human → 3-finger G1 hand or 16-DoF Allegro. Functional (object-centric) tracking beats hand-pose tracking in every comparison (DexMachina, Human2Sim2Robot, REGRIND), but object-only rewards lose style/naturalness — the body-prior trick (Omnigrasp/PULSE) has no mature hand equivalent.
3. **Contact-rich sim-to-real**: GPU-sim contact models (PhysX, Newton/MuJoCo-Warp) still disagree with reality on friction/compliance; tactile sensing is nearly absent from tracking frameworks (Dex-X [arXiv:2609.07747](https://arxiv.org/abs/2609.07747) is an early exception).
4. **Articulated and deformable objects**: only DexMachina (ARCTIC) and HAIC (underactuated carts) go beyond rigid bodies.
5. **Long-horizon composition**: SkillMimic's HLC and MaskedManipulator's goal conditioning are the only released multi-skill HOI controllers; nothing comparable to ASE/MaskedMimic latent spaces exists for hands.
6. **Adversarial (AMP-style) objectives for manipulation** are barely explored — AdvDex is the only 2026 example — so MimicKit's AMP/ADD/SMP code has an open niche.
7. **No multi-engine HOI tracking codebase**: InterMimic covers Isaac Gym + Lab; nobody has Newton/MuJoCo-Warp HOI tracking released. That is exactly the slot a MimicKit extension would fill.

---

## 8. Recommendations: what to replicate first

**(1) PhysHOI → SkillMimic reward inside MimicKit (first milestone, 2–4 weeks).**
Rationale: it is DeepMimic plus three reward factors and an object channel, on a codebase (ASE) whose structure MimicKit already mirrors, with released SMPL-X-with-fingers assets and the BallPlay data. Concretely: add a `DeepMimicHOIEnv` deriving from `DeepMimicEnv` + `StaticObjectsEnv` logic but with `fix_root=False` and an object track in the motion file; implement `r = r_body · r_obj · r_rel · r_cg`; extend RSI to reset object pose/velocity. Validate on one BallPlay clip, then ParaHome household clips (book grasp, cube reorient) which are the actual "manipulation" targets. Add SkillMimic-V2's ε-NSI and ATS next — both are ~50-line changes to motion sampling. Cite: PhysHOI [arXiv:2312.04393](https://arxiv.org/abs/2312.04393), SkillMimic [arXiv:2408.15270](https://arxiv.org/abs/2408.15270), V2 [arXiv:2505.02094](https://arxiv.org/abs/2505.02094).

**(2) InterMimic as the reference implementation for Isaac Lab + G1-with-hands.**
Rationale: it is the only released system doing exactly your target (whole-body HOI tracking, SMPL-X and G1, Isaac Lab 2.3), MIT-licensed, with processed OMOMO data and the InterAct cleaning pipeline. Run it as-is first to get a working G1+object tracking baseline and reference numbers; then port its teacher-student + contact reward into MimicKit's Isaac Lab engine so you can compare DeepMimic/AMP/ADD objectives on the same data. Its retargeting-free G1 result ("directly learned from mocap") is worth reproducing before investing in OmniRetarget. Cite: [arXiv:2502.20390](https://arxiv.org/abs/2502.20390), [github.com/Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic).

**(3) ManipTrans (or Dexplore) for the hand-only / robot-hand branch.**
Rationale: if the goal includes real dexterous hands rather than SMPL-X fingers, ManipTrans gives a pretrained hand-trajectory imitator, six hand URDFs, DexManipNet (3.3k bimanual episodes from OakInk-V2/FAVOR/GRAB) and the "residual per task" recipe; Dexplore is the NVIDIA alternative with cleaner soft-scope tracking and four hands on GRAB. Either can be re-implemented as a MimicKit "character" that is a floating-wrist hand (Dexplore's 6-DoF floating base is the trick) tracking a hand+object reference — which also makes DexMachina's decaying virtual-object-controller curriculum a one-line engine force. Cite: ManipTrans [arXiv:2503.21860](https://arxiv.org/abs/2503.21860), Dexplore [arXiv:2509.09671](https://arxiv.org/abs/2509.09671), DexMachina [arXiv:2505.24853](https://arxiv.org/abs/2505.24853).

Supporting tools to adopt rather than build: `dex-retargeting` for hand IK, GMR/OmniRetarget for body+object retargeting to G1, SPIDER for physics-cleaning references, and ProtoMotions' SceneLib as the design reference for object motion files.

**Verification note.** Items marked [unverified] were taken from search snippets or memory and should be checked before citing; everything else was confirmed from the arXiv abstract, project page or GitHub README during this survey. I did not run any of the released code.
