# MimicKit → Manipulation-from-Demonstration: Gap Analysis & Design

Repo: `/home/vawale/my_things/codes/MimicKit` (fork of xbpeng/MimicKit, branch `main`). All paths below are relative to the repo root; assets/motions under `data/assets/*` and `data/motions/*` are symlinks into `/data/user_data/vawale/MimicKit/data/`.

## 1. Current architecture map

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

## 2. Motion format spec (exact, as implemented)

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

## 3. Gaps for manipulation

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

## 4. Proposed minimal-change design

### 4.1 Motion file v2 (backward compatible)

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

### 4.2 `HOIMotionLib(MotionLib)` — new file `mimickit/anim/hoi_motion_lib.py`

- `_load_motion_pkl`: call super logic, additionally read `objects`; require every clip to expose the same ordered object-name list (or pad with `active=0` tracks); stack `_frame_obj_pos [F,O,3]`, `_frame_obj_rot [F,O,4]`, `_frame_contact [F,O,S]`, `_frame_obj_active [F,O]`.
- `_process_data`: after super, finite-difference `_frame_obj_vel`, `_frame_obj_ang_vel` exactly like L176-183, and build `_motion_wrap_delta` broadcasting to objects.
- `calc_obj_frame(motion_ids, times) -> (obj_pos, obj_rot, obj_vel, obj_ang_vel, contact, active)`: reuse `_calc_frame_blend` L123-136 and `_calc_loop_offset` L138-147; lerp/slerp pose, nearest-frame contact/active.
- `get_num_objects()`, `get_object_names()`, `get_object_assets()`, `get_num_contact_sites()`.

### 4.3 `HOIMixin` + `DeepMimicHOIEnv` — new file `mimickit/envs/deepmimic_hoi_env.py`

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

### 4.4 Object asset / env config schema

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

### 4.5 Hand asset: `data/assets/g1/g1_dex3.xml`

Append Dex3-1 (7 DOF/hand: thumb 3, index 2, middle 2) subtrees under `left_wrist_yaw_link` (`g1.xml:204`) / `right_wrist_yaw_link` (L252), each finger segment as a nested `<body>` with exactly one `<joint type="hinge">`, `pos="0 0 0"`, `range`, `stiffness/damping` (hand gains ~1-2 N·m/rad, 0.05-0.1), `actuatorfrcrange`, plus a `<motor>` per joint. Fingertip collision geoms need `condim="3"` and friction ≥ 1 (the body default is `condim="1"`, `humanoid.xml:7`). D becomes 29+14 = 43; the frame is 49 wide; `init_pose` and `joint_err_w` (30 → 44 entries) must be regenerated; verify DFS order with `env_name: char_dof_test` before any training. Isaac Lab additionally needs `g1_dex3.usd` with the same body names (`_validate_envs`, `char_env.py:230-238`). Shadow/Allegro/Inspire work the same way, except coupled/mimic joints must be exposed as independent hinges (no mimic support in `kin_char_model.py`).

### 4.6 AMP variant

`AMPHOIEnv(HOIMixin, AMPEnv)` extends `_build_disc_obs_buffers` (`amp_env.py:98-158`) with object circular buffers, `_update_disc_hist` (L165-184) pushes object pose/vel and hand-object vectors, `_fetch_disc_demo_data` (L67-90) calls `calc_obj_frame`, and `compute_disc_obs` (L336-359) concatenates object features expressed in the reference-root heading frame. `AMPAgent`/`AMPModel` need no change because the disc input size comes from `get_disc_obs_space()`.

### 4.7 `tools/convert_hoi_to_mimickit.py`

Inputs: (i) retargeted body+hand pose for the target MJCF — from GMR (`{fps, root_pos, root_rot, dof_pos}`) or from SMPL-X via the SMPL path extended with dex-retargeting for fingers; (ii) object 6-DoF tracks per frame from the source dataset (GRAB/ARCTIC/OMOMO/BEHAVE/HOI-M3) in the *same* world frame; (iii) contact labels (dataset-provided or derived: fingertip-to-mesh distance < 5 mm). Steps: resample everything to one fps; apply the exact global transform used for the human (Y-up→Z-up as in `smpl_to_mimickit.py:47-48`, floor calibration L161-167, GMR scale) to object tracks; if the retarget rescaled limbs, re-anchor objects to the hand pose during labelled contact frames so the hand-object relative transform is preserved; convert object rotations to exp-maps; write the v2 pickle with `loop_mode=CLAMP`, and emit the matching `objects:` yaml snippet. Validate with the `view_motion_hoi` env.

## 5. Phased implementation checklist

**Phase 0 — Data plumbing (no RL).** ☐ Motion v2 loader/saver in `motion.py` (optional keys, old files unchanged). ☐ `HOIMotionLib` with unit tests on a synthetic clip (box translated along x, slerp sanity, velocity sanity, loop offset). ☐ `ViewMotionHOIEnv` (extend `view_motion_env.py:53-78` to also `set_root_pos/rot` objects) + `data/envs/view_motion_hoi_humanoid_env.yaml`. ☐ Register env names in `env_builder.py`.

**Phase 1 — Whole-body + object, no hand DOF.** ☐ `HOIMixin`, `DeepMimicHOIEnv`; use the existing humanoid (sphere hands) and a synthetic carry/push clip. ☐ Object obs, four reward terms, object termination, ghost objects. ☐ Config `engine:` overrides (240 Hz, `ground_contact_height`). ☐ Smoke-train 512 envs on a `katef` GPU via sbatch (see `experiments/*/train.sbatch`), check `Test_Return` rises and object deviation logs (add to `record_diagnostics`, cf. `deepmimic_env.py:63-75`).

**Phase 2 — Articulated hands.** ☐ `g1_dex3.xml`/`.usd`; validate joint order with `char_dof_test`. ☐ Regenerate `init_pose`, `joint_err_w`, `key_bodies`, `contact_bodies`. ☐ Hand-specific `action_std`/gain tuning; consider zeroing `joint_err_w` for fingers early (pose term) and relying on `r_rel`/`r_contact`.

**Phase 3 — Real demonstrations.** ☐ `convert_hoi_to_mimickit.py` for one dataset; dataset yaml with fixed object slots. ☐ Train DeepMimicHOI on 3-5 pick-and-place clips; RSI with `rand_reset` across the clip including post-grasp phases.

**Phase 4 — Priors and robustness.** ☐ `AMPHOIEnv` with object-conditioned discriminator. ☐ Engine API additions: pair-filtered contact sensors (Isaac Lab/Newton), object mass/friction/scale randomisation, `set_obj_color`. ☐ Curriculum: kinematic object (`fix_root=True`, driven from the reference) → dynamic object. ☐ Optional transformer/set encoder for variable object counts (replace `fc_*` net in `learning/nets/`).

**Risks to flag now.** Isaac Gym's height-masked ground contact (`isaac_gym_engine.py:311-320`) will misfire for floor-level manipulation; every engine requires identical object slots per env, so mixed-object datasets need placeholders; the MJCF parser's "3 hinges in one body = spherical" rule (`mjcf_char_model.py:150-180`) would silently fuse a 3-hinge finger root into an exp-map joint, so finger joints must live in separate nested bodies.
