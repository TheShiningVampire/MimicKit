# 1. Survey: datasets of humans manipulating objects (for a MimicKit/DeepMimic-style physics imitation pipeline)

Verified on 2026-10-12 from a Babel compute node. "Open" = direct URL / ungated HF; "Reg" = account + license click-through; "Form" = Google/Office form with manual approval; "Email" = write to authors. Items marked **[unverified]** come from secondary sources only.

## 1.1 Comparison table

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

## 1.2 Already processed for physics imitation (most important for MimicKit)

1. **InterMimic** (CVPR'25) — `OMOMO_new.zip` (929 MB, open Drive) and `OMOMO_retarget.zip` (753 MB) are SMPL-X humanoid reference clips `Tensor[T,591]` @30 fps: root pos/quat, 51×3 DoF, 52 body positions + quats, object pos/quat, object-contact flag and per-body contact flags. Also ships Unitree G1 (29-DoF + 3-finger hands) configs. InterAct's `simulation/interact2mimic.py` converts any InterAct sub-dataset (OMOMO, BEHAVE, InterCap, GRAB, NeuralDome, IMHD², CHAIRS, ParaHome, ARCTIC) into this format + auto-generated object URDFs. Authorization form: https://docs.google.com/forms/d/e/1FAIpQLScMCfdd8BXzDBZ3iw0x5zA3KSTlD1F2GTaO8ylDG9Cj1upaPw/viewform. An HF mirror `guand1/interact-smplh` (2.6 GB, BEHAVE/GRAB/IMHD/NeuralDome SMPL-H) exists but is manually gated.
2. **HOI-Retarget** (ETH, 2026, HF `leggedrobotics/hoi-retarget`, open) — 13,904 robot-space clips (Unitree G1 + H2) from OMOMO/ParaHome/NeuralDome/CoRoleHOI/IMHD² with contact flags and object poses; `shinben0327/hoi-retarget-sapiens-k1` adds Robotis K1. 13 OMOMO object URDFs included.
3. **SkillMimic BallPlay-M** (subset on GitHub, Apache-2.0) and **PhysHOI BallPlay-V**: Isaac-Gym-ready `.pt` with 52-joint humanoid + ball; full BallPlay-M still "coming soon".
4. **ManipTrans DexManipNet** (HF `LiKailin/DexManipNet`, GPL-3.0, 8.3 GB) — 3.3k successful RL rollouts (Inspire hand, 12-DoF) on OakInk2 + FAVOR, with object URDFs; hdf5 with q/dq/state/contact tip forces at 60 fps.
5. **OmniGrasp** — no release of processed GRAB; processing scripts (`process_grab_raw.py`, `convert_grab_smplx.py`) require GRAB registration.
6. **HOIDiNi** (HF `Roey/hoidini`, open) — "GRAB_RETARGETED_compressed" (2.7 GB) + SMPL/MANO models redistributed (license caveat).
7. **DexCanvas** — 28-DoF MANO-URDF `urdf_dof_target` + contact forces: directly usable as dexterous-hand tracking targets.
8. **InterPrior** (CVPR'26) and **InterReal** (2026) build on InterMimic data; no new dataset released. **DreamMimic**, **DeVI** (2026) likewise **[unverified]**.

## 1.3 Notable 2025–2026 additions I added

GigaHands (2025, open), EgoDex (2025, open), HOT3D-Clips (open HF), CORE4D (open HF), ParaHome SMPL-X (2025), HUMOTO (2025), PA-HOI (2025), IMHD² on HF (2026), HODome SMPL-X re-release (2026-05), HOI-Retarget (2026), HandX (2026), DexCanvas (2025/26), DexterCap/DexterHand (2026), HiPHI (2026), OpenEgo (2025; aggregator of 6 ego sets, 21-joint MANO in camera frame; download via openegocentric.com, HDF5 **[unverified]**), HO-Cap (2024, open HF `JWRoboticsVision/HO-Cap-Dataset`, CC BY 4.0, 119 GB), MMHOI (WACV'26, Zenodo CC BY-SA), Glove2Hand/ART-Glove (2026, tactile gloves, release status **[unverified]**).

## 1.4 Access notes (registration-only sets)

- GRAB / ARCTIC / InterCap / AMASS / MANO / SMPL-X: create account at `*.is.tue.mpg.de`, accept license, then scripted download (ARCTIC: env vars `ARCTIC_USERNAME/PASSWORD`, `SMPLX_*`, `MANO_*`, `raw_seqs` 215 MB, `meta` 91 MB suffice for motion).
- InterAct: Google form above; HIMO: https://docs.google.com/forms/d/e/1FAIpQLSdl5adeyKxBSBFZpgs0A7-dAouRkMFAGUP5iz3zxGDj_PhB1w/viewform; IMHD²: https://forms.gle/3MDh3b4szhFwcYa26 (or HF); CHAIRS: https://forms.gle/t4SjmJS4RPx7AFvFA; HOGraspNet: https://forms.gle/UqH15zN2PiBGQDUs7; HUMOTO full: https://forms.office.com/r/3a7jgFi5UH (CMU is on the licensed-institution list — ask the lab's designated contact); Ego-Exo4D: ego4d.dev/request/ego-exo4d (AWS creds); Motion-X: Google form; HOT3D full: projectaria.com email signup; HiPHI / Assembly101 / interact-smplh: HF "accept conditions" button.

# 2. Download report

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

# 3. Recommendations

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
