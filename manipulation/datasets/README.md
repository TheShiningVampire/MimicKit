# Sample datasets

Small, committed samples that exercise the motion format v2 and the converters. The full downloaded
material (about 1.4 GB across 20 sources) lives outside the repo at
`/data/user_data/vawale/MimicKit/manip_data/` with a `MANIFEST.md` and a per-file `_SCHEMA_DUMP.txt`.
The dataset survey with licences, access procedures and schemas is
[../report/04_datasets_survey.md](../report/04_datasets_survey.md).

| Folder | What | Source and licence | Use |
|---|---|---|---|
| `samples/synthetic/` | two humanoid clips (walk, punch) with a box attached to one hand for the middle of the clip; v2 pkl + preview gif | generated from MimicKit's own humanoid clips by `tools/make_synthetic_hoi_clip.py`; no licence constraints | unit tests of the HOI motion library, env, viewer |
| `samples/hoi_retarget_g1/` | four Unitree G1 29-DoF clips (OMOMO clothes-stand carry x2, ParaHome cup, ParaHome kettle) converted to v2, with contact flags, the original preview mp4s, the clothes-stand URDF + mesh, and a skeleton gif | HOI-Retarget, huggingface.co/datasets/leggedrobotics/hoi-retarget, CC BY-NC-SA 4.0 (see `NOTICE.hoi_retarget.md`); converted by `tools/convert_hoiretarget_to_mimickit.py` | **Track A phase 0/1 data**: joint order matches `data/assets/g1/g1.xml`, so these load into a G1 HOI env directly |
| `samples/skillmimic_ballplay/` | one BallPlay-M clip (`rebound`, tensor `[T, 337]`, 60 fps) plus the SMPL-X-like 52-joint MJCF (box-hand variant) and the ball URDF | SkillMimic, github.com/wyhuai/SkillMimic, Apache-2.0 | reference for the PhysHOI/SkillMimic file layout and contact-graph rewards; the MJCF needs a MimicKit-compatible rewrite (one hinge per nested body) before it can be loaded |

The ParaHome clips reference `parahome/_assets/cup.urdf` and `kettle.urdf`, which were not part of the
downloaded sample; fetch them from the HF repo if you need the object meshes.

## What is in a v2 file

```
{"loop_mode": 0, "fps": 30, "frames": [N, 6+D],
 "objects": [{"name", "asset", "frames": [N, 6], "contacts": [N, S], "active": [N]}],
 "contact_sites": [S names], "meta": {...}}
```

Validate any file with:

```bash
python manipulation/tools/hoi_motion.py manipulation/datasets/samples/hoi_retarget_g1/*.pkl
```
