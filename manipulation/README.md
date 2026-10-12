# MimicKit for manipulation: learning from human demonstrations

Research package for extending MimicKit's state-action RL motion imitation (DeepMimic, AMP, ASE, ADD, SMP)
from whole-body control to **manipulation learned from human hand+object data**. Produced 2026-10-12.

Start with the plan, then the explainer video.

| Item | Path |
|---|---|
| Working plan (tracks, phases, tests, compute, risks, questions) | [plan/00_working_plan.md](plan/00_working_plan.md) |
| Literature survey: methods | [report/01_literature_survey_methods.md](report/01_literature_survey_methods.md) |
| MimicKit gap analysis and `DeepMimicHOIEnv` design (file:line level) | [report/02_mimickit_gap_analysis_and_design.md](report/02_mimickit_gap_analysis_and_design.md) |
| Open-source codebase survey (InterMimic, SkillMimic, ManipTrans, HDMI, ...) | [report/03_codebase_survey.md](report/03_codebase_survey.md) |
| Dataset survey (licences, access, schemas) | [report/04_datasets_survey.md](report/04_datasets_survey.md) |
| Whole-body control vs manipulation: the differences | [report/05_differences_whole_body_vs_manipulation.md](report/05_differences_whole_body_vs_manipulation.md) |
| Deep research with adversarial claim verification (reward design, data repair, curricula, sim-to-real) | [report/06_deep_research_hoi_rl.md](report/06_deep_research_hoi_rl.md) |
| Sample datasets (small, committed) and manifest | [datasets/](datasets/) |
| Tools: motion format v2, converters, synthetic clip, viewer | [tools/](tools/) |
| Explainer video (3Blue1Brown style, manim + TTS) | [video/](video/) |

## Motion format v2 (hand+object clips)

Backward compatible with MimicKit's pickle `{loop_mode, fps, frames[N, 6+D]}`. Adds an optional `objects`
block with one 6-vector pose track per object (world position + exponential-map rotation, the same
convention as the character root), optional per-frame contact labels, and an `active` flag.
See [tools/hoi_motion.py](tools/hoi_motion.py) for the loader, validator and writer, and
[datasets/samples/](datasets/samples/) for example files.

## Quick start

```bash
# validate and summarize an HOI motion file
python manipulation/tools/hoi_motion.py manipulation/datasets/samples/synthetic/humanoid_carry_box_synthetic.pkl

# render a quick matplotlib preview (character key bodies + object) to mp4/gif
python manipulation/tools/preview_hoi_motion.py <file.pkl> --char data/assets/g1/g1.xml --out preview.gif

# convert HOI-Retarget G1+object clips (parquet, joint order == data/assets/g1/g1.xml) to MimicKit v2
python manipulation/tools/convert_hoiretarget_to_mimickit.py <rows.parquet> --out_dir <dir> --video

# build a synthetic carry clip from any MimicKit humanoid clip
python manipulation/tools/make_synthetic_hoi_clip.py --motion data/motions/humanoid/humanoid_walk.pkl \
    --char data/assets/humanoid/humanoid.xml --hand right_hand --out <clip.pkl>
```

Large downloaded samples live outside the repo in `/data/user_data/vawale/MimicKit/manip_data/` (see its `MANIFEST.md`).
