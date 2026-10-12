# Explainer video: "From mimicking motion to mimicking manipulation"

A 3Blue1Brown-style explainer (about 3.5 minutes) on what changes when MimicKit's motion-imitation
recipe is applied to hand-object manipulation. Narration is synthesized with piper TTS; the animation
is manim Community Edition.

- Final video: `manipulation_explained.mp4` (1080p30, in this directory once built; also at
  `/data/user_data/vawale/MimicKit/manip_video/manipulation_explained.mp4`)
- Narration script: [src/narration.py](src/narration.py) (one list of sentences per scene)
- Scenes: [src/scenes.py](src/scenes.py) (one manim `Scene` per section, timed to the narration)
- Build: [src/build.sh](src/build.sh)

## Sections

1. Title: the question
2. The MimicKit loop for whole-body motion (reference, simulator, policy, reward, RL)
3. Difference 1: the goal is the object, not the pose
4. Difference 2: contact (few flat footsteps vs many staggered fingertip contacts)
5. Difference 3: what the policy sees (object state, future reference)
6. Difference 4: the data (capture, occlusion, datasets)
7. Retargeting: human hand to robot hand, object as the anchor frame
8. The recipe: dataset, retarget + motion file, HOI environment, RL, distill
9. The plan: four phases with tests
10. Close

## Rebuilding

```bash
# on a compute node (CPU is enough); TTS env and manim env are on /data/user_data
bash manipulation/video/src/build.sh h      # 1080p30 (about 10 minutes on 8 cores)
bash manipulation/video/src/build.sh l      # 480p15 quick check
```

Environments: `/data/user_data/vawale/envs/tts` (piper, voice `en_US-ryan-high`) and
`/data/user_data/vawale/envs/manim` (manim 0.22 from conda-forge). Manim's `Text` is used throughout,
so no LaTeX is required.
