"""Make a synthetic human-object interaction clip in MimicKit motion format v2.

Takes an existing MimicKit character clip (e.g. humanoid_walk.pkl), runs forward kinematics with the
character's MJCF model, and attaches a box to one hand for a middle segment of the clip: the box
rests on the floor before the "pick" frame, rides with the hand between pick and release, and rests
where it was released afterwards. Contact labels are 1 for the carrying hand while the box is held.

This is a stand-in for real HOI data: it exercises the v2 schema, the HOI motion library, object
observation/reward code, and the viewers, with no licence constraints.

Run with the mimickit env from the repo root:

    python manipulation/tools/make_synthetic_hoi_clip.py \
        --motion data/motions/humanoid/humanoid_walk.pkl --char data/assets/humanoid/humanoid.xml \
        --hand right_hand --pick 0.25 --release 0.75 \
        --out manipulation/datasets/samples/synthetic/humanoid_carry_box_synthetic.pkl
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import anim.mjcf_char_model as mjcf_char_model  # noqa: E402
import util.torch_util as torch_util  # noqa: E402
from hoi_motion import CLAMP, HOIMotion, HOIObjectTrack  # noqa: E402


def fk_body_pos(char_file, frames):
    """World positions and rotations of all bodies for every frame: [N, B, 3], [N, B, 4] (xyzw)."""
    model = mjcf_char_model.MJCFCharModel(device="cpu")
    model.load(char_file)
    f = torch.tensor(frames, dtype=torch.float32)
    root_pos = f[:, 0:3]
    root_rot = torch_util.exp_map_to_quat(f[:, 3:6])
    joint_rot = model.dof_to_rot(f[:, 6:])
    body_pos, body_rot = model.forward_kinematics(root_pos, root_rot, joint_rot)
    return body_pos.numpy(), body_rot.numpy(), model.get_body_names()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--char", required=True)
    ap.add_argument("--hand", default="right_hand")
    ap.add_argument("--pick", type=float, default=0.25, help="fraction of the clip where the hand picks the box")
    ap.add_argument("--release", type=float, default=0.75, help="fraction of the clip where the hand releases it")
    ap.add_argument("--box", type=float, default=0.12, help="box edge length [m]")
    ap.add_argument("--asset", default="data/assets/objects/dodgeball.xml",
                    help="asset path stored in the file (any small rigid object works for a test)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = HOIMotion.load(a.motion)
    frames = src.frames.copy()
    n_bad = int((~np.isfinite(frames)).sum())
    if n_bad:
        # some shipped clips carry NaN in unused joint columns; MimicKit's loader tolerates them, we zero them
        frames[~np.isfinite(frames)] = 0.0
        print(f"note: replaced {n_bad} non-finite values in the source frames with 0")
    n = src.num_frames
    body_pos, body_rot, names = fk_body_pos(a.char, frames)
    hid = names.index(a.hand)
    hand_p = body_pos[:, hid]       # [N, 3]
    hand_q = body_rot[:, hid]       # [N, 4]

    i0 = int(round(a.pick * (n - 1)))
    i1 = int(round(a.release * (n - 1)))
    assert 0 <= i0 < i1 < n, "pick must come before release inside the clip"

    half = a.box / 2.0
    obj = np.zeros((n, 6), dtype=np.float32)
    # held segment: box centre offset 6 cm below the hand body along world -z (a palm-down carry)
    held_p = hand_p[i0:i1 + 1] + np.array([0.0, 0.0, -0.06], dtype=np.float32)
    held_p[:, 2] = np.maximum(held_p[:, 2], half)        # never below the floor
    obj[i0:i1 + 1, 0:3] = held_p
    obj[i0:i1 + 1, 3:6] = torch_util.quat_to_exp_map(torch.tensor(hand_q[i0:i1 + 1])).numpy()
    # resting before the pick: on the floor directly under the pick position
    rest0 = held_p[0].copy(); rest0[2] = half
    obj[:i0, 0:3] = rest0
    # resting after release: on the floor under the release position
    rest1 = held_p[-1].copy(); rest1[2] = half
    obj[i1 + 1:, 0:3] = rest1

    # smooth the two transitions over a few frames so velocities are finite
    k = max(2, int(0.1 * src.fps))
    for j0 in (i0, i1 + 1):
        lo, hi = max(0, j0 - k), min(n - 1, j0 + k)
        if hi > lo:
            t = np.linspace(0, 1, hi - lo + 1)[:, None]
            obj[lo:hi + 1, 0:3] = (1 - t) * obj[lo, 0:3] + t * obj[hi, 0:3]

    sites = ["left_hand", "right_hand"]
    contacts = np.zeros((n, len(sites)), dtype=np.float32)
    contacts[i0:i1 + 1, sites.index(a.hand)] = 1.0
    active = np.ones(n, dtype=np.float32)

    track = HOIObjectTrack("box", a.asset, obj, contacts, active)
    out = HOIMotion(CLAMP, src.fps, frames, [track], sites,
                    meta={"source": os.path.relpath(a.motion, ROOT), "char": os.path.relpath(a.char, ROOT),
                          "hand": a.hand, "pick_frame": i0, "release_frame": i1, "box_size_m": a.box,
                          "kind": "synthetic carry; object rigidly attached to the hand while held"})
    errs = out.validate()
    assert not errs, errs
    out.save(a.out)
    print(out.summary())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
