"""Quick matplotlib preview of an HOI motion file (v2): character skeleton + object box, to mp4 or gif.

No simulator needed. Uses MimicKit's MJCF kinematic model for forward kinematics.

    python manipulation/tools/preview_hoi_motion.py <clip.pkl> --char data/assets/humanoid/humanoid.xml --out preview.mp4
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402
import torch  # noqa: E402

import anim.mjcf_char_model as mjcf_char_model  # noqa: E402
import util.torch_util as torch_util  # noqa: E402
from hoi_motion import HOIMotion  # noqa: E402


def box_corners(center, exp_map, size):
    q = torch_util.exp_map_to_quat(torch.tensor(exp_map, dtype=torch.float32)[None])[0]
    h = size / 2.0
    c = np.array([[sx, sy, sz] for sx in (-h, h) for sy in (-h, h) for sz in (-h, h)], dtype=np.float32)
    w = torch_util.quat_rotate(q[None].expand(8, 4), torch.tensor(c)).numpy() + center
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return w, edges


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--char", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=2, help="use every k-th frame")
    ap.add_argument("--box", type=float, default=0.12)
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()

    m = HOIMotion.load(a.file)
    frames = m.frames.copy()
    frames[~np.isfinite(frames)] = 0.0
    model = mjcf_char_model.MJCFCharModel(device="cpu")
    model.load(a.char)
    f = torch.tensor(frames)
    body_pos, _ = model.forward_kinematics(f[:, 0:3], torch_util.exp_map_to_quat(f[:, 3:6]), model.dof_to_rot(f[:, 6:]))
    body_pos = body_pos.numpy()
    parents = [model.get_parent_id(j) for j in range(model.get_num_joints())]

    idx = list(range(0, m.num_frames, a.stride))
    allp = body_pos.reshape(-1, 3)
    for o in m.objects:
        allp = np.concatenate([allp, o.frames[:, :3]], 0)
    lo, hi = allp.min(0), allp.max(0)
    ctr = (lo + hi) / 2
    rad = max((hi - lo).max() / 2, 1.0) + 0.3

    fig = plt.figure(figsize=(8, 6), dpi=a.dpi)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.patch.set_facecolor("#0f1420")
    ax.set_facecolor("#0f1420")

    def draw(i):
        ax.cla()
        ax.set_facecolor("#0f1420")
        ax.set_xlim(ctr[0] - rad, ctr[0] + rad); ax.set_ylim(ctr[1] - rad, ctr[1] + rad); ax.set_zlim(0, 1.5 * rad)
        ax.set_box_aspect((1, 1, 0.75))
        ax.view_init(elev=18, azim=-55)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0.06, 0.08, 0.13, 1.0))
        ax.tick_params(colors="#96a0b6", labelsize=7)
        bp = body_pos[i]
        for j in range(1, len(parents)):
            p = parents[j]
            ax.plot([bp[p, 0], bp[j, 0]], [bp[p, 1], bp[j, 1]], [bp[p, 2], bp[j, 2]], color="#78aaff", lw=2.5)
        ax.scatter(bp[:, 0], bp[:, 1], bp[:, 2], color="#78aaff", s=10)
        for o in m.objects:
            held = o.contacts is not None and o.contacts[i].max() > 0.5
            col = "#ffcd5a" if held else "#96a0b6"
            w, edges = box_corners(o.frames[i, :3], o.frames[i, 3:6], a.box)
            for e0, e1 in edges:
                ax.plot([w[e0, 0], w[e1, 0]], [w[e0, 1], w[e1, 1]], [w[e0, 2], w[e1, 2]], color=col, lw=2)
        ax.set_title(f"{os.path.basename(a.file)}   t = {i / m.fps:5.2f} s", color="#eceff6", fontsize=10)

    anim_ = animation.FuncAnimation(fig, draw, frames=idx, interval=1000 * a.stride / m.fps)
    fps_out = max(1, int(round(m.fps / a.stride)))
    if a.out.endswith(".gif"):
        anim_.save(a.out, writer="pillow", fps=fps_out)
    else:
        anim_.save(a.out, writer="ffmpeg", fps=fps_out, savefig_kwargs={"facecolor": "#0f1420"})
    print("wrote", a.out)


if __name__ == "__main__":
    main()
