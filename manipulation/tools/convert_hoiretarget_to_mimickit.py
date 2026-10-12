"""Convert HOI-Retarget (HF: leggedrobotics/hoi-retarget) Unitree G1 clips to MimicKit motion format v2.

HOI-Retarget rows carry, at 30 fps, a G1 29-DoF robot-space human-object interaction retargeted from
OMOMO / ParaHome / NeuralDome / IMHD2 / CoRoleHOI, plus the object 6D pose, object velocity, per-link
contact flags (two palm pads, two ankles) and the contact points in the object frame. Licence: CC BY-NC-SA 4.0.

The joint order in `dof_names` is identical to MimicKit's `data/assets/g1/g1.xml`, but the converter maps
by name anyway. Quaternions in the parquet are wxyz; MimicKit uses xyzw and stores rotations as exp-maps.

    python manipulation/tools/convert_hoiretarget_to_mimickit.py \
        /data/user_data/vawale/MimicKit/manip_data/hoi_retarget/data/omomo/unitree_g1/sample_2rows.parquet \
        --out_dir manipulation/datasets/samples/hoi_retarget_g1 [--clip omomo/unitree_g1/sub10_clothesstand_000] [--video]
"""
import argparse
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "mimickit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

import util.torch_util as torch_util  # noqa: E402
from hoi_motion import CLAMP, HOIMotion, HOIObjectTrack  # noqa: E402

G1_XML = os.path.join(ROOT, "data/assets/g1/g1.xml")


def mimickit_joint_order(char_file=G1_XML):
    names = re.findall(r'<joint\s+name="([^"]+)"', open(char_file).read())
    return [n for n in names if n != "root"]


def wxyz_to_expmap(q_wxyz):
    q = np.asarray(q_wxyz, dtype=np.float32)
    q_xyzw = np.concatenate([q[:, 1:4], q[:, 0:1]], axis=1)
    return torch_util.quat_to_exp_map(torch.tensor(q_xyzw)).numpy()


def convert_row(row, joint_order):
    T = int(row["n_frames"])
    fps = float(row["fps"])
    dof_names = list(row["dof_names"])
    dof = np.array([np.asarray(x, dtype=np.float32) for x in row["dof_pos"]])          # [T, 29]
    idx = [dof_names.index(j) for j in joint_order]
    dof_mk = dof[:, idx]
    root_pos = np.array([np.asarray(x, dtype=np.float32) for x in row["root_pos"]])
    root_rot = wxyz_to_expmap(np.array([np.asarray(x, dtype=np.float32) for x in row["root_rot"]]))
    frames = np.concatenate([root_pos, root_rot, dof_mk], axis=1).astype(np.float32)   # [T, 35]

    obj_pos = np.array([np.asarray(x, dtype=np.float32) for x in row["object_pos"]])
    obj_rot = wxyz_to_expmap(np.array([np.asarray(x, dtype=np.float32) for x in row["object_rot"]]))
    obj_frames = np.concatenate([obj_pos, obj_rot], axis=1)
    sites = list(row["contact_link_names"])
    contacts = np.array([np.asarray(x, dtype=np.float32) for x in row["per_link_contact_flags"]]).reshape(T, len(sites))
    active = np.ones(T, dtype=np.float32)

    asset = "manipulation/datasets/samples/hoi_retarget_g1/" + str(row["object_model_path"])
    meta = {
        "source": "HOI-Retarget (huggingface.co/datasets/leggedrobotics/hoi-retarget), CC BY-NC-SA 4.0",
        "clip_id": row["clip_id"], "dataset": row["dataset"], "robot": row["robot"], "object": row["object"],
        "object_scale": float(row["object_scale"]), "object_model_path": row["object_model_path"],
        "qc_pass": bool(row["qc_pass"]), "contact_points_in_object_frame": "see parquet column "
        "fixed_contact_points_per_frame_in_object_frame (NaN = no contact)",
        "note": "root/object quaternions converted wxyz->xyzw->exp-map; joint order mapped by name to g1.xml",
    }
    m = HOIMotion(CLAMP, fps, frames, [HOIObjectTrack(str(row["object"]), asset, obj_frames, contacts, active)],
                  sites, meta)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet", nargs="+")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--clip", default=None, help="only this clip_id (substring match)")
    ap.add_argument("--max_rows", type=int, default=5)
    ap.add_argument("--video", action="store_true", help="also dump the embedded preview mp4")
    ap.add_argument("--char", default=G1_XML)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    joint_order = mimickit_joint_order(a.char)
    os.makedirs(a.out_dir, exist_ok=True)
    n_done = 0
    for f in a.parquet:
        t = pq.read_table(f)
        cols = [c for c in t.schema.names if c != "video"]
        for i in range(t.num_rows):
            if n_done >= a.max_rows:
                break
            row = t.slice(i, 1).select(cols).to_pylist()[0]
            if a.clip and a.clip not in row["clip_id"]:
                continue
            m = convert_row(row, joint_order)
            errs = m.validate()
            if errs:
                print("SKIP", row["clip_id"], errs)
                continue
            name = row["clip_id"].replace("/", "__")
            out = os.path.join(a.out_dir, name + ".pkl")
            m.save(out)
            print("==", out)
            print(m.summary())
            if a.video:
                v = t.slice(i, 1).select(["video"]).to_pylist()[0]["video"]
                if v and v.get("bytes"):
                    with open(os.path.join(a.out_dir, name + "_preview.mp4"), "wb") as vf:
                        vf.write(v["bytes"])
            n_done += 1
    print(f"converted {n_done} clips")


if __name__ == "__main__":
    main()
