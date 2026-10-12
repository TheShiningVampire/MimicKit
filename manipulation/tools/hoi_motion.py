"""MimicKit motion format v2: character clip + object tracks (human-object interaction).

The v1 MimicKit motion file is a pickle::

    {"loop_mode": int, "fps": float, "frames": [N, 6 + D]}
    frame = [root_pos(3), root_rot_expmap(3), joint_dofs(D)]       # Z-up world, metres, radians

v2 keeps those keys unchanged (so every existing MimicKit loader still works) and adds::

    "objects": [
        {"name": str,
         "asset": str,                 # path of the object asset (MJCF/URDF/OBJ) relative to the repo
         "frames": [N, 6],             # world pos(3) + rotation exp-map(3), same convention as the root
         "contacts": [N, S] | None,    # optional per contact-site label (0/1 or float)
         "active": [N] | None},        # optional 0/1: object is present / relevant at this frame
        ...],
    "contact_sites": [str, ...],       # body names of the character used as contact sites (S of them)
    "meta": {...}                      # free-form provenance (source dataset, subject, conversion args)

Rotation convention: exponential map (axis * angle), converted with MimicKit's
``util.torch_util.exp_map_to_quat`` at load time. Quaternions in this file are xyzw if ever present.

Run as a script to validate and summarize a file::

    python manipulation/tools/hoi_motion.py path/to/clip.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

CLAMP, WRAP = 0, 1


class HOIObjectTrack:
    def __init__(self, name, asset, frames, contacts=None, active=None):
        self.name = name
        self.asset = asset
        self.frames = np.asarray(frames, dtype=np.float32)
        self.contacts = None if contacts is None else np.asarray(contacts, dtype=np.float32)
        self.active = None if active is None else np.asarray(active, dtype=np.float32)

    def to_dict(self):
        return {
            "name": self.name,
            "asset": self.asset,
            "frames": self.frames.tolist(),
            "contacts": None if self.contacts is None else self.contacts.tolist(),
            "active": None if self.active is None else self.active.tolist(),
        }


class HOIMotion:
    def __init__(self, loop_mode, fps, frames, objects=(), contact_sites=(), meta=None):
        self.loop_mode = int(loop_mode)
        self.fps = float(fps)
        self.frames = np.asarray(frames, dtype=np.float32)
        self.objects = list(objects)
        self.contact_sites = list(contact_sites)
        self.meta = dict(meta or {})

    # --- basic queries -----------------------------------------------------------
    @property
    def num_frames(self):
        return self.frames.shape[0]

    @property
    def num_dofs(self):
        return self.frames.shape[1] - 6

    def get_length(self):
        return (self.num_frames - 1) / self.fps

    def root_pos(self):
        return self.frames[:, 0:3]

    def root_rot(self):
        return self.frames[:, 3:6]

    def joint_dofs(self):
        return self.frames[:, 6:]

    # --- io ------------------------------------------------------------------------
    def to_dict(self):
        d = {"loop_mode": self.loop_mode, "fps": self.fps, "frames": self.frames.tolist()}
        if self.objects:
            d["objects"] = [o.to_dict() for o in self.objects]
            d["contact_sites"] = list(self.contact_sites)
        if self.meta:
            d["meta"] = self.meta
        return d

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        objs = [HOIObjectTrack(o["name"], o.get("asset", ""), o["frames"], o.get("contacts"), o.get("active"))
                for o in d.get("objects", [])]
        return HOIMotion(d["loop_mode"], d["fps"], d["frames"], objs, d.get("contact_sites", []), d.get("meta"))

    # --- validation ----------------------------------------------------------------
    def validate(self):
        """Returns a list of problems (empty list == valid)."""
        errs = []
        n = self.num_frames
        if self.frames.ndim != 2 or self.frames.shape[1] < 6:
            errs.append(f"frames must be [N, 6+D], got {self.frames.shape}")
        if self.loop_mode not in (CLAMP, WRAP):
            errs.append(f"loop_mode must be 0 (clamp) or 1 (wrap), got {self.loop_mode}")
        if not (self.fps > 0):
            errs.append(f"fps must be positive, got {self.fps}")
        if not np.all(np.isfinite(self.frames)):
            errs.append("frames contain NaN/inf")
        s = len(self.contact_sites)
        names = set()
        for o in self.objects:
            if o.name in names:
                errs.append(f"duplicate object name {o.name}")
            names.add(o.name)
            if o.frames.shape != (n, 6):
                errs.append(f"object {o.name}: frames must be [{n}, 6], got {o.frames.shape}")
            if o.contacts is not None and o.contacts.shape != (n, s):
                errs.append(f"object {o.name}: contacts must be [{n}, {s}], got {o.contacts.shape}")
            if o.active is not None and o.active.shape != (n,):
                errs.append(f"object {o.name}: active must be [{n}], got {o.active.shape}")
            if not np.all(np.isfinite(o.frames)):
                errs.append(f"object {o.name}: frames contain NaN/inf")
        if self.loop_mode == WRAP and self.objects:
            for o in self.objects:
                drift = np.linalg.norm(o.frames[-1, :3] - o.frames[0, :3])
                if drift > 0.05:
                    errs.append(f"object {o.name}: WRAP clip but object drifts {drift:.2f} m over the loop "
                                "(MimicKit applies the root xy loop offset; object offset must match)")
        return errs

    def summary(self):
        lines = [
            f"frames: {self.num_frames}  dofs: {self.num_dofs}  fps: {self.fps:g}  length: {self.get_length():.2f} s  "
            f"loop: {'wrap' if self.loop_mode == WRAP else 'clamp'}",
            f"root xy travel: {np.linalg.norm(self.root_pos()[-1, :2] - self.root_pos()[0, :2]):.2f} m   "
            f"root height: {self.root_pos()[:, 2].min():.2f}..{self.root_pos()[:, 2].max():.2f} m",
        ]
        for o in self.objects:
            p = o.frames[:, :3]
            v = np.linalg.norm(np.diff(p, axis=0), axis=1) * self.fps if self.num_frames > 1 else np.zeros(1)
            ln = (f"object '{o.name}' asset={o.asset or '-'}  pos range x[{p[:,0].min():.2f},{p[:,0].max():.2f}] "
                  f"y[{p[:,1].min():.2f},{p[:,1].max():.2f}] z[{p[:,2].min():.2f},{p[:,2].max():.2f}]  "
                  f"max speed {v.max():.2f} m/s")
            if o.contacts is not None:
                frac = o.contacts.mean(axis=0)
                ln += "  contact fraction per site: " + ", ".join(f"{s}={f:.2f}" for s, f in zip(self.contact_sites, frac))
            if o.active is not None:
                ln += f"  active {o.active.mean():.2f}"
            lines.append(ln)
        if self.meta:
            lines.append("meta: " + ", ".join(f"{k}={v}" for k, v in self.meta.items()))
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    bad = 0
    for f in a.files:
        m = HOIMotion.load(f)
        errs = m.validate()
        print(f"== {f}")
        print(m.summary())
        for e in errs:
            print("  ERROR:", e)
        bad += bool(errs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
