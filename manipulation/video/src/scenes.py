"""Manim scenes for "From mimicking motion to mimicking manipulation".

Render (from this directory, with the manim env):
    manim -qm --disable_caching -a scenes.py          # all scenes, 720p30
Each scene reads its narration timing from timing.json produced by gen_audio.py and
embeds the matching wav files with Scene.add_sound, so every rendered clip carries
its own narration.  Scenes are concatenated afterwards by build.sh.
"""
import json
import os

from manim import *

AUDIO = os.environ.get("NARRATION_DIR", "/data/user_data/vawale/MimicKit/manip_video/audio")
TIMING = json.load(open(os.path.join(AUDIO, "timing.json")))

config.background_color = "#0f1420"

BLUE_ = "#78aaff"
GREEN_ = "#96e146"
YELLOW_ = "#ffcd5a"
RED_ = "#ff6e64"
GREY_ = "#96a0b6"
WHITE_ = "#eceff6"
PANEL = "#1c2438"
FONT = "DejaVu Sans"

PAD = 0.45   # silence after each sentence


def T(s, size=30, color=WHITE_, weight=NORMAL, **kw):
    return Text(s, font=FONT, font_size=size, color=color, weight=weight, **kw)


class Narrated(Scene):
    """A scene whose timeline is driven by narration audio.

    say() starts the next sentence's audio and returns its duration.  Animations
    played afterwards count against that duration; rest() waits for whatever is left.
    """
    name = None

    def setup(self):
        self.timing = TIMING[self.name]
        self.idx = 0
        self.t_say = 0.0

    def now(self):
        return self.renderer.time

    def say(self):
        dur = self.timing[self.idx]
        self.add_sound(os.path.join(AUDIO, f"{self.name}_{self.idx:02d}.wav"))
        self.idx += 1
        self.t_say = self.now()
        self.dur = dur
        return dur

    def rest(self, pad=PAD):
        left = self.dur - (self.now() - self.t_say)
        self.wait(max(left, 0.0) + pad)

    def caption(self, text, size=26):
        c = T(text, size=size, color=GREY_).to_edge(DOWN, buff=0.35)
        return c


# --------------------------------------------------------------------------- drawing

def stickman(color=BLUE_, scale=1.0, stroke=6):
    """Simple upright humanoid as lines; returns VGroup centred at the hip."""
    p = {
        "head": [0, 1.65], "neck": [0, 1.3], "hip": [0, 0.0],
        "ls": [-0.35, 1.25], "rs": [0.35, 1.25],
        "le": [-0.6, 0.65], "re": [0.6, 0.65],
        "lh": [-0.55, 0.05], "rh": [0.55, 0.05],
        "lk": [-0.25, -0.7], "rk": [0.25, -0.7],
        "lf": [-0.3, -1.4], "rf": [0.3, -1.4],
    }
    P = {k: np.array([v[0] * scale, v[1] * scale, 0]) for k, v in p.items()}
    segs = [("neck", "hip"), ("ls", "rs"), ("ls", "le"), ("le", "lh"), ("rs", "re"), ("re", "rh"),
            ("hip", "lk"), ("lk", "lf"), ("hip", "rk"), ("rk", "rf")]
    g = VGroup(*[Line(P[a], P[b], color=color, stroke_width=stroke) for a, b in segs])
    g.add(Circle(radius=0.22 * scale, color=color, stroke_width=stroke).move_to(P["head"]))
    for k in ["ls", "rs", "le", "re", "lh", "rh", "lk", "rk", "hip", "neck"]:
        g.add(Dot(P[k], radius=0.05 * scale, color=color))
    return g


def cup(color=YELLOW_, scale=1.0):
    body = RoundedRectangle(corner_radius=0.08 * scale, width=0.6 * scale, height=0.7 * scale,
                            color=color, fill_color=color, fill_opacity=0.25, stroke_width=4)
    handle = Arc(radius=0.18 * scale, start_angle=-PI / 2, angle=PI, color=color, stroke_width=4)
    handle.move_to(body.get_right() + RIGHT * 0.12 * scale)
    return VGroup(body, handle)


def hand(color=BLUE_, scale=1.0, curl=0.0, stroke=5):
    """Schematic hand: palm + 5 fingers. curl in [0,1] bends the fingers inward."""
    palm = RoundedRectangle(corner_radius=0.12 * scale, width=0.9 * scale, height=0.8 * scale,
                            color=color, stroke_width=stroke, fill_color=color, fill_opacity=0.12)
    g = VGroup(palm)
    base_y = 0.4 * scale
    xs = [-0.32, -0.11, 0.1, 0.31]
    lens = [0.55, 0.65, 0.6, 0.5]
    for x, L in zip(xs, lens):
        L *= scale
        p0 = np.array([x * scale, base_y, 0])
        a1 = PI / 2 - curl * 1.1
        a2 = a1 - curl * 1.2
        p1 = p0 + L * 0.55 * np.array([np.cos(a1), np.sin(a1), 0])
        p2 = p1 + L * 0.45 * np.array([np.cos(a2), np.sin(a2), 0])
        g.add(Line(p0, p1, color=color, stroke_width=stroke), Line(p1, p2, color=color, stroke_width=stroke),
              Dot(p1, radius=0.04 * scale, color=color), Dot(p2, radius=0.05 * scale, color=color))
    # thumb
    p0 = np.array([-0.45 * scale, -0.05 * scale, 0])
    a1 = PI * 0.8 - curl * 0.9
    a2 = a1 - curl * 1.3
    p1 = p0 + 0.35 * scale * np.array([np.cos(a1), np.sin(a1), 0])
    p2 = p1 + 0.3 * scale * np.array([np.cos(a2), np.sin(a2), 0])
    g.add(Line(p0, p1, color=color, stroke_width=stroke), Line(p1, p2, color=color, stroke_width=stroke),
          Dot(p1, radius=0.04 * scale, color=color), Dot(p2, radius=0.05 * scale, color=color))
    return g


def panel(w, h, title=None, color=GREY_):
    r = RoundedRectangle(corner_radius=0.2, width=w, height=h, color=color, stroke_width=2,
                         fill_color=PANEL, fill_opacity=0.9)
    g = VGroup(r)
    if title:
        g.add(T(title, size=24, color=color, weight=BOLD).next_to(r.get_top(), DOWN, buff=0.2))
    return g


def box(text, color=BLUE_, w=2.6, h=0.9, size=24):
    r = RoundedRectangle(corner_radius=0.15, width=w, height=h, color=color, stroke_width=3,
                         fill_color=PANEL, fill_opacity=1.0)
    t = T(text, size=size).move_to(r)
    return VGroup(r, t)


def reward_bar(label, color, width=3.0):
    track = Rectangle(width=width, height=0.35, color=GREY_, stroke_width=2)
    fill = Rectangle(width=0.01, height=0.35, color=color, fill_color=color, fill_opacity=0.9, stroke_width=0)
    fill.align_to(track, LEFT)
    lab = T(label, size=24).next_to(track, LEFT, buff=0.3)
    return VGroup(track, fill, lab)


def set_bar(bar, frac):
    track, fill = bar[0], bar[1]
    w = max(0.01, frac * track.width)
    fill.stretch_to_fit_width(w)
    fill.align_to(track, LEFT)
    return bar


# --------------------------------------------------------------------------- scenes

class S01_Title(Narrated):
    name = "S01_Title"

    def construct(self):
        title = T("From mimicking motion", size=48, weight=BOLD)
        title2 = T("to mimicking manipulation", size=48, weight=BOLD, color=YELLOW_)
        tg = VGroup(title, title2).arrange(DOWN, buff=0.25).to_edge(UP, buff=0.8)
        man = stickman(BLUE_, scale=0.9).shift(DOWN * 0.8 + LEFT * 2.5)
        floor = Line(LEFT * 6, RIGHT * 6, color=GREY_, stroke_width=2).shift(DOWN * 2.1)

        self.say()
        self.play(Write(tg), run_time=2.0)
        self.play(Create(floor), FadeIn(man, shift=UP), run_time=1.2)
        self.rest()

        self.say()
        self.play(Rotate(man, angle=TAU, about_point=man.get_center() + UP * 0.5), run_time=2.2)
        self.rest()

        self.say()
        q = T("?", size=90, color=YELLOW_).shift(RIGHT * 2.5 + DOWN * 0.4)
        self.play(FadeIn(q, scale=0.5), run_time=0.8)
        self.rest()

        self.say()
        c = cup(YELLOW_, 1.3).move_to(q)
        self.play(ReplacementTransform(q, c), run_time=1.0)
        self.play(man.animate.shift(RIGHT * 1.0), run_time=1.0)
        self.rest()
        self.play(FadeOut(VGroup(tg, man, floor, c)), run_time=0.6)


class S02_Recap(Narrated):
    name = "S02_Recap"

    def construct(self):
        head = T("The MimicKit loop (whole body)", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.2)
        self.rest()

        # reference ghost + simulated body
        floor = Line(LEFT * 1.0, RIGHT * 6.5, color=GREY_, stroke_width=2).shift(DOWN * 2.6)
        ref = stickman(GREEN_, 0.85).shift(RIGHT * 2.2 + DOWN * 1.2)
        ref.set_opacity(0.55)
        lab_ref = T("reference q̂(t)", size=22, color=GREEN_).next_to(ref, LEFT, buff=0.25)
        self.say()
        self.play(Create(floor), FadeIn(ref), Write(lab_ref), run_time=1.5)
        self.rest()

        sim = stickman(BLUE_, 0.85).shift(RIGHT * 4.2 + DOWN * 1.2)
        lab_sim = T("simulated q(t)", size=22, color=BLUE_).next_to(sim, RIGHT, buff=0.25)
        simg = VGroup(sim, lab_sim)
        pol = box("policy π(a | s)", BLUE_, w=3.0).shift(LEFT * 4.0 + UP * 1.2)
        simbox = box("physics simulator", GREY_, w=3.0).shift(LEFT * 4.0 + DOWN * 1.2)
        a1 = Arrow(pol.get_bottom(), simbox.get_top(), buff=0.1, color=YELLOW_)
        a1l = T("joint torques", size=20, color=YELLOW_).next_to(a1, RIGHT, buff=0.1)
        a2 = CurvedArrow(simbox.get_left() + LEFT * 0.05, pol.get_left() + LEFT * 0.05, angle=-PI / 1.6, color=GREY_)
        a2l = T("state s", size=20, color=GREY_).move_to(a2.point_from_proportion(0.5) + RIGHT * 0.75)
        self.say()
        self.play(FadeIn(sim), Write(lab_sim), run_time=1.0)
        self.play(FadeIn(pol), FadeIn(simbox), GrowArrow(a1), Write(a1l), run_time=1.4)
        self.play(Create(a2), Write(a2l), run_time=1.0)
        self.rest()

        # reward
        rew = T("r = exp( −‖q − q̂‖² )", size=30, color=YELLOW_).shift(RIGHT * 3.2 + UP * 2.3)
        bar = reward_bar("r", YELLOW_, 2.6).next_to(rew, DOWN, buff=0.25)
        self.say()
        self.play(Write(rew), run_time=1.2)
        self.play(FadeIn(bar), run_time=0.5)
        # body drifts toward reference, bar rises
        self.play(simg.animate.shift(LEFT * 1.0), UpdateFromAlphaFunc(bar, lambda m, a: set_bar(m, 0.2 + 0.6 * a)),
                  run_time=1.8)
        self.rest()

        self.say()
        rl = T("RL: ∇ E[ Σ r ]", size=28, color=YELLOW_).next_to(pol, UP, buff=0.3)
        self.play(Write(rl), Indicate(pol, color=YELLOW_), run_time=1.5)
        self.rest()

        self.say()
        self.play(simg.animate.shift(LEFT * 0.95), UpdateFromAlphaFunc(bar, lambda m, a: set_bar(m, 0.8 + 0.2 * a)),
                  run_time=1.5)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S03_WhatChanges(Narrated):
    name = "S03_WhatChanges"

    def construct(self):
        head = T("Difference 1: the goal is the object, not the pose", size=34, weight=BOLD).to_edge(UP, buff=0.4)
        table = Line(LEFT * 5.5, RIGHT * 5.5, color=GREY_, stroke_width=3).shift(DOWN * 2.2)
        shelf = Line(RIGHT * 2.5, RIGHT * 5.5, color=GREY_, stroke_width=3).shift(UP * 0.6)
        shelf_l = T("shelf", size=20, color=GREY_).next_to(shelf, UP, buff=0.1)
        c = cup(YELLOW_, 1.1).next_to(table, UP, buff=0).shift(LEFT * 1.5)
        self.say()
        self.play(Write(head), Create(table), Create(shelf), Write(shelf_l), run_time=1.2)
        self.play(FadeIn(c, shift=DOWN), run_time=0.8)
        self.rest()

        ref = hand(GREEN_, 0.9, curl=0.55).next_to(c, LEFT, buff=0.05).shift(UP * 0.1)
        ref.set_opacity(0.55)
        sim = hand(BLUE_, 0.9, curl=0.55).move_to(ref).shift(UP * 0.35 + LEFT * 0.25)
        self.say()
        self.play(FadeIn(ref), FadeIn(sim), run_time=1.0)
        self.play(Indicate(sim, color=BLUE_, scale_factor=1.05), run_time=1.0)
        self.rest()

        self.say()
        self.play(Indicate(c, color=YELLOW_, scale_factor=1.3), run_time=1.2)
        self.rest()

        # case A: hand a bit off, cup reaches shelf
        self.say()
        hc = VGroup(sim, c)
        target = shelf.get_center() + UP * 0.45 + LEFT * 0.5
        self.play(hc.animate.move_to(target), run_time=1.8)
        ok = T("✓", size=60, color=GREEN_).next_to(shelf, RIGHT, buff=0.3)
        self.play(FadeIn(ok, scale=0.5), run_time=0.5)
        self.rest()

        # case B: hand perfect, cup falls
        self.say()
        self.play(FadeOut(ok), hc.animate.move_to(ref).shift(RIGHT * 0.45), run_time=1.2)
        self.play(sim.animate.move_to(ref), run_time=0.6)
        self.play(c.animate.shift(DOWN * 1.6).rotate(-PI / 3), rate_func=rush_into, run_time=0.8)
        bad = T("✗", size=60, color=RED_).next_to(c, RIGHT, buff=0.3)
        self.play(FadeIn(bad, scale=0.5), run_time=0.4)
        self.rest()

        self.say()
        r_old = T("r = exp(−‖q − q̂‖²)", size=28, color=GREY_).shift(RIGHT * 2.3 + UP * 2.0)
        r_new = T("r = r_object · r_hand · r_contact", size=28, color=YELLOW_).move_to(r_old)
        self.play(Write(r_old), run_time=1.0)
        self.play(ReplacementTransform(r_old, r_new), run_time=1.2)
        self.play(Indicate(r_new[2:10], color=YELLOW_), run_time=1.0)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S04_Contacts(Narrated):
    name = "S04_Contacts"

    def construct(self):
        head = T("Difference 2: contact", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        self.rest()

        # left: walking, feet on floor, periodic timeline
        left = panel(6.0, 4.6, "locomotion").shift(LEFT * 3.3 + DOWN * 0.4)
        floor = Line(LEFT * 2.4, RIGHT * 2.4, color=GREY_, stroke_width=3).move_to(left[0].get_center() + DOWN * 0.3)
        man = stickman(BLUE_, 0.6).next_to(floor, UP, buff=0).shift(LEFT * 0.8)
        axis = Line(LEFT * 2.4, RIGHT * 2.4, color=GREY_, stroke_width=2).move_to(left[0].get_center() + DOWN * 1.6)
        tlab = T("time", size=18, color=GREY_).next_to(axis, RIGHT, buff=0.1)
        marks = VGroup(*[Rectangle(width=0.7, height=0.25, color=BLUE_, fill_color=BLUE_, fill_opacity=0.7, stroke_width=0)
                         .move_to(axis.get_left() + RIGHT * (0.5 + i * 1.1) + UP * 0.15) for i in range(4)])
        self.say()
        self.play(FadeIn(left), Create(floor), FadeIn(man), run_time=1.0)
        self.play(Create(axis), Write(tlab), LaggedStart(*[FadeIn(m) for m in marks], lag_ratio=0.3), run_time=1.6)
        self.rest()

        # right: grasp, 5 contacts staggered
        right = panel(6.0, 4.6, "grasp").shift(RIGHT * 3.3 + DOWN * 0.4)
        ball = Circle(radius=0.55, color=YELLOW_, fill_color=YELLOW_, fill_opacity=0.25, stroke_width=4)
        ball.move_to(right[0].get_center() + UP * 0.2)
        angs = [200, 150, 100, 50, 10]
        tips = VGroup(*[Dot(ball.get_center() + 0.55 * np.array([np.cos(np.radians(a)), np.sin(np.radians(a)), 0]),
                            radius=0.08, color=RED_) for a in angs])
        h = hand(BLUE_, 0.75, curl=0.65).move_to(ball.get_center() + LEFT * 0.75 + DOWN * 0.1)
        axis2 = Line(LEFT * 2.4, RIGHT * 2.4, color=GREY_, stroke_width=2).move_to(right[0].get_center() + DOWN * 1.6)
        offs = [0.3, 0.7, 0.55, 0.95, 1.2]
        marks2 = VGroup(*[Rectangle(width=2.6 - o, height=0.12, color=RED_, fill_color=RED_, fill_opacity=0.8, stroke_width=0)
                          .align_to(axis2.get_left() + RIGHT * (0.3 + o), LEFT).shift(UP * (0.12 + 0.17 * i))
                          for i, o in enumerate(offs)])
        for i, m in enumerate(marks2):
            m.align_to(axis2.get_left() + RIGHT * (0.3 + offs[i]), LEFT)
            m.set_y(axis2.get_y() + 0.15 + 0.17 * i)
        self.say()
        self.play(FadeIn(right), FadeIn(ball), FadeIn(h), run_time=1.0)
        self.play(LaggedStart(*[FadeIn(t, scale=2) for t in tips], lag_ratio=0.25), run_time=1.2)
        self.play(Create(axis2), LaggedStart(*[FadeIn(m) for m in marks2], lag_ratio=0.2), run_time=1.5)
        self.rest()

        # 1 mm comparison
        self.say()
        mm1 = T("1 mm error in a step", size=24, color=GREY_).move_to(left[0].get_bottom() + UP * 0.35)
        self.play(Write(mm1), run_time=1.0)
        self.rest()
        self.say()
        mm2 = T("1 mm error in a pinch", size=24, color=RED_).move_to(right[0].get_bottom() + UP * 0.35)
        self.play(Write(mm2), run_time=1.0)
        self.play(Wiggle(tips[0]), Wiggle(tips[-1]), run_time=1.2)
        self.rest()

        self.say()
        fine = T("finer time step · finer meshes · finer rewards", size=28, color=YELLOW_).to_edge(DOWN, buff=0.35)
        self.play(Write(fine), run_time=1.5)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S05_Observation(Narrated):
    name = "S05_Observation"

    def construct(self):
        head = T("Difference 3: what the policy sees", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        self.rest()

        pol = box("policy π", BLUE_, w=2.4, h=1.2, size=30).shift(RIGHT * 4.0)
        obs = VGroup(
            T("joint angles  q", size=26, color=WHITE_),
            T("joint velocities  q̇", size=26, color=WHITE_),
            T("heading / goal direction", size=26, color=WHITE_),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).shift(LEFT * 3.0 + UP * 1.2)
        arrows = VGroup(*[Arrow(o.get_right(), pol.get_left(), buff=0.15, color=GREY_, stroke_width=3) for o in obs])
        self.say()
        self.play(FadeIn(pol), run_time=0.6)
        self.play(LaggedStart(*[AnimationGroup(Write(o), GrowArrow(a)) for o, a in zip(obs, arrows)], lag_ratio=0.3),
                  run_time=2.2)
        self.rest()

        obj = VGroup(
            T("object position  p_obj − p_hand", size=26, color=YELLOW_),
            T("object rotation  R_obj", size=26, color=YELLOW_),
            T("object velocity  v_obj, ω_obj", size=26, color=YELLOW_),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(obs, DOWN, buff=0.4, aligned_edge=LEFT)
        arrows2 = VGroup(*[Arrow(o.get_right(), pol.get_left(), buff=0.15, color=YELLOW_, stroke_width=3) for o in obj])
        c = cup(YELLOW_, 0.8).next_to(pol, UP, buff=0.4)
        self.say()
        self.play(FadeIn(c, shift=DOWN), run_time=0.6)
        self.play(LaggedStart(*[AnimationGroup(Write(o), GrowArrow(a)) for o, a in zip(obj, arrows2)], lag_ratio=0.3),
                  run_time=2.5)
        self.rest()

        fut = T("future reference  q̂(t+1 … t+k),  p̂_obj(t+1 … t+k)", size=26, color=GREEN_)
        fut.next_to(obj, DOWN, buff=0.4, aligned_edge=LEFT)
        a3 = Arrow(fut.get_right(), pol.get_left(), buff=0.15, color=GREEN_, stroke_width=3)
        ghosts = VGroup(*[hand(GREEN_, 0.45, curl=0.15 + 0.2 * i).set_opacity(0.6 - 0.12 * i)
                          .next_to(pol, DOWN, buff=0.3).shift(RIGHT * (0.55 * i - 0.8)) for i in range(4)])
        self.say()
        self.play(Write(fut), GrowArrow(a3), run_time=1.5)
        self.play(LaggedStart(*[FadeIn(g, shift=LEFT * 0.3) for g in ghosts], lag_ratio=0.3), run_time=1.5)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S06_Data(Narrated):
    name = "S06_Data"

    def construct(self):
        head = T("Difference 4: the data", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        self.rest()

        # left: body mocap
        left = panel(5.6, 4.4, "whole-body mocap").shift(LEFT * 3.4 + DOWN * 0.5)
        man = stickman(GREEN_, 0.75).move_to(left[0].get_center() + DOWN * 0.1)
        cams = VGroup(*[Square(0.22, color=GREY_).move_to(left[0].get_center() + 2.0 * np.array([np.cos(a), np.sin(a) * 0.8, 0]))
                        for a in np.linspace(0, TAU, 8, endpoint=False)])
        clips = T("hundreds of clean clips", size=22, color=GREEN_).move_to(left[0].get_bottom() + UP * 0.35)
        self.say()
        self.play(FadeIn(left), FadeIn(man), LaggedStart(*[FadeIn(c) for c in cams], lag_ratio=0.1), run_time=1.5)
        self.play(Write(clips), run_time=1.0)
        self.rest()

        # right: hand + object with occlusion
        right = panel(5.6, 4.4, "hand + object capture").shift(RIGHT * 3.4 + DOWN * 0.5)
        c = cup(YELLOW_, 1.1).move_to(right[0].get_center() + RIGHT * 0.3)
        h = hand(BLUE_, 0.85, curl=0.6).move_to(c.get_center() + LEFT * 0.75)
        occl = Rectangle(width=1.2, height=1.1, color=PANEL, fill_color=PANEL, fill_opacity=0.85, stroke_width=0).move_to(h).shift(RIGHT * 0.5)
        occl_t = T("occluded", size=20, color=RED_).move_to(occl)
        self.say()
        self.play(FadeIn(right), FadeIn(c), FadeIn(h), run_time=1.0)
        self.play(FadeIn(occl), Write(occl_t), run_time=1.0)
        self.rest()

        self.say()
        self.play(FadeOut(occl), FadeOut(occl_t), run_time=0.5)
        rig = VGroup(*[Square(0.2, color=GREY_).move_to(right[0].get_center() + 1.9 * np.array([np.cos(a), np.sin(a) * 0.8, 0]))
                       for a in np.linspace(0, TAU, 10, endpoint=False)])
        glove = T("gloves · markers · many cameras", size=20, color=GREY_).move_to(right[0].get_bottom() + UP * 0.35)
        pose6 = T("+ 6D object pose", size=22, color=YELLOW_).next_to(c, UP, buff=0.25)
        self.play(LaggedStart(*[FadeIn(r) for r in rig], lag_ratio=0.08), Write(glove), run_time=1.5)
        self.play(Write(pose6), run_time=0.8)
        self.rest()

        # dataset list
        self.say()
        self.play(FadeOut(Group(left, man, cams, clips)), run_time=0.5)
        names = VGroup(*[T(n, size=28, color=YELLOW_ if i < 2 else WHITE_) for i, n in
                         enumerate(["GRAB · body + hands + object", "ARCTIC · bimanual, articulated",
                                    "OakInk2 · long-horizon tasks", "HOI4D · egocentric RGB-D", "EgoDex · 829 h egocentric"])])
        names.arrange(DOWN, aligned_edge=LEFT, buff=0.28).to_edge(LEFT, buff=0.6).shift(DOWN * 0.5)
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.3) for n in names], lag_ratio=0.2), run_time=2.5)
        self.rest()

        self.say()
        per = T("per frame:  body/hand pose  +  object (p, R)", size=26, color=GREEN_).to_edge(DOWN, buff=0.35)
        self.play(Write(per), run_time=1.5)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S07_Retarget(Narrated):
    name = "S07_Retarget"

    def construct(self):
        head = T("Retargeting: human hand → robot hand", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        hh = hand(GREEN_, 1.1, curl=0.45).shift(LEFT * 3.8 + DOWN * 0.3)
        hl = T("human (21 keypoints)", size=22, color=GREEN_).next_to(hh, DOWN, buff=0.4)
        rh = hand(BLUE_, 1.1, curl=0.2).shift(RIGHT * 3.8 + DOWN * 0.3)
        rh[0].become(Rectangle(width=1.0, height=0.95, color=BLUE_, stroke_width=5, fill_color=BLUE_, fill_opacity=0.12).move_to(rh[0]))
        rl = T("robot (16 DOF)", size=22, color=BLUE_).next_to(rh, DOWN, buff=0.4)
        self.play(FadeIn(hh), Write(hl), FadeIn(rh), Write(rl), run_time=1.5)
        self.rest()

        self.say()
        arr = Arrow(hh.get_right(), rh.get_left(), buff=0.4, color=YELLOW_)
        al = T("retarget", size=24, color=YELLOW_).next_to(arr, UP, buff=0.1)
        self.play(GrowArrow(arr), Write(al), run_time=1.2)
        self.rest()

        self.say()
        # fingertip correspondences
        tips_h = [m for m in hh if isinstance(m, Dot) and m.radius > 0.045]
        tips_r = [m for m in rh if isinstance(m, Dot) and m.radius > 0.045]
        corr = VGroup(*[DashedLine(a.get_center(), b.get_center(), color=YELLOW_, stroke_width=2, dash_length=0.08)
                        for a, b in zip(tips_h, tips_r)])
        cost = T("min_θ  Σ_tips ‖ p_tip^robot(θ) − p_tip^human ‖²", size=28, color=YELLOW_).shift(UP * 1.9)
        self.play(FadeOut(arr), FadeOut(al), Create(corr), run_time=1.0)
        self.play(Write(cost), run_time=1.5)
        rh2 = hand(BLUE_, 1.1, curl=0.45).move_to(rh)
        rh2[0].become(rh[0].copy())
        self.play(Transform(rh, rh2), run_time=1.8)
        self.rest()

        self.say()
        c = cup(YELLOW_, 0.9).move_to(ORIGIN + DOWN * 0.3)
        anchor = T("object = anchor frame", size=24, color=YELLOW_).next_to(c, DOWN, buff=0.3)
        self.play(FadeOut(corr), FadeIn(c), Write(anchor), run_time=1.2)
        self.rest()

        self.say()
        self.play(Indicate(c, color=YELLOW_, scale_factor=1.2), rh.animate.shift(LEFT * 0.6), hh.animate.shift(RIGHT * 0.6), run_time=1.6)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S08_Pipeline(Narrated):
    name = "S08_Pipeline"

    def construct(self):
        head = T("The recipe", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        self.rest()

        steps = [
            ("1  human HOI dataset", GREEN_),
            ("2  retarget + motion file\n     (body, hand, object)", GREEN_),
            ("3  HOI environment\n  spawn · see · reward", YELLOW_),
            ("4  RL tracking (PPO)", BLUE_),
            ("5  distill → deploy", BLUE_),
        ]
        boxes = VGroup(*[box(t, c, w=3.6 if i != 2 else 4.0, h=1.1, size=20) for i, (t, c) in enumerate(steps)])
        boxes[0].move_to(LEFT * 4.7 + UP * 1.4)
        boxes[1].move_to(LEFT * 4.7 + DOWN * 0.6)
        boxes[2].move_to(ORIGIN + UP * 0.4)
        boxes[3].move_to(RIGHT * 4.7 + UP * 1.4)
        boxes[4].move_to(RIGHT * 4.7 + DOWN * 0.6)
        arrows = [Arrow(boxes[0].get_bottom(), boxes[1].get_top(), buff=0.1, color=GREY_),
                  Arrow(boxes[1].get_right(), boxes[2].get_left(), buff=0.1, color=GREY_),
                  Arrow(boxes[2].get_right(), boxes[3].get_left(), buff=0.1, color=GREY_),
                  Arrow(boxes[3].get_bottom(), boxes[4].get_top(), buff=0.1, color=GREY_)]
        for i in range(5):
            self.say()
            anims = [FadeIn(boxes[i], shift=UP * 0.2)]
            if i > 0:
                anims.append(GrowArrow(arrows[i - 1]))
            self.play(*anims, run_time=1.0)
            if i == 2:
                self.play(Indicate(boxes[2], color=YELLOW_), run_time=1.0)
            self.rest()
        loop = CurvedArrow(boxes[3].get_top() + UP * 0.05, boxes[2].get_top() + UP * 0.05, angle=PI / 2.2, color=BLUE_)
        self.play(Create(loop), run_time=0.8)
        self.wait(0.4)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S09_Plan(Narrated):
    name = "S09_Plan"

    def construct(self):
        head = T("The plan", size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.say()
        self.play(Write(head), run_time=1.0)
        line = Line(LEFT * 5.5, RIGHT * 5.5, color=GREY_, stroke_width=3).shift(DOWN * 0.2)
        self.play(Create(line), run_time=0.8)
        self.rest()

        phases = [
            ("Phase 0", "reproduce\nexisting method", "its demo clip\ntracks in sim", GREY_),
            ("Phase 1", "objects in\nMimicKit", "one GRAB clip\nsuccess > 90%", GREEN_),
            ("Phase 2", "dataset +\nrobot hand", "100 clips,\none policy", YELLOW_),
            ("Phase 3", "real robot", "pick-and-place\non hardware", BLUE_),
        ]
        xs = [-4.1, -1.4, 1.4, 4.1]
        for (name, what, test, col), x in zip(phases, xs):
            self.say()
            dot = Dot(np.array([x, -0.2, 0]), radius=0.12, color=col)
            n = T(name, size=24, color=col, weight=BOLD).next_to(dot, UP, buff=0.3)
            w = T(what, size=22, color=WHITE_, line_spacing=0.9).next_to(n, UP, buff=0.2)
            t = T(test, size=20, color=col, line_spacing=0.9).next_to(dot, DOWN, buff=0.35)
            self.play(FadeIn(dot, scale=0.5), Write(n), run_time=0.7)
            self.play(FadeIn(w, shift=DOWN * 0.2), FadeIn(t, shift=UP * 0.2), run_time=0.9)
            self.rest()

        self.say()
        tl = T("test ↑ above the line: what it builds · below: how we know it works", size=22, color=GREY_).to_edge(DOWN, buff=0.35)
        self.play(Write(tl), run_time=1.5)
        self.rest()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.6)


class S10_Close(Narrated):
    name = "S10_Close"

    def construct(self):
        man = stickman(BLUE_, 0.9).shift(LEFT * 3.0 + DOWN * 0.5)
        self.say()
        self.play(FadeIn(man), run_time=0.8)
        self.play(Rotate(man, angle=TAU, about_point=man.get_center() + UP * 0.5), run_time=2.0)
        self.rest()

        self.say()
        h = hand(BLUE_, 1.0, curl=0.5).shift(RIGHT * 2.3 + DOWN * 0.5)
        c = cup(YELLOW_, 1.1).next_to(h, RIGHT, buff=0.05)
        self.play(FadeIn(h), FadeIn(c), run_time=1.0)
        self.play(VGroup(h, c).animate.shift(UP * 1.0), run_time=1.2)
        self.rest()

        self.say()
        line = T("same loop · new reward · much finer contact", size=36, color=YELLOW_, weight=BOLD).to_edge(UP, buff=1.0)
        self.play(Write(line), run_time=1.8)
        self.rest()

        self.say()
        repo = T("manipulation/  in the MimicKit fork", size=28, color=GREY_).to_edge(DOWN, buff=0.6)
        self.play(Write(repo), run_time=1.5)
        self.rest()
        self.wait(0.8)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
