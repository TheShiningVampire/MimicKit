# Deep research: physics-based human-object interaction imitation with RL (2023 to 2026)

Generated 2026-10-12 by a fan-out research workflow: 5 search angles, top sources fetched, every extracted claim put to a 3-vote adversarial verification (2 of 3 refutes kill a claim), then synthesized. Source search and claim extraction ran on Claude Fable 5.1, verification votes on Sonnet, synthesis on Opus. Confidence labels come from vote unanimity and source quality. Companion to reports 01 to 05; see the comparison note at the end.

## Question

Physics-based human-object interaction (HOI) imitation with reinforcement learning: simulated humanoids or robot hands tracking human mocap/video clips that include objects (PhysHOI, SkillMimic, InterMimic, Omnigrasp, ManipTrans, DexMachina, Dexplore, HDMI, ResMimic, 2023-2026). Focus on: (1) reward and observation design (object-centric terms, contact-graph rewards, multiplicative vs additive composition, future reference windows); (2) handling noisy or sparse reference data (teacher correction, physics-based retargeting, state-initialization and trajectory-sampling tricks); (3) curricula such as virtual object controllers and residual policies; (4) simulators, training cost, and reported success rates; (5) sim-to-real results on humanoids (Unitree G1) and dexterous hands, and the main open problems. Context: the findings will guide extending the MimicKit motion-imitation framework (DeepMimic/AMP in Isaac Gym / Isaac Lab / Newton) from whole-body control to manipulation.

## Executive summary

Across 2023-2026, physics-based HOI imitation has converged on a small, reusable recipe: (1) an object-centric observation (humanoid proprioception + contact forces + full object state + joint-to-object vectors) and a reward built from exp(-lambda*error) terms that includes explicit object-tracking and contact-graph terms, with the PhysHOI/SkillMimic line composing those terms multiplicatively (SkillMimic's own ablation shows additive composition collapsing BallPlay-V success from 82.4% to 7.5%) while the newer InterMimic/HumanX/InterReal line keeps an additive sum and buys back interaction fidelity with other machinery (interaction-specific early termination, or a learned meta-policy that multiplicatively reweights the reward from tracking-error trends); (2) noisy or sparse references are not trusted as ground truth but repaired - subject-specific RL teachers mimic-retarget-refine imperfect MoCap and are distilled into one student (InterMimic's "perfect first, then scale up"), monocular video is replaced by physics-synthesized, force-closure-refined object trajectories (HumanX), MoCap is reduced to adaptive spatial scopes used as soft guidance inside a single retarget+track RL loop (Dexplore), and the reference set is augmented (stitched masked bridging trajectories and epsilon-neighborhood initialization in SkillMimic-V2, contact-constrained object-offset grids in InterReal, object-geometry/trajectory variation in HumanX); (3) the curricula that matter in practice are teacher-student distillation with a DAgger-to-PPO schedule, reward-weight meta-learning, reward-driven adaptive clip sampling, and initialization-buffer tricks (PSI) rather than the virtual-object-controller or residual-correction curricula implied by the question - Dexplore explicitly deletes the residual-correction stage; and (4) everything still trains in Isaac Gym with PPO, with sim-to-real demonstrated on a Unitree G1 only by self-reported preprint results (InterReal: 96.41% box-picking, 87.45% box-pushing, sim-to-sim in MuJoCo, FoundationPose for object pose) and no independently replicated numbers, no wall-clock/compute costs, and no surviving evidence at all on Omnigrasp, ManipTrans, DexMachina, HDMI or ResMimic.

## Findings

### 1. Multiplicative reward composition is the defining design choice of the PhysHOI/SkillMimic line, and its removal is catastrophic rather than cosmetic: total reward is a product of body-motion, object-motion, relative/interaction-graph and contact-graph factors, each an exp(-lambda*error) term.

**Confidence:** high · **Votes:** 3-0 on each of the four constituent claims

PhysHOI: r_t = r^b * r^o * r^ig * r^cg, with r^b itself r^p*r^r*r^pv*r^rv and per-dataset lambdas (BallPlay lambda_p=50, lambda_r=20, lambda_op=1, lambda_ig=20, contact-graph 5/5/5). SkillMimic: r_t = r^b * r^o * r^rel * r^reg * r^cg, with the authors justifying multiplication over summation as forcing balanced learning and avoiding local optima. SkillMimic-V2 inherits the same product (r^b*r^o*r^rel*r^cg) and explicitly contrasts it with an additive DeepMimic-style baseline. The ablation is the load-bearing evidence: on BallPlay-V, replacing the product with a sum drops success 82.4% -> 7.5% and explodes object position error 82.9mm -> 1662.5mm, while removing just the contact-graph reward drops success to 13.6% and raises contact error 0.087 -> 0.412. For MimicKit, this is the single cheapest high-value change: the existing DeepMimic-style additive tracking reward should be refactored into a product of exp-kernel factors before any object term is added, since an added object factor in a sum can be ignored by the policy. Merged from claims 0, 2, 3, 8.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/abs/2505.02094>

### 2. Additive composition is still used by the newest systems, but only when paired with extra machinery that forces the policy to care about the interaction - interaction-specific early termination, or a learned meta-policy that multiplicatively reallocates reward weights.

**Confidence:** high · **Votes:** 3-0 on claims 14 and 11

HumanX's privileged teacher uses r_t = r^body + r^obj + r^rel + r^c + r^reg with sub-rewards following a shared gamma*exp(-lambda*e) kernel template and a per-key-body contact-graph term exp(-sum_j lambda_j * e^cg[j]); the paper's own motivation for its Interaction Termination trick is that additive rewards let the policy ignore interaction-specific learning in favour of body-motion imitation. InterReal keeps a per-term exponential reward but replaces hand-tuned weighting with a meta-policy, Theta' = Theta_0 * sigma(t) * mu_psi(Theta_t | u_t), driven by the change in joint-position, object-position and link-position tracking error over a window Delta_t, on top of an interaction-graph contact reward r^ig = exp(-theta^ig * e^ig). InterMimic likewise maps each cost to exp(-lambda*E) and sums. Practical reading for MimicKit: additive is viable but then you owe the system either an interaction-gated termination condition or automated weight adaptation; multiplicative is the lower-engineering path. Merged from claims 14, 11, and the reward parts of 6/18.

Sources: <https://arxiv.org/html/2602.02473v1>, <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2502.20390>

### 3. Observations are object-centric in a consistent way across the field: humanoid proprioception plus contact/force signals plus the object's full 6-DoF state and velocities, plus explicit relational features (joint-to-object vectors, interaction graph) and a binary contact graph that is also what the contact reward supervises.

**Confidence:** high · **Votes:** 3-0 on each constituent claim

PhysHOI: s_t = {g_t, h_hat_{t+1}} where g_t is global root height, local body position/rotation/linear+angular velocity, net contact forces on contact bodies, and object local position/rotation/velocities. SkillMimic: s_t = {o^prop, o^f, o^obj} = proprioception + fingertip contact forces + object position/rotation/velocity/angular velocity, with r^cg = exp(-sum_j lambda^cg[j] * e^cg[j]) penalizing simulated-vs-reference contact mismatch. InterMimic adds vectors from joints to nearest object points and contact markers to the observation, distance-weighted joint error terms (weights inversely proportional to joint-object distance), object position/rotation costs E^o_p = ||p_hat^o - p^o|| and E^o_theta, and a three-level contact reward (promotion / penalty / neutral) whose levels are discretized by distance thresholds specifically to tolerate MoCap contact error. HumanX supervises a binary per-key-body contact vector. Concretely for MimicKit: the observation extension is proprioception + object state + joint-to-object vectors + per-contact-body force/contact bits, and the contact term should be tri-level rather than binary if the reference contacts come from noisy retargeting. Merged from claims 1, 4, 6, 18, and the contact-graph part of 14.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/html/2602.02473v1>

### 4. Practice on future reference windows is split and the split tracks task difficulty: single-frame lookahead suffices for a tightly tracked single skill, while multi-frame sparse keyframe sets are used once one policy must cover many interactions, and the student gets a denser set than the teacher.

**Confidence:** high · **Votes:** 3-0 on each constituent claim

PhysHOI feeds only the single next reference frame h_hat_{t+1} (human motion, object motion, interaction graph, contact graph) - explicitly not a multi-frame window. InterMimic instead sets keyframe offsets K = {1,16} for the 17 subject-specific teachers and K = {1,2,4,16} for the distilled student, i.e. a sparse, log-spaced horizon rather than a dense window, with the denser set reserved for the policy that must generalize across subjects and objects. Design implication: start MimicKit's HOI observation with next-frame-only parity with PhysHOI, and move to {1,2,4,16}-style sparse keyframes when moving from per-clip policies to one multi-clip policy. Merged from claims 1, 6, 18.

Sources: <https://arxiv.org/abs/2312.04393>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/abs/2502.20390>

### 5. The dominant answer to noisy MoCap is to let the simulator repair the data: train per-subject RL teachers that mimic, retarget and refine imperfect references, then distill them into one student that is supervised both by online teacher actions and by the teachers' physics-corrected rollouts, and finally RL-fine-tune the student.

**Confidence:** high · **Votes:** 3-0 on claims 5, 17, 12

InterMimic states the strategy as 'perfect first, then scale up': 17 subject-specific MLP teachers each imitate and refine a small subset of interactions (the physics simulator inherently corrects MoCap inaccuracies), subject 14 held out as test; the teachers are distilled into a single Transformer student by Reference Distillation (teacher rollouts become the student's goal states and reward references) and Policy Distillation (behavior cloning on actions), with training beginning as DAgger and transitioning to PPO via w = min(t/beta, 1), followed by RL fine-tuning to surpass demonstration replication. The pattern is reused downstream rather than merely proposed: InterReal adopts InterMimic as its physics-based HOI tracker to verify SMPL->G1 retargeted motions by enforcing physical constraints in Isaac Gym. For MimicKit this maps onto a two-stage pipeline - per-clip (or per-source) teacher tracking policies whose rollouts become the cleaned motion dataset, then one distilled multi-skill policy. Merged from claims 5, 17, and the verification half of 12.

Sources: <https://arxiv.org/abs/2502.20390>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/html/2603.07516>

### 6. Reference data is treated as something to augment, not just to clean; four distinct augmentation mechanisms are in use, all aimed at robustness to perturbation and at bridging sparse clips.

**Confidence:** high · **Votes:** 3-0 on claims 9, 12, 15

SkillMimic-V2 builds a Stitched Trajectory Graph that inserts masked intermediate states between demonstration clips when state similarity crosses a threshold (N = min(-floor(log10 beta), N_max) masked frames), and a State Transition Field via epsilon-Neighborhood State Initialization: sample s_new uniformly from the epsilon-neighborhood of a reference state s^i and connect it back with masked frames, {s_new, s_empty x N, s^j, ..., s^T}. InterReal applies IK-preserved hand-object contact constraints while sweeping object offsets Delta_x, Delta_y over linspace(-eps, eps, sqrt(c3)) to generate augmented multi-motions that improve policy stability under object perturbation. HumanX augments by scaling or substituting object geometry at mesh-acquisition time and by applying translation/scaling to the contact-phase object trajectory plus non-contact-phase velocity randomization. These are cheap to port: object-offset grids and epsilon-neighborhood initialization need no new policy machinery, only a dataset/initialization layer in MimicKit's motion loader. Merged from claims 9, 12, 15.

Sources: <https://arxiv.org/abs/2505.02094>, <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2602.02473v1>

### 7. Initialization and clip-sampling are used as the main implicit curriculum: a FIFO buffer mixing MoCap references with high-return simulated states replaces plain Reference State Initialization, and clips are sampled inversely to how well they are currently tracked.

**Confidence:** medium · **Votes:** 3-0 on claims 7, 19, 10 (with PSI's advantage over RSI explicitly scoped as the authors' claim)

InterMimic's Physical State Initialization (PSI) creates an initialization buffer holding MoCap reference states and simulation states from prior rollouts; each rollout starts from a state drawn at random from the buffer, and completed trajectories whose expected discounted return exceeds a threshold are admitted FIFO while older or lower-quality ones are discarded. The paper's own framing is that PSI collects trajectories for policy updates that RSI does not effectively utilize - an authors' claim, not an independently replicated result, which is why this finding is medium rather than high. SkillMimic-V2 supplies the complementary curriculum: clip sampling probability p_i = exp(-lambda_s * r_bar_i) / sum_j exp(-lambda_s * r_bar_j) with r_bar_i the average per-frame reward from initializing at s^i, so badly-tracked segments are visited more, plus a behavior-cloning-pretrained History Encoder h_t = theta(s_{t-k},...,s_{t-1}) frozen during RL and appended to the policy input. Both are small, self-contained additions to an existing DeepMimic-style trainer. Merged from claims 7, 19, 10.

Sources: <https://arxiv.org/abs/2502.20390>, <https://arxiv.org/html/2502.20390>, <https://arxiv.org/abs/2505.02094>

### 8. Interaction-specific early termination plus disturbed initialization and object-property domain randomization are what make an additive-reward HOI policy actually attend to the object.

**Confidence:** medium · **Votes:** 3-0

HumanX monitors, on reference frames that involve contact, the relative position error between the object and predefined key bodies, and terminates the episode probabilistically when that error exceeds a threshold - deliberately prioritizing interaction success over body-motion matching, which the paper motivates by the failure mode of additive rewards. It also applies disturbed initialization (random perturbations to root rotation/displacement, joint angles and object pose) and randomizes object size, mass, coefficient of restitution and robot friction. Single-source (CoRL 2026, public repo at github.com/wyhuai/HumanX), hence medium, but it is the most directly portable trick in the set: MimicKit already has early-termination machinery, and gating it on object-to-key-body error is a few lines. Claim 16.

Sources: <https://arxiv.org/html/2602.02473v1>

### 9. For dexterous hands, the newest work deletes the retarget -> track -> residual-correct pipeline entirely, treating MoCap as soft guidance via adaptive spatial scopes inside one RL loop, and modulates the reward by hand-object distance so tracking relaxes near contact while effort is penalized more there.

**Confidence:** medium · **Votes:** 3-0 on claims 20 and 21

Dexplore (CoRL 2025) argues the standard three-stage workflow leaves demonstrations underused and compounds errors across stages, and replaces it with a unified single-loop optimization that jointly retargets and tracks, deriving adaptive spatial scopes from raw trajectories and training with RL to stay in-scope while minimizing control effort and accomplishing the task. Its per-timestep reward is a state-reference matching term plus an energy regularization term, where the kinematic matching weights lambda_R^h and lambda_J^h are proportional to hand-object surface distance (via w(D) = min(1, D/0.20m)) while the energy penalty is inversely proportional - so as D -> 0 the matching weight shrinks and the effort penalty grows, which the paper describes as encouraging smoother, lower-energy motions when tracking constraints are relaxed. Two independently verified claims from one primary peer-reviewed source, hence medium. This directly contradicts the residual-policy framing in the research question: the trend is away from residual correction, not toward it. Merged from claims 20, 21.

Sources: <https://arxiv.org/abs/2509.09671>

### 10. Simulator and optimizer choices are uniform and boring - Isaac Gym with PPO - and the reported success rates that survived verification come from two sources only, with the most-cited success-rate table (InterMimic's) failing verification.

**Confidence:** medium · **Votes:** 3-0 on claims 13 and 3; the competing InterMimic success-rate claim was refuted 1-2

InterReal trains in Isaac Gym with PPO for 5e4 epochs and validates sim-to-sim in MuJoCo, reporting 96.41% box-picking success (MPJPE 0.1076 rad vs 0.1634 for the authors' ASAP* reproduction and 0.1984 for InterMimic*) and 87.45% box-pushing. SkillMimic reports 82.4% on BallPlay-V for the full method (E_b-mpjpe 56.8mm, E_o-mpjpe 82.9mm, E_cg 0.087). InterMimic's own OMOMO/BEHAVE numbers (90.7%/95.5% success, 42.6s vs 12.2s sustained tracking) did NOT survive verification (1-2 vote) and are excluded. No claim in the surviving set reports wall-clock time, GPU count, or sample count, so training cost is effectively unknown from this evidence. Confidence is medium because every number is a self-reported, mostly preprint figure with no independent replication, and the baselines in InterReal's table are the authors' own adaptations (marked ASAP*, InterMimic*). Merged from claims 13, 3; refuted InterMimic numbers noted.

Sources: <https://arxiv.org/html/2603.07516>, <https://arxiv.org/abs/2408.15270>, <https://arxiv.org/html/2502.20390>

### 11. Sim-to-real HOI on a Unitree G1 is demonstrated but thin: one preprint deploys on hardware with FoundationPose for object pose after a SMPL->G1 retarget + physics-verify + augment pipeline, and one CoRL paper drives the same class of skills from monocular human video, with no third-party replication of either.

**Confidence:** medium · **Votes:** 3-0 on claims 12, 13, 15

InterReal retargets SMPL HOI data to G1 shape parameters, verifies it with InterMimic as a physics-based HOI tracker, augments with contact-constrained object-offset variants, trains in Isaac Gym, validates sim-to-sim in MuJoCo, and deploys on a real Unitree G1 using FoundationPose to extract object-related observations; notably the authors defer the SMPL->G1 retargeting algorithm itself to a later publication, so that step is not reproducible from the paper. HumanX (CoRL 2026) handles monocular video by physics-based synthesis of object trajectories with frame-by-frame force-closure optimization yielding refined robot and object poses, explicitly arguing that physically plausible interaction matters more than photometrically faithful reconstruction. Together these define the realistic target for a MimicKit manipulation extension - object pose from a 6-DoF pose estimator at deployment, references repaired in simulation beforehand - but the evidence base is two self-reported systems, hence medium. Merged from claims 12, 13, 15.

Sources: <https://arxiv.org/html/2603.07516>, <https://arxiv.org/html/2602.02473v1>

## Caveats

- Evidence quality: every surviving claim is sourced to the primary paper describing its own method, which is appropriate for design descriptions but means all quantitative results are self-reported, and the only head-to-head baselines (ASAP*, InterMimic* in InterReal) are the authors' own reproductions rather than the original authors' numbers. Three papers are preprints without peer review (InterReal arXiv 2603.07516, PhysHOI, SkillMimic-V2 was later SIGGRAPH 2025); InterMimic (CVPR 2025), SkillMimic (CVPR 2025), Dexplore (CoRL 2025) and HumanX (CoRL 2026) are venue-published. One notable refutation: InterMimic's widely quoted OMOMO/BEHAVE success rates and tracking durations failed verification (1-2), as did an asymmetric actor-critic claim about InterReal (0-3) and a contact-graph topology claim about PhysHOI (0-3) - do not reuse those numbers from secondary summaries. Coverage gap: of the systems named in the research question, only PhysHOI, SkillMimic, SkillMimic-V2, InterMimic, Dexplore, HumanX and InterReal produced surviving claims; nothing survived on Omnigrasp, ManipTrans, DexMachina, HDMI or ResMimic, so any statement about those (including about virtual object controllers and residual policies, which this evidence set only touches via Dexplore's rejection of residual correction) is unsupported here. Training cost is entirely unanswered - no wall-clock, GPU-count, or sample-count figure survived. Time sensitivity is real but mild: the two most recent sources are dated Feb and Mar 2026 relative to today (2026-10-12), so later work may already have superseded the InterReal/HumanX G1 results, and Isaac Gym is deprecated upstream even though every paper here still uses it - a MimicKit port to Isaac Lab or Newton will not find published HOI baselines on those backends.

## Open questions

- Multiplicative vs additive composition is unresolved across generations: SkillMimic's ablation shows additive collapsing to 7.5% success, yet InterMimic, HumanX and InterReal all ship additive rewards with extra machinery. Is the multiplicative advantage specific to single-skill, dense-contact tracking (basketball), and does it survive when one policy must cover many objects and subjects?
- What is the actual training cost of these systems (GPU-hours, env steps, number of teacher policies) - none of the surviving claims reports it, and that number decides whether the teacher-distillation recipe is affordable on a shared SLURM cluster before MimicKit adopts it.
- How is reference contact state obtained for a robot embodiment rather than SMPL? InterReal defers its SMPL->G1 retargeting algorithm to a later paper, and the three-level contact discretization in InterMimic assumes MoCap-derived contact markers - an open question for building MimicKit's G1 HOI data layer.
- Do any of these methods report sim-to-real success for dexterous hands (as opposed to a G1's grippers/links), and does the Dexplore single-loop, soft-guidance formulation transfer to a whole-body humanoid where the "hand-object surface distance" that modulates its reward is far less informative?

## Claims refuted by verification

- (0-3) The contact graph is a complete graph whose nodes are objects and (optionally aggregated) humanoid body parts, with binary edge labels for contact; for basketball the two hands are aggregated as one node, the rest of the body as another, and the ball as the third. The contact-graph reward is reported as critical: removing it drops success from 84.8% to 20.7% on the Toss skill and to roughly 13.6-38.6% across ablations. — source https://arxiv.org/abs/2312.04393
- (1-2) In Isaac Gym at 30 Hz control, the full InterMimic model reports 90.7% success on OMOMO-Train and 95.5% on OMOMO-Test (success = proportion of references imitated at least once), with human/object tracking errors of roughly 5.5/9.7 cm (train) and 5.4/11.9 cm (test); on BEHAVE it sustains 42.6 s average tracking vs 12.2 s for SkillMimic, with object error 9.2 cm vs 13.4 cm, and PhysHOI yields clearly shorter tracking durations. — source https://arxiv.org/html/2502.20390
- (0-3) The policy uses an asymmetric actor-critic: the actor sees only deployable observations (proprioception, object position, interaction-graph features) while the critic receives privileged object velocity and rotation. — source https://arxiv.org/html/2603.07516

## How this compares with the overnight survey (reports 01 to 05)

- **Confirms** the central design choice in the plan: object-centric observations, contact-graph rewards, and
  a product of exp-kernel factors in the PhysHOI/SkillMimic line, with SkillMimic's ablation (82.4% to 7.5%
  success when the product becomes a sum) as the strongest single piece of evidence.
- **Adds a nuance the plan should carry**: InterMimic, HumanX and InterReal ship *additive* rewards and get
  interaction fidelity from other machinery (interaction-specific early termination, learned reward-weight
  meta-policy, distance-weighted joint terms). So phase 1 should run the additive-vs-multiplicative ablation
  as planned, but also implement object-deviation and contact-mismatch termination from day one, since that is
  what makes the additive variant work at all.
- **Confirms** the data-repair stance: teacher correction (InterMimic), physics-synthesized references
  (HumanX), soft-scope tracking (Dexplore), and augmentation (SkillMimic-V2 STG/STF, InterReal offset grids).
- **Weakens** two items the overnight reports took from READMEs and secondary summaries: InterMimic's quoted
  OMOMO/BEHAVE success rates failed verification (1 to 2), and the PhysHOI contact-graph topology description
  failed (0 to 3). Do not reuse those numbers; cite the papers' own tables after reading them.
- **Coverage gap, not contradiction**: the verified claim set contains nothing on Omnigrasp, ManipTrans,
  DexMachina, HDMI or ResMimic, so the plan's virtual-object-controller and residual-policy curricula rest on
  the overnight README-level survey only. Dexplore argues for deleting the residual stage. Treat those two
  curricula as experiments, not defaults.
- **Still missing everywhere**: training cost in GPU-hours. The plan's compute estimates remain extrapolations
  from MimicKit's own whole-body runs.
