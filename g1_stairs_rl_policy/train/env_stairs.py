"""G1 stairs env. Subclasses Isaac Lab's own G1RoughEnvCfg — NOT written from
scratch. That gives us, already wired and upstream-maintained:
  isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg.G1RoughEnvCfg
  (-> LocomotionVelocityRoughEnvCfg -> ManagerBasedRLEnvCfg)
  - G1 asset (isaaclab_assets.G1_MINIMAL_CFG), action scaling
  - height-scanner RayCaster sensor (mounted, just repointed + repatterned below)
  - velocity tracking rewards, base-contact termination, reset/push/domain-rand events

We change three things:
  1. terrain: swap in the stairs curriculum (terrain_stairs.py) + repoint/repattern
     the height-scanner to match grid/spec.py's LOCKED contract exactly, so train
     and deploy read the identical grid.
  2. rewards: full stairs reward table below (task / stairs-specific / regularization
     / motion-style). Source per term is cited inline; nothing here is unsourced.
  3. terminations: keep base_contact, add a base-pitch limit.

Base class content + terrain/height-scan API verified against the real
isaaclab-tasks source (fetched 2026-08-19). The reward functions are ports/
adaptations, NOT run against a live isaaclab install — check the API shapes
used (ContactSensor.data.net_forces_w, RayCaster.data.ray_hits_w,
Articulation body_pos_w/body_lin_vel_w/root_quat_w/root_pos_w) on the training
box before trusting a training run to them.

=== Reward table (design source, not just this file's docstring) ===

Task (inherited from G1Rewards / LocomotionVelocityRoughEnvCfg, untouched):
  lin. velocity tracking   exp(-||v_cmd,xy - v_xy||^2 / sigma)   unitree_rl_lab / isaaclab
  ang. velocity tracking   exp(-(w_cmd - w_z)^2 / sigma)         unitree_rl_lab / isaaclab
  termination (fall)       inherited termination_penalty (weight -200, isaaclab's own
                            episode-length/reward-scale convention; NOT the same number as
                            "-1 on done" from other codebases — see caveats below)
  alive                    ADDED here, base_mdp.is_alive, weight 0.15

Stairs additions (the new code, 5 rows):
  swing clearance    sum_f |h_f - h_terr - 0.10| * ||v_f,xy||        -1.0   TACT-ful (arXiv)
  edge proximity     grows as swing foot nears a riser edge,          -0.5   Jiang et al., dense
                      pre-contact, from local grid slope                     unsafe-stepping (arXiv 2603.07928)
  foot collision     anticipatory: swing-foot velocity component      -0.5   same paper
                      aimed AT the rising slope (riser face), pre-contact
  stumble            1{||F_xy|| > 4|F_z|} * 1{||v_f,xy|| > 0.15}      -1.0   TACT-ful's formulation
                      (picked over VB-Com's plain 3x ratio: the velocity      chosen explicitly — the
                      gate is what stops it firing on legitimate hard        velocity gate is what
                      landings)                                              separates edge-strikes
                                                                              from hard landings
  base height         (z_pelvis - z_feet_mean - 0.728)^2              -10.0  VB-Com G1 target (0.728m)
  (feet-relative, not ground-relative)                                       + BeamDojo's feet-relative fix

Regularization / motion-style: LEFT AS G1RoughEnvCfg's OWN inherited, already-
tuned values (lin_vel_z_l2, ang_vel_xy_l2, flat_orientation_l2, dof_acc_l2,
dof_torques_l2, action_rate_l2, dof_pos_limits, feet_slide, feet_air_time) —
per the explicit caveat that reward weights don't copy-paste across codebases
(different dt, action scale, reward normalization). Cross-check numbers from
VB-Com exist in the design doc but are NOT applied verbatim here; only the 5
stairs rows above (genuinely new terms with no inherited equivalent) use the
literature's numbers directly. Three exceptions, deliberately bumped because
they ARE this task's actual upper-body/collision story, not just a cross-check:
  - joint_deviation_arms / joint_deviation_hip: -0.1 -> -0.5 ("the whole
    upper-body story" — arms/hips have no task in a velocity-tracking policy,
    PPO should find a small natural swing/stance, not be pulled hard to
    default; -0.5 lets that happen without going limp)
  - joint_deviation_torso -> WAIST_DEVIATION_WEIGHT, kept heavy (29-DOF G1's
    3-DOF waist; unitree_rl_lab penalizes it hard for upright posture) but
    flagged as the FIRST knob to soften if climbing looks stiff on steep
    flights — torso needs some pitch on stairs that it doesn't need on flat.
  - re-enabled non-foot collision (disabled by G1RoughEnvCfg's own
    __post_init__ via undesired_contacts=None) as `collision_penalty`, a port
    of chengxuxin/extreme-parkour's real _reward_collision, weight -15.0.
  - feet_lateral_distance added (small weight, prevents leg-crossing).
  Skipped as redundant with an inherited term: VB-Com's "feet stumble" 3x
  variant (TACT-ful's 4x+velocity-gate above supersedes it) and a standalone
  "foot slip" term (already covered by inherited feet_slide).

Not implemented, deliberately: footstep-guidance (-ln||dx|| to a recommended
stepping point, from the parkour literature) — kept as an escape hatch if the
dense penalties above don't converge, not built until that's proven needed.
AMP/motion-prior style terms — solving a style problem this task doesn't have.
"""
import torch

from isaaclab.envs import mdp as base_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import (
    G1Rewards,
    G1RoughEnvCfg,
)

from grid import spec as grid_spec
from .terrain_stairs import make_terrain_generator_cfg

FOOT_BODY_RE = ".*_ankle_roll_link"           # same regex G1Rewards already uses for feet
NON_FOOT_COLLISION_BODY_RE = ".*(knee|elbow|shoulder|wrist).*"  # verify against G1's actual link names
# torso_link deliberately excluded: it's also the inherited base_contact
# TERMINATION body — a torso hit usually ends the episode before this reward
# term gets a chance to act, so including it here mostly just double-books

SWING_CLEARANCE_TARGET = 0.10       # m, TACT-ful's target foot-above-terrain height
STUMBLE_FORCE_RATIO = 4.0           # TACT-ful's horizontal/vertical contact-force ratio
STUMBLE_VEL_GATE = 0.15             # m/s, TACT-ful's velocity gate (separates edge-strikes from hard landings)
BASE_HEIGHT_TARGET = 0.728          # m, VB-Com's G1 pelvis-above-feet target
FEET_MIN_LATERAL_DIST = 0.18        # m, VB-Com's G1 min left/right foot separation
RISER_SLOPE_REF = 1.5               # m/m, local grid slope treated as "fully a riser face" for edge/collision terms
BASE_PITCH_LIMIT = 0.7              # rad, ~40deg, termination threshold

# waist deviation: 29-DOF G1 specific (3-DOF waist; unitree_rl_lab penalizes it
# heavily for upright posture on flat ground). FIRST knob to soften if the
# policy climbs stiffly or refuses steep flights — soften one at a time,
# starting here before touching orientation.
WAIST_DEVIATION_WEIGHT = -1.0
WAIST_JOINT_RE = ".*waist.*"        # generalizes G1Rewards' single "torso_joint" match for the 3-DOF waist


def _yaw_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _foot_grid_cells(env, asset_cfg: SceneEntityCfg):
    """Per-foot (idx_x, idx_y) into the locked height-scan grid, base-yaw frame."""
    robot = env.scene[asset_cfg.name]
    base_xy = robot.data.root_pos_w[:, :2]                       # (E, 2)
    yaw = _yaw_from_quat(robot.data.root_quat_w)                  # (E,)
    cos_y, sin_y = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)

    foot_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]   # (E, F, 2)
    d = foot_xy - base_xy.unsqueeze(1)
    local_x = d[..., 0] * cos_y + d[..., 1] * sin_y               # matches grid FRAME == "base", yaw-aligned
    local_y = -d[..., 0] * sin_y + d[..., 1] * cos_y

    res, nx, ny = grid_spec.RESOLUTION, grid_spec.NX, grid_spec.NY
    idx_x = torch.clamp(torch.round((local_x + grid_spec.EXTENT_X / 2) / res).long(), 0, nx - 1)
    idx_y = torch.clamp(torch.round((local_y + grid_spec.EXTENT_Y / 2) / res).long(), 0, ny - 1)
    return idx_x, idx_y


def _height_grid(env) -> torch.Tensor:
    """(E, NX, NY) absolute world-z terrain height, straight off the height-scanner.

    RayCaster writes +-inf for missed rays (robot near a patch edge after a
    push/reset, mid-air during a reset). One inf here poisons every reward
    that reads it (-> NaN -> a wrecked PPO batch), unlike the inherited
    height_scan OBSERVATION term, which survives because it clips. Clamp raw
    hits to a band around the robot's own current height instead.
    """
    height_scanner: RayCaster = env.scene.sensors["height_scanner"]
    heights = height_scanner.data.ray_hits_w[..., 2].reshape(-1, grid_spec.NX, grid_spec.NY)
    base_z = env.scene["robot"].data.root_pos_w[:, 2].view(-1, 1, 1)
    return torch.nan_to_num(heights, nan=0.0, posinf=1e6, neginf=-1e6).clamp(base_z - 2.0, base_z + 2.0)


def _grid_slope(heights: torch.Tensor, idx_x: torch.Tensor, idx_y: torch.Tensor):
    """Central-difference gradient (slope_x, slope_y) at each (idx_x, idx_y) cell."""
    nx, ny, res = grid_spec.NX, grid_spec.NY, grid_spec.RESOLUTION
    idx_x_lo, idx_x_hi = torch.clamp(idx_x - 1, 0, nx - 1), torch.clamp(idx_x + 1, 0, nx - 1)
    idx_y_lo, idx_y_hi = torch.clamp(idx_y - 1, 0, ny - 1), torch.clamp(idx_y + 1, 0, ny - 1)
    env_idx = torch.arange(heights.shape[0], device=heights.device).unsqueeze(-1).expand_as(idx_x)
    slope_x = (heights[env_idx, idx_x_hi, idx_y] - heights[env_idx, idx_x_lo, idx_y]) / (2 * res)
    slope_y = (heights[env_idx, idx_x, idx_y_hi] - heights[env_idx, idx_x, idx_y_lo]) / (2 * res)
    return slope_x, slope_y


def _in_contact(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3], dim=-1) > 1.0


def _assert_same_body_order(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Every reward below elementwise-combines a ContactSensor tensor (indexed by
    sensor_cfg.body_ids, the sensor's own body-id space) with an Articulation
    tensor (indexed by asset_cfg.body_ids, a DIFFERENT id space) — same regex on
    both does NOT guarantee the same resolved order. If they disagree, contact
    gating/forces get silently swapped left<->right (symmetric, so it won't look
    obviously broken). Checked once per process, not every step."""
    if getattr(env, "_stairs_body_order_checked", False):
        return
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    sensor_names = [contact_sensor.body_names[i] for i in sensor_cfg.body_ids]
    asset_names = [asset.body_names[i] for i in asset_cfg.body_ids]
    assert sensor_names == asset_names, (
        f"ContactSensor/Articulation body order mismatch: {sensor_names} vs {asset_names}. "
        "Reward terms in env_stairs.py assume these line up; fix by reindexing one to the other."
    )
    env._stairs_body_order_checked = True


# --- stairs additions: the 5 new rows --------------------------------------

def swing_clearance_reward(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                            target: float = SWING_CLEARANCE_TARGET):
    """TACT-ful: |h_f - h_terr - target| * ||v_f,xy||. Terrain-relative (via our
    own height-scan grid), not a fixed flat-ground target — the improvement
    over the plain unitree_rl_gym version this replaces. Gated by ~in_contact:
    a planted foot sliding (push recovery, slight slip) shouldn't double-count
    against the already-inherited feet_slide term."""
    _assert_same_body_order(env, sensor_cfg, asset_cfg)
    robot = env.scene[asset_cfg.name]
    idx_x, idx_y = _foot_grid_cells(env, asset_cfg)
    heights = _height_grid(env)
    env_idx = torch.arange(heights.shape[0], device=heights.device).unsqueeze(-1).expand_as(idx_x)
    h_terr = heights[env_idx, idx_x, idx_y]

    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_speed_xy = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    in_contact = _in_contact(env, sensor_cfg)
    clearance_err = torch.abs(foot_z - h_terr - target) * foot_speed_xy * (~in_contact)
    return torch.sum(clearance_err, dim=-1)


EDGE_HEIGHT_DECAY = 8.0   # 1/m, how fast edge_proximity/foot_collision fall off with foot height above terrain


def _height_gate(foot_z: torch.Tensor, h_terr: torch.Tensor) -> torch.Tensor:
    """A foot swinging WELL ABOVE the terrain (which is the correct way to
    clear a riser) shouldn't get the full edge/collision penalty — without
    this, these terms fight the exact maneuver stairs require. Decays with
    height above the local terrain cell, not gated to only contact."""
    return torch.exp(-EDGE_HEIGHT_DECAY * torch.clamp(foot_z - h_terr, min=0.0))


def edge_proximity_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Simplified adaptation of Jiang et al.'s (arXiv 2603.07928) dense
    edge-contact term — no code release exists for that paper, so this keeps
    its core idea (penalty grows continuously as a swing foot nears a riser
    edge, BEFORE contact) using local slope on our own height-scan grid as the
    proximity signal, instead of their Sobel edge-centroid extraction.
    Height-gated (see _height_gate) so clearing a riser doesn't get punished."""
    _assert_same_body_order(env, sensor_cfg, asset_cfg)
    robot = env.scene[asset_cfg.name]
    idx_x, idx_y = _foot_grid_cells(env, asset_cfg)
    heights = _height_grid(env)
    slope_x, slope_y = _grid_slope(heights, idx_x, idx_y)
    slope_mag = torch.sqrt(slope_x**2 + slope_y**2 + 1e-8)
    proximity = torch.clamp(slope_mag / RISER_SLOPE_REF, 0.0, 1.0)

    env_idx = torch.arange(heights.shape[0], device=heights.device).unsqueeze(-1).expand_as(idx_x)
    h_terr = heights[env_idx, idx_x, idx_y]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    gate = _height_gate(foot_z, h_terr)

    in_contact = _in_contact(env, sensor_cfg)
    return torch.sum(proximity * gate * (~in_contact), dim=-1)


def foot_collision_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg):
    """Simplified adaptation of the same paper's foot-collision term: penalize
    swing-foot velocity aimed INTO a riser face (the up-gradient direction on
    our grid), not just proximity to one — distinguishes "approaching a step
    edge" (edge_proximity_penalty) from "about to kick the riser"
    (this term). Height-gated for the same reason as edge_proximity_penalty."""
    _assert_same_body_order(env, sensor_cfg, asset_cfg)
    robot = env.scene[asset_cfg.name]
    idx_x, idx_y = _foot_grid_cells(env, asset_cfg)
    heights = _height_grid(env)
    slope_x, slope_y = _grid_slope(heights, idx_x, idx_y)
    slope_mag = torch.sqrt(slope_x**2 + slope_y**2 + 1e-8)
    grad_hat_x, grad_hat_y = slope_x / slope_mag, slope_y / slope_mag

    foot_vel_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    v_toward = torch.clamp(foot_vel_xy[..., 0] * grad_hat_x + foot_vel_xy[..., 1] * grad_hat_y, min=0.0)
    proximity = torch.clamp(slope_mag / RISER_SLOPE_REF, 0.0, 1.0)

    env_idx = torch.arange(heights.shape[0], device=heights.device).unsqueeze(-1).expand_as(idx_x)
    h_terr = heights[env_idx, idx_x, idx_y]
    foot_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    gate = _height_gate(foot_z, h_terr)

    in_contact = _in_contact(env, sensor_cfg)
    return torch.sum(v_toward * proximity * gate * (~in_contact), dim=-1)


def stumble_penalty(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg,
                     ratio: float = STUMBLE_FORCE_RATIO, vel_gate: float = STUMBLE_VEL_GATE):
    """TACT-ful's formulation (picked over VB-Com's plain 3x ratio, per the
    design doc): force-ratio check AND a velocity gate, so a foot that's
    already stopped (hard landing, not a riser-edge strike) doesn't fire this.
    Uses net_forces_w_history (not instantaneous net_forces_w) if available —
    upstream biped terms (e.g. feet_air_time_positive_biped) do the same, to
    survive a strike lasting less than one policy decimation step."""
    _assert_same_body_order(env, sensor_cfg, asset_cfg)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_hist = getattr(contact_sensor.data, "net_forces_w_history", None)
    forces = (forces_hist[:, :, sensor_cfg.body_ids, :3].abs().amax(dim=1)
              if forces_hist is not None else contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3])
    horiz_force = torch.norm(forces[..., :2], dim=-1)
    vert_force = torch.abs(forces[..., 2])
    force_flag = horiz_force > ratio * vert_force

    robot = env.scene[asset_cfg.name]
    foot_speed_xy = torch.norm(robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    vel_flag = foot_speed_xy > vel_gate

    return torch.any(force_flag & vel_flag, dim=-1).float()


def base_height_feet_relative_penalty(env, asset_cfg: SceneEntityCfg, feet_cfg: SceneEntityCfg,
                                       target: float = BASE_HEIGHT_TARGET):
    """VB-Com's G1 target (0.728m) + BeamDojo's feet-relative fix: measure
    pelvis height above the STANCE foot (min z, not mean) — mean drags 0.1-0.15m
    up every step the swing foot lifts, making the "correct" pelvis height
    oscillate every stride; min tracks whichever foot is actually load-bearing."""
    robot = env.scene[asset_cfg.name]
    pelvis_z = robot.data.root_pos_w[:, 2]
    feet_z = robot.data.body_pos_w[:, feet_cfg.body_ids, 2].amin(dim=-1)
    return torch.square(pelvis_z - feet_z - target)


# --- re-enabled / new regularization terms ---------------------------------

def collision_penalty(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0):
    """Port of chengxuxin/extreme-parkour's real _reward_collision: count of
    non-foot bodies with contact force above threshold. G1RoughEnvCfg disables
    the inherited undesired_contacts term (sets it None); this replaces it for
    stairs, where shin/knee-vs-riser contact is a real failure mode.
    threshold=1.0 (not extreme-parkour's 0.1 — their number lives in a
    different force/normalization regime; 0.1N is trigger-happy on isaaclab's
    ContactSensor, use the same 1.0N floor as _in_contact instead)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :3]
    return torch.sum((torch.norm(forces, dim=-1) > threshold).float(), dim=-1)


def feet_lateral_distance_penalty(env, asset_cfg: SceneEntityCfg, min_dist: float = FEET_MIN_LATERAL_DIST):
    """VB-Com's G1 min separation (0.18m) — prevents leg-crossing. Measures
    BASE-FRAME lateral (y) separation, not raw xy distance: feet are normally
    split fore-aft during a stride, so plain ||p_L - p_R|| stays above 0.18m
    even when the feet are laterally crossed, making the original xy-distance
    version unable to catch the thing it's meant to prevent."""
    robot = env.scene[asset_cfg.name]
    yaw = _yaw_from_quat(robot.data.root_quat_w)
    feet_xy = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    delta = feet_xy[:, 0] - feet_xy[:, 1]
    lateral = -delta[..., 0] * torch.sin(yaw) + delta[..., 1] * torch.cos(yaw)
    return torch.clamp(min_dist - torch.abs(lateral), min=0.0)


@configclass
class G1StairsRewards(G1Rewards):
    # --- task: add alive, everything else (tracking, termination) inherited ---
    alive = RewTerm(func=base_mdp.is_alive, weight=0.15)

    # --- stairs additions: the 5 new rows ---
    swing_clearance = RewTerm(
        func=swing_clearance_reward,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_RE),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )
    edge_proximity = RewTerm(
        func=edge_proximity_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_RE),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )
    foot_collision = RewTerm(
        func=foot_collision_penalty,
        weight=-0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_RE),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )
    stumble = RewTerm(
        func=stumble_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_RE),
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )
    base_height = RewTerm(
        func=base_height_feet_relative_penalty,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot"),
                "feet_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )

    # --- re-enabled / new regularization ---
    collision = RewTerm(
        func=collision_penalty,
        weight=-15.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=NON_FOOT_COLLISION_BODY_RE)},
    )
    feet_lateral_distance = RewTerm(
        func=feet_lateral_distance_penalty,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_RE)},
    )


@configclass
class G1StairsEnvCfg(G1RoughEnvCfg):
    rewards: G1StairsRewards = G1StairsRewards()

    def __post_init__(self):
        super().__post_init__()

        # 1. terrain: stairs curriculum instead of ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator = make_terrain_generator_cfg(num_rows=10, num_cols=20)

        # height-scanner: match grid/spec.py's LOCKED contract exactly (frame,
        # extent, resolution, ordering) so train and deploy read identical grids.
        # grid.spec.query_points() locks index = ix*NY+iy (x OUTER loop, y INNER
        # loop). IsaacLab's GridPatternCfg.ordering="xy" means outer-x/inner-y
        # (its own docstring example: X=(0,1,2), Y=(3,4) -> [(0,3),(0,4),(1,3),...]),
        # which is exactly that — "xy" is the match, NOT "yx" (previous version of
        # this file had that backwards; verify with the runtime assertion below
        # before trusting it further).
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"  # grid_spec.FRAME == "base"
        self.scene.height_scanner.pattern_cfg = GridPatternCfg(
            resolution=grid_spec.RESOLUTION,
            size=(grid_spec.EXTENT_X, grid_spec.EXTENT_Y),
            ordering="xy",
        )
        self.observations.policy.height_scan.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.observations.policy.height_scan.clip = (grid_spec.CLIP_MIN, grid_spec.CLIP_MAX)
        # RUN ONCE on the training box before trusting any of the above:
        #   pts = env.scene.sensors["height_scanner"].cfg.pattern_cfg  # or inspect
        #   ray_starts, _ = patterns.grid_pattern(pts, "cpu")
        #   assert torch.allclose(ray_starts[:, :2], torch.as_tensor(grid_spec.query_points()))
        # If it doesn't match, flip back to "yx" — don't trust the docstring math above.

        # 2. rewards: bump upper-body / waist per the design doc (see module
        # docstring for why these three deviate from "leave inherited as-is").
        self.rewards.joint_deviation_arms.weight = -0.5
        self.rewards.joint_deviation_hip.weight = -0.5
        self.rewards.joint_deviation_torso.weight = WAIST_DEVIATION_WEIGHT
        self.rewards.joint_deviation_torso.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=WAIST_JOINT_RE
        )

        # 3. terminations: keep base_contact (inherited), add a tilt limit.
        # NOTE: mdp.bad_orientation limits total tilt via projected gravity
        # (roll+pitch combined), not pure pitch — fine as a fall guard, just
        # don't read BASE_PITCH_LIMIT as "pitch-only".
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": BASE_PITCH_LIMIT}
        )


@configclass
class G1StairsPlayEnvCfg(G1StairsEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# --- fixed after a fable-agent static review (2026-08-19), see git history for
# the pre-fix version --------------------------------------------------------
# - GridPatternCfg ordering was backwards ("yx" instead of "xy") — would have
#   silently scrambled train vs deploy obs. Fixed; still needs the runtime
#   assertion noted above the pattern_cfg to be actually confirmed.
# - `mdp.terminations_cfg.TerminationTermCfg` doesn't exist — crashed at cfg
#   construction. Fixed to `isaaclab.managers.TerminationTermCfg`.
# - No inf/NaN guard on raw ray_hits_w (missed rays -> inf -> NaN reward ->
#   poisoned PPO batch). Fixed in _height_grid via nan_to_num + clamp.
# - ContactSensor vs Articulation body-id spaces silently assumed aligned.
#   Added _assert_same_body_order(), called once per process in every reward
#   that mixes both.
# - feet_lateral_distance_penalty measured raw xy distance (can't catch
#   leg-crossing — feet are normally split fore-aft). Fixed to base-frame
#   lateral-only separation.
# - edge_proximity/foot_collision had no vertical gating — penalized clearing
#   a riser, the correct climbing maneuver. Added _height_gate().
#
# --- still open, verify on the training box, in order -----------------------
# 1. `from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg
#     import G1Rewards, G1RoughEnvCfg` — confirm exact module path (isaaclab version-
#     dependent; fetched from isaac-sim/IsaacLab main on 2026-08-19).
# 2. `mdp.bad_orientation` / `base_mdp.is_alive` and their param names — confirm
#    against installed isaaclab.envs.mdp.
# 3. `RayCaster.data.ray_hits_w`, `ContactSensor.data.{net_forces_w,net_forces_w_history,body_names}`,
#    `Articulation.data.{root_pos_w,root_quat_w,body_pos_w,body_lin_vel_w,body_names}`
#    shapes/field names — confirm against installed isaaclab.sensors/assets.
# 4. NON_FOOT_COLLISION_BODY_RE, WAIST_JOINT_RE — confirm against G1's actual
#    link/joint names (isaaclab_assets.G1_MINIMAL_CFG), these are guesses.
# 5. Run the GridPatternCfg assertion noted above pattern_cfg — don't trust the
#    "xy" derivation without it actually running once.
# 6. Run register.py's Play-v0 in the viewer: check swing_clearance/edge_proximity/
#    foot_collision values near a riser vs mid-tread, and watch a few episodes
#    for stiff/refusing-to-pitch behavior — if so, soften WAIST_DEVIATION_WEIGHT
#    first, then flat_orientation_l2 (inherited), one at a time, per the design doc.
