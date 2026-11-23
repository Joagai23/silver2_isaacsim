"""
Standalone headless GPU verification suite for Warp hydrodynamics ocean currents.

Tests the actual Warp CUDA kernel and CUDA Graph wrapper across:
  1. Zero-current equilibrium & above-surface gating
  2. Galilean invariance (moving body in still water vs. stationary body in opposite current)
  3. Upstream Center of Pressure (CoP), drag direction, and lift sign
  4. Munk moment, isotropic Coriolis cancellation, and full 6-DOF Fossen added-mass dynamics
  5. Per-environment RL randomization, partial env_ids resets, and CUDA Graph pointer stability
  6. SILVER2 JSON config loading (`from_json`) across batched environments

Run either directly or with pytest:
    python src/scripts/hydrodynamics/tests/test_warp_hydrodynamics_currents.py
    python -m pytest -v src/scripts/hydrodynamics/tests/test_warp_hydrodynamics_currents.py
"""

import math
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import warp as wp

# ---------------------------------------------------------------------------
# Path resolution (works from src/scripts/hydrodynamics/, tests/, or repo root)
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in THIS_FILE.parents if (p / "src" / "scripts" / "hydrodynamics").is_dir()),
    THIS_FILE.parent,
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_FILE.parent) not in sys.path:
    sys.path.insert(0, str(THIS_FILE.parent))

try:
    from src.scripts.hydrodynamics.warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper
except ImportError:
    from warp_hydrodynamics_wrapper import WarpHydrodynamicsWrapper

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Helpers & Independent Analytical Reference (NumPy / float64)
# ---------------------------------------------------------------------------
def _require_cuda() -> None:
    wp.init()
    if not torch.cuda.is_available() or not wp.is_cuda_available():
        msg = "CUDA GPU is required to run Warp CUDA Graph hydrodynamics tests."
        try:
            import pytest
            pytest.skip(msg)
        except ImportError:
            raise RuntimeError(msg)


def _quat_from_axis_angle_xyzw(axis: Tuple[float, float, float], angle_rad: float) -> np.ndarray:
    """Returns a normalized Warp quaternion in (x, y, z, w) order."""
    ax = np.asarray(axis, dtype=np.float64)
    ax = ax / np.linalg.norm(ax)
    half = 0.5 * angle_rad
    s = math.sin(half)
    c = math.cos(half)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, c], dtype=np.float64)


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for Warp (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    q = np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rotmat_xyzw(q: np.ndarray) -> np.ndarray:
    """Converts a unit Warp (x, y, z, w) quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _reference_fossen_added_mass(
    width: float,
    depth: float,
    height: float,
    water_density: float,
    linear_mass_coeff: float,
    angular_mass_coeff: float,
    q_xyzw: np.ndarray,
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    lin_acc: np.ndarray,
    ang_acc: np.ndarray,
    current_vel: np.ndarray,
    sub_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Independent float64 reference for tau_A = -M_A * dot(nu_r) - C_A(nu_r) * nu_r."""
    vol = width * depth * height
    displaced_mass = vol * water_density
    ref_area = (vol ** (2.0 / 3.0)) + 1e-6

    ma_lin = np.array([
        displaced_mass * linear_mass_coeff * ((depth * height) / ref_area),
        displaced_mass * linear_mass_coeff * ((width * height) / ref_area),
        displaced_mass * linear_mass_coeff * ((width * depth) / ref_area),
    ], dtype=np.float64)

    ma_ang = np.array([
        (1.0 / 12.0) * displaced_mass * (depth ** 2 + height ** 2) * angular_mass_coeff,
        (1.0 / 12.0) * displaced_mass * (width ** 2 + height ** 2) * angular_mass_coeff,
        (1.0 / 12.0) * displaced_mass * (width ** 2 + depth ** 2) * angular_mass_coeff,
    ], dtype=np.float64)

    R = _quat_to_rotmat_xyzw(q_xyzw)
    v_r_world = lin_vel - current_vel
    v_r_b = R.T @ v_r_world
    w_b = R.T @ ang_vel

    # Body-frame relative accelerations for steady irrotational current (a_c = 0, omega_c = 0)
    dot_v_r_b = R.T @ lin_acc - np.cross(w_b, v_r_b)
    dot_w_b = R.T @ ang_acc

    p_b = ma_lin * v_r_b
    l_b = ma_ang * w_b

    f_local = -(ma_lin * dot_v_r_b) - np.cross(w_b, p_b)
    t_local = -(ma_ang * dot_w_b) - (np.cross(v_r_b, p_b) + np.cross(w_b, l_b))

    return (R @ f_local) * sub_ratio, (R @ t_local) * sub_ratio


def _run_kernel(
    wrapper: WarpHydrodynamicsWrapper,
    pos: torch.Tensor,
    quat: torch.Tensor,
    lin_vel: torch.Tensor,
    ang_vel: torch.Tensor,
    lin_acc: torch.Tensor,
    ang_acc: torch.Tensor,
    current_vel=None,
):
    """Runs the CUDA Graph kernel and returns cloned float32 tensors after GPU sync."""
    outputs = wrapper.calculate_hydrodynamic_forces(
        pos.contiguous(),
        quat.contiguous(),
        lin_vel.contiguous(),
        ang_vel.contiguous(),
        lin_acc.contiguous(),
        ang_acc.contiguous(),
        t_current_velocity=current_vel,
    )
    wp.synchronize_device(wrapper.device)
    return tuple(t.clone() for t in outputs)


# ---------------------------------------------------------------------------
# 1. Zero-Current Equilibrium & Above-Surface Gating
# ---------------------------------------------------------------------------
def test_zero_current_and_above_surface_gating():
    _require_cuda()
    w, d, h = 1.0, 0.5, 0.4
    rho, g = 1025.0, 9.81

    hydro = WarpHydrodynamicsWrapper(
        width=w, depth=d, height=h,
        linear_drag_coefficient=1.2, angular_drag_coefficient=0.5,
        linear_damping=10.0, angular_damping=2.0,
        linear_mass_coeff=0.3, angular_mass_coeff=0.15,
        lift_coefficient=0.4,
        water_density=rho, gravity=g,
        num_envs=1, device=DEVICE,
        ocean_current=(0.0, 0.0, 0.0),
    )

    q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    # Fully submerged & stationary in still water
    pos_sub = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, device=DEVICE)
    f_b, f_d, f_l, t_d, f_am, t_am, cob, cop = _run_kernel(
        hydro, pos_sub, q_id, zeros, zeros, zeros, zeros
    )
    expected_buoy = torch.tensor([[0.0, 0.0, rho * (w * d * h) * g]], dtype=torch.float32, device=DEVICE)
    assert torch.allclose(f_b, expected_buoy, rtol=1e-5, atol=1e-4)
    assert torch.allclose(cob, pos_sub, atol=1e-5)
    assert torch.allclose(cop, pos_sub, atol=1e-5)
    for tensor in (f_d, f_l, t_d, f_am, t_am):
        assert torch.allclose(tensor, zeros, atol=1e-6)

    # Completely above water surface with strong current -> zero forces/torques
    hydro.set_ocean_current((2.5, -1.5, 0.5))
    pos_air = torch.tensor([[0.0, 0.0, +2.0]], dtype=torch.float32, device=DEVICE)
    f_b, f_d, f_l, t_d, f_am, t_am, _, _ = _run_kernel(
        hydro, pos_air, q_id, zeros, zeros, zeros, zeros
    )
    for tensor in (f_b, f_d, f_l, t_d, f_am, t_am):
        assert torch.allclose(tensor, zeros, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Galilean Invariance (Moving Body vs. Opposite Current & General Boost)
# ---------------------------------------------------------------------------
def test_galilean_invariance():
    _require_cuda()
    hydro = WarpHydrodynamicsWrapper(
        width=1.4, depth=0.45, height=0.35,
        linear_drag_coefficient=1.15, angular_drag_coefficient=0.4,
        linear_damping=8.5, angular_damping=1.8,
        linear_mass_coeff=0.35, angular_mass_coeff=0.2,
        lift_coefficient=0.55,
        num_envs=1, device=DEVICE,
    )

    # Non-trivial 3D orientation (yaw + pitch + roll)
    q = _quat_mul_xyzw(
        _quat_from_axis_angle_xyzw((0.0, 0.0, 1.0), 0.7),
        _quat_mul_xyzw(
            _quat_from_axis_angle_xyzw((0.0, 1.0, 0.0), -0.35),
            _quat_from_axis_angle_xyzw((1.0, 0.0, 0.0), 0.25),
        ),
    )

    pos = torch.tensor([[1.2, -0.8, -4.5]], dtype=torch.float32, device=DEVICE)
    quat = torch.tensor([q], dtype=torch.float32, device=DEVICE)
    ang_vel = torch.tensor([[0.3, -0.4, 0.6]], dtype=torch.float32, device=DEVICE)
    lin_acc = torch.tensor([[0.5, -0.2, 0.1]], dtype=torch.float32, device=DEVICE)
    ang_acc = torch.tensor([[-0.1, 0.25, -0.3]], dtype=torch.float32, device=DEVICE)
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    v_rel = torch.tensor([[0.85, -0.55, 0.25]], dtype=torch.float32, device=DEVICE)

    # Case A: Body moves at +v_rel in still water (v_c = 0)
    hydro.set_ocean_current((0.0, 0.0, 0.0))
    out_move = _run_kernel(hydro, pos, quat, v_rel, ang_vel, lin_acc, ang_acc)

    # Case B: Stationary body (v = 0) in opposite ocean current (v_c = -v_rel)
    hydro.set_ocean_current(-v_rel[0])
    out_curr = _run_kernel(hydro, pos, quat, zeros, ang_vel, lin_acc, ang_acc)

    # Case C: General Galilean velocity boost (v = v_rel + u, v_c = u)
    u_boost = torch.tensor([[-1.3, 0.9, -0.4]], dtype=torch.float32, device=DEVICE)
    hydro.set_ocean_current(u_boost[0])
    out_boost = _run_kernel(hydro, pos, quat, v_rel + u_boost, ang_vel, lin_acc, ang_acc)

    names = [
        "buoyancy_force", "drag_force", "lift_force", "drag_torque",
        "added_mass_force", "added_mass_torque", "cob", "cop",
    ]
    for name, a, b, c in zip(names, out_move, out_curr, out_boost):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-4), f"Move vs Current mismatch on {name}: {a} vs {b}"
        assert torch.allclose(a, c, rtol=1e-5, atol=1e-4), f"Move vs Boost mismatch on {name}: {a} vs {c}"


# ---------------------------------------------------------------------------
# 3. Upstream Center of Pressure (CoP), Drag Direction & Lift Sign
# ---------------------------------------------------------------------------
def test_upstream_cop_drag_and_lift_signs():
    _require_cuda()
    w, d, h = 2.0, 0.6, 0.4
    rho = 1025.0
    cd = 1.2
    c_lin = 15.0

    hydro = WarpHydrodynamicsWrapper(
        width=w, depth=d, height=h,
        linear_drag_coefficient=cd, angular_drag_coefficient=0.5,
        linear_damping=c_lin, angular_damping=1.0,
        linear_mass_coeff=0.2, angular_mass_coeff=0.1,
        lift_coefficient=0.8,
        water_density=rho,
        num_envs=1, device=DEVICE,
    )

    pos = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, device=DEVICE)
    q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    # Steady +x ocean current on stationary body -> v_r = (-1, 0, 0)
    # Upstream face is -x face at x = -w/2 = -1.0; drag pushes body downstream in +x.
    speed = 1.2
    hydro.set_ocean_current((speed, 0.0, 0.0))
    _, f_drag, f_lift, _, _, _, _, cop = _run_kernel(hydro, pos, q_id, zeros, zeros, zeros, zeros)

    expected_cop_upstream = torch.tensor([[-w / 2.0, 0.0, -5.0]], dtype=torch.float32, device=DEVICE)
    expected_drag_x = 0.5 * rho * (speed ** 2) * cd * (d * h) + c_lin * speed
    assert torch.allclose(cop, expected_cop_upstream, atol=1e-5), f"Expected upstream CoP {expected_cop_upstream}, got {cop}"
    assert math.isclose(f_drag[0, 0].item(), expected_drag_x, rel_tol=1e-5, abs_tol=1e-3)
    assert torch.allclose(f_drag[0, 1:], torch.zeros(2, device=DEVICE), atol=1e-5)
    assert torch.allclose(f_lift, zeros, atol=1e-5)

    # Nose-up link (pitched -20 deg about +y so local +x tilts toward +z, local +z tilts toward -x)
    # sitting stationary in an oncoming -x current (flow toward -x => v_r = (+1, 0, 0))
    # must experience upward (+z) lift.
    q_nose_up = _quat_from_axis_angle_xyzw((0.0, 1.0, 0.0), -math.radians(20.0))
    quat_pitch = torch.tensor([q_nose_up], dtype=torch.float32, device=DEVICE)
    hydro.set_ocean_current((-1.0, 0.0, 0.0))
    _, f_drag_p, f_lift_p, _, _, _, _, _ = _run_kernel(hydro, pos, quat_pitch, zeros, zeros, zeros, zeros)
    assert f_drag_p[0, 0].item() < 0.0, "Drag should push downstream (-x)"
    assert f_lift_p[0, 2].item() > 0.0, f"Nose-up link in oncoming -x current should get +z lift, got {f_lift_p}"


# ---------------------------------------------------------------------------
# 4. Munk Moment, Isotropic Cancellation & Full 6-DOF Fossen Added Mass
# ---------------------------------------------------------------------------
def test_munk_moment_and_fossen_added_mass():
    _require_cuda()
    w, d, h = 2.0, 0.4, 0.4
    rho = 1025.0
    lm_c, am_c = 0.5, 0.25

    hydro = WarpHydrodynamicsWrapper(
        width=w, depth=d, height=h,
        linear_drag_coefficient=0.0, angular_drag_coefficient=0.0,
        linear_damping=0.0, angular_damping=0.0,
        linear_mass_coeff=lm_c, angular_mass_coeff=am_c,
        lift_coefficient=0.0,
        water_density=rho,
        num_envs=1, device=DEVICE,
    )

    pos = torch.tensor([[0.0, 0.0, -5.0]], dtype=torch.float32, device=DEVICE)
    q_id = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE)
    zeros = torch.zeros((1, 3), dtype=torch.float32, device=DEVICE)

    # 4a. Stationary elongated body at 45 deg to steady horizontal current (-1, -1, 0)
    # -> v_r = (+1, +1, 0), M_Ay > M_Ax => destabilizing clockwise yaw torque (tau_z < 0)
    vc_oblique = np.array([-1.0, -1.0, 0.0], dtype=np.float64)
    hydro.set_ocean_current(vc_oblique)
    _, _, _, _, f_am, t_am, _, _ = _run_kernel(hydro, pos, q_id, zeros, zeros, zeros, zeros)

    ref_f, ref_t = _reference_fossen_added_mass(
        w, d, h, rho, lm_c, am_c, np.array([0.0, 0.0, 0.0, 1.0]),
        np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), vc_oblique
    )
    assert t_am[0, 2].item() < -10.0, f"Expected strong negative Munk yaw torque, got {t_am}"
    assert torch.allclose(f_am[0], torch.tensor(ref_f, dtype=torch.float32, device=DEVICE), atol=1e-4)
    assert torch.allclose(t_am[0], torch.tensor(ref_t, dtype=torch.float32, device=DEVICE), rtol=1e-5, atol=1e-3)

    # Purely axial or purely broadside flow -> zero Munk torque
    for vc_aligned in ((-1.5, 0.0, 0.0), (0.0, -1.5, 0.0)):
        hydro.set_ocean_current(vc_aligned)
        _, _, _, _, _, t_aligned, _, _ = _run_kernel(hydro, pos, q_id, zeros, zeros, zeros, zeros)
        assert torch.allclose(t_aligned, zeros, atol=1e-5)

    # Rotational equivariance of Munk moment: rotating body & current by q rotates tau by q
    q_rot = _quat_from_axis_angle_xyzw((0.2, -0.5, 0.8), 1.1)
    R_rot = _quat_to_rotmat_xyzw(q_rot)
    hydro.set_ocean_current(R_rot @ vc_oblique)
    _, _, _, _, _, t_rot, _, _ = _run_kernel(
        hydro, pos, torch.tensor([q_rot], dtype=torch.float32, device=DEVICE),
        zeros, zeros, zeros, zeros,
    )
    expected_t_rot = torch.tensor(R_rot @ ref_t, dtype=torch.float32, device=DEVICE)
    assert torch.allclose(t_rot[0], expected_t_rot, rtol=1e-5, atol=1e-3)

    # 4b. General 6-DOF analytical Fossen check (arbitrary orientation, v, v_c, omega, a, alpha)
    lin_vel_np = np.array([0.7, -0.4, 0.2], dtype=np.float64)
    ang_vel_np = np.array([-0.5, 0.8, -0.6], dtype=np.float64)
    lin_acc_np = np.array([1.2, -0.9, 0.4], dtype=np.float64)
    ang_acc_np = np.array([0.3, -0.7, 0.5], dtype=np.float64)
    curr_np = np.array([-0.6, 0.35, -0.15], dtype=np.float64)

    hydro.set_ocean_current(curr_np)
    _, _, _, _, f_6d, t_6d, _, _ = _run_kernel(
        hydro,
        pos,
        torch.tensor([q_rot], dtype=torch.float32, device=DEVICE),
        torch.tensor([lin_vel_np], dtype=torch.float32, device=DEVICE),
        torch.tensor([ang_vel_np], dtype=torch.float32, device=DEVICE),
        torch.tensor([lin_acc_np], dtype=torch.float32, device=DEVICE),
        torch.tensor([ang_acc_np], dtype=torch.float32, device=DEVICE),
    )
    ref_f6, ref_t6 = _reference_fossen_added_mass(
        w, d, h, rho, lm_c, am_c, q_rot, lin_vel_np, ang_vel_np, lin_acc_np, ang_acc_np, curr_np
    )
    assert torch.allclose(f_6d[0], torch.tensor(ref_f6, dtype=torch.float32, device=DEVICE), rtol=1e-5, atol=1e-3)
    assert torch.allclose(t_6d[0], torch.tensor(ref_t6, dtype=torch.float32, device=DEVICE), rtol=1e-5, atol=1e-3)

    # 4c. Isotropic cube spinning in constant straight-line motion -> zero phantom Coriolis/Munk
    cube = WarpHydrodynamicsWrapper(
        width=1.0, depth=1.0, height=1.0,
        linear_drag_coefficient=0.0, angular_drag_coefficient=0.0,
        linear_damping=0.0, angular_damping=0.0,
        linear_mass_coeff=0.5, angular_mass_coeff=0.5,
        lift_coefficient=0.0,
        num_envs=1, device=DEVICE,
        ocean_current=(-0.4, 0.7, -0.2),
    )
    _, _, _, _, f_iso, t_iso, _, _ = _run_kernel(
        cube,
        pos,
        torch.tensor([q_rot], dtype=torch.float32, device=DEVICE),
        torch.tensor([lin_vel_np], dtype=torch.float32, device=DEVICE),
        torch.tensor([ang_vel_np], dtype=torch.float32, device=DEVICE),
        zeros,
        zeros,
    )
    assert torch.allclose(f_iso, zeros, atol=1e-4), f"Phantom linear Coriolis on isotropic cube: {f_iso}"
    assert torch.allclose(t_iso, zeros, atol=1e-4), f"Phantom angular Coriolis on isotropic cube: {t_iso}"


# ---------------------------------------------------------------------------
# 5. Batched Per-Env RL Randomization, Partial Resets & CUDA Graph Stability
# ---------------------------------------------------------------------------
def test_batched_rl_randomization_and_cuda_graph_stability():
    _require_cuda()
    num_envs = 16
    num_links = 19  # Matches SILVER2 link count
    total_bodies = num_envs * num_links

    # Heterogenous per-link dimensions to verify env-major indexing alignment
    widths = [0.26] + [0.06] * 6 + [0.06] * 6 + [0.06] * 6
    depths = [0.26] + [0.06] * 6 + [0.09] * 6 + [0.09] * 6
    heights = [0.27] + [0.09] * 6 + [0.06] * 6 + [0.06] * 6

    hydro = WarpHydrodynamicsWrapper(
        width=widths, depth=depths, height=heights,
        linear_drag_coefficient=1.2, angular_drag_coefficient=0.05,
        linear_damping=0.5, angular_damping=0.05,
        linear_mass_coeff=0.25, angular_mass_coeff=0.1,
        lift_coefficient=0.1,
        num_envs=num_envs, device=DEVICE,
        ocean_current=(0.4, -0.2, 0.0),
    )

    # Identical submerged pose & velocity across all environments
    pos = torch.tensor([[0.0, 0.0, -3.0]], dtype=torch.float32, device=DEVICE).repeat(total_bodies, 1)
    q_yaw = _quat_from_axis_angle_xyzw((0.0, 0.0, 1.0), 0.5)
    quat = torch.tensor([q_yaw], dtype=torch.float32, device=DEVICE).repeat(total_bodies, 1)
    lin_vel = torch.tensor([[0.15, -0.05, 0.0]], dtype=torch.float32, device=DEVICE).repeat(total_bodies, 1)
    ang_vel = torch.zeros((total_bodies, 3), dtype=torch.float32, device=DEVICE)
    lin_acc = torch.zeros((total_bodies, 3), dtype=torch.float32, device=DEVICE)
    ang_acc = torch.zeros((total_bodies, 3), dtype=torch.float32, device=DEVICE)

    # 5a. First launch captures CUDA Graph; verify constant current produces identical env outputs
    out_const = _run_kernel(hydro, pos, quat, lin_vel, ang_vel, lin_acc, ang_acc)
    assert hydro.graph is not None, "CUDA Graph should be captured on first launch"
    graph_id = id(hydro.graph)
    buf_ptr = hydro.t_in_current_vel.data_ptr()

    drag_const_env = out_const[1].view(num_envs, num_links, 3)
    for e in range(1, num_envs):
        assert torch.allclose(drag_const_env[e], drag_const_env[0], atol=1e-6)

    # 5b. Full per-env RL randomization
    min_v = (-0.8, -0.6, -0.1)
    max_v = (0.8, 0.6, 0.1)
    gen = torch.Generator(device=DEVICE).manual_seed(12345)
    sampled_all = hydro.randomize_ocean_current(min_v, max_v, generator=gen)

    assert sampled_all.shape == (num_envs, 3)
    assert hydro.t_in_current_vel.data_ptr() == buf_ptr, "Buffer pointer changed (breaks CUDA Graph)"
    assert torch.allclose(hydro.get_ocean_current(), sampled_all)

    # Verify every link within env e shares env e's sampled current vector
    cur_env_view = hydro.t_in_current_vel.view(num_envs, num_links, 3)
    for e in range(num_envs):
        assert torch.allclose(cur_env_view[e], sampled_all[e].unsqueeze(0).expand(num_links, 3))

    # Bounds check
    t_min = torch.tensor(min_v, dtype=torch.float32, device=DEVICE)
    t_max = torch.tensor(max_v, dtype=torch.float32, device=DEVICE)
    assert (sampled_all >= t_min - 1e-6).all() and (sampled_all <= t_max + 1e-6).all()

    out_rand = _run_kernel(hydro, pos, quat, lin_vel, ang_vel, lin_acc, ang_acc)
    assert id(hydro.graph) == graph_id, "CUDA Graph should be replayed, not recaptured"

    # Cross-check a couple of environments against an independent single-env wrapper
    single_ref = WarpHydrodynamicsWrapper(
        width=widths, depth=depths, height=heights,
        linear_drag_coefficient=1.2, angular_drag_coefficient=0.05,
        linear_damping=0.5, angular_damping=0.05,
        linear_mass_coeff=0.25, angular_mass_coeff=0.1,
        lift_coefficient=0.1,
        num_envs=1, device=DEVICE,
    )
    for check_env in (0, 7, 15):
        single_ref.set_ocean_current(sampled_all[check_env])
        ref_out = _run_kernel(
            single_ref,
            pos[:num_links], quat[:num_links], lin_vel[:num_links],
            ang_vel[:num_links], lin_acc[:num_links], ang_acc[:num_links],
        )
        for t_batch, t_single in zip(out_rand, ref_out):
            env_slice = t_batch.view(num_envs, num_links, 3)[check_env]
            assert torch.allclose(env_slice, t_single, rtol=1e-5, atol=1e-5)

    # 5c. Partial RL episode reset: only reset_ids should change
    reset_ids = torch.tensor([1, 5, 11, 15], dtype=torch.long, device=DEVICE)
    keep_ids = [e for e in range(num_envs) if e not in set(reset_ids.tolist())]

    currents_before = hydro.get_ocean_current().clone()
    drag_before = out_rand[1].view(num_envs, num_links, 3).clone()
    am_torque_before = out_rand[5].view(num_envs, num_links, 3).clone()

    sampled_subset = hydro.randomize_ocean_current(min_v, max_v, env_ids=reset_ids, generator=gen)
    assert sampled_subset.shape == (len(reset_ids), 3)
    assert hydro.t_in_current_vel.data_ptr() == buf_ptr

    out_after_reset = _run_kernel(hydro, pos, quat, lin_vel, ang_vel, lin_acc, ang_acc)
    assert id(hydro.graph) == graph_id

    currents_after = hydro.get_ocean_current()
    drag_after = out_after_reset[1].view(num_envs, num_links, 3)
    am_torque_after = out_after_reset[5].view(num_envs, num_links, 3)

    for e in keep_ids:
        assert torch.equal(currents_after[e], currents_before[e]), f"Non-reset env {e} current mutated!"
        assert torch.equal(drag_after[e], drag_before[e]), f"Non-reset env {e} drag mutated!"
        assert torch.equal(am_torque_after[e], am_torque_before[e]), f"Non-reset env {e} added-mass torque mutated!"

    for idx, e in enumerate(reset_ids.tolist()):
        assert torch.allclose(currents_after[e], sampled_subset[idx])
        assert not torch.allclose(drag_after[e], drag_before[e], atol=1e-4), f"Reset env {e} drag did not update!"

    # 5d. Empty env_ids must be a safe no-op
    empty_out = hydro.randomize_ocean_current(min_v, max_v, env_ids=[])
    assert empty_out.shape == (0, 3)
    assert torch.equal(hydro.get_ocean_current(), currents_after)

    # 5e. Optional per-step t_current_velocity override in calculate_hydrodynamic_forces
    override_curr = torch.tensor([0.25, 0.25, 0.0], dtype=torch.float32, device=DEVICE)
    out_override = _run_kernel(
        hydro, pos, quat, lin_vel, ang_vel, lin_acc, ang_acc, current_vel=override_curr
    )
    assert hydro.t_in_current_vel.data_ptr() == buf_ptr
    assert id(hydro.graph) == graph_id
    drag_override_env = out_override[1].view(num_envs, num_links, 3)
    for e in range(1, num_envs):
        assert torch.allclose(drag_override_env[e], drag_override_env[0], atol=1e-6)


# ---------------------------------------------------------------------------
# 6. SILVER2 JSON Config Integration (if silver_hydro_config.json is present)
# ---------------------------------------------------------------------------
def test_silver2_json_batched_loading():
    _require_cuda()
    candidates = [
        PROJECT_ROOT / "src" / "scripts" / "hydrodynamics" / "silver_hydro_config.json",
        THIS_FILE.parent / "silver_hydro_config.json",
    ]
    json_path = next((p for p in candidates if p.is_file()), None)
    if json_path is None:
        msg = "silver_hydro_config.json not found near test script or repo root."
        try:
            import pytest
            pytest.skip(msg)
        except ImportError:
            print(f"  [SKIP] test_silver2_json_batched_loading ({msg})")
            return

    link_paths = (
        ["/World/SILVER2/Body"]
        + [f"/World/SILVER2/Coxa_{i}" for i in range(6)]
        + [f"/World/SILVER2/Femur_{i}" for i in range(6)]
        + [f"/World/SILVER2/Tibia_{i}" for i in range(6)]
    )
    num_envs = 8
    hydro = WarpHydrodynamicsWrapper.from_json(
        json_path=json_path,
        link_names_or_paths=link_paths,
        num_envs=num_envs,
        device=DEVICE,
    )
    assert hydro.num_links == 19
    assert hydro.total_bodies == num_envs * 19

    # Explicit override beats JSON default
    override = (0.35, -0.15, 0.0)
    hydro_override = WarpHydrodynamicsWrapper.from_json(
        json_path=json_path,
        link_names_or_paths=link_paths,
        num_envs=num_envs,
        device=DEVICE,
        ocean_current=override,
    )
    expected = torch.tensor(override, dtype=torch.float32, device=DEVICE).unsqueeze(0).expand(num_envs, 3)
    assert torch.allclose(hydro_override.get_ocean_current(), expected)

    # Execute a batched GPU step and verify all outputs are finite and non-zero drag occurs
    n = hydro_override.total_bodies
    pos = torch.tensor([[0.0, 0.0, -2.0]], dtype=torch.float32, device=DEVICE).repeat(n, 1)
    quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=DEVICE).repeat(n, 1)
    zeros = torch.zeros((n, 3), dtype=torch.float32, device=DEVICE)

    outputs = _run_kernel(hydro_override, pos, quat, zeros, zeros, zeros, zeros)
    for t in outputs:
        assert torch.isfinite(t).all()
    # Body pushed downstream (+x, -y) by current (+0.35, -0.15, 0)
    f_drag = outputs[1]
    assert (f_drag[:, 0] > 0.0).all()
    assert (f_drag[:, 1] < 0.0).all()