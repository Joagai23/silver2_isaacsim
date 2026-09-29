import torch
import numpy as np
from typing import Tuple, Union
from silver2_constants.isaac_constants import SILVER2_MOUNTS, SILVER2_LINKS, GAIT_CONFIGS

class OmnidirectionalGaitController:
    def __init__(
        self,
        gait_type: str = "tripod",
        step_height: float = 0.10,
        stance_height: float = 0.30,
        gait_period: float = 2.0,
        device: str = "cuda:0"
    ):
        self.step_height = step_height
        self.stance_height = stance_height
        self.T = gait_period
        self.device = device

        # Unpack robot link lengths from standardized constants
        self.link_lengths = SILVER2_LINKS
        if len(self.link_lengths) == 3:
            self.l0, self.l2, self.l3 = self.link_lengths
            self.l1 = 0.0  # Zero lateral offset in SILVER2 USD CAD model
        elif len(self.link_lengths) == 4:
            self.l0, self.l1, self.l2, self.l3 = self.link_lengths

        # Joint limits in radians [coxa, femur, tibia]
        self.q_min = torch.tensor([-np.pi / 2.0, -np.pi / 2.0, -np.pi / 4.0], device=self.device)
        self.q_max = torch.tensor([ np.pi / 2.0,  np.pi / 2.0,  0.75 * np.pi], device=self.device)

        # Leg chirality / mirroring vector
        self.mir = torch.tensor([1.0, 1.0, -1.0, -1.0, -1.0, 1.0], device=self.device)

        # Duty cycle: 0.5 for tripod, 2/3 for tetrapod, 5/6 for wave
        if gait_type not in GAIT_CONFIGS:
            raise ValueError(f"Unknown gait type: {gait_type}. Available: {list(GAIT_CONFIGS.keys())}")
        config = GAIT_CONFIGS[gait_type]
        self.beta = config["epsilon"]
        self.phases = torch.tensor(config["phases"], dtype=torch.float32, device=self.device)

        # Ingest nominal mounting coordinates from constants
        leg_names = list(SILVER2_MOUNTS.keys())
        self.mount_pos = torch.tensor(
            [SILVER2_MOUNTS[name]["pos"] for name in leg_names],
            dtype=torch.float32, device=self.device
        )
        self.mount_yaw = torch.tensor(
            [SILVER2_MOUNTS[name]["yaw"] for name in leg_names],
            dtype=torch.float32, device=self.device
        )

    def body_to_local_frame(self, p_body: torch.Tensor) -> torch.Tensor:
        """
        Transforms foot positions from Body (Chassis) Frame [..., 6, 3]
        to Local Coxa Mount Frames [..., 6, 3].
        """
        dx = p_body[..., 0] - self.mount_pos[:, 0]
        dy = p_body[..., 1] - self.mount_pos[:, 1]
        dz = p_body[..., 2] - self.mount_pos[:, 2]

        c_y = torch.cos(-self.mount_yaw)
        s_y = torch.sin(-self.mount_yaw)

        px = dx * c_y - dy * s_y
        py = dx * s_y + dy * c_y
        pz = dz
        return torch.stack([px, py, pz], dim=-1)

    def local_to_body_frame(self, p_local: torch.Tensor) -> torch.Tensor:
        """
        Transforms foot positions from Local Coxa Mount Frames [..., 6, 3]
        to Body (Chassis) Frame [..., 6, 3].
        """
        c_y = torch.cos(self.mount_yaw)
        s_y = torch.sin(self.mount_yaw)

        x_b = self.mount_pos[:, 0] + p_local[..., 0] * c_y - p_local[..., 1] * s_y
        y_b = self.mount_pos[:, 1] + p_local[..., 0] * s_y + p_local[..., 1] * c_y
        z_b = self.mount_pos[:, 2] + p_local[..., 2]
        return torch.stack([x_b, y_b, z_b], dim=-1)

    def batch_leg_inv_kine(self, feet_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse Kinematics: maps local Cartesian feet [..., 6, 3] to joint targets [..., 6, 3]
        and an admissibility mask.
        """
        has_batch = (feet_local.dim() == 3)
        if not has_batch:
            feet_local = feet_local.unsqueeze(0)

        x = feet_local[..., 0]
        y = feet_local[..., 1]
        z = feet_local[..., 2]

        L_sq = x**2 + y**2
        L = torch.sqrt(torch.clamp(L_sq, min=1e-8))

        valid_radial = L >= abs(self.l1)
        planar_span_sq = torch.clamp(L_sq - self.l1**2, min=0.0)
        planar_span = torch.sqrt(planar_span_sq)

        beta = torch.atan2(y, x)
        gamma = torch.atan2(torch.tensor(self.l1, device=self.device), planar_span)
        q1 = beta - self.mir * gamma

        x3 = planar_span - self.l0
        y3 = z
        a_sq = x3**2 + y3**2
        a = torch.sqrt(torch.clamp(a_sq, min=1e-8))

        cos_q2_arg = (self.l2**2 + a_sq - self.l3**2) / (2.0 * self.l2 * a)
        cos_q3_arg = (self.l2**2 + self.l3**2 - a_sq) / (2.0 * self.l2 * self.l3)

        valid_triangle = (cos_q2_arg.abs() <= 1.0) & (cos_q3_arg.abs() <= 1.0)

        q2_cos = torch.clamp(cos_q2_arg, -1.0, 1.0)
        q3_cos = torch.clamp(cos_q3_arg, -1.0, 1.0)

        q2 = torch.acos(q2_cos) + torch.atan2(y3, x3)
        q3 = torch.pi - torch.acos(q3_cos)

        q = torch.stack([q1, q2, q3], dim=-1)

        within_limits = (q >= self.q_min).all(dim=-1) & (q <= self.q_max).all(dim=-1)
        admissible = valid_radial & valid_triangle & within_limits

        return (q, admissible) if has_batch else (q.squeeze(0), admissible.squeeze(0))

    def batch_leg_for_kine(self, q: torch.Tensor) -> torch.Tensor:
        """
        Forward Kinematics: maps joint angles [..., 6, 3] to local Cartesian feet [..., 6, 3].
        """
        has_batch = (q.dim() == 3)
        if not has_batch:
            q = q.unsqueeze(0)

        q1 = q[..., 0]
        q2 = q[..., 1]
        q3 = q[..., 2]

        r_planar = self.l0 + self.l2 * torch.cos(q2) + self.l3 * torch.cos(q2 - q3)
        z = self.l2 * torch.sin(q2) + self.l3 * torch.sin(q2 - q3)

        lat_offset = self.mir * self.l1
        x = r_planar * torch.cos(q1) - lat_offset * torch.sin(q1)
        y = r_planar * torch.sin(q1) + lat_offset * torch.cos(q1)

        feet_pos = torch.stack([x, y, z], dim=-1)
        return feet_pos if has_batch else feet_pos.squeeze(0)

    def admissible(self, feet_local: torch.Tensor) -> torch.Tensor:
        """
        Direct analytical reachability check for foot positions in local mount frames [..., 6, 3].
        """
        _, is_admissible = self.batch_leg_inv_kine(feet_local)
        return is_admissible

    def compute_foot_trajectories(
        self,
        t: float,
        vx: float,
        vy: float,
        wz: float,
        default_feet_body: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes instantaneous 3D Cartesian foot targets in Body Frame for all 6 legs.
        """
        tau = torch.remainder(t / self.T + self.phases, 1.0)

        # Linear + yaw displacement per leg
        sx_lin = vx * self.T
        sy_lin = vy * self.T
        sx_yaw = -wz * self.mount_pos[:, 1] * self.T
        sy_yaw =  wz * self.mount_pos[:, 0] * self.T

        sx_total = sx_lin + sx_yaw
        sy_total = sy_lin + sy_yaw

        is_stance = tau < self.beta
        u_stance = tau / self.beta
        u_swing = (tau - self.beta) / (1.0 - self.beta)

        dx = torch.where(
            is_stance,
            (0.5 - u_stance) * sx_total,
            (u_swing - 0.5) * sx_total
        )
        dy = torch.where(
            is_stance,
            (0.5 - u_stance) * sy_total,
            (u_swing - 0.5) * sy_total
        )
        dz = torch.where(
            is_stance,
            torch.zeros(6, device=self.device),
            self.step_height * torch.sin(torch.pi * u_swing)
        )

        feet_targets = default_feet_body.clone()
        feet_targets[:, 0] += dx
        feet_targets[:, 1] += dy
        feet_targets[:, 2] += dz

        return feet_targets

    def change_configuration(
        self,
        q_des: torch.Tensor,
        q_cur: torch.Tensor,
        gait_t: float = 3.0,
        lift_height: float = 0.05
    ) -> Union[bool, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, float]]:
        """
        Statically stable 3-phase posture transition planner.
        """
        q_des = torch.as_tensor(q_des, dtype=torch.float32, device=self.device).reshape(6, 3)
        q_cur = torch.as_tensor(q_cur, dtype=torch.float32, device=self.device).reshape(6, 3)

        if torch.allclose(q_des, q_cur, atol=1e-4):
            print("Same pose as before")
            return True

        # Forward kinematics for all 6 legs in local mount frames
        op = self.batch_leg_for_kine(q_cur)
        np_pos = self.batch_leg_for_kine(q_des)

        nh = np_pos[:, 2].min()
        oh = op[:, 2].min()

        if torch.abs(nh - oh) < 0.001:
            n1 = 0
            n2 = 30
        else:
            n1 = 60
            n2 = 30

        n = n1 + 2 * n2
        delta_t = gait_t / n
        t = torch.linspace(0.0, 1.0, n2, device=self.device)

        Qq = torch.zeros((18, n), dtype=torch.float32, device=self.device)
        Qdot = torch.zeros((18, n), dtype=torch.float32, device=self.device)
        admiss = torch.ones(6, dtype=torch.bool, device=self.device)

        tripod_1_legs = [0, 2, 4]

        for i in range(6):
            op_i = op[i]
            np_i = np_pos[i]

            # Phase 1: Vertical adjustment
            if n1 > 0:
                xi_1 = op_i[0].repeat(n1)
                yi_1 = op_i[1].repeat(n1)
                zi_1 = torch.linspace(op_i[2], np_i[2], n1, device=self.device)
            else:
                xi_1 = torch.empty(0, device=self.device)
                yi_1 = torch.empty(0, device=self.device)
                zi_1 = torch.empty(0, device=self.device)

            # Phase 2: Tripod 1 swings, Tripod 2 grounded
            if i in tripod_1_legs:
                xi_2 = torch.linspace(op_i[0], np_i[0], n2, device=self.device)
                yi_2 = torch.linspace(op_i[1], np_i[1], n2, device=self.device)
                zi_2 = np_i[2] + lift_height * torch.sin(torch.pi * t)
            else:
                xi_2 = op_i[0].repeat(n2)
                yi_2 = op_i[1].repeat(n2)
                zi_2 = np_i[2].repeat(n2)

            # Phase 3: Tripod 1 grounded, Tripod 2 swings
            if i in tripod_1_legs:
                xi_3 = np_i[0].repeat(n2)
                yi_3 = np_i[1].repeat(n2)
                zi_3 = np_i[2].repeat(n2)
            else:
                xi_3 = torch.linspace(op_i[0], np_i[0], n2, device=self.device)
                yi_3 = torch.linspace(op_i[1], np_i[1], n2, device=self.device)
                zi_3 = np_i[2] + lift_height * torch.sin(torch.pi * t)

            T_i = torch.stack([
                torch.cat([xi_1, xi_2, xi_3]),
                torch.cat([yi_1, yi_2, yi_3]),
                torch.cat([zi_1, zi_2, zi_3])
            ], dim=0)  # [3, n]

            # Replicate coordinates across all 6 legs to batch-evaluate leg i
            eval_pts = T_i.T.unsqueeze(1).repeat(1, 6, 1)  # [n, 6, 3]
            q_traj, valid_mask = self.batch_leg_inv_kine(eval_pts)

            leg_valid = valid_mask[:, i].all().item()
            admiss[i] = leg_valid

            if leg_valid:
                Qq[3 * i : 3 * i + 3, :] = q_traj[:, i, :].T
                Qdot[3 * i : 3 * i + 3, 0] = torch.zeros(3, device=self.device)
                Qdot[3 * i : 3 * i + 3, 1:n] = torch.abs(
                    (Qq[3 * i : 3 * i + 3, 1:n] - Qq[3 * i : 3 * i + 3, 0 : n - 1]) / delta_t
                )
            else:
                print(f"Non admissible trajectory for leg_id: {i}")

        return Qq, Qdot, admiss, n, delta_t