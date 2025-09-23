"""
TD3-based Active Suspension Controller for Enhanced Nonlinear Quarter-Car Model

"""

from typing import Tuple, Dict, List
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random, pickle, warnings
from scipy.linalg import solve_continuous_are  # CARE

warnings.filterwarnings('ignore')
np.random.seed(42); torch.manual_seed(42); random.seed(42)

# ============ Enhanced config ============
MAX_STEPS_PER_EPISODE = 2500  
FAST_PREVIEW_POINTS   = 12    
EXPLORATION_EPISODES  = 180   
CURRICULUM_MILESTONES = (30, 60, 120)  
LEARNING_STARTS_STEPS = 25_000

# Stabilization & optimization
POLICY_NOISE_BASE = 0.10
NOISE_CLIP_BASE   = 0.25
NOISE_DECAY_ITERS = 400_000
GRAD_CLIP_NORM    = 1.0
WEIGHT_DECAY      = 1e-4

BC_MAX = 0.25; BC_MIN = 0.03; BC_FADE_ITERS = 400_000; SAFETY_BC_MAX = 0.30
DEMO_FLOOR_BASE = 0.20; SAFETY_DEMO_FLOOR = 0.45
ADAPTIVE_DEMO_FLOOR = DEMO_FLOOR_BASE; DEMO_FLOOR_TARGET = DEMO_FLOOR_BASE

SAFETY_ENABLED = True; SAFETY_FORCE_BC = False
SAFETY_COOLDOWN_EPISODES = 60; SAFETY_DURATION_EPISODES = 60; ROLLBACK_WINDOW = 3
ROLLBACK_DELTA_BASE = 40.0; ROLLBACK_DELTA_MIN = 15.0; ROLLBACK_TIGHTEN_EP = 700

# Late-phase "push window"
PUSH_EP_START = 400
PUSH_ITERS = 100_000
PUSH_ACTOR_LR = 1.0e-4
PUSH_CRITIC_LR = 2.0e-4
BIAS_HIGH_SPEED_AFTER_EP = 400  # bias 55 km/h a bit

def dynamic_rollback_delta(ep: int) -> float:
    t = min(1.0, max(0.0, ep / ROLLBACK_TIGHTEN_EP))
    return (1.0 - t) * ROLLBACK_DELTA_BASE + t * ROLLBACK_DELTA_MIN

def smooth_update(current: float, target: float, tau: float) -> float:
    return (1.0 - tau) * current + tau * target

# =========================
# Enhanced Nonlinear Quarter-car environment
# =========================
class NonlinearQuarterCarModel:
    """
    Heavily nonlinear quarter-car model with multiple nonlinear effects
    designed to create scenarios where ML can outperform LQR.
    """

    # ---- Enhanced physical parameters ----
    ms_base: float = 400.0      # base sprung mass [kg] - will vary with load
    mu: float = 50.0            # unsprung mass [kg]
    ks_base: float = 20_000.0   # base spring rate [N/m]
    cs_base: float = 1_800.0    # base damping [N*s/m]
    kt_base: float = 180_000.0  # base tire stiffness [N/m]
    dt: float = 0.001

    # ---- Stronger nonlinearities ----
    k_nl: float = 5e6           # 5x stronger cubic hardening
    k_progressive: float = 2e5  # progressive spring term
    bump_limit: float = 0.12    # larger travel (±12 cm)
    k_bump: float = 8.0e6       # much stiffer bumpstops

    # Hysteretic friction
    friction_breakaway: float = 800.0    # breakaway friction [N]
    friction_coulomb: float = 600.0      # sliding friction [N]
    friction_viscous: float = 50.0       # viscous friction [N*s/m]
    stiction_velocity: float = 0.001     # stiction threshold [m/s]
    friction_tau: float = 0.02           # <-- time constant for friction "state" (stable)

    # Variable damping
    damping_freq_dep: float = 0.3        # frequency-dependent factor
    damping_amp_dep: float = 0.5         # amplitude-dependent factor

    # Velocity-squared drag (aero-like damper term)
    drag_coeff: float = 2.0              # [N*s²/m²]

    # Load variation
    load_variation_range: float = 0.6    # ±60% mass variation

    # Speed-dependent tire
    tire_speed_factor: float = 0.02      # tire stiffness changes with speed

    # Enhanced actuator with smooth saturation
    max_force: float = 3000.0            # higher max force
    tau_actuator: float = 0.008          # slightly faster
    saturation_smooth: float = 500.0     # smooth saturation zone [N]

    # Enhanced road generation
    road_len_m: float = 1200.0           # longer roads
    road_dx: float = 0.015               # higher resolution
    preview_distance: float = 15.0       # longer preview
    preview_points: int = FAST_PREVIEW_POINTS

    def __init__(self):
        # Dynamic state [z_s, z_s_dot, z_u, z_u_dot, friction_state]
        self.state = np.zeros(5, dtype=np.float64)
        self.u_applied = 0.0
        self.last_action = 0.0
        self.last_accel = 0.0

        # Enhanced state tracking
        self.velocity_history = deque(maxlen=10)    # for frequency estimation
        self.amplitude_estimate = 0.0               # for amplitude-dependent damping

        # Variable parameters
        self.current_load_factor = 1.0
        self.ms = self.ms_base
        self.ks = self.ks_base
        self.cs = self.cs_base
        self.kt = self.kt_base

        self.s_pos = 0.0
        self.speed = 25.0  

        self.road_x = None
        self.road_profile = self.generate_enhanced_road()
        self.r_u_scale = 1.0

        # LQR baseline (computed on nominal base params)
        self.K_lqr = self._compute_baseline_lqr_gain()

    # -------- Enhanced road generation --------
    def generate_enhanced_road(self, class_mix: List[str] = None) -> np.ndarray:
        """Generate mixed-class road with harsh segments"""
        if class_mix is None:
            class_mix = ['C', 'D', 'D', 'C', 'E']  # Include very rough E-class

        x = np.arange(0.0, self.road_len_m, self.road_dx, dtype=np.float64)
        self.road_x = x

        # Enhanced roughness including E-class (4x Class C)
        Gd0_map = {'A':16e-6, 'B':64e-6, 'C':256e-6, 'D':1024e-6, 'E':4096e-6}

        segment_len = len(x) // len(class_mix)
        full_road = np.zeros_like(x)

        for i, road_class in enumerate(class_mix):
            start_idx = i * segment_len
            end_idx = min((i + 1) * segment_len, len(x))
            segment_x = x[start_idx:end_idx]
            if len(segment_x) == 0:
                continue

            Gd0 = Gd0_map.get(road_class.upper(), 1024e-6)
            n0 = 0.1
            n_min, n_max = 0.01, min(12.0, 0.45/self.road_dx)
            Nf = 512

            n = np.linspace(n_min, n_max, Nf)
            Gd = Gd0 * (n / n0) ** (-2.0)
            dn = n[1] - n[0]
            amp = np.sqrt(2.0 * Gd * dn)
            phases = np.random.uniform(0.0, 2.0*np.pi, size=Nf)

            segment_road = np.zeros_like(segment_x)
            two_pi_seg = 2.0*np.pi*segment_x
            for j in range(Nf):
                segment_road += amp[j] * np.cos(two_pi_seg * n[j] + phases[j])

            full_road[start_idx:end_idx] = segment_road

        return full_road.astype(np.float64)

    def road_height_at(self, s: float) -> float:
        if self.road_x is None or len(self.road_profile) == 0:
            return 0.0
        s = float(s)
        if s <= self.road_x[0]: return float(self.road_profile[0])
        if s >= self.road_x[-1]: return float(self.road_profile[-1])
        return float(np.interp(s, self.road_x, self.road_profile))

    # -------- Enhanced reset with load variation --------
    def reset(self, speed: float = 25.0, road_profile: np.ndarray = None, load_factor: float = None):
        self.state[:] = 0.0
        self.u_applied = 0.0
        self.last_action = 0.0
        self.last_accel = 0.0
        self.speed = float(speed)
        self.s_pos = 0.0
        self.velocity_history.clear()
        self.amplitude_estimate = 0.0

        # Randomize load if not specified
        if load_factor is None:
            load_factor = 1.0 + np.random.uniform(-self.load_variation_range, self.load_variation_range)

        self.current_load_factor = float(np.clip(load_factor, 0.4, 1.6))
        self.ms = self.ms_base * self.current_load_factor

        # Adjust spring rate slightly with load (realistic)
        self.ks = self.ks_base * (1.0 + 0.1 * (self.current_load_factor - 1.0))

        # Speed-dependent tire stiffness
        v_ms = max(self.speed / 3.6, 1.0)
        self.kt = self.kt_base * (1.0 + self.tire_speed_factor * v_ms)
        self.cs = self.cs_base  # base damping each reset

        if road_profile is None:
            self.road_profile = self.generate_enhanced_road()
        else:
            self.road_profile = np.asarray(road_profile, dtype=np.float64)
            if self.road_x is None or len(self.road_x) != len(road_profile):
                self.road_x = np.arange(0.0, self.road_len_m, self.road_dx)[:len(road_profile)]

        return self.get_observation()

    def get_road_preview(self) -> np.ndarray:
        v = max(self.speed/3.6, 1e-6)
        Tprev = self.preview_distance / v
        t_samples = np.linspace(0.0, Tprev, self.preview_points, dtype=np.float64)
        s_samples = self.s_pos + v * t_samples
        return np.array([self.road_height_at(s) for s in s_samples], dtype=np.float32)

    def get_observation(self) -> np.ndarray:
        z_r_now = self.road_height_at(self.s_pos)
        preview = self.get_road_preview()

        # Include friction "state" and dynamics cues for the policy
        obs = np.concatenate([
            self.state[:4].astype(np.float64),             # main suspension state
            np.array([z_r_now], dtype=np.float64),         # current road
            preview.astype(np.float64),                    # road preview
            np.array([self.speed/80.0], dtype=np.float64), # normalized speed
            np.array([self.current_load_factor], dtype=np.float64),   # load info
            np.array([self.amplitude_estimate/0.1], dtype=np.float64),# dynamics info
            np.array([self.state[4]/1000.0], dtype=np.float64),       # friction state (scaled)
        ]).astype(np.float32)
        return obs

    # -------- Enhanced nonlinear forces --------
    def _enhanced_suspension_force(self, defl: float) -> float:
        """Nonlinear spring: linear + cubic + progressive + bumpstops"""
        f_linear = self.ks * defl
        f_cubic = self.k_nl * (defl ** 3)
        # Progressive term ~ |defl| * defl (softer than pure cubic but stiffening)
        f_progressive = self.k_progressive * defl * abs(defl)

        if defl > self.bump_limit:
            over = defl - self.bump_limit
            f_bump = self.k_bump * over * (1.0 + over / self.bump_limit)
        elif defl < -self.bump_limit:
            over = -defl - self.bump_limit
            f_bump = -self.k_bump * over * (1.0 + over / self.bump_limit)
        else:
            f_bump = 0.0

        return f_linear + f_cubic + f_progressive + f_bump

    def _enhanced_damping_force(self, vel: float) -> float:
        """Complex nonlinear damping with frequency & amplitude effects"""
        f_linear_damp = self.cs * vel
        f_drag = self.drag_coeff * vel * abs(vel)

        self.velocity_history.append(abs(vel))
        if len(self.velocity_history) >= 5:
            vel_variance = float(np.var(list(self.velocity_history)))
            freq_factor = 1.0 + self.damping_freq_dep * vel_variance
            amp_factor = 1.0 + self.damping_amp_dep * min(self.amplitude_estimate/0.05, 2.0)
        else:
            freq_factor = amp_factor = 1.0

        f_adaptive = self.cs * vel * freq_factor * amp_factor
        # Mix to avoid over-amplification
        return f_linear_damp + f_drag + (f_adaptive - f_linear_damp) * 0.3

    def _hysteretic_friction(self, vel: float, friction_state: float) -> float:
        """Friction force with stiction & sliding; returns instantaneous desired friction"""
        abs_vel = abs(vel)
        if abs_vel < self.stiction_velocity:
            # Stiction region: oppose motion up to breakaway
            if abs_vel == 0:
                friction_force = friction_state  # hold
            else:
                friction_force = -np.sign(vel) * self.friction_breakaway * min(abs_vel/self.stiction_velocity, 1.0)
        else:
            # Sliding: Coulomb + viscous
            coulomb = -np.sign(vel) * self.friction_coulomb
            viscous = -self.friction_viscous * vel
            friction_force = coulomb + viscous
        return friction_force

    def _smooth_actuator_saturation(self, u_cmd: float) -> float:
        """Smooth saturation to avoid sharp clipping"""
        if abs(u_cmd) <= self.max_force - self.saturation_smooth:
            return u_cmd
        excess = abs(u_cmd) - (self.max_force - self.saturation_smooth)
        saturation_ratio = excess / self.saturation_smooth
        smooth_factor = 1.0 - saturation_ratio + 0.5 * (saturation_ratio ** 2)
        smooth_factor = max(0.1, smooth_factor)
        return np.sign(u_cmd) * (self.max_force - self.saturation_smooth +
                                 self.saturation_smooth * smooth_factor)

    def _derivatives(self, x: np.ndarray, u: float, z_r: float) -> np.ndarray:
        z_s, z_s_dot, z_u, z_u_dot, friction_state = x

        # Update amplitude estimate (EMA)
        travel = abs(z_s - z_u)
        self.amplitude_estimate = 0.9 * self.amplitude_estimate + 0.1 * travel

        # Forces
        susp_defl = z_s - z_u
        susp_vel = z_s_dot - z_u_dot
        tire_defl = z_u - z_r
        tire_vel = z_u_dot

        F_spring = self._enhanced_suspension_force(susp_defl)
        F_damp   = self._enhanced_damping_force(susp_vel)
        F_fric_desired = self._hysteretic_friction(susp_vel, friction_state)

        # Friction state dynamics (FIRST-ORDER, stable)
        dfric_dt = (F_fric_desired - friction_state) / self.friction_tau
        F_friction = friction_state  # use current state as applied friction

        # Nonlinear tire with speed dependency
        F_tire = self.kt * tire_defl * (1.0 + 0.1 * (tire_defl / self.bump_limit) ** 2)
        F_tire_damp = 120.0 * tire_vel

        # Equations of motion
        z_s_ddot = (-F_spring - F_damp - F_friction + u) / self.ms
        z_u_ddot = ( F_spring + F_damp + F_friction - F_tire - F_tire_damp - u) / self.mu

        return np.array([z_s_dot, z_s_ddot, z_u_dot, z_u_ddot, dfric_dt])

    def _rk4(self, x: np.ndarray, u: float, dt: float) -> np.ndarray:
        v = self.speed / 3.6
        s = self.s_pos

        z1 = self.road_height_at(s)
        k1 = self._derivatives(x, u, z1)

        z2 = self.road_height_at(s + 0.5 * v * dt)
        k2 = self._derivatives(x + 0.5 * dt * k1, u, z2)

        z3 = self.road_height_at(s + 0.5 * v * dt)
        k3 = self._derivatives(x + 0.5 * dt * k2, u, z3)

        z4 = self.road_height_at(s + v * dt)
        k4 = self._derivatives(x + dt * k3, u, z4)

        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _apply_actuator(self, u_cmd: float) -> float:
        u_smooth = self._smooth_actuator_saturation(u_cmd)
        # Actuator lag
        self.u_applied += (self.dt/self.tau_actuator) * (u_smooth - self.u_applied)
        return self.u_applied

    # -------- Reward --------
    def _step_cost(self, z_s_ddot: float, travel: float, u: float, du: float, load_factor: float) -> float:
        # Load-aware comfort weighting
        q_acc = 8.0 * (1.0 + 0.5 * abs(load_factor - 1.0))
        q_trv = 600.0 * (1.0 + 0.3 * abs(load_factor - 1.0))
        r_u = 2e-5 * self.r_u_scale
        r_du = 2e-6
        return (q_acc * z_s_ddot**2 + q_trv * travel**2 + r_u * u**2 + r_du * du**2)

    def calculate_reward(self, u_applied: float, z_r: float) -> float:
        dz = self._derivatives(self.state, u_applied, z_r)
        z_s_ddot = float(dz[1])
        travel = float(self.state[0] - self.state[2])
        du = float(u_applied - self.last_action)

        keepout_bonus = 0.03 if abs(travel) < 0.8 * self.bump_limit else 0.0
        friction_penalty = 0.01 * (self.state[4] / 1000.0) ** 2
        load_adaptation_bonus = 0.02 if abs(z_s_ddot) < 3.0 * abs(self.current_load_factor - 1.0) + 2.0 else 0.0

        step_cost = self._step_cost(z_s_ddot, travel, u_applied, du, self.current_load_factor)
        r = -(step_cost * self.dt) + keepout_bonus + load_adaptation_bonus - friction_penalty
        return float(np.clip(r, -2000.0, 0.3))

    def step(self, u_cmd: float):
        u_applied = self._apply_actuator(float(u_cmd))

        # integrate with road varying across the step
        self.state = self._rk4(self.state, u_applied, self.dt)

        # advance along the road
        v = self.speed / 3.6
        self.s_pos += v * self.dt

        # compute accel at the *current* position
        z_r_now = self.road_height_at(self.s_pos)
        self.last_accel = float(self._derivatives(self.state, u_applied, z_r_now)[1])
        self.last_action = float(u_applied)

        done = self.s_pos >= (self.road_x[-1] - 10.0)
        r = self.calculate_reward(u_applied, z_r_now)
        return self.get_observation(), r, done

    # -------- Baseline LQR (linearized around operating point) --------
    def _compute_baseline_lqr_gain(self, q_weights: np.ndarray = None, r_weight: float = 2e-5) -> np.ndarray:
        ms, mu = self.ms_base, self.mu
        ks, cs = self.ks_base, self.cs_base
        kt, ct = self.kt_base, 120.0  # tire damping

        A = np.array([
            [0,      1,       0,      0     ],
            [-ks/ms, -cs/ms,  ks/ms,  cs/ms ],
            [0,      0,       0,      1     ],
            [ks/mu,  cs/mu,   -(ks+kt)/mu, -(cs+ct)/mu]
        ], dtype=np.float64)

        B = np.array([
            [0      ],
            [1/ms   ],
            [0      ],
            [-1/mu  ]
        ], dtype=np.float64)

        if q_weights is None:
            q_weights = np.array([2e6, 125.0, 5e5, 75.0], dtype=np.float64)

        Q = np.diag(q_weights)
        R = np.array([[r_weight]], dtype=np.float64)

        try:
            P = solve_continuous_are(A, B, Q, R)
            K = (np.linalg.inv(R) @ B.T @ P).flatten()
        except Exception as e:
            print(f"LQR computation failed: {e}")
            K = np.array([8000.0, 2000.0, -3000.0, -800.0])
        return K

    def get_lqr_action(self) -> float:
        x = self.state[:4]
        u_cmd = -float(np.dot(self.K_lqr, x))
        return float(np.clip(u_cmd, -self.max_force, self.max_force))


# =========================
# Replay Buffer
# =========================
class ReplayBuffer:
    def __init__(self, capacity: int = 1_200_000):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, s2, d, is_demo: bool = False):
        self.buffer.append((s, a, r, s2, d, is_demo))
    def sample(self, batch_size: int):
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples. Have {len(self.buffer)}, need {batch_size}")
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d, m = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(s),
            torch.FloatTensor(a).unsqueeze(-1) if a.ndim == 1 else torch.FloatTensor(a),
            torch.FloatTensor(r).unsqueeze(-1) if r.ndim == 1 else torch.FloatTensor(r),
            torch.FloatTensor(s2),
            torch.BoolTensor(d).unsqueeze(-1) if d.ndim == 1 else torch.BoolTensor(d),
            torch.BoolTensor(m).unsqueeze(-1)
        )
    def __len__(self): return len(self.buffer)


# =========================
# TD3 Networks & Agent
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        super().__init__()
        self.max_action = max_action
        self.l1 = nn.Linear(state_dim, 320)
        self.l2 = nn.Linear(320, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, action_dim)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); m.bias.data.fill_(0.01)
    def forward(self, s):
        x = F.relu(self.l1(s)); x = F.relu(self.l2(x)); x = F.relu(self.l3(x))
        return self.max_action * torch.tanh(self.l4(x))

class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 320)
        self.l2 = nn.Linear(320, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 1)
        self.l5 = nn.Linear(state_dim + action_dim, 320)
        self.l6 = nn.Linear(320, 256)
        self.l7 = nn.Linear(256, 128)
        self.l8 = nn.Linear(128, 1)
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); m.bias.data.fill_(0.01)
    def forward(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = F.relu(self.l3(q1)); q1 = self.l4(q1)
        q2 = F.relu(self.l5(sa)); q2 = F.relu(self.l6(q2)); q2 = F.relu(self.l7(q2)); q2 = self.l8(q2)
        return q1, q2
    def Q1(self, s, a):
        sa = torch.cat([s, a], 1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = F.relu(self.l3(q1)); q1 = self.l4(q1)
        return q1

class TD3:
    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=8e-5, weight_decay=WEIGHT_DECAY)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=1.5e-4, weight_decay=WEIGHT_DECAY)

        # Cosine Warm Restarts — we'll pause these during the "push window"
        self.actor_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.actor_optimizer, T_0=150_000, T_mult=2, eta_min=1e-6
        )
        self.critic_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.critic_optimizer, T_0=150_000, T_mult=2, eta_min=2e-6
        )
        self._sched_enabled = True
        self._push_until_it = None
        self._base_actor_lr = self.actor_optimizer.param_groups[0]['lr']
        self._base_critic_lr = self.critic_optimizer.param_groups[0]['lr']

        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005

        self.policy_noise0 = POLICY_NOISE_BASE * max_action
        self.noise_clip0 = NOISE_CLIP_BASE * max_action
        self.policy_freq = 2
        self.total_it = 0

    def begin_push_window(self, iters: int = PUSH_ITERS, actor_lr: float = PUSH_ACTOR_LR, critic_lr: float = PUSH_CRITIC_LR):
        for g in self.actor_optimizer.param_groups:  g['lr'] = actor_lr
        for g in self.critic_optimizer.param_groups: g['lr'] = critic_lr
        self._sched_enabled = False
        self._push_until_it = self.total_it + iters
        print(f"⚡ Push window started for {iters} iters: LR(a/c)={actor_lr:.2e}/{critic_lr:.2e}")

    def maybe_end_push_window(self):
        if self._push_until_it is not None and self.total_it >= self._push_until_it:
            # restore base LR and resume schedulers
            for g in self.actor_optimizer.param_groups:  g['lr'] = self._base_actor_lr
            for g in self.critic_optimizer.param_groups: g['lr'] = self._base_critic_lr
            self._sched_enabled = True
            self._push_until_it = None
            print("✅ Push window ended; schedulers resumed.")

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        s = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        a = self.actor(s).cpu().data.numpy().flatten()
        if add_noise:
            noise = np.random.normal(0, 0.12 * self.max_action, size=a.shape)
            a = np.clip(a + noise, -self.max_action, self.max_action)
        return a

    def train(self, buffer: ReplayBuffer, batch_size: int = 256):
        self.total_it += 1
        s, a, r, s2, d, m = buffer.sample(batch_size)
        s, a, r, s2, d, m = s.to(self.device), a.to(self.device), r.to(self.device), s2.to(self.device), d.to(self.device), m.to(self.device)

        decay = max(0.30, math.exp(-self.total_it / NOISE_DECAY_ITERS))
        noise_scale = self.policy_noise0 * decay
        noise_clip = self.noise_clip0

        with torch.no_grad():
            noise = (torch.randn_like(a) * noise_scale).clamp(-noise_clip, noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-self.max_action, self.max_action)
            q1t, q2t = self.critic_target(s2, a2)
            qt = torch.min(q1t, q2t)
            qt = r + self.discount * qt * (~d).float()

        q1, q2 = self.critic(s, a)
        if qt.shape != q1.shape: qt = qt.view_as(q1)

        critic_loss = F.smooth_l1_loss(q1, qt) + F.smooth_l1_loss(q2, qt)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), GRAD_CLIP_NORM)
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            pi = self.actor(s)
            actor_loss = -self.critic.Q1(s, pi).mean()

            # TD3+BC with safety boost
            demo_mask = m.float()
            if bool(demo_mask.any().item()):
                progress = min(1.0, self.total_it / BC_FADE_ITERS)
                bc_weight = max(BC_MIN, BC_MAX * (1.0 - progress))
                if SAFETY_FORCE_BC:
                    bc_weight = max(bc_weight, SAFETY_BC_MAX)
                bc_loss = ((pi - a) ** 2).mean(dim=1, keepdim=True)
                actor_loss = actor_loss + bc_weight * (demo_mask * bc_loss).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
            self.actor_optimizer.step()

            # Soft updates
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau*p.data + (1-self.tau)*tp.data)

        # Schedulers (paused during push window)
        if self._sched_enabled:
            self.actor_sched.step(self.total_it)
            self.critic_sched.step(self.total_it)

        # Maybe end push window
        self.maybe_end_push_window()

    def save(self, path: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_it': self.total_it
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer_state_dict'])
        self.total_it = ckpt['total_it']


# Demo mix schedule
def lqr_mix_by_visit(n_visits_for_speed: int, speed: float, after_exploration: bool = False) -> float:
    floor = (ADAPTIVE_DEMO_FLOOR if after_exploration else 0.35)  # slightly higher floor pre-explore
    base = 0.95 * np.exp(-n_visits_for_speed / 60.0) + floor  # slower decay
    if   speed >= 55.0: boost = 0.15
    elif speed >= 45.0: boost = 0.10
    elif speed >= 35.0: boost = 0.05
    else:               boost = 0.0
    return float(np.clip(base + boost, floor, 0.95))


# Evaluation and utilities
def evaluate_controller(env: NonlinearQuarterCarModel, agent: TD3, speeds: List[float], episodes: int = 3) -> Dict:
    results = {'td3': {}, 'lqr': {}}
    for speed in speeds:
        print(f"Evaluating at {speed} km/h...")
        td3_rewards, lqr_rewards = [], []
        td3_rms_accel, lqr_rms_accel = [], []
        td3_travel, lqr_travel = [], []

        for ep in range(episodes):
            load_factor = 1.0 + 0.4 * np.sin(ep * 1.3)

            # TD3
            state = env.reset(speed=speed, load_factor=load_factor)
            r_sum, accs, travs, steps = 0.0, [], [], 0
            while True:
                a = agent.select_action(state, add_noise=False)
                state, r, done = env.step(float(a[0]))
                r_sum += r
                accs.append(abs(env.last_accel))
                travs.append(abs(env.state[0] - env.state[2]))
                steps += 1
                if done or steps > MAX_STEPS_PER_EPISODE: break
            td3_rewards.append(r_sum)
            td3_rms_accel.append(float(np.sqrt(np.mean(np.square(accs))) if accs else 0.0))
            td3_travel.append(float(np.max(travs) if travs else 0.0))

            # LQR
            state = env.reset(speed=speed, load_factor=load_factor)
            r_sum, accs, travs, steps = 0.0, [], [], 0
            while True:
                a = env.get_lqr_action()
                state, r, done = env.step(a)
                r_sum += r
                accs.append(abs(env.last_accel))
                travs.append(abs(env.state[0] - env.state[2]))
                steps += 1
                if done or steps > MAX_STEPS_PER_EPISODE: break
            lqr_rewards.append(r_sum)
            lqr_rms_accel.append(float(np.sqrt(np.mean(np.square(accs))) if accs else 0.0))
            lqr_travel.append(float(np.max(travs) if travs else 0.0))

        results['td3'][speed] = {
            'reward': np.mean(td3_rewards), 'reward_std': np.std(td3_rewards),
            'rms_accel': np.mean(td3_rms_accel), 'max_travel': np.mean(td3_travel),
        }
        results['lqr'][speed] = {
            'reward': np.mean(lqr_rewards), 'reward_std': np.std(lqr_rewards),
            'rms_accel': np.mean(lqr_rms_accel), 'max_travel': np.mean(lqr_travel),
        }
    return results


def choose_speed(ep: int, speed_visits: dict) -> float:
    e0, e1, e2 = CURRICULUM_MILESTONES
    if ep < e0:          eligible = [25.0]
    elif ep < e1:        eligible = [25.0, 35.0]
    elif ep < e2:        eligible = [25.0, 35.0, 45.0]
    else:                eligible = [25.0, 35.0, 45.0, 55.0]

    # After a while, emphasize harder speed 55 a bit more
    if ep >= BIAS_HIGH_SPEED_AFTER_EP and 55.0 in eligible:
        w = np.array([1.0/(1.0 + speed_visits[s]) for s in eligible], dtype=np.float64)
        idx55 = eligible.index(55.0)
        w[idx55] *= 2.0  # mild upweight
        w = w / w.sum()
        return float(np.random.choice(eligible, p=w))

    w = np.array([1.0/(1.0 + speed_visits[s]) for s in eligible], dtype=np.float64)
    w = w / np.sum(w)
    return float(np.random.choice(eligible, p=w))


# Main training loop
def main():
    print("="*70)
    print("TD3 ENHANCED NONLINEAR ACTIVE SUSPENSION CONTROLLER")
    print("Complex Nonlinear Quarter-Car with Multiple Nonlinear Effects")
    print("="*70)

    env = NonlinearQuarterCarModel()
    state_dim = len(env.get_observation())
    action_dim = 1
    max_action = env.max_force

    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Load variation: ±{env.load_variation_range*100:.0f}%")
    print(f"Enhanced nonlinearities: Hysteretic friction, Adaptive damping, Load variation")
    print(f"Road profiles: Mixed Class C/D/E segments")

    agent = TD3(state_dim, action_dim, max_action)
    replay = ReplayBuffer()

    max_episodes = 1000
    batch_size = 256
    exploration_episodes = EXPLORATION_EPISODES
    checkpoint_interval = 100

    episode_rewards = []
    best_perf = -np.inf
    speed_visits = {25.0: 0, 35.0: 0, 45.0: 0, 55.0: 0}

    # Safety bookkeeping
    global ADAPTIVE_DEMO_FLOOR, DEMO_FLOOR_TARGET, SAFETY_FORCE_BC
    last_safety_ep = -10_000
    safety_active_until_ep = -1
    gap_hist = deque(maxlen=ROLLBACK_WINDOW)

    print(f"\nStarting enhanced TD3 training...")
    print(f"Curriculum: 25 → 35 → 45 → 55 km/h at episodes {CURRICULUM_MILESTONES}")

    for ep in range(max_episodes):
        # Push phase: after ep hits threshold, lower demo floor + warm restart LR
        if ep == PUSH_EP_START:
            DEMO_FLOOR_TARGET = 0.10
            agent.begin_push_window(iters=PUSH_ITERS, actor_lr=PUSH_ACTOR_LR, critic_lr=PUSH_CRITIC_LR)

        # Smooth demo floor tracking
        ADAPTIVE_DEMO_FLOOR = smooth_update(ADAPTIVE_DEMO_FLOOR, DEMO_FLOOR_TARGET, tau=0.08)

        # Auto-release safety when window ends
        if ep >= safety_active_until_ep and SAFETY_FORCE_BC:
            DEMO_FLOOR_TARGET = DEMO_FLOOR_BASE
            SAFETY_FORCE_BC = False

        speed = choose_speed(ep, speed_visits)
        s = env.reset(speed=speed)  # random load factor

        # Effort warm-up
        env.r_u_scale = 0.6 if speed_visits[speed] < 3 else 1.0

        ep_reward, steps, sat_cnt, demo_cnt = 0.0, 0, 0, 0
        policy_limit = max_action * (0.7 if speed_visits[speed] < 3 else 1.0)
        noisy_lqr_std = 0.15 * max_action

        while True:
            after_exploration = (ep >= exploration_episodes)
            p_demo = lqr_mix_by_visit(speed_visits[speed], speed, after_exploration)

            if np.random.random() < p_demo:
                a = np.array([env.get_lqr_action()], dtype=np.float64)
                used_lqr = True
            else:
                if ep < exploration_episodes:
                    base = env.get_lqr_action()
                    noisy = base + np.random.normal(0.0, noisy_lqr_std)
                    a = np.array([np.clip(noisy, -policy_limit, policy_limit)], dtype=np.float64)
                    used_lqr = False
                else:
                    a = agent.select_action(s)
                    a[0] = float(np.clip(a[0], -policy_limit, policy_limit))
                    used_lqr = False

            s2, r, done = env.step(float(a[0]))
            if abs(env.u_applied) >= (env.max_force - 10.0):
                sat_cnt += 1

            replay.push(s, a, r, s2, done, is_demo=used_lqr)

            if (len(replay) >= LEARNING_STARTS_STEPS) and (ep >= exploration_episodes // 3):
                try:
                    agent.train(replay, batch_size)
                except Exception as e:
                    print(f"Training error at episode {ep}: {e}")

            if used_lqr: demo_cnt += 1
            s = s2; ep_reward += r; steps += 1
            if done or steps > MAX_STEPS_PER_EPISODE: break

        episode_rewards.append(ep_reward)
        speed_visits[speed] += 1

        if ep % 10 == 0:
            demo_ratio = demo_cnt / max(steps, 1)
            sat_rate = sat_cnt / max(steps, 1)
            shown_p_demo = lqr_mix_by_visit(speed_visits[speed], speed, after_exploration=(ep >= exploration_episodes))

            if ep % 50 == 0:
                lr_a = agent.actor_optimizer.param_groups[0]['lr']
                lr_c = agent.critic_optimizer.param_groups[0]['lr']
                lr_info = f", LR(a/c): {lr_a:.2e}/{lr_c:.2e}"
            else:
                lr_info = ""

            print(f"Episode {ep:3d} [{'Exploring' if ep < exploration_episodes else 'Learning'}], "
                  f"Speed: {speed:4.1f} km/h, Reward: {ep_reward:8.0f}, Load: {env.current_load_factor:.2f}, "
                  f"Buffer: {len(replay):6d}, Visits: {speed_visits[speed]} (p_demo≈{shown_p_demo:.2f}), "
                  f"Demo: {demo_ratio:.2f}, Sat: {sat_rate:.2f}, Floor: {ADAPTIVE_DEMO_FLOOR:.2f}, "
                  f"SafetyBC: {int(SAFETY_FORCE_BC)}{lr_info}")

        if ep % checkpoint_interval == 0 and ep > 0:
            os.makedirs("checkpoints_nonlinear", exist_ok=True)
            path = os.path.join("checkpoints_nonlinear", f"td3_episode_{ep}.pth")
            agent.save(path)
            print(f"✓ Checkpoint saved at episode {ep}")

            try:
                quick = evaluate_controller(env, agent, [25.0, 35.0, 45.0, 55.0], episodes=2)
                avg_td3 = np.mean([quick['td3'][s]['reward'] for s in [25.0, 35.0, 45.0, 55.0]])
                avg_lqr = np.mean([quick['lqr'][s]['reward'] for s in [25.0, 35.0, 45.0, 55.0]])
                gap = float(avg_td3 - avg_lqr)
                gap_hist.append(gap)
                gap_avg = float(np.mean(gap_hist))
                margin = dynamic_rollback_delta(ep)

                print(f"Quick eval - TD3: {avg_td3:.2f}, LQR: {avg_lqr:.2f} | "
                      f"gap={gap:.2f}, gap_avg({len(gap_hist)}): {gap_avg:.2f}, margin={margin:.1f}")

                # Safety system
                can_trigger = (ep - last_safety_ep) >= SAFETY_COOLDOWN_EPISODES and ep >= 200
                if SAFETY_ENABLED and can_trigger and (gap_avg < -margin):
                    DEMO_FLOOR_TARGET = SAFETY_DEMO_FLOOR
                    SAFETY_FORCE_BC = True
                    last_safety_ep = ep
                    safety_active_until_ep = ep + SAFETY_DURATION_EPISODES
                    best_path = os.path.join("checkpoints_nonlinear", "best_model.pth")
                    if os.path.exists(best_path):
                        agent.load(best_path)
                        print("⚠️  Safety TRIGGERED: Complex system underperforming - increased guidance")
                    else:
                        print("⚠️  Safety TRIGGERED: Increased demo guidance for complex system")

                # Save best model when TD3 >= LQR
                if (avg_td3 >= avg_lqr) and (avg_td3 > best_perf):
                    best_perf = avg_td3
                    agent.save(os.path.join("checkpoints_nonlinear", "best_model.pth"))
                    print(f"★ New best model saved! Performance: {best_perf:.2f}")

            except Exception as e:
                print(f"Evaluation error: {e}")

    print("\n" + "="*80)
    print("ENHANCED NONLINEAR TRAINING COMPLETE")
    print("="*80)
    print(f"Final buffer size: {len(replay)}")
    print(f"Training iterations: {agent.total_it}")

    # Final comprehensive evaluation
    print("\nStarting comprehensive evaluation with load variation...")
    speeds = [25, 35, 45, 55]
    results = evaluate_controller(env, agent, speeds, episodes=5)

    print(f"{'Speed (km/h)':<12} {'Controller':<10} {'Reward':<12} {'RMS Accel':<12} {'Max Travel':<12}")
    print("-" * 80)
    td3_wins = 0

    for sp in speeds:
        td3 = results['td3'][sp]; lqr = results['lqr'][sp]
        print(f"{sp:<12} {'TD3':<10} {td3['reward']:<12.2f} {td3['rms_accel']:<12.4f} {td3['max_travel']:<12.4f}")
        print(f"{'':12} {'LQR':<10} {lqr['reward']:<12.2f} {lqr['rms_accel']:<12.4f} {lqr['max_travel']:<12.4f}")
        if td3['reward'] > lqr['reward']:
            td3_wins += 1
            print(f"{'':12} Winner: TD3 (+{td3['reward'] - lqr['reward']:.1f})")
        else:
            print(f"{'':12} Winner: LQR (+{lqr['reward'] - td3['reward']:.1f})")
        print("-" * 80)

    avg_td3_r = np.mean([results['td3'][s]['reward'] for s in speeds])
    avg_lqr_r = np.mean([results['lqr'][s]['reward'] for s in speeds])
    avg_td3_a = np.mean([results['td3'][s]['rms_accel'] for s in speeds])
    avg_lqr_a = np.mean([results['lqr'][s]['rms_accel'] for s in speeds])

    reward_impr = ((avg_td3_r - avg_lqr_r)/max(abs(avg_lqr_r), 1e-8)) * 100
    accel_impr = ((avg_lqr_a - avg_td3_a)/max(avg_lqr_a, 1e-8)) * 100

    print(f"\nENHANCED SYSTEM PERFORMANCE SUMMARY:")
    print(f"TD3 wins: {td3_wins}/{len(speeds)} ({100*td3_wins/len(speeds):.1f}%)")
    print(f"Average reward improvement: {reward_impr:.1f}%")
    print(f"Average RMS acceleration improvement: {accel_impr:.1f}%")
    print(f"Nonlinear advantages demonstrated: {reward_impr > 5.0}")

    return results, episode_rewards

if __name__ == "__main__":
    try:
        from google.colab import files  # type: ignore
        print("Running in Google Colab")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("Running locally")
        import matplotlib.pyplot as plt


    results, rewards = main()
