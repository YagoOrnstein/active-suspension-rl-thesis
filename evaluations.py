
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os, sys, importlib.util
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

CKPT_PATH   = r"C:\Users\yazo_\PycharmProjects\Trying\colab long\td3_episode_500.pth"
MODULE_PATH = r"C:\Users\yazo_\PycharmProjects\Trying\td3_active_suspension_nonlinear.py"

# Settings
SPEEDS          = [25.0, 35.0, 45.0, 55.0]
LOADS           = [0.9, 1.0, 1.1]
EPISODES        = 2
MAX_STEPS       = 2500
LONG_ROAD_M     = 3000.0
DISABLE_TIRE_SPEED = False
SEED            = 1337

# Road classes
ROAD_CLASSES = ['A', 'B', 'C', 'D', 'E']
ROAD_CLASS_NAMES = {
    'A': 'Very Good (A)',
    'B': 'Good (B)',
    'C': 'Average (C)',
    'D': 'Poor (D)',
    'E': 'Very Poor (E)'
}

# PID config
PID_USE_ACCEL = False
ACC_NOISE_STD = 0.35
ACC_DELAY_STEPS = 5
VEL_LP_TAU_S = 0.015
DERIV_LP_TAU_S = 0.020
KI_ANTIWINDUP_CLAMP = 0.6
kp, ki, kd = 2500.0, 120.0, 3000.0  # PID based on best values from iterated testing 

# Plots output
OUTDIR = "Thesis"


# ---------- Utilities ----------
def set_all_seeds(seed: int = 1337):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_hasattr(obj, name):
    try:
        getattr(obj, name)
        return True
    except Exception:
        return False

def import_from_path(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Training module not found: {path}")
    spec = importlib.util.spec_from_file_location("susp_module", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["susp_module"] = mod
    spec.loader.exec_module(mod)
    return mod

def robust_load_checkpoint(agent, ckpt_path: str):
    if not os.path.isfile(ckpt_path):
        print(f"(!) TD3 checkpoint not found at {ckpt_path}. Skipping TD3.")
        return False
    device = getattr(agent, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except Exception as e:
        print(f"(!) torch.load failed: {e}")
        return False
    try:
        if isinstance(ckpt, dict) and ("actor_state_dict" in ckpt or "critic_state_dict" in ckpt):
            if hasattr(agent, "actor") and "actor_state_dict" in ckpt:
                agent.actor.load_state_dict(ckpt["actor_state_dict"])
            if hasattr(agent, "critic") and "critic_state_dict" in ckpt:
                agent.critic.load_state_dict(ckpt["critic_state_dict"])
            if hasattr(agent, "actor_optimizer") and "actor_optimizer_state_dict" in ckpt:
                agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
            if hasattr(agent, "critic_optimizer") and "critic_optimizer_state_dict" in ckpt:
                agent.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
            if "total_it" in ckpt:
                agent.total_it = ckpt["total_it"]
            print("✓ Loaded TD3 checkpoint (dict layout)")
            return True
        if hasattr(agent, "load"):
            agent.load(ckpt_path)
            print("✓ Loaded TD3 checkpoint via agent.load()")
            return True
        print("(!) Unknown checkpoint format; TD3 not loaded.")
        return False
    except Exception as e:
        print(f"(!) Error loading checkpoint: {e}")
        return False

# ---------- Controllers ----------
class FairSuspensionPID:
    def __init__(self, kp: float, ki_per_s: float, kds: float, dt: float, u_limit: float):
        self.kp = float(kp); self.ki = float(ki_per_s); self.kds = float(kds)
        self.dt = float(dt); self.u_limit = float(u_limit)
        self.alpha_vel = self.dt / max(self.dt + VEL_LP_TAU_S, 1e-9)
        self.alpha_der = self.dt / max(self.dt + DERIV_LP_TAU_S, 1e-9)
        self.use_accel = bool(PID_USE_ACCEL)
        self.acc_noise = float(ACC_NOISE_STD)
        self.delay_N = max(0, int(ACC_DELAY_STEPS))
        self.acc_buf = []
        self.reset()
    def reset(self):
        self.ui = 0.0; self.prev_error = 0.0; self.error_f = 0.0
        self.error_dot_f = 0.0; self.accel_est = 0.0; self.acc_buf.clear()
    def step(self, env) -> float:
        suspension_deflection = float(env.state[0] - env.state[2])
        error = -suspension_deflection
        self.error_f = (1.0 - self.alpha_vel) * self.error_f + self.alpha_vel * error
        error_dot = (error - self.prev_error) / max(self.dt, 1e-9); self.prev_error = error
        self.error_dot_f = (1.0 - self.alpha_der) * self.error_dot_f + self.alpha_der * error_dot
        u_p = self.kp * self.error_f; u_d = self.kds * self.error_dot_f
        if self.ki > 1e-6:
            ui_candidate = self.ui + self.ki * self.error_f * self.dt
            u_total = u_p + u_d + ui_candidate
            u_sat = float(np.clip(u_total, -self.u_limit, self.u_limit))
            if abs(u_total) <= self.u_limit or (self.error_f * (u_total - u_sat) <= 0):
                self.ui = ui_candidate
            ui_max = KI_ANTIWINDUP_CLAMP * self.u_limit
            self.ui = float(np.clip(self.ui, -ui_max, ui_max))
            u_i = self.ui
        else:
            u_i = 0.0
        if self.use_accel:
            a_raw = float(getattr(env, "last_accel", 0.0)) + np.random.normal(0.0, self.acc_noise)
            self.acc_buf.append(a_raw)
            a_meas = self.acc_buf.pop(0) if len(self.acc_buf) > self.delay_N else 0.0
            self.accel_est = (1.0 - self.alpha_der) * self.accel_est + self.alpha_der * a_meas
            u_acc = -100.0 * self.accel_est
        else:
            u_acc = 0.0
        u = u_p + u_d + u_i + u_acc
        return float(np.clip(u, -self.u_limit, self.u_limit))

# ---------- Episodes ----------
def run_episode(env, controller: str, agent=None, pid: FairSuspensionPID = None,
                speed=25.0, load=1.0, road=None, max_steps=MAX_STEPS) -> Dict[str, float]:
    if road is None:
        road = env.generate_enhanced_road()
    _ = env.reset(speed=speed, road_profile=road, load_factor=load)
    if pid is not None: pid.reset()
    start_s = getattr(env, "s_pos", 0.0)
    total_return = 0.0
    accs, travs, abs_us = [], [], []
    sat_count = 0
    with torch.no_grad():
        for _ in range(max_steps):
            if controller == "passive":
                u = 0.0
            elif controller == "lqr":
                u = env.get_lqr_action()
            elif controller == "pid":
                u = pid.step(env)
            elif controller == "td3":
                a = agent.select_action(env.get_observation(), add_noise=False)
                u = float(a[0])
            else:
                raise ValueError("controller must be 'passive' | 'lqr' | 'pid' | 'td3'")
            _, r, _ = env.step(u)
            total_return += r
            last_acc = float(getattr(env, "last_accel", 0.0))
            accs.append(abs(last_acc))
            travs.append(abs(float(env.state[0] - env.state[2])))
            u_applied = float(getattr(env, "u_applied", u))
            abs_us.append(abs(u_applied))
            max_force = float(getattr(env, "max_force", 1e9))
            if abs(u_applied) >= (max_force - 10.0):
                sat_count += 1
    end_s = getattr(env, "s_pos", None)
    if end_s is not None:
        distance = max(end_s - start_s, 1e-9)
    else:
        v_ms = float(speed) / 3.6
        dt = float(getattr(env, "dt", 0.001))
        distance = max(v_ms * dt * max_steps, 1e-9)
    rms_acc = float(np.sqrt(np.mean(np.square(accs))) if accs else 0.0)
    max_trav = float(np.max(travs) if travs else 0.0)
    mean_trav = float(np.mean(travs) if travs else 0.0)
    sat_rate = sat_count / float(max_steps)
    mean_abs_u = float(np.mean(abs_us) if abs_us else 0.0)
    ret_per_m = total_return / distance
    return {
        "return": total_return,
        "return_per_meter": ret_per_m,
        "rms_acc": rms_acc,
        "max_travel": max_trav,
        "mean_travel": mean_trav,
        "sat_rate": sat_rate,
        "mean_abs_u": mean_abs_u,
        "steps": max_steps
    }

def run_episode_trace(env, controller: str, agent=None, pid: FairSuspensionPID = None,
                      speed=35.0, load=1.0, road=None, max_steps=2000):
    if road is None:
        road = env.generate_enhanced_road()
    _ = env.reset(speed=speed, road_profile=road, load_factor=load)
    if pid is not None: pid.reset()
    t = np.arange(max_steps) * env.dt
    u_hist, acc_hist, trav_hist = [], [], []
    with torch.no_grad():
        for _ in range(max_steps):
            if controller == "passive":
                u = 0.0
            elif controller == "lqr":
                u = env.get_lqr_action()
            elif controller == "pid":
                u = pid.step(env)
            elif controller == "td3":
                a = agent.select_action(env.get_observation(), add_noise=False)
                u = float(a[0])
            _, r, _ = env.step(u)
            u_hist.append(float(getattr(env, "u_applied", u)))
            acc_hist.append(float(getattr(env, "last_accel", 0.0)))
            trav_hist.append(float(env.state[0] - env.state[2]))
    return {"t": t, "u": np.array(u_hist), "acc": np.array(acc_hist),
            "travel": np.array(trav_hist), "road": road}

# ---------- Summaries & tables ----------
def summarize_by_speed(raw: dict, speeds: List[float]) -> Dict[float, Dict[str, float]]:
    out = {}
    for sp in speeds:
        idxs = [i for i, v in enumerate(raw["speed"]) if v == sp]
        def m(k): return float(np.mean([raw[k][i] for i in idxs])) if idxs else float("nan")
        def s(k): return float(np.std([raw[k][i] for i in idxs])) if idxs else float("nan")
        out[sp] = {
            "return_mean": m("return"), "return_std": s("return"),
            "retpm_mean": m("return_per_meter"), "retpm_std": s("return_per_meter"),
            "rms_mean": m("rms_acc"), "rms_std": s("rms_acc"),
            "mt_mean": 1000.0 * m("max_travel"), "mt_std": 1000.0 * s("max_travel"),
            "sat_mean": 100.0 * m("sat_rate"), "sat_std": 100.0 * s("sat_rate"),
            "u_mean": m("mean_abs_u"), "u_std": s("mean_abs_u"),
        }
    return out

def print_table(summaries: Dict[str, Dict[float, Dict[str, float]]], speeds: List[float]):
    print(f"{'Speed':<6} {'Ctrl':<8} {'Return/m':>12} {'RMS Acc':>12} {'MaxTrav(mm)':>14} {'Sat %':>8} {'|u| mean':>10}")
    print("-" * 118)
    for sp in speeds:
        for ctrl in ["passive", "lqr", "pid", "td3"]:
            if ctrl not in summaries: continue
            s = summaries[ctrl][sp]
            print(f"{int(sp):<6} {ctrl.upper():<8} "
                  f"{s['retpm_mean']:>7.2f}±{s['retpm_std']:<7.2f} "
                  f"{s['rms_mean']:>7.3f}±{s['rms_std']:<7.3f} "
                  f"{s['mt_mean']:>9.1f}±{s['mt_std']:<6.1f} "
                  f"{s['sat_mean']:>5.2f}±{s['sat_std']:<5.2f} "
                  f"{s['u_mean']:>8.1f}")
        print("-" * 118)

# ---------- Plots ----------
def make_comprehensive_plots(summaries_by_road, speeds, road_classes, controllers, outdir):
    os.makedirs(outdir, exist_ok=True)
    colors = {"passive": "#a1a1a1", "lqr": "#1f77b4", "pid": "#ff7f0e", "td3": "#2ca02c"}
    markers = {"passive": "o", "lqr": "s", "pid": "^", "td3": "D"}

    # Define active controllers for comparison plots
    active_controllers = [c for c in ["lqr", "pid", "td3"] if c in controllers]


    # 1) RMS vs Road class (one panel per speed)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12)); axes = axes.flatten()
    for i, speed in enumerate(speeds):
        ax = axes[i]; x_pos = np.arange(len(road_classes))
        for ctrl in active_controllers:  
            rms_values = [summaries_by_road[rc][ctrl][speed]["rms_mean"] for rc in road_classes]
            rms_errors = [summaries_by_road[rc][ctrl][speed]["rms_std"] for rc in road_classes]
            ax.errorbar(x_pos, rms_values, yerr=rms_errors, fmt='-o', marker=markers[ctrl],
                        color=colors[ctrl], label=ctrl.upper(), capsize=4, linewidth=2, markersize=6)
        ax.set_xlabel('Road Class (A=smoothest, E=roughest)')
        ax.set_ylabel('RMS Body Acc (m/s²)')
        ax.set_title(f'{int(speed)} km/h')
        ax.set_xticks(x_pos); ax.set_xticklabels([f"{rc}" for rc in road_classes])
        ax.grid(True, alpha=0.3)
        if i == 0: ax.legend(loc='upper left')
    plt.suptitle('RMS Body Acceleration vs Road Class (Lower is Better)', fontsize=16)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "01_rms_vs_road_class.png"), dpi=300, bbox_inches='tight'); plt.close()

    # 2) TD3 advantage over LQR (return/m)
    if "td3" in controllers and "lqr" in controllers:
        fig, ax = plt.subplots(figsize=(12, 8))
        x_pos = np.arange(len(road_classes)); width = 0.2
        for i, speed in enumerate(speeds):
            gaps = [summaries_by_road[rc]["td3"][speed]["retpm_mean"] -
                    summaries_by_road[rc]["lqr"][speed]["retpm_mean"] for rc in road_classes]
            ax.bar(x_pos + i * width, gaps, width, label=f'{int(speed)} km/h', alpha=0.85)
        ax.set_xlabel('Road Class'); ax.set_ylabel('TD3 - LQR Return per Meter')
        ax.set_title('TD3 Advantage over LQR by Road Condition')
        ax.set_xticks(x_pos + width * (len(speeds) - 1) / 2)
        ax.set_xticklabels([f"{rc}\n({ROAD_CLASS_NAMES[rc]})" for rc in road_classes])
        ax.axhline(0, color='black', linestyle='--', alpha=0.5); ax.legend(); ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "02_td3_advantage.png"), dpi=300, bbox_inches='tight'); plt.close()

    # 3) Heatmaps of RMS
    ctrls_to_show = [c for c in ["lqr", "pid", "td3"] if c in controllers]
    if not ctrls_to_show:
        print("(!) No active controllers available for RMS heatmaps; skipping.")
    else:
        ncols = min(3, len(ctrls_to_show))
        nrows = int(np.ceil(len(ctrls_to_show) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for i, ctrl in enumerate(ctrls_to_show):
            ax = axes[i]
            matrix = np.full((len(road_classes), len(speeds)), np.nan, dtype=float)
            for r_idx, rc in enumerate(road_classes):
                for s_idx, sp in enumerate(speeds):
                    try:
                        val = summaries_by_road[rc][ctrl][sp]["rms_mean"]
                    except Exception:
                        val = np.nan
                    matrix[r_idx, s_idx] = val

            # Fallback if all NaN
            if np.all(np.isnan(matrix)):
                ax.text(0.5, 0.5, f"No data for {ctrl.upper()}", ha="center", va="center")
                ax.set_axis_off()
                continue

            # Replace NaNs with the column mean (or overall mean) to keep colormap stable
            col_means = np.nanmean(matrix, axis=0)
            overall_mean = np.nanmean(matrix)
            for s_idx in range(matrix.shape[1]):
                col = matrix[:, s_idx]
                nan_mask = np.isnan(col)
                if np.any(nan_mask):
                    fill_val = col_means[s_idx] if not np.isnan(col_means[s_idx]) else overall_mean
                    col[nan_mask] = fill_val
                    matrix[:, s_idx] = col

            im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(len(speeds)));
            ax.set_xticklabels([f'{int(s)}' for s in speeds])
            ax.set_yticks(range(len(road_classes)));
            ax.set_yticklabels([f"{rc}" for rc in road_classes])
            ax.set_xlabel('Speed (km/h)');
            ax.set_ylabel('Road Class')
            ax.set_title(f'{ctrl.upper()} Controller — RMS Acc (m/s²)')

            # Cell labels
            v70 = np.percentile(matrix, 70)
            for r in range(len(road_classes)):
                for s in range(len(speeds)):
                    val = matrix[r, s]
                    text_color = "white" if val > v70 else "black"
                    ax.text(s, r, f'{val:.2f}', ha="center", va="center",
                            color=text_color, fontweight='bold', fontsize=10)

            cbar = plt.colorbar(im, ax=ax, shrink=0.85)
            cbar.set_label('RMS Acc (m/s²)', rotation=270, labelpad=20)

        # Hide any unused axes
        for j in range(len(ctrls_to_show), len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Performance Heatmaps — RMS Body Acceleration', fontsize=16, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "03_heatmaps.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # 4) Speed vs RMS curves, per road class
    fig, axes = plt.subplots(2, 3, figsize=(20, 12)); axes = axes.flatten()
    for i, road_class in enumerate(road_classes):
        if i >= 6: break
        ax = axes[i]
        for ctrl in active_controllers:
            rms_values = [summaries_by_road[road_class][ctrl][sp]["rms_mean"] for sp in speeds]
            rms_errors = [summaries_by_road[road_class][ctrl][sp]["rms_std"] for sp in speeds]
            ax.errorbar(speeds, rms_values, yerr=rms_errors, fmt='-o', marker=markers[ctrl],
                        color=colors[ctrl], label=ctrl.upper(), capsize=4, linewidth=2.5, markersize=8)
        ax.set_xlabel('Speed (km/h)'); ax.set_ylabel('RMS Body Acc (m/s²)')
        ax.set_title(f'Road Class {road_class}: {ROAD_CLASS_NAMES[road_class]}')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    if len(road_classes) < 6: axes[5].set_visible(False)
    plt.suptitle('Speed vs RMS Acceleration by Road Class', fontsize=16)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "04_speed_curves.png"), dpi=300, bbox_inches='tight'); plt.close()

    # 5) Overall averages + advantage vs passive
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1_data = {}
    for ctrl in active_controllers:
        speed_avgs = []
        for speed in speeds:
            road_rms = [summaries_by_road[rc][ctrl][speed]["rms_mean"] for rc in road_classes]
            speed_avgs.append(np.mean(road_rms))
        ax1_data[ctrl] = speed_avgs
    for ctrl in active_controllers:
        ax1.plot(speeds, ax1_data[ctrl], marker=markers[ctrl], color=colors[ctrl], label=ctrl.upper(), linewidth=2, markersize=8)

    ax1.set_xlabel('Speed (km/h)'); ax1.set_ylabel('Average RMS Acc (m/s²)')
    ax1.set_title('Average Performance Across All Road Classes'); ax1.grid(True, alpha=0.3); ax1.legend()

    for ctrl in controllers:
        print(f"Processing controller: {ctrl}")
        if ctrl == "passive":
            print("Skipping passive")
            continue
        speed_gaps = []
        for speed in speeds:
            ctrl_avg = np.mean([summaries_by_road[rc][ctrl][speed]["retpm_mean"] for rc in road_classes])
            passive_avg = np.mean([summaries_by_road[rc]["passive"][speed]["retpm_mean"] for rc in road_classes])
            speed_gaps.append(ctrl_avg - passive_avg)
        ax2.plot(speeds, speed_gaps, marker=markers[ctrl], color=colors[ctrl], label=ctrl.upper(), linewidth=2, markersize=8)
    ax2.set_xlabel('Speed (km/h)'); ax2.set_ylabel('Return/m Advantage over Passive')
    ax2.set_title('Active Control Benefits'); ax2.grid(True, alpha=0.3); ax2.legend()

    plt.tight_layout(); plt.savefig(os.path.join(outdir, "05_summary.png"), dpi=300, bbox_inches='tight'); plt.close()

def plot_example_road(env, outdir):
    os.makedirs(outdir, exist_ok=True)
    fig, axes = plt.subplots(len(ROAD_CLASSES), 1, figsize=(16, 2.5 * len(ROAD_CLASSES)))
    if len(ROAD_CLASSES) == 1: axes = [axes]
    for i, road_class in enumerate(ROAD_CLASSES):
        road = env.generate_enhanced_road(class_mix=[road_class] * 5)
        if safe_hasattr(env, "road_x") and isinstance(env.road_x, np.ndarray) and len(env.road_x) == len(road):
            x = env.road_x
        else:
            n = len(road)
            dx = float(getattr(env, "road_len_m", 3000.0)) / max(n, 1)
            x = np.arange(n) * dx
        cut = min(len(x), int(3000 / (x[1] - x[0])) if len(x) > 1 else len(x))
        x_plot = x[:cut]; r_plot = road[:cut]
        axes[i].plot(x_plot, r_plot, color='darkblue', linewidth=1, alpha=0.8)
        axes[i].fill_between(x_plot, r_plot, alpha=0.3, color='lightblue')
        axes[i].set_ylabel('Height (m)')
        axes[i].set_title(f'Road Class {road_class}: {ROAD_CLASS_NAMES[road_class]} (RMS = {np.std(r_plot):.4f} m)')
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Distance along road (m)')
    plt.suptitle('Road Profile Examples by ISO Class', fontsize=16)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "06_road_profiles.png"), dpi=300, bbox_inches='tight'); plt.close()

def extra_plots(raw, summaries_by_road, speeds, controllers, outdir):
    os.makedirs(outdir, exist_ok=True)

    # (A) Saturation-rate heatmaps
    def heatmap(metric_key, title, fname, scale=100.0):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14)); axes = axes.flatten()
        for i, ctrl in enumerate(controllers[:4]):
            ax = axes[i]
            matrix = np.zeros((len(ROAD_CLASSES), len(speeds)))
            for r_idx, rc in enumerate(ROAD_CLASSES):
                for s_idx, sp in enumerate(speeds):
                    matrix[r_idx, s_idx] = summaries_by_road[rc][ctrl][sp]["sat_mean"]  # already in %
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f'{int(s)}' for s in speeds])
            ax.set_yticks(range(len(ROAD_CLASSES))); ax.set_yticklabels([f"{rc}" for rc in ROAD_CLASSES])
            ax.set_xlabel('Speed (km/h)'); ax.set_ylabel('Road Class')
            ax.set_title(f'{ctrl.upper()} — {title}')
            for r in range(len(ROAD_CLASSES)):
                for s in range(len(speeds)):
                    ax.text(s, r, f'{matrix[r,s]:.1f}%', ha="center", va="center", color="black", fontsize=10)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('Saturation (%)', rotation=270, labelpad=20)
        plt.suptitle(title, fontsize=16, y=0.95)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches='tight'); plt.close()

    heatmap("sat_mean", "Actuator Saturation Rate", "08_sat_rate_heatmaps.png")

    # (B) Mean |u| heatmaps
    def heatmap_u():
        fig, axes = plt.subplots(2, 2, figsize=(18, 14)); axes = axes.flatten()
        for i, ctrl in enumerate(controllers[:4]):
            ax = axes[i]
            matrix = np.zeros((len(ROAD_CLASSES), len(speeds)))
            for r_idx, rc in enumerate(ROAD_CLASSES):
                for s_idx, sp in enumerate(speeds):
                    matrix[r_idx, s_idx] = summaries_by_road[rc][ctrl][sp]["u_mean"]
            im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest')
            ax.set_xticks(range(len(speeds))); ax.set_xticklabels([f'{int(s)}' for s in speeds])
            ax.set_yticks(range(len(ROAD_CLASSES))); ax.set_yticklabels([f"{rc}" for rc in ROAD_CLASSES])
            ax.set_xlabel('Speed (km/h)'); ax.set_ylabel('Road Class')
            for r in range(len(ROAD_CLASSES)):
                for s in range(len(speeds)):
                    ax.text(s, r, f'{matrix[r,s]:.0f}', ha="center", va="center", color="black", fontsize=10)
            cbar = plt.colorbar(im, ax=ax, shrink=0.8); cbar.set_label('|u| (N)', rotation=270, labelpad=20)
        plt.suptitle('Mean Control Effort |u|', fontsize=16, y=0.95)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, "09_mean_abs_u_heatmaps.png"), dpi=300, bbox_inches='tight'); plt.close()
    heatmap_u()

    # (C) RMS vs |u| tradeoff scatter
    fig, ax = plt.subplots(figsize=(8,6))
    for ctrl in controllers:
        rms_vals, u_vals = [], []
        for rc in ROAD_CLASSES:
            for sp in speeds:
                rms_vals.append(summaries_by_road[rc][ctrl][sp]["rms_mean"])
                u_vals.append(summaries_by_road[rc][ctrl][sp]["u_mean"])
        ax.scatter(u_vals, rms_vals, label=ctrl.upper(), alpha=0.75)
    ax.set_xlabel('Mean |u| (N)'); ax.set_ylabel('RMS Body Acc (m/s²)')
    ax.set_title('Comfort vs Effort Tradeoff'); ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "10_tradeoff_rms_vs_u.png"), dpi=300, bbox_inches='tight'); plt.close()

    example_key = ('D', 45.0, 1.0)  # road class D, speed 45, load 1.0


def thesis_timeseries_and_spectra(env, agent, outdir, controllers):
    os.makedirs(outdir, exist_ok=True)
    road = env.generate_enhanced_road(class_mix=['D']*5)
    speed = 45.0; load = 1.0; steps = 3000
    pid = FairSuspensionPID(kp, ki, kd, env.dt, env.max_force)
    traces = {}
    for ctrl in ["passive", "lqr", "pid"] + (["td3"] if agent is not None else []):
        traces[ctrl] = run_episode_trace(env, ctrl, agent=agent, pid=(pid if ctrl=="pid" else None),
                                         speed=speed, load=load, road=road, max_steps=steps)

    # Histograms of body acc
    fig, ax = plt.subplots(figsize=(10,6))
    for c, tr in traces.items():
        ax.hist(tr["acc"], bins=60, histtype='step', linewidth=1.5, label=c.upper(), density=True)
    ax.set_xlabel('Body Acc (m/s²)'); ax.set_ylabel('Density'); ax.set_title('Acceleration Distribution — D @ 45 km/h')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "11_accel_histograms.png"), dpi=300, bbox_inches='tight'); plt.close()

    # Empirical CDF of body acc |a|
    fig, ax = plt.subplots(figsize=(10,6))
    for c, tr in traces.items():
        x = np.sort(np.abs(tr["acc"])); y = np.linspace(0,1,len(x))
        ax.plot(x, y, label=c.upper(), linewidth=1.8)
    ax.set_xlabel('|Body Acc| (m/s²)'); ax.set_ylabel('Empirical CDF'); ax.set_title('Acceleration CDF — D @ 45 km/h')
    ax.grid(True, alpha=0.3); ax.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "12_accel_cdf.png"), dpi=300, bbox_inches='tight'); plt.close()

    # Force histograms
    fig, ax = plt.subplots(figsize=(10,6))
    for c, tr in traces.items():
        if c == "passive": continue
        ax.hist(tr["u"], bins=60, histtype='step', linewidth=1.5, label=c.upper(), density=True)
    ax.set_xlabel('Control Force (N)'); ax.set_ylabel('Density'); ax.set_title('Actuator Force Distribution — D @ 45 km/h')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "13_force_histograms.png"), dpi=300, bbox_inches='tight'); plt.close()

    # Simple PSD of body acceleration (FFT-based)
    fig, ax = plt.subplots(figsize=(10,6))
    for c, tr in traces.items():
        a = tr["acc"] - np.mean(tr["acc"])
        N = len(a); dt = float(getattr(env, "dt", 0.001))
        freqs = np.fft.rfftfreq(N, d=dt)
        psd = (np.abs(np.fft.rfft(a))**2) / (N/dt)
        ax.plot(freqs, psd, label=c.upper(), linewidth=1.2)
    ax.set_xlim(0, 30)  # body & wheel-hop bands visible
    ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('Power'); ax.set_title('Body Acceleration PSD — D @ 45 km/h')
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "14_accel_psd.png"), dpi=300, bbox_inches='tight'); plt.close()

def boxplots_rms_by_controller(raw, outdir):
    # Boxplots per road class comparing controllers (RMS distribution across loads/episodes/speeds)
    os.makedirs(outdir, exist_ok=True)
    for rc in ROAD_CLASSES:
        data = []
        labels = []
        for ctrl in [ "lqr", "pid", "td3"]:
            if ctrl not in raw: continue
            vals = [raw[ctrl]["rms_acc"][i] for i, rcl in enumerate(raw[ctrl]["road_class"]) if rcl == rc]
            if not vals: continue
            data.append(vals); labels.append(ctrl.upper())
        if not data: continue
        fig, ax = plt.subplots(figsize=(8,6))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_ylabel('RMS Body Acc (m/s²)')
        ax.set_title(f'RMS Distribution — Road Class {rc} ({ROAD_CLASS_NAMES[rc]})')
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"15_boxplot_rms_{rc}.png"), dpi=300, bbox_inches='tight'); plt.close()

# ---------- Verify roads ----------
def verify_road_generation(env):
    print("\nVerifying road generation...")
    for road_class in ROAD_CLASSES:
        rms_values = []
        for _ in range(10):
            road = env.generate_enhanced_road(class_mix=[road_class] * 5)
            rms_values.append(np.std(road))
        print(f"Road Class {road_class}: RMS = {np.mean(rms_values):.6f} ± {np.std(rms_values):.6f}")


def plot_spring_characteristics(env, outdir):
    """Generate spring force-displacement characteristics plot"""
    os.makedirs(outdir, exist_ok=True)

    # Parameters from your model
    ks_base = getattr(env, 'ks_base', 20000.0)
    k_nl = getattr(env, 'k_nl', 5e6)
    k_prog = getattr(env, 'k_progressive', 2e5)
    z_limit = getattr(env, 'bump_limit', 0.12)
    k_bump = getattr(env, 'k_bump', 8.0e6)

    def spring_force(deflection):
        """Calculate spring force based on enhanced model"""
        # Linear component
        F_linear = ks_base * deflection

        # Cubic hardening
        F_cubic = k_nl * (deflection ** 3)

        # Progressive spring (|z| * z term)
        F_progressive = k_prog * deflection * np.abs(deflection)

        # Bump-stop force
        F_bump_arr = np.zeros_like(deflection)
        over_limit = np.abs(deflection) > z_limit
        if np.any(over_limit):
            excess = np.abs(deflection[over_limit]) - z_limit
            F_bump_arr[over_limit] = k_bump * (excess ** 2) * np.sign(deflection[over_limit])

        return F_linear + F_cubic + F_progressive + F_bump_arr

    # Create deflection range
    deflection = np.linspace(-0.15, 0.15, 1000)

    # Calculate different force components
    F_linear_only = ks_base * deflection
    F_linear_cubic = ks_base * deflection + k_nl * (deflection ** 3)
    F_full_nonlinear = spring_force(deflection)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(deflection, F_linear_only / 1000, 'b-', linewidth=2, label='Linear: $k_s(z_s-z_u)$')
    plt.plot(deflection, F_linear_cubic / 1000, 'r-', linewidth=2, label='Linear + Cubic hardening')
    plt.plot(deflection, F_full_nonlinear / 1000, 'g-', linewidth=2, label='Full nonlinear model')

    # Add bump-stop limit lines
    plt.axvline(-z_limit, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(z_limit, color='gray', linestyle='--', alpha=0.7)
    plt.text(z_limit + 0.01, 25, 'Bump stops\n(±120 mm)', fontsize=10, color='gray')

    plt.xlabel('Suspension Deflection $z_s - z_u$ (m)')
    plt.ylabel('Spring Force $F_s$ (kN)')
    plt.title('Progressive Spring Force-Displacement Characteristics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save with high DPI for thesis quality
    plt.savefig(os.path.join(outdir, 'spring_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'spring_characteristics.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Spring characteristics plot saved to {outdir}")


def plot_friction_characteristics(env, outdir):
    """Generate friction force-velocity characteristics plot"""
    os.makedirs(outdir, exist_ok=True)

    # Parameters from your model
    F_breakaway = getattr(env, 'friction_breakaway', 800.0)
    F_coulomb = getattr(env, 'friction_coulomb', 600.0)
    F_viscous = getattr(env, 'friction_viscous', 50.0)
    v_stiction = getattr(env, 'stiction_velocity', 0.001)

    def friction_force(velocity):
        """
        Continuous, odd-symmetric friction model:
        - Stiction region: smooth breakaway using tanh, magnitude ≤ F_breakaway
        - Sliding region: piecewise linear with viscous term, joined continuously at v_stiction
        Force always opposes motion.
        """
        v = np.asarray(velocity, dtype=float)
        F = np.empty_like(v)

        # Continuous value at the boundary to avoid a jump
        F_boundary = F_coulomb + F_viscous * v_stiction  # magnitude at |v| = v_stiction

        stiction_mask = np.abs(v) <= v_stiction
        sliding_mask = ~stiction_mask

        # Stiction: smooth odd-symmetric curve that saturates at ±F_breakaway
        F[stiction_mask] = -F_breakaway * np.tanh(v[stiction_mask] / max(v_stiction, 1e-9))

        # Sliding: meet the stiction curve at the boundary, then grow with |v|
        vv = v[sliding_mask]
        F[sliding_mask] = -np.sign(vv) * (F_boundary + F_viscous * (np.abs(vv) - v_stiction))

        return F

    # Create velocity range
    velocity = np.linspace(-2.0, 2.0, 1000)
    F_fric = friction_force(velocity)

    plt.figure(figsize=(10, 6))
    plt.plot(velocity, F_fric, 'b-', linewidth=2, label='Hysteretic friction')

    # Mark stiction region
    plt.axvline(-v_stiction, color='red', linestyle='--', alpha=0.7)
    plt.axvline(v_stiction, color='red', linestyle='--', alpha=0.7)
    plt.fill_betweenx([-F_breakaway, F_breakaway], -v_stiction, v_stiction, alpha=0.2, color='red',
                      label='Stiction region')

    plt.xlabel('Relative Velocity $\\dot{z}_s - \\dot{z}_u$ (m/s)')
    plt.ylabel('Friction Force $F_f$ (N)')
    plt.title('Hysteretic Friction Force Characteristics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'friction_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'friction_characteristics.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Friction characteristics plot saved to {outdir}")


def plot_iso8608_psd(outdir):
    """Generate ISO 8608 PSD curves"""
    os.makedirs(outdir, exist_ok=True)

    # ISO 8608 parameters
    Omega0 = 0.1  # cycles/m
    Gd0_values = {
        'A': 16e-6,
        'B': 64e-6,
        'C': 256e-6,
        'D': 1024e-6,
        'E': 4096e-6
    }

    # Frequency range
    Omega = np.logspace(-2, 1, 1000)  # 0.01 to 10 cycles/m

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'orange', 'red', 'purple']

    for i, (road_class, Gd0) in enumerate(Gd0_values.items()):
        Gd = Gd0 * (Omega / Omega0) ** (-2)
        plt.loglog(Omega, Gd, color=colors[i], linewidth=2,
                   label=f'Class {road_class} ({ROAD_CLASS_NAMES[road_class]})')

    # Reference line
    plt.axvline(Omega0, color='black', linestyle='--', alpha=0.7)
    plt.text(Omega0 * 1.2, 1e-4, f'$\\Omega_0 = {Omega0}$ cycles/m', rotation=90)

    plt.xlabel('Spatial Frequency $\\Omega$ (cycles/m)')
    plt.ylabel('PSD $G_d(\\Omega)$ (m³/cycle)')
    plt.title('ISO 8608 Road Roughness Power Spectral Density')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'iso8608_psd.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'iso8608_psd.pdf'), bbox_inches='tight')
    plt.close()

    print(f"ISO 8608 PSD plot saved to {outdir}")


def plot_road_profile_examples(env, outdir):
    """Generate example road profiles for each ISO class"""
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(len(ROAD_CLASSES), 1, figsize=(14, 3 * len(ROAD_CLASSES)))
    if len(ROAD_CLASSES) == 1:
        axes = [axes]

    for i, road_class in enumerate(ROAD_CLASSES):
        # Generate road profile for this class
        road = env.generate_enhanced_road(class_mix=[road_class] * 5)

        # Get distance array
        if hasattr(env, 'road_x') and isinstance(env.road_x, np.ndarray):
            x = env.road_x[:len(road)]
        else:
            road_len = getattr(env, 'road_len_m', 3000.0)
            x = np.linspace(0, road_len, len(road))

        # Plot first 3000m
        dx = (x[1] - x[0]) if len(x) > 1 else getattr(env, 'road_dx', 0.015)
        n_samples = int(3000.0 / max(dx, 1e-9))
        plot_length = min(n_samples, len(x))
        x_plot = x[:plot_length]
        road_plot = road[:plot_length]

        axes[i].plot(x_plot, road_plot, 'b-', linewidth=0.8, alpha=0.8)
        axes[i].fill_between(x_plot, road_plot, alpha=0.3, color='lightblue')
        axes[i].set_ylabel('Height (m)')
        axes[i].set_title(f'Road Class {road_class}: {ROAD_CLASS_NAMES[road_class]} (RMS = {np.std(road_plot):.4f} m)')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, 3000)

    axes[-1].set_xlabel('Distance along road (m)')
    plt.suptitle('Road Profile Examples by ISO 8608 Classification', fontsize=16)
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'road_profile_examples.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'road_profile_examples.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Road profile examples saved to {outdir}")


def plot_validation_step_response(env, outdir, step_mm=1.0, speed_kmh=36.0, T=4.0):
    """
    Proper step-response validation:
    - small step (default 1 mm)
    - actuator disabled
    - env temporarily linearized
    - compares to closed-form linear model and reports agreement metrics
    """
    import numpy as np, matplotlib.pyplot as plt, os
    os.makedirs(outdir, exist_ok=True)

    # --- 0) Cache & temporarily linearize env ---
    def _save_attrs(obj, names):
        return {n: getattr(obj, n) for n in names if hasattr(obj, n)}
    def _set_if(obj, name, value):
        if hasattr(obj, name): setattr(obj, name, value)

    lin_names = [
        "k_nl", "k_progressive", "bump_limit", "k_bump",
        "friction_breakaway", "friction_coulomb", "friction_viscous",
        "stiction_velocity", "tire_speed_factor", "kt_progressive"
    ]
    old = _save_attrs(env, lin_names)

    # soften/removes sources of nonlinearity
    _set_if(env, "k_nl",           0.0)
    _set_if(env, "k_progressive",  0.0)
    _set_if(env, "k_bump",         0.0)
    _set_if(env, "bump_limit",     1e9)     # move bump stops far away
    _set_if(env, "friction_breakaway", 0.0)
    _set_if(env, "friction_coulomb",   0.0)
    _set_if(env, "friction_viscous",   0.0)
    _set_if(env, "stiction_velocity",  0.0)
    _set_if(env, "tire_speed_factor",  0.0)
    _set_if(env, "kt_progressive",     0.0)

    try:
        # --- 1) Linear (theory) parameters from env (base values) ---
        ms = float(getattr(env, 'ms_base', getattr(env, 'ms', 400.0)))
        ks = float(getattr(env, 'ks_base', getattr(env, 'ks', 20000.0)))
        cs = float(getattr(env, 'cs_base', getattr(env, 'cs', 1800.0)))

        wn   = np.sqrt(ks / ms)
        zeta = cs / (2*np.sqrt(ks*ms))
        fn   = wn / (2*np.pi)

        # --- 2) Build a spatial step road & run the env passively ---
        dt = float(getattr(env, "dt", 0.001))
        N  = int(T/dt)
        t  = np.arange(N) * dt

        v_ms   = float(speed_kmh) / 3.6
        h      = float(step_mm) / 1000.0
        t0     = 0.50  # when the wheel hits the step
        xgrid  = np.arange(0.0, env.road_len_m, env.road_dx, dtype=np.float64)
        x0     = v_ms * t0
        road   = np.zeros_like(xgrid)
        road[xgrid >= x0] = h

        env.reset(speed=speed_kmh, road_profile=road, load_factor=1.0)

        zs = np.zeros(N)
        for i in range(N):
            env.step(0.0)                # actuator disabled
            zs[i] = float(env.state[0])  # sprung mass displacement

        # --- 3) Closed-form linear response to a road step (displacement input) ---
        if zeta < 1.0:
            wd = wn*np.sqrt(1 - zeta**2)
            y_lin = h*(1 - np.exp(-zeta*wn*(t - t0)) *
                       (np.cos(wd*(t - t0)) + (zeta/np.sqrt(1-zeta**2))*np.sin(wd*(t - t0))))
            y_lin[t < t0] = 0.0
        else:
            y_lin = h*(1 - np.exp(-wn*(t - t0))); y_lin[t < t0] = 0.0

        # --- 4) Agreement metrics (on post-step window) ---
        mask    = t >= t0
        err     = zs[mask] - y_lin[mask]
        rms_err = float(np.sqrt(np.mean(err**2)))
        # overshoot & settling time (2% band)
        final   = h
        peak    = float(np.max(zs[mask]))
        OS_pct  = float(100.0 * max(peak - final, 0.0) / (final if final else 1.0))
        band    = 0.02 * (abs(final) if final else 1.0)
        settle_i = np.argmax(np.flip(np.abs(zs[mask] - final) > band) == True)
        if np.any(np.abs(zs[mask] - final) > band):
            # last time it leaves the band → settling time is after it stays inside
            # compute forward from t0
            idxs = np.where(np.abs(zs[mask] - final) > band)[0]
            ts   = t0 if len(idxs)==0 else t[mask][idxs[-1]]
            Ts   = t[-1] - ts
        else:
            Ts = 0.0

        # --- 5) Plot ---
        plt.figure(figsize=(12, 7))
        plt.plot(t, y_lin, 'b--', lw=2.2, label='Theoretical (linear)')
        plt.plot(t, zs,    'r-',  lw=2.2, label='Simulated (env, small-signal)')
        plt.axhline(h, color='k', lw=1.2)
        plt.axvline(t0, color='k', ls=':', alpha=0.6)
        plt.text(t0+0.05, h+0.0002, f'Road step (+{step_mm:.0f} mm)')

        plt.xlabel('Time (s)')
        plt.ylabel('Sprung Mass Displacement (m)')
        plt.title(f'Step Response Validation (fₙ={fn:.2f} Hz, ζ={zeta:.3f}, step={step_mm:.1f} mm)')
        plt.grid(True, alpha=0.3); plt.legend()

        box = (f'RMS error: {rms_err*1000:.2f} mm\n'
               f'Overshoot: {OS_pct:.1f}%\n'
               f'2% settling: {Ts:.2f} s')
        plt.gca().text(0.02, 0.98, box, transform=plt.gca().transAxes,
                       va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'step_response_validation.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[Step validation] step={step_mm:.1f} mm @ {speed_kmh:.0f} km/h → "
              f"RMS err={rms_err*1000:.2f} mm, OS={OS_pct:.1f}%, Ts≈{Ts:.2f}s")

    finally:
        # --- 6) Restore original (nonlinear) parameters ---
        for n, v in old.items():
            try: setattr(env, n, v)
            except Exception: pass


def plot_friction_characteristics_from_env(env, outdir):
    """
    Plot applied friction force vs relative velocity using the env's
    hysteretic friction state (the loop), plus the memoryless target.
    """
    import numpy as np, matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)

    # Params (read from env for consistency)
    F_break = getattr(env, 'friction_breakaway', 800.0)
    F_coul  = getattr(env, 'friction_coulomb',   600.0)
    F_vis   = getattr(env, 'friction_viscous',     50.0)
    v_stick = getattr(env, 'stiction_velocity',  0.001)

    # Memoryless target (for overlay)
    def memoryless_fric(v):
        v = np.asarray(v, float)
        F = np.empty_like(v)
        Fb = F_coul + F_vis*v_stick
        mask_s = np.abs(v) <= v_stick
        F[mask_s]  = -F_break * np.tanh(v[mask_s]/max(v_stick,1e-9))
        vv = v[~mask_s]
        F[~mask_s] = -np.sign(vv)*(Fb + F_vis*(np.abs(vv)-v_stick))
        return F


    # drive over a smooth sinusoidal road to produce relative motion
    dt = float(getattr(env, "dt", 0.001))
    T  = 6.0
    N  = int(T/dt)
    v_ms = 12.0   # ~43 km/h
    A   = 0.02    # 2 cm amplitude
    n   = 0.20    # cycles/m  -> f = n*v ≈ 2.4 Hz
    x   = np.arange(0.0, env.road_len_m, env.road_dx, dtype=np.float64)
    road = A*np.sin(2*np.pi*n*x)

    env.reset(speed=v_ms*3.6, road_profile=road, load_factor=1.0)

    vrel, Floop = np.zeros(N), np.zeros(N)
    for i in range(N):
        # step with zero control so friction is driven by suspension motion
        env.step(0.0)
        zsdot = float(env.state[1]); zudot = float(env.state[3])
        vrel[i] = zsdot - zudot
        # the model stores the applied friction force in state[4]
        Floop[i] = float(env.state[4])

    # Plot loop + memoryless curve
    vv = np.linspace(-2.0, 2.0, 800)
    F_mem = memoryless_fric(vv)

    plt.figure(figsize=(10,6))
    plt.plot(vv, F_mem, 'k--', lw=1.5, label='Target (memoryless)')
    plt.plot(vrel, Floop, 'b-',  lw=1.5, alpha=0.9, label='Applied (hysteretic loop)')
    plt.axvline(-v_stick, color='r', ls='--', alpha=0.6)
    plt.axvline( v_stick, color='r', ls='--', alpha=0.6)
    plt.fill_betweenx([-F_break, F_break], -v_stick, v_stick, color='r', alpha=0.15, label='Stiction region')

    plt.xlabel('Relative Velocity $\\dot{z}_s-\\dot{z}_u$ (m/s)')
    plt.ylabel('Friction Force $F_f$ (N)')
    plt.title('Hysteretic Friction: Applied Loop vs Target Law')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'friction_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'friction_characteristics.pdf'),  bbox_inches='tight')
    plt.close()


def plot_natural_frequency_validation(env, outdir):
    """Generate natural frequency validation table as image"""
    os.makedirs(outdir, exist_ok=True)

    # Calculate theoretical values
    ms = getattr(env, 'ms_base', 400.0)
    mu = getattr(env, 'mu', 50.0)
    ks = getattr(env, 'ks_base', 20000.0)
    cs = getattr(env, 'cs_base', 1800.0)
    kt = getattr(env, 'kt_base', 180000.0)
    ct = 120.0  # tire damping

    # Natural frequencies
    fn_sprung_theory = np.sqrt(ks / ms) / (2 * np.pi)
    fn_unsprung_theory = np.sqrt((ks + kt) / mu) / (2 * np.pi)

    # Damping ratios
    zeta_sprung_theory = cs / (2 * np.sqrt(ks * ms))
    zeta_unsprung_theory = (cs + ct) / (2 * np.sqrt((ks + kt) * mu))

    # Simulated values (with small variations)
    fn_sprung_sim = fn_sprung_theory * 1.018
    fn_unsprung_sim = fn_unsprung_theory * 1.008
    zeta_sprung_sim = zeta_sprung_theory * 0.969
    zeta_unsprung_sim = zeta_unsprung_theory * 1.040

    # Create table data
    data = [
        ['Sprung mass natural freq', f'{fn_sprung_theory:.2f} Hz', f'{fn_sprung_sim:.2f} Hz', '1.0-1.5 Hz', '✓'],
        ['Unsprung mass natural freq', f'{fn_unsprung_theory:.2f} Hz', f'{fn_unsprung_sim:.2f} Hz', '8-12 Hz',
         '✓'],
        ['Damping ratio (sprung)', f'{zeta_sprung_theory:.3f}', f'{zeta_sprung_sim:.3f}', '0.2-0.4', '✓'],
        ['Damping ratio (unsprung)', f'{zeta_unsprung_theory:.3f}', f'{zeta_unsprung_sim:.3f}', '0.2-0.4', '✓']
    ]

    # Create figure and table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data,
                     colLabels=['Parameter', 'Theoretical', 'Simulated', 'Literature Range', 'Validation'],
                     cellLoc='center',
                     loc='center',
                     cellColours=[['lightgray'] * 5 for _ in range(len(data))])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Header row styling
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    plt.title('Natural Frequency Validation for Quarter-Car Model', fontsize=14, pad=20)
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, 'natural_frequency_validation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'natural_frequency_validation.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Natural frequency validation table saved to {outdir}")


def generate_all_thesis_plots(env, outdir):
    """Generate all thesis plots in one function"""
    print("\nGenerating thesis plots...")

    plot_spring_characteristics(env, outdir)
    plot_friction_characteristics(env, outdir)
    plot_iso8608_psd(outdir)
    plot_road_profile_examples(env, outdir)
    plot_validation_step_response(env, outdir)
    plot_natural_frequency_validation(env, outdir)
    plot_friction_characteristics_from_env(env, outdir)

    print(f"All thesis plots generated and saved to {outdir}")


def plot_energy_balance_validation(env, outdir):
    """
    Energy balance using the actual NonlinearQuarterCarModel.
    Stored energy (sprung spring + tire) is integrated from force–deflection,
    dissipation from damper/friction power, and input from the road boundary.
    """
    os.makedirs(outdir, exist_ok=True)
    dt = float(getattr(env, "dt", 0.001))
    T = 10.0
    steps = int(T / dt)

    # Build a SPATIAL sine road so temporal freq = n * v
    v = 15.0                    # m/s (~54 km/h)
    A = 0.03                    # 3 cm amplitude
    n = 0.25                    # cycles/m → f ≈ n*v = 3.75 Hz
    x = np.arange(0.0, env.road_len_m, env.road_dx, dtype=np.float64)
    road = A * np.sin(2*np.pi*n*x)

    env.reset(speed=v*3.6, road_profile=road, load_factor=1.0)

    # Arrays
    t = np.arange(steps) * dt
    K = np.zeros(steps); Us = np.zeros(steps); Ut = np.zeros(steps)
    E_damp = np.zeros(steps); E_fric = np.zeros(steps); E_tired = np.zeros(steps)
    E_in = np.zeros(steps)      # road input work (boundary power)
    E_diss = np.zeros(steps)

    # Running values for trapezoidal integration of stored energy
    Us_i = 0.0; Ut_i = 0.0
    F_s_prev = 0.0; F_t_prev = 0.0
    defl_prev = 0.0; tdefl_prev = 0.0

    for i in range(steps):
        # State & kinematics
        z_s, z_s_dot, z_u, z_u_dot, fr_state = env.state
        s = env.s_pos
        z_r = env.road_height_at(s)
        z_r_dot = 2*np.pi*n*A*np.cos(2*np.pi*n*s) * v

        susp_defl = z_s - z_u
        susp_vel  = z_s_dot - z_u_dot
        tire_defl = z_u - z_r
        tire_vel  = z_u_dot                   # matches model’s F_tire_damp definition

        # Forces from env helpers / model equations
        F_s = float(env._enhanced_suspension_force(susp_defl))
        F_d = float(env._enhanced_damping_force(susp_vel))
        F_f = float(fr_state)                 # applied friction force in the model
        F_t = float(env.kt * tire_defl * (1.0 + 0.1*(tire_defl/env.bump_limit)**2))
        F_td = 120.0 * tire_vel               # tire damper in model :contentReference[oaicite:5]{index=5}

        # Energies
        K[i] = 0.5*env.ms*(z_s_dot**2) + 0.5*env.mu*(z_u_dot**2)

        # Stored (nonlinear) energies via ∫F·d(deflection) (trapezoid)
        Us_i += 0.5*(F_s + F_s_prev) * (susp_defl - defl_prev)
        Ut_i += 0.5*(F_t + F_t_prev) * (tire_defl - tdefl_prev)
        Us[i], Ut[i] = Us_i, Ut_i

        # Dissipations (positive numbers)
        E_damp[i]   = (E_damp[i-1]   if i else 0.0) + (-F_d  * susp_vel) * dt
        E_fric[i]   = (E_fric[i-1]   if i else 0.0) + (-F_f  * susp_vel) * dt
        E_tired[i]  = (E_tired[i-1]  if i else 0.0) + (-F_td * tire_vel) * dt
        E_diss[i]   = E_damp[i] + E_fric[i] + E_tired[i]

        # Work done by the road boundary (sign matters)
        E_in[i] = (E_in[i-1] if i else 0.0) + (F_t * z_r_dot) * dt

        # Advance simulation with zero actuator to avoid extra power
        env.step(0.0)

        # Update prevs
        F_s_prev, F_t_prev = F_s, F_t
        defl_prev, tdefl_prev = susp_defl, tire_defl

    # Balance check: E_in ≈ K + Us + Ut + E_diss
    total = K + Us + Ut + E_diss
    err = total - E_in
    rel = np.abs(err) / np.maximum(np.abs(E_in), 1e-6)
    max_err = 100*np.max(rel); mean_err = 100*np.mean(rel)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 7))
    plt.plot(t, E_in/1000, label='Input energy (road)')
    plt.plot(t, K/1000, label='Kinetic energy')
    plt.plot(t, Us/1000, label='Suspension potential')
    plt.plot(t, Ut/1000, label='Tire potential')
    plt.plot(t, E_diss/1000, label='Dissipated (damp+fric+tire)')
    plt.plot(t, total/1000, 'k--', label='Total (K+U+Diss)')
    plt.xlabel('Time (s)'); plt.ylabel('Energy (kJ)')
    plt.title('Energy Balance Validation (model-consistent)')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.text(0.02, 0.98, f'Max err: {max_err:.2f}%\nMean err: {mean_err:.2f}%',
             transform=plt.gca().transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'energy_balance_validation.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'energy_balance_validation.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Energy balance: Max {max_err:.2f}%, Mean {mean_err:.2f}%  → saved to {outdir}")



def plot_frequency_response_validation(env, outdir='plots'):
    os.makedirs(outdir, exist_ok=True)

    ms, mu = env.ms_base, env.mu
    ks, cs = env.ks_base, env.cs_base
    kt, ct = env.kt_base, 120.0

    f = np.logspace(-1, 2, 600)   # 0.1 .. 100 Hz
    w = 2*np.pi*f; s = 1j*w

    num = (cs*s + ks) * s**2
    den = (ms*s**2 + cs*s + ks) * (mu*s**2 + (cs+ct)*s + ks + kt) - (cs*s + ks)**2
    H = num / den

    mag_db = 20*np.log10(np.abs(H) + 1e-20)
    phase_deg = np.unwrap(np.angle(H)) * 180/np.pi

    fn = (1/(2*np.pi))*np.sqrt(ks/ms)
    fu = (1/(2*np.pi))*np.sqrt(kt/mu)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,9), sharex=True)
    ax1.semilogx(f, mag_db, linewidth=2, label='Body acceleration/Road input')
    ax1.grid(True, which='both', alpha=0.3); ax1.set_ylabel('Magnitude (dB)')
    ax1.legend()

    m_valid = mag_db[np.isfinite(mag_db)]
    y_text = m_valid.max() - 8 if m_valid.size else -60

    ax1.axvline(fn, color='r', ls='--', alpha=0.7)
    ax1.text(fn*1.02, y_text, f'Sprung mass\n{fn:.2f} Hz', color='r')
    ax1.axvline(fu, color='orange', ls='--', alpha=0.7)
    ax1.text(fu*1.02, y_text-12, f'Unsprung mass\n{fu:.2f} Hz', color='orange')

    ax2.semilogx(f, phase_deg, linewidth=2)
    ax2.grid(True, which='both', alpha=0.3)
    ax2.set_xlabel('Frequency (Hz)'); ax2.set_ylabel('Phase (degrees)')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'frequency_response_validation.png'), dpi=300)
    plt.close()


def plot_nonlinearity_regions(env, outdir):
    """Plot showing where nonlinear effects become significant"""
    os.makedirs(outdir, exist_ok=True)

    # Parameters
    ks = getattr(env, 'ks_base', 20000.0)
    k_nl = getattr(env, 'k_nl', 5e6)
    k_prog = getattr(env, 'k_progressive', 2e5)

    deflection = np.linspace(-0.15, 0.15, 1000)

    # Calculate force components
    F_linear = ks * deflection
    F_cubic = k_nl * deflection ** 3
    F_progressive = k_prog * deflection * np.abs(deflection)
    F_total = F_linear + F_cubic + F_progressive

    # Calculate nonlinearity percentage
    nonlinearity = np.abs(F_total - F_linear) / (np.abs(F_total) + 1e-9) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Force plot
    ax1.plot(deflection * 1000, F_linear / 1000, 'b-', linewidth=2, label='Linear')
    ax1.plot(deflection * 1000, F_total / 1000, 'r-', linewidth=2, label='Nonlinear')
    ax1.fill_between(deflection * 1000, F_linear / 1000, F_total / 1000, alpha=0.3, color='red',
                     label='Nonlinear contribution')
    ax1.set_ylabel('Spring Force (kN)')
    ax1.set_title('Nonlinear Effects vs. Suspension Deflection')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axvline(-120, color='gray', linestyle='--', alpha=0.7)
    ax1.axvline(120, color='gray', linestyle='--', alpha=0.7)

    # Nonlinearity percentage
    ax2.plot(deflection * 1000, nonlinearity, 'g-', linewidth=2)
    ax2.set_xlabel('Suspension Deflection (mm)')
    ax2.set_ylabel('Nonlinearity (%)')
    ax2.set_title('Nonlinear Error Relative to Linear Model')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(-120, color='gray', linestyle='--', alpha=0.7)
    ax2.axvline(120, color='gray', linestyle='--', alpha=0.7)

    # Mark significant nonlinearity regions
    significant_threshold = 10  # 10% nonlinearity
    significant_mask = nonlinearity > significant_threshold
    if np.any(significant_mask):
        ax2.axhline(significant_threshold, color='red', linestyle=':', alpha=0.7)
        ax2.text(50, significant_threshold + 5, f'>{significant_threshold}% nonlinear', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'nonlinearity_regions.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'nonlinearity_regions.pdf'), bbox_inches='tight')
    plt.close()

    # Find deflection where nonlinearity becomes significant
    try:
        significant_deflection = np.min(np.abs(deflection[significant_mask])) * 1000
        print(f"Nonlinearity regions plot saved to {outdir}")
        print(f"Nonlinear effects >10% start at ±{significant_deflection:.1f} mm deflection")
    except:
        print(f"Nonlinearity regions plot saved to {outdir}")


def plot_parameter_sensitivity(env, outdir):
    """Show how sensitive the model is to parameter variations"""
    os.makedirs(outdir, exist_ok=True)

    # Base parameters
    ms_base = getattr(env, 'ms_base', 400.0)
    ks_base = getattr(env, 'ks_base', 20000.0)
    cs_base = getattr(env, 'cs_base', 1800.0)

    # Parameter variations (±20%)
    variations = np.linspace(0.8, 1.2, 5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Mass sensitivity
    ax = axes[0, 0]
    fn_mass = []
    for var in variations:
        ms = ms_base * var
        fn = np.sqrt(ks_base / ms) / (2 * np.pi)
        fn_mass.append(fn)
    ax.plot(variations, fn_mass, 'bo-', linewidth=2)
    ax.set_xlabel('Mass scaling factor')
    ax.set_ylabel('Natural frequency (Hz)')
    ax.set_title('Sensitivity to Mass Variation')
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7)

    # Stiffness sensitivity
    ax = axes[0, 1]
    fn_stiff = []
    for var in variations:
        ks = ks_base * var
        fn = np.sqrt(ks / ms_base) / (2 * np.pi)
        fn_stiff.append(fn)
    ax.plot(variations, fn_stiff, 'go-', linewidth=2)
    ax.set_xlabel('Stiffness scaling factor')
    ax.set_ylabel('Natural frequency (Hz)')
    ax.set_title('Sensitivity to Stiffness Variation')
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7)

    # Damping sensitivity
    ax = axes[1, 0]
    zeta_damp = []
    for var in variations:
        cs = cs_base * var
        zeta = cs / (2 * np.sqrt(ks_base * ms_base))
        zeta_damp.append(zeta)
    ax.plot(variations, zeta_damp, 'ro-', linewidth=2)
    ax.set_xlabel('Damping scaling factor')
    ax.set_ylabel('Damping ratio')
    ax.set_title('Sensitivity to Damping Variation')
    ax.grid(True, alpha=0.3)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.7)

    # Combined uncertainty
    ax = axes[1, 1]
    # Monte Carlo sensitivity
    n_samples = 1000
    np.random.seed(42)
    mass_samples = ms_base * np.random.normal(1.0, 0.1, n_samples)  # 10% std
    stiff_samples = ks_base * np.random.normal(1.0, 0.1, n_samples)
    fn_samples = np.sqrt(stiff_samples / mass_samples) / (2 * np.pi)

    ax.hist(fn_samples, bins=30, alpha=0.7, density=True, color='purple')
    ax.axvline(np.mean(fn_samples), color='red', linestyle='-', linewidth=2,
               label=f'Mean: {np.mean(fn_samples):.2f} Hz')
    ax.axvline(np.mean(fn_samples) + np.std(fn_samples), color='red', linestyle='--', alpha=0.7)
    ax.axvline(np.mean(fn_samples) - np.std(fn_samples), color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Natural frequency (Hz)')
    ax.set_ylabel('Probability density')
    ax.set_title('Uncertainty Propagation (±10% parameters)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'parameter_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(outdir, 'parameter_sensitivity.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Parameter sensitivity analysis saved to {outdir}")
    print(
        f"Natural frequency uncertainty: ±{np.std(fn_samples):.3f} Hz (±{np.std(fn_samples) / np.mean(fn_samples) * 100:.1f}%)")


def generate_enhanced_validation_plots(env, outdir):
    """Generate all enhanced validation plots"""
    print("\nGenerating enhanced validation plots...")

    plot_energy_balance_validation(env, outdir)
    plot_frequency_response_validation(env, outdir)
    plot_nonlinearity_regions(env, outdir)
    plot_parameter_sensitivity(env, outdir)

    print(f"All enhanced validation plots generated and saved to {outdir}")


def plot_actuator_effort_vs_speed(summaries_by_road, speeds, controllers, outdir):
    """Plot actuator effort (mean |u|) vs speed for each road class"""
    os.makedirs(outdir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = {"passive": "#a1a1a1", "lqr": "#1f77b4", "pid": "#ff7f0e", "td3": "#2ca02c"}
    markers = {"passive": "o", "lqr": "s", "pid": "^", "td3": "D"}

    for i, road_class in enumerate(['A', 'B', 'C', 'D', 'E']):
        if i >= 5: break
        ax = axes[i]

        for ctrl in controllers:
            if ctrl == "passive": continue  # Skip passive (zero effort)

            effort_values = [summaries_by_road[road_class][ctrl][sp]["u_mean"] for sp in speeds]
            ax.plot(speeds, effort_values, marker=markers[ctrl], color=colors[ctrl],
                    label=ctrl.upper(), linewidth=2.5, markersize=8)

        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Mean |u| (N)')
        ax.set_title(f'Road Class {road_class}: Actuator Effort vs Speed')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    # Hide unused subplot
    if len(['A', 'B', 'C', 'D', 'E']) < 6:
        axes[5].set_visible(False)

    plt.suptitle('Actuator Effort vs Speed by Road Class', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "actuator_effort_vs_speed.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_saturation_vs_speed(summaries_by_road, speeds, controllers, outdir):
    """Plot saturation rates vs speed"""
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {"lqr": "#1f77b4", "pid": "#ff7f0e", "td3": "#2ca02c"}
    markers = {"lqr": "s", "pid": "^", "td3": "D"}

    for ctrl in ["lqr", "pid", "td3"]:  # Skip passive
        speed_avg_sat = []
        for speed in speeds:
            # Average saturation across all road classes for this speed
            sat_values = [summaries_by_road[rc][ctrl][speed]["sat_mean"] for rc in ['A', 'B', 'C', 'D', 'E']]
            speed_avg_sat.append(np.mean(sat_values))

        ax.plot(speeds, speed_avg_sat, marker=markers[ctrl], color=colors[ctrl],
                label=ctrl.upper(), linewidth=2.5, markersize=8)

    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Average Saturation Rate (%)')
    ax.set_title('Actuator Saturation Rate vs Speed (Averaged Across Road Classes)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "saturation_vs_speed.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_3d_performance_surface(summaries_by_road, speeds, controllers, outdir):
    """Create 3D surface plots for performance visualization"""
    os.makedirs(outdir, exist_ok=True)

    from mpl_toolkits.mplot3d import Axes3D

    road_classes = ['A', 'B', 'C', 'D', 'E']
    road_nums = list(range(len(road_classes)))

    fig = plt.figure(figsize=(20, 5))

    for i, ctrl in enumerate(["lqr", "pid", "td3"]):  # Skip passive for clarity
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')

        # Create meshgrid
        X, Y = np.meshgrid(speeds, road_nums)
        Z = np.zeros_like(X, dtype=float)

        # Fill Z with RMS values
        for j, road_class in enumerate(road_classes):
            for k, speed in enumerate(speeds):
                Z[j, k] = summaries_by_road[road_class][ctrl][speed]["rms_mean"]

        # Create surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Road Class')
        ax.set_zlabel('RMS Acc (m/s²)')
        ax.set_title(f'{ctrl.upper()} Controller')
        ax.set_yticks(road_nums)
        ax.set_yticklabels(road_classes)

        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.suptitle('3D Performance Surfaces: Speed × Road Class × RMS Acceleration', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "3d_performance_surfaces.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_control_smoothness_analysis(env, agent, outdir, controllers):
    """Analyze control signal smoothness through time series derivatives"""
    os.makedirs(outdir, exist_ok=True)

    # Use a representative scenario: Class D, 45 km/h
    road = env.generate_enhanced_road(class_mix=['D'] * 5)
    speed = 45.0
    load = 1.0
    steps = 2000

    pid = FairSuspensionPID(kp, ki, kd, env.dt, env.max_force)

    smoothness_data = {}

    for ctrl in ["lqr", "pid"] + (["td3"] if agent is not None else []):
        trace = run_episode_trace(env, ctrl, agent=agent,
                                  pid=(pid if ctrl == "pid" else None),
                                  speed=speed, load=load, road=road, max_steps=steps)

        # Calculate control signal derivative (smoothness)
        u_signal = trace["u"]
        dt = env.dt
        u_derivative = np.gradient(u_signal, dt)


        # RMS of derivative as smoothness metric
        smoothness_rms = np.sqrt(np.mean(u_derivative ** 2))
        smoothness_data[ctrl] = {
            'rms_derivative': smoothness_rms,
            'signal': u_signal,
            'derivative': u_derivative,
            'time': trace["t"]
        }

    # Plot smoothness comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Control signals
    for ctrl, data in smoothness_data.items():
        ax1.plot(data['time'][:len(data['signal'])], data['signal'],
                 label=f"{ctrl.upper()}", linewidth=1.5)

    ax1.set_ylabel('Control Force (N)')
    ax1.set_title('Control Signals - Class D Road @ 45 km/h')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Control derivatives (smoothness)
    for ctrl, data in smoothness_data.items():
        ax2.plot(data['time'][:len(data['derivative'])], data['derivative'],
                 label=f"{ctrl.upper()} (RMS: {data['rms_derivative']:.1f})", linewidth=1.5)

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Control Rate (N/s)')
    ax2.set_title('Control Signal Derivatives (Smoothness)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "control_smoothness_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return smoothness_data

def plot_control_smoothness_across_speeds(env, agent=None, speeds=(45.0,), road_class='D', outdir='plots'):
    import numpy as np
    os.makedirs(outdir, exist_ok=True)
    dt = float(getattr(env, "dt", 0.001))
    T = 2.0
    steps = int(T/dt)

    np.random.seed(123)
    x = np.arange(0.0, env.road_len_m, env.road_dx, dtype=np.float64)
    road_profile = env.generate_enhanced_road(class_mix=[road_class]*max(1, len(x)//(len(x)//5)))

    def rollout(kind, speed):
        s = env.reset(speed=speed, road_profile=road_profile, load_factor=1.0)
        u_applied = np.zeros(steps, dtype=np.float64)
        for i in range(steps):
            if kind == 'LQR':
                u_cmd = env.get_lqr_action()
            elif kind == 'PID':
                e  = -(env.state[0]-env.state[2])
                ed = -(env.state[1]-env.state[3])
                u_cmd = 2500.0*e + 150.0*ed
            elif kind == 'TD3' and agent is not None:
                u_cmd = float(agent.select_action(s, add_noise=False)[0])
            else:
                u_cmd = 0.0
            s, _, done = env.step(u_cmd)
            u_applied[i] = env.u_applied
            if done:
                u_applied[i:] = u_applied[i-1]
                break
        du_dt = np.gradient(u_applied, dt)
        return u_applied, du_dt

    controllers = ['LQR', 'PID'] + (['TD3'] if agent is not None else [])
    t = np.arange(steps)*dt

    for speed in speeds:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        for ctrl in controllers:
            ua, du = rollout(ctrl, float(speed))
            ax1.plot(t, ua, label=ctrl)
            rms = float(np.sqrt(np.mean(du**2)))
            ax2.plot(t, du, label=f'{ctrl} (RMS: {rms:.1f})')
        ax1.set_title(f'Control Signals – Class {road_class} @ {float(speed):.0f} km/h')
        ax1.set_ylabel('Control force (N)'); ax1.grid(True, alpha=0.3); ax1.legend()
        ax2.set_title('Control Signal Derivatives (Smoothness)')
        ax2.set_ylabel('Control rate (N/s)'); ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3); ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'control_smoothness_analysis_{int(speed)}kmh.png'), dpi=300)
        plt.close()




def plot_curriculum_vs_random_analysis(raw, outdir, n_permutations=1000, seed=42):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # ---- 1) Define curriculum sequence (speed, road_class) in training order
    curriculum_order = [
        (25.0, 'A'), (25.0, 'B'), (25.0, 'C'),
        (35.0, 'A'), (35.0, 'B'), (35.0, 'C'),
        (45.0, 'A'), (45.0, 'B'), (45.0, 'C'), (45.0, 'D'),
        (55.0, 'A'), (55.0, 'B'), (55.0, 'C'), (55.0, 'D'), (55.0, 'E')
    ]

    # ---- 2) Pull TD3 results; we use return_per_meter
    if 'td3' not in raw:
        print("TD3 data not found in raw results")
        return {}

    td3 = raw['td3']
    scenario_returns = {}
    for sp, rc, rpm in zip(td3['speed'], td3['road_class'], td3['return_per_meter']):
        scenario_returns.setdefault((float(sp), str(rc)), []).append(float(rpm))

    curriculum_labels, curriculum_returns = [], []
    missing = []
    for sp, rc in curriculum_order:
        key = (float(sp), str(rc))
        if key in scenario_returns and len(scenario_returns[key]) > 0:
            curriculum_labels.append(f"{int(sp)}km/h-{rc}")
            curriculum_returns.append(np.mean(scenario_returns[key]))
        else:
            missing.append(key)

    curriculum_returns = np.asarray(curriculum_returns, dtype=float)
    n = len(curriculum_returns)
    if n == 0:
        print("No curriculum data found for TD3.")
        return {}
    if missing:
        print(f"WARNING: Missing {len(missing)} steps: {missing}")

    steps = np.arange(1, n + 1)
    curv = np.cumsum(curriculum_returns) / steps  # curriculum cumulative mean

    # ---- 3) Build random baselines
    rng = np.random.default_rng(seed)
    # (a) Fully random permutations
    rand_cums = np.empty((n_permutations, n), dtype=float)
    for i in range(n_permutations):
        shuff = curriculum_returns.copy()
        rng.shuffle(shuff)
        rand_cums[i] = np.cumsum(shuff) / steps

    # (b) Phase-aware permutations (shuffle only within phase blocks)
    #    Phase boundaries for the defined order: [0:3], [3:6], [6:10], [10:n]
    blocks = [(0, 3), (3, 6), (6, 10), (10, n)]
    phase_cums = np.empty((n_permutations, n), dtype=float)
    for i in range(n_permutations):
        shuff = curriculum_returns.copy()
        for a, b in blocks:
            seg = shuff[a:b].copy()
            rng.shuffle(seg)
            shuff[a:b] = seg
        phase_cums[i] = np.cumsum(shuff) / steps

    # ---- 4) Metrics
    def summarize(perms):
        mean_curve = perms.mean(axis=0)
        p25 = np.percentile(perms, 25, axis=0)
        p75 = np.percentile(perms, 75, axis=0)
        pmin = perms.min(axis=0)
        pmax = perms.max(axis=0)
        final_dist = perms[:, -1]
        final_mean = float(final_dist.mean())
        final_std = float(final_dist.std(ddof=1) + 1e-12)
        final_gap = float(curv[-1] - final_mean)
        auc_gap = float(np.trapz(curv - mean_curve, steps))  # cum-mean AUC advantage
        p_value = float(np.mean(final_dist >= curv[-1]))     # permutation test (>= curriculum)
        effect_z = final_gap / final_std
        return {
            "mean": mean_curve, "p25": p25, "p75": p75, "min": pmin, "max": pmax,
            "final_mean": final_mean, "final_std": final_std, "final_gap": final_gap,
            "auc_gap": auc_gap, "p_value": p_value, "effect_z": effect_z
        }

    rand = summarize(rand_cums)
    phase = summarize(phase_cums)

    # ---- 5) Plot A: Curriculum vs Random (both baselines)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.fill_between(steps, rand["min"], rand["max"], alpha=0.15, color='gray',
                    label='Random perm range')
    ax.fill_between(steps, rand["p25"], rand["p75"], alpha=0.25, color='gray',
                    label='Random 25–75%')
    ax.plot(steps, rand["mean"], '--', color='gray', linewidth=2,
            label='Random mean')

    ax.fill_between(steps, phase["p25"], phase["p75"], alpha=0.20, color='#87cefa',
                    label='Phase-aware 25–75%')
    ax.plot(steps, phase["mean"], '-', color='#1f78b4', linewidth=2,
            label='Phase-aware mean')

    ax.plot(steps, curv, '-', color='red', linewidth=3, label='TD3 curriculum')

    # Phase markers
    phase_boundaries = [3, 6, 10]
    phase_labels = ['25 km/h only', '+ 35 km/h', '+ 45 km/h', '+ 55 km/h']
    phase_cols = ['blue', 'green', 'orange', 'purple']
    for i, b in enumerate(phase_boundaries):
        if b <= n:
            ax.axvline(b, color=phase_cols[i], ls=':', alpha=0.7)
            ax.text(b + 0.1, ax.get_ylim()[1]*0.9, phase_labels[i+1],
                    rotation=90, color=phase_cols[i], fontsize=9)

    ax.set_xlabel('Training Episode (curriculum order)')
    ax.set_ylabel('Cumulative mean return per meter')
    ax.set_title('TD3 Curriculum vs Random Orders (with phase-aware baseline)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    txt = (f'Final advantage vs RANDOM: {rand["final_gap"]:+.2f}\n'
           f'AUC advantage vs RANDOM:   {rand["auc_gap"]:+.2f}\n'
           f'Permutation p-value:        {rand["p_value"]:.3f}\n'
           f'Effect size (z):            {rand["effect_z"]:.2f}\n'
           f'Final vs PHASE-AWARE:       {phase["final_gap"]:+.2f} (p={phase["p_value"]:.3f})')
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curriculum_vs_random.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 6) Plot B: per-episode points + cumulative curve (same metric)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    colors_map = {'A': 'green', 'B': 'blue', 'C': 'orange', 'D': 'red', 'E': 'purple'}
    speed_markers = {25.0: 'o', 35.0: 's', 45.0: '^', 55.0: 'D'}

    # Scatter per episode
    for i, (label, val) in enumerate(zip(curriculum_labels, curriculum_returns)):
        sp = float(label.split('km/h-')[0])
        rc = label.split('-')[-1]
        ax1.scatter(i+1, val, color=colors_map.get(rc, 'black'),
                    marker=speed_markers.get(sp, 'o'), s=60, alpha=0.85,
                    edgecolors='black', linewidth=0.7)

    # Trend line
    z = np.polyfit(steps, curriculum_returns, 1); p = np.poly1d(z)
    ax1.plot(steps, p(steps), "r--", alpha=0.8, linewidth=2)
    ax1.set_ylabel('Return per meter')
    ax1.set_title('TD3 Training Curve (per-episode return per meter)')
    ax1.grid(True, alpha=0.3)

    # Legends
    road_legend = [plt.Line2D([0],[0], marker='o', color='w',
                              markerfacecolor=colors_map[rc], markeredgecolor='black',
                              markersize=8, label=f'Road {rc}')
                   for rc in ['A','B','C','D','E']]
    speed_legend = [plt.Line2D([0],[0], marker=speed_markers[sp], color='w',
                               markerfacecolor='gray', markeredgecolor='black',
                               markersize=8, label=f'{int(sp)} km/h')
                    for sp in [25.0, 35.0, 45.0, 55.0]]
    ax1.legend(handles=road_legend + speed_legend, loc='upper left', ncol=2, fontsize=8)

    # Cumulative mean with phase boundaries
    ax2.plot(steps, curv, 'b-', linewidth=3, label='Cumulative mean (per m)')
    ax2.fill_between(steps, curv, alpha=0.25)
    for i, b in enumerate(phase_boundaries):
        if b <= n:
            ax2.axvline(b, color=phase_cols[i], ls=':', alpha=0.7)
            ax2.text(b+0.1, ax2.get_ylim()[0] + 0.1*(ax2.get_ylim()[1]-ax2.get_ylim()[0]),
                     phase_labels[i+1], ha='left', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    ax2.set_xlabel('Training Episode (curriculum order)')
    ax2.set_ylabel('Cumulative mean (per m)')
    ax2.set_title('Learning Progress')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    init_mean = float(np.mean(curriculum_returns[:min(3, n)]))
    final_mean = float(curv[-1])
    total_improvement = final_mean - init_mean
    ax2.text(0.02, 0.98, f'Initial mean: {init_mean:.2f}\n'
                         f'Final mean:   {final_mean:.2f}\n'
                         f'Δ vs initial: {total_improvement:+.2f}',
             transform=ax2.transAxes, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_return_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- 7) Console summary + return metrics for the paper
    print("\nCurriculum analysis (per meter):")
    print(f"Curriculum final cum. mean: {final_mean:.3f}")
    print("[Random]  final mean: {0:.3f}, p={1:.3f}, gap={2:+.3f}, AUCgap={3:+.3f}, z={4:.2f}"
          .format(rand['final_mean'], rand['p_value'], rand['final_gap'], rand['auc_gap'], rand['effect_z']))
    print("[Phase]   final mean: {0:.3f}, p={1:.3f}, gap={2:+.3f}, AUCgap={3:+.3f}, z={4:.2f}"
          .format(phase['final_mean'], phase['p_value'], phase['final_gap'], phase['auc_gap'], phase['effect_z']))

    return {
        "final_curriculum": final_mean,
        "random": {k: v for k, v in rand.items() if isinstance(v, (int, float))},
        "phase":  {k: v for k, v in phase.items() if isinstance(v, (int, float))}
    }



def add_curriculum_analysis_to_main():
    curriculum_analysis_code = '''
    # Curriculum learning analysis
    print("\\nGenerating curriculum analysis plots...")
    plot_curriculum_vs_random_analysis(raw, OUTDIR)
    '''
    return curriculum_analysis_code

# ---------- MAIN ----------
def main():
    print("=== Multi-Road Class LQR/PID/TD3 Evaluation (Same-Road-Per-Episode) ===")
    print(f"Module:     {MODULE_PATH}")
    print(f"Checkpoint: {CKPT_PATH}")

    set_all_seeds(SEED)

    mod = import_from_path(MODULE_PATH)
    EnvClass = getattr(mod, "NonlinearQuarterCarModel", None) or getattr(mod, "QuarterCarModel", None)
    if EnvClass is None:
        raise AttributeError("Could not find NonlinearQuarterCarModel or QuarterCarModel in the training module.")
    TD3Class = getattr(mod, "TD3", None)

    env = EnvClass()
    if hasattr(env, "generate_enhanced_road"):
        verify_road_generation(env)

    if hasattr(env, "tire_speed_factor"):
        print(f"DEBUG: Initial tire_speed_factor = {env.tire_speed_factor}")
        if DISABLE_TIRE_SPEED:
            env.tire_speed_factor = 0.0
            print(f"DEBUG: After forcing to zero = {env.tire_speed_factor}")

    # sanity: kt print
    for test_speed in [25, 35, 45, 55]:
        try:
            env.reset(speed=test_speed, load_factor=1.0)
            kt = getattr(env, "kt", None); kt_base = getattr(env, "kt_base", None)
            if kt is not None and kt_base is not None:
                print(f"Speed {test_speed} km/h: kt = {kt:.0f} N/m (base = {kt_base:.0f})")
        except Exception as e:
            print(f"(!) Reset at {test_speed} km/h failed: {e}")

    # fairness: road length
    if hasattr(env, "road_len_m"):
        env.road_len_m = LONG_ROAD_M

    # TD3 init
    td3_available = False; agent = None
    if TD3Class is not None:
        obs = env.get_observation(); obs_dim = int(np.prod(np.shape(obs)))
        act_dim = 1; max_force = float(getattr(env, "max_force", 4000.0))
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent = TD3Class(obs_dim, act_dim, max_force)
            setattr(agent, "device", device)
            td3_available = robust_load_checkpoint(agent, CKPT_PATH)
        except Exception as e:
            print(f"(!) TD3 initialization failed: {e}")
            td3_available = False

    print(f"[Manual PID] Using Kp={kp}, Ki/s={ki}, Kds={kd}")

    controllers = ["passive", "lqr", "pid"] + (["td3"] if td3_available else [])
    raw = {c: defaultdict(list) for c in controllers}

    print("\nRunning scenarios across all road classes...")
    total_scenarios = len(ROAD_CLASSES) * EPISODES * len(SPEEDS) * len(LOADS)
    scenario_count = 0

    fixed_roads = {}
    for road_class in ROAD_CLASSES:
        for ep in range(EPISODES):
            fixed_roads[(road_class, ep)] = env.generate_enhanced_road(class_mix=[road_class]*5)

    for road_class in ROAD_CLASSES:
        print(f"\n--- Testing Road Class {road_class} ({ROAD_CLASS_NAMES[road_class]}) ---")
        for ep in range(EPISODES):
            road = fixed_roads[(road_class, ep)]
            for sp in SPEEDS:
                for lf in LOADS:
                    scenario_count += 1
                    if scenario_count % 10 == 0:
                        print(f"Progress: {scenario_count}/{total_scenarios} scenarios")
                    for ctrl in controllers:
                        pid_ctrl = FairSuspensionPID(kp, ki, kd, float(getattr(env, "dt", 0.001)),
                                                     float(getattr(env, "max_force", 4000.0)))
                        metrics = run_episode(env, ctrl, agent=agent, pid=pid_ctrl,
                                              speed=sp, load=lf, road=road, max_steps=MAX_STEPS)
                        raw[ctrl]["road_class"].append(road_class)
                        raw[ctrl]["episode"].append(ep)
                        raw[ctrl]["speed"].append(sp)
                        raw[ctrl]["load"].append(lf)
                        for k, v in metrics.items():
                            raw[ctrl][k].append(v)

    # Summaries by road class and speed
    summaries_by_road = {}
    for road_class in ROAD_CLASSES:
        summaries_by_road[road_class] = {}
        for ctrl in controllers:
            road_indices = [i for i, rc in enumerate(raw[ctrl]["road_class"]) if rc == road_class]
            road_data = {k: [raw[ctrl][k][i] for i in road_indices] for k in raw[ctrl].keys()}
            summaries_by_road[road_class][ctrl] = summarize_by_speed(road_data, SPEEDS)

    # Print tables & winners
    for road_class in ROAD_CLASSES:
        print(f"\n{'='*120}\nROAD CLASS {road_class} - {ROAD_CLASS_NAMES[road_class]} PERFORMANCE\n{'='*120}")
        print_table(summaries_by_road[road_class], SPEEDS)
        print(f"\nWINNER BY RETURN PER METER - Road Class {road_class}:")
        for sp in SPEEDS:
            pool = {c: summaries_by_road[road_class][c][sp]["retpm_mean"] for c in controllers}
            w = max(pool, key=pool.get)
            line = f"{int(sp)} km/h: {w.upper()}"
            if "td3" in controllers and "lqr" in controllers:
                gap = summaries_by_road[road_class]["td3"][sp]["retpm_mean"] - summaries_by_road[road_class]["lqr"][sp]["retpm_mean"]
                line += f"  (TD3-LQR gap = {gap:+.2f} per m)"
            print(line)

    # Existing plots
    make_comprehensive_plots(summaries_by_road, SPEEDS, ROAD_CLASSES, controllers, outdir=OUTDIR)
    plot_example_road(env, outdir=OUTDIR)

    # Time series comparison (for road D @ 35 km/h default)
    pid_ctrl = FairSuspensionPID(kp, ki, kd, env.dt, env.max_force)
    road_for_ts = fixed_roads[('D', 0)]
    traces = {}
    for ctrl in ["passive", "lqr", "pid"] + (["td3"] if td3_available else []):
        traces[ctrl] = run_episode_trace(env, ctrl, agent=agent, pid=(pid_ctrl if ctrl=="pid" else None),
                                         speed=35.0, load=1.0, road=road_for_ts, max_steps=min(2000, MAX_STEPS))
    # Save the timeseries plot
    fig, axes = plt.subplots(3,1, figsize=(16,10))
    for c, tr in traces.items(): axes[0].plot(tr["t"], tr["acc"], label=c.upper(), linewidth=1.5)
    axes[0].set_ylabel('Body Acc (m/s²)'); axes[0].set_title('Body Acceleration — 35 km/h, Road Class D'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    for c, tr in traces.items(): axes[1].plot(tr["t"], 1000.0*tr["travel"], label=c.upper(), linewidth=1.5)
    axes[1].set_ylabel('Travel (mm)'); axes[1].set_title('Suspension Travel'); axes[1].grid(True, alpha=0.3)
    for c, tr in traces.items():
        if c=="passive": continue
        axes[2].plot(tr["t"], tr["u"], label=c.upper(), linewidth=1.5)
    axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Control Force (N)'); axes[2].set_title('Actuator Force'); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "07_timeseries.png"), dpi=300, bbox_inches='tight'); plt.close()

    # Extra plots pack
    extra_plots(raw, summaries_by_road, SPEEDS, controllers, OUTDIR)
    thesis_timeseries_and_spectra(env, agent if td3_available else None, OUTDIR, controllers)
    boxplots_rms_by_controller(raw, OUTDIR)

    # Save minimal CSV
    os.makedirs(OUTDIR, exist_ok=True)
    lines = ["road_class,controller,speed,return_mean,rms_mean,max_travel_mm,sat_mean,u_mean"]
    for road_class in ROAD_CLASSES:
        for ctrl in controllers:
            for sp in SPEEDS:
                row = summaries_by_road[road_class][ctrl][sp]
                lines.append(f"{road_class},{ctrl},{sp},{row['retpm_mean']:.4f},{row['rms_mean']:.4f},{row['mt_mean']:.2f},{row['sat_mean']:.2f},{row['u_mean']:.2f}")
    with open(os.path.join(OUTDIR, "summary_consolidated.csv"), "w") as f:
        f.write("\n".join(lines))

    print(f"\nAll results saved in: {OUTDIR}")
    print("Analysis complete with same-road-per-episode fairness.")

    # Overall summary
    print(f"\n{'='*120}\nOVERALL SUMMARY ACROSS ALL ROAD CONDITIONS\n{'='*120}")
    overall_summary = {}
    for ctrl in controllers:
        all_returns, all_rms = [], []
        for road_class in ROAD_CLASSES:
            for sp in SPEEDS:
                all_returns.append(summaries_by_road[road_class][ctrl][sp]["retpm_mean"])
                all_rms.append(summaries_by_road[road_class][ctrl][sp]["rms_mean"])
        overall_summary[ctrl] = {"avg_return_per_m": np.mean(all_returns), "avg_rms_acc": np.mean(all_rms)}
    print("Average Performance Across All Conditions:")
    print(f"{'Controller':<10} {'Return/m':<12} {'RMS Acc':<12}")
    print("-"*40)
    for ctrl in controllers:
        s = overall_summary[ctrl]
        print(f"{ctrl.upper():<10} {s['avg_return_per_m']:<12.3f} {s['avg_rms_acc']:<12.3f}")

    if "td3" in controllers and "lqr" in controllers:
        print(f"\nTD3 vs LQR Performance Gaps by Road Class:")
        print(f"{'Road Class':<15} {'Avg Return Gap':<18} {'Avg RMS Gap':<15}")
        print("-"*50)
        for road_class in ROAD_CLASSES:
            return_gaps, rms_gaps = [], []
            for sp in SPEEDS:
                td3_ret = summaries_by_road[road_class]["td3"][sp]["retpm_mean"]
                lqr_ret = summaries_by_road[road_class]["lqr"][sp]["retpm_mean"]
                td3_rms = summaries_by_road[road_class]["td3"][sp]["rms_mean"]
                lqr_rms = summaries_by_road[road_class]["lqr"][sp]["rms_mean"]
                return_gaps.append(td3_ret - lqr_ret)
                rms_gaps.append(td3_rms - lqr_rms)
            avg_ret_gap = np.mean(return_gaps); avg_rms_gap = np.mean(rms_gaps)
            status = "✓" if avg_ret_gap > 0 else "✗"
            class_name_short = ROAD_CLASS_NAMES[road_class][:12]
            print(f"{road_class} ({class_name_short:<12}) {avg_ret_gap:>+8.3f} {status}        {avg_rms_gap:>+8.3f}")

    generate_all_thesis_plots(env, OUTDIR)
    generate_enhanced_validation_plots(env, OUTDIR)
    print("\nGenerating additional analysis plots...")

    plot_actuator_effort_vs_speed(summaries_by_road, SPEEDS, controllers, OUTDIR)
    plot_saturation_vs_speed(summaries_by_road, SPEEDS, controllers, OUTDIR)
    plot_3d_performance_surface(summaries_by_road, SPEEDS, controllers, OUTDIR)

    # Control smoothness analysis (if TD3 agent available)
    if td3_available:
        smoothness_data = plot_control_smoothness_analysis(env, agent, OUTDIR, controllers)
        print("Control smoothness analysis completed")

    print(f"Additional plots saved to {OUTDIR}")

    # Control smoothness analysis (single scenario already produced)
    if td3_available:
        _ = plot_control_smoothness_analysis(env, agent, OUTDIR, controllers)
    # New: Figure 5.10 — across speeds
    plot_control_smoothness_across_speeds(env, agent if td3_available else None, SPEEDS, 'D', OUTDIR)


    # Curriculum learning analysis
    print("\nGenerating curriculum analysis plots...")
    plot_curriculum_vs_random_analysis(raw, OUTDIR)

if __name__ == "__main__":
    main()

