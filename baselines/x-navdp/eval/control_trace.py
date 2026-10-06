"""Common command/response logging and plots for direct and MPC evaluation."""

import csv
from pathlib import Path

import numpy as np


FIELDS = (
    "mode", "sim_time_s", "episode_time_s", "env_id", "episode_idx",
    "plan_id", "command_valid", "command_v_mps", "command_w_radps",
    "measured_vx_mps", "measured_speed_mps", "measured_w_radps",
    "goal_distance_m",
)


class ControlTraceRecorder:
    """Stream control traces to CSV and render a diagnostic figure at shutdown."""

    def __init__(self, output_dir, mode):
        self.output_dir = Path(output_dir)
        self.mode = str(mode)
        self.csv_path = self.output_dir / "control_output.csv"
        self.figure_path = self.output_dir / "control_output.png"
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        self.rows = []

    def write_batch(self, sim_time, episode_time, episode_indices, plan_id,
                    command_valid, command, linear_velocity, angular_velocity,
                    goal_distance):
        command = np.asarray(command, dtype=np.float32)
        linear_velocity = np.asarray(linear_velocity, dtype=np.float32)
        angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
        episode_time = np.asarray(episode_time)
        goal_distance = np.asarray(goal_distance)
        batch = command.shape[0]
        for env_id in range(batch):
            episode_idx = episode_indices[env_id]
            row = {
                "mode": self.mode,
                "sim_time_s": float(sim_time),
                "episode_time_s": float(episode_time[env_id]),
                "env_id": int(env_id),
                "episode_idx": -1 if episode_idx is None else int(episode_idx),
                "plan_id": -1 if plan_id is None else int(plan_id),
                "command_valid": int(bool(command_valid)),
                "command_v_mps": float(command[env_id, 0]),
                "command_w_radps": float(command[env_id, 1]),
                "measured_vx_mps": float(linear_velocity[env_id, 0]),
                "measured_speed_mps": float(np.linalg.norm(linear_velocity[env_id, :2])),
                "measured_w_radps": float(angular_velocity[env_id, 2]),
                "goal_distance_m": float(goal_distance[env_id]),
            }
            self._writer.writerow(row)
            self.rows.append(row)
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
        if self.rows:
            plot_control_trace(self.rows, self.figure_path, self.mode)


def plot_control_trace(rows, output_path, mode=None):
    """Plot commanded and measured linear/angular velocities against sim time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    mode = mode or rows[0].get("mode", "control")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)
    env_ids = sorted({int(row["env_id"]) for row in rows})
    colors = plt.cm.tab10(np.linspace(0., 1., max(1, min(10, len(env_ids)))))
    for color, env_id in zip(colors, env_ids):
        selected = [row for row in rows if int(row["env_id"]) == env_id]
        t = np.asarray([float(row["sim_time_s"]) for row in selected])
        cmd_v = np.asarray([float(row["command_v_mps"]) for row in selected])
        cmd_w = np.asarray([float(row["command_w_radps"]) for row in selected])
        measured_v = np.asarray([float(row["measured_vx_mps"]) for row in selected])
        measured_w = np.asarray([float(row["measured_w_radps"]) for row in selected])
        suffix = f" env {env_id}" if len(env_ids) > 1 else ""
        axes[0].plot(t, cmd_v, color=color, linewidth=1.0,
                     label=f"command{suffix}")
        axes[0].plot(t, measured_v, color=color, linewidth=.9, alpha=.62,
                     linestyle="--", label=f"measured{suffix}")
        axes[1].plot(t, cmd_w, color=color, linewidth=1.0,
                     label=f"command{suffix}")
        axes[1].plot(t, measured_w, color=color, linewidth=.9, alpha=.62,
                     linestyle="--", label=f"measured{suffix}")

        episode = [int(row["episode_idx"]) for row in selected]
        for index in range(1, len(selected)):
            if episode[index] != episode[index-1]:
                for axis in axes:
                    axis.axvline(t[index], color="0.82", linewidth=.55, zorder=0)

    axes[0].axhline(0., color="0.4", linewidth=.6)
    axes[1].axhline(0., color="0.4", linewidth=.6)
    axes[0].set_ylabel("linear velocity v (m/s)")
    axes[1].set_ylabel("angular velocity w (rad/s)")
    axes[1].set_xlabel("simulation time (s); gray lines mark episode resets")
    axes[0].set_title(f"Control output and robot response — {mode}")
    for axis in axes:
        axis.grid(True, alpha=.2)
        axis.legend(loc="upper right", ncol=min(4, 2*len(env_ids)), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
