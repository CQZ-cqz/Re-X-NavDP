"""Opt-in recording subclass. The original policy agent is not edited."""

import rexnavdp  # noqa: F401  (sys.path bootstrap)
import time
import numpy as np
import torch

from .fm_data import SCHEMA, save_record, validate_observation
from eval.src.policy_agent import NavDP_Agent


def recording_agent_class(run_dir, manifest):
    class RecordingAgent(NavDP_Agent):
        def reset(self, batch_size, *args, **kwargs):
            if int(batch_size) != 1:
                raise ValueError("FM minimal capture supports num_envs=1 only")
            super().reset(batch_size, *args, **kwargs)
            self._fm_reset_history()

        def reset_env(self, i):
            super().reset_env(i)
            self._fm_reset_history()

        def _fm_reset_history(self):
            self._fm_episode = getattr(self, "_fm_episode", 0)+1
            self._fm_step = 0
            self._fm_times = np.full(self.history_window, -1., dtype=np.float64)
            self._fm_steps = np.full(self.history_window, -1, dtype=np.int64)

        def _update_and_sample_history(self, process_images, num_samples=None):
            count = self.memory_size-1 if num_samples is None else num_samples
            indices = np.linspace(0, self.history_window-1, count-1).astype(np.int64)
            self._fm_rgb_times = np.append(self._fm_times[indices], self._fm_feedback["sim_time"])
            self._fm_rgb_steps = np.append(self._fm_steps[indices], self._fm_step)
            result = super()._update_and_sample_history(process_images, num_samples)
            if ((self.frame_count[0]-1) % self.frame_interval == 0) and self.frame_count[0] > 0:
                self._fm_times[:-1] = self._fm_times[1:].copy()
                self._fm_steps[:-1] = self._fm_steps[1:].copy()
                self._fm_times[-1] = self._fm_feedback["sim_time"]
                self._fm_steps[-1] = self._fm_step
            return result

        def _save_observation(self, rgb_history, depth, goal, robot_pos, robot_quat,
                              prev_action=None, valid_segment_len=None):
            if robot_pos is None or robot_quat is None or prev_action is None or valid_segment_len is None:
                raise ValueError("FM capture requires actual pose and history")
            scene = self.current_scene_name
            if scene != manifest["scene"]:
                raise ValueError(f"scene {scene!r} differs from frozen manifest")
            def array(value):
                if torch.is_tensor(value):
                    value = value.detach().cpu().numpy()
                return np.asarray(value).copy()
            arrays = {"rgb": array(rgb_history[0]), "depth": array(depth[0]),
                      "pointgoal": array(goal[0]), "prev_action": array(prev_action[0]),
                      "robot_pos": array(robot_pos[0]), "robot_quat": array(robot_quat[0]),
                      "rgb_history_times": self._fm_rgb_times.copy(),
                      "rgb_history_steps": self._fm_rgb_steps.copy()}
            metadata = {"schema": SCHEMA, "kind": "observation", "run_id": manifest["run_id"],
                        "episode_id": f'{manifest["run_id"]}:{self._fm_episode}',
                        "step": self._fm_step, "scene": scene, "split": manifest["split"],
                        "sample_idx": int(self.sample_idx_list[0]), "embodiment": int(self.embodiment),
                        "sim_time": float(self._fm_feedback["sim_time"]), "wall_time": time.time(),
                        "valid_segment_len": int(valid_segment_len[0]), "stuck": bool(self.is_stuck[0]),
                        "stuck_diagnostic": self.stuck_diagnostics[0],
                        "guidance_factor": self._build_guidance_factor_batch(self.is_stuck)[0].tolist(),
                        "execution_feedback": self._fm_feedback,
                        "online_rtc": bool(self.navi_former.rtc_enabled),
                        "history_source": "actual_agent_state", "history_padding_sentinel": -1}
            validate_observation(arrays, metadata)
            self._fm_pending = arrays, metadata

        def step_pointgoal_with_guidance(self, goals, images, depths, robot_pos, robot_quat,
                                         execution_feedback=None):
            if not execution_feedback or execution_feedback.get("sim_time") is None:
                raise ValueError("FM capture requires simulation-time execution feedback")
            self._fm_feedback = execution_feedback
            self._fm_pending = None
            result = super().step_pointgoal_with_guidance(
                goals, images, depths, robot_pos, robot_quat, execution_feedback)
            if self._fm_pending is None:
                raise RuntimeError("agent did not expose preprocessed observation")
            arrays, meta = self._fm_pending
            arrays.update(online_returned_trajectory=result[0][0].copy(),
                          online_candidates=result[1][0].copy(), online_scores=result[2][0].copy())
            meta["recovery_diagnostic"] = self.recovery_diagnostics[0]
            # Returned plan is not proof of execution; later feedback is authoritative.
            meta["online_selection_semantics"] = "returned_to_client_not_execution_ack"
            path = run_dir / "observations" / f"obs_{self._fm_episode:04d}_{self._fm_step:06d}.npz"
            save_record(path, arrays, meta)
            self._fm_step += 1
            self._fm_pending = None
            return result
    return RecordingAgent
