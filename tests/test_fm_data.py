"""P0/P1 CPU-only checks; no checkpoint load, CUDA call or simulator launch."""

# Relocation bootstrap: resolve only within this independent repository.
import sys as _rex_sys
from pathlib import Path as _RexPath
_REX_ROOT = _RexPath(__file__).resolve().parents[1]
_REX_BASE = _REX_ROOT / "baselines/x-navdp"
for _rex_path in (_REX_ROOT, _REX_BASE):
    if str(_rex_path) not in _rex_sys.path:
        _rex_sys.path.insert(0, str(_rex_path))

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import DDPMScheduler

from FM_distillation.src.fm_capture_agent import recording_agent_class
from FM_distillation.src.fm_data import (SCHEMA, load_record, save_record, sha256,
                              teacher_tap, validate_label, validate_observation)
from eval.src.policy_agent import NavDP_Agent
from eval.src.policy_network_embodiment import NavDP_Policy_Embodiment
from bridge.recovery import RecoverySelector

BASE = _REX_BASE
from FM_distillation.src import dataset as cli


def observation():
    arrays = {"rgb": np.zeros((8, 224, 224, 3), np.float32),
              "depth": np.ones((224, 224, 1), np.float32), "pointgoal": np.array([2., 0, 0]),
              "prev_action": np.zeros((24, 3), np.float32), "robot_pos": np.zeros(3),
              "robot_quat": np.array([0., 0, 0, 1]),
              "rgb_history_times": np.array([-1.]*7+[0.]),
              "rgb_history_steps": np.array([-1]*7+[0])}
    meta = dict(schema=SCHEMA, kind="observation", run_id="test", episode_id="test:1",
                scene="easy_0", split="train", step=0, sim_time=0., embodiment=1,
                valid_segment_len=0, stuck=False, guidance_factor=[.5]*6+[.05]*2)
    return arrays, meta


def label_fixture(src, digest):
    delta = np.ones((8, 24, 3), np.float32)*.1
    paths = np.cumsum(delta/4, axis=1)
    scores = np.arange(8, dtype=np.float32)
    arrays = dict(raw_action_deltas=delta.copy(), smoothed_actions=delta,
                  q_path=paths.copy(), trajectories=paths, q1=scores, q2=scores,
                  scores=scores, top_indices=np.array([7, 6]), top_trajectories=paths[[7, 6]],
                  rgbd_embed=np.zeros((128, 384), np.float32), goal_embed=np.zeros((1, 384), np.float32),
                  initial_noise=np.zeros_like(delta))
    meta = dict(src, kind="label", rtc_enabled=False, candidates=8, teacher_sha256="teacher",
                observation="obs.npz", observation_sha256=digest,
                target_space="raw_pre_smoothing_action_deltas", tap_equivalence=True, q_recompute_verified=True)
    return arrays, meta


class FMDataTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_schema_roundtrip_no_overwrite_and_invalid_inputs(self):
        arrays, meta = observation()
        validate_observation(arrays, meta)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/"observation.npz"
            save_record(path, arrays, meta)
            loaded, metadata = load_record(path)
            self.assertEqual(metadata, meta)
            np.testing.assert_array_equal(loaded["rgb"], arrays["rgb"])
            with self.assertRaises(FileExistsError):
                save_record(path, arrays, meta)
            with self.assertRaises(ValueError):
                save_record(Path(d)/"bad.npz", arrays, {"nan": float("nan")})
            self.assertFalse((Path(d)/"bad.npz").exists())
        with self.assertRaises(ValueError):
            validate_observation(arrays, dict(meta, schema="legacy"))
        arrays["rgb_history_times"][0] = 1
        with self.assertRaises(ValueError):
            validate_observation(arrays, meta)

    def test_label_coordinates_and_index_corruption(self):
        _, src = observation()
        arrays, meta = label_fixture(src, "digest")
        validate_label(arrays, meta)
        arrays["top_indices"] = np.array([0, 1])
        with self.assertRaises(AssertionError):
            validate_label(arrays, meta)
        arrays, meta = label_fixture(src, "digest")
        arrays["trajectories"] = arrays["trajectories"]*4
        with self.assertRaises(AssertionError):
            validate_label(arrays, meta)
        with self.assertRaises(ValueError):
            validate_label(arrays, dict(meta, rtc_enabled=True))

    def test_tap_real_sampling_equivalence_and_restore_on_error(self):
        model = NavDP_Policy_Embodiment.__new__(NavDP_Policy_Embodiment)
        torch.nn.Module.__init__(model)
        model.device, model.predict_size, model.ft_step = "cpu", 24, 6
        model.sampler, model.rtc_enabled = "ddpm", False
        model.distinguish_embodiment = True
        model.noise_scheduler = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2")
        model.sampling_timesteps = torch.arange(9, -1, -1)
        model.rgbd_encoder = lambda *a: torch.zeros(1, 128, 384)
        model.point_encoder = torch.nn.Linear(3, 384)
        model.predict_noise = lambda x, *a: .1*x
        model.predict_noise_ft = lambda x, *a: .1*x
        model.predict_pointgoal_q = lambda path, *a, **kw: (path.sum((1,2)), path.sum((1,2))+.1)
        model.weight = np.ones(25)
        model.eval()
        arrays, _ = observation()
        def sample():
            torch.manual_seed(12)
            return model.predict_pointgoal_action_with_guidance(
                arrays["pointgoal"][None], None, None, 8, np.array([0]),
                arrays["prev_action"][None], 0, 23, np.zeros(8), embodiment=1)
        original = model.predict_pointgoal_q
        ref = sample()
        with teacher_tap(model) as tapped:
            actual = sample()
        for a, b in zip(ref[:3], actual[:3]):
            np.testing.assert_array_equal(a, b)
        self.assertIs(model.predict_pointgoal_q, original)
        self.assertNotIn("smooth_trajectory", model.__dict__)
        np.testing.assert_allclose(actual[1][0], ((tapped["q1"]+tapped["q2"])/2).numpy())
        torch.testing.assert_close(tapped["q_path"], model.smooth_cumulative_trajectory(
            torch.cumsum(tapped["raw"]/4,1), num_points=25, smooth_factor=.5, weight=model.weight))
        with self.assertRaises(RuntimeError):
            with teacher_tap(model):
                raise RuntimeError("test error")
        self.assertIs(model.predict_pointgoal_q, original)
        self.assertNotIn("smooth_trajectory", model.__dict__)

    def test_default_gpu_resolution_does_not_follow_inherited_gpu0(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch.object(
                cli.subprocess, "check_output", return_value="GPU-physical-one\n") as query:
            self.assertEqual(cli.default_gpu(), "GPU-physical-one")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "GPU-physical-one")
            self.assertEqual(os.environ["ISAAC_ACTIVE_GPU"], "1")
            self.assertEqual(os.environ["ISAAC_PHYSICS_GPU"], "0")
            self.assertIn("--id=1", query.call_args.args[0])

    def test_prepare_split_guard_snapshot_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            checkpoint = Path(d)/"fake.ckpt"
            checkpoint.write_bytes(b"fixture-not-real-weights")
            args = argparse.Namespace(config=str(BASE/"eval/config/eval_pointgoal/humanoid_clutter_easy.yaml"),
                checkpoint=str(checkpoint), run=str(Path(d)/"run"), scene="easy_6", split="train", seed=0)
            with self.assertRaises(ValueError):
                cli.prepare(args)
            self.assertFalse(Path(args.run).exists())
            args.scene = "easy_0"
            with contextlib.redirect_stdout(io.StringIO()):
                cli.prepare(args)
            meta = cli.check_frozen(Path(args.run))
            self.assertEqual(meta["scene"], "easy_0")
            self.assertEqual(meta["physical_gpu"], 0)
            self.assertTrue((Path(args.run)/"source_snapshot.zip").is_file())
            with self.assertRaises(FileExistsError):
                cli.prepare(args)
            (Path(args.run)/"selected_scene.json").write_text("{}")
            with self.assertRaises(RuntimeError):
                cli.check_frozen(Path(args.run))

    def test_offline_validation_hash_and_incomplete_guard(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run/"observations").mkdir()
            (run/"labels_smoke").mkdir()
            arrays, src = observation()
            save_record(run/"observations/obs.npz", arrays, src)
            digest = sha256(run/"observations/obs.npz")
            arrays, meta = label_fixture(src, digest)
            save_record(run/"labels_smoke/obs.npz", arrays, meta)
            cli.dump(run/"manifest.json", {"run_id":"test", "scene":"easy_0", "split":"train", "teacher_sha256":"teacher"})
            cli.dump(run/"labels_smoke/label_manifest.json", {"teacher_sha256":"teacher", "sources":{"obs.npz":digest}})
            args = argparse.Namespace(run=str(run), name="labels_smoke")
            with self.assertRaises(FileNotFoundError):
                cli.validate(args)
            cli.dump(run/"labels_smoke/COMPLETE.json", {"count":1, "status":"complete",
                     "sha256":{"obs.npz":sha256(run/"labels_smoke/obs.npz")}})
            with contextlib.redirect_stdout(io.StringIO()):
                cli.validate(args)
            report = json.loads((run/"labels_smoke/validation_report.json").read_text())
            self.assertEqual(report["records"], 1)
            self.assertEqual(report["candidate_count"], 8)
            with (run/"labels_smoke/obs.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                cli.validate(args)


class CaptureTests(unittest.TestCase):
    def agent(self, cls):
        agent = cls.__new__(cls)
        agent.recovery = RecoverySelector("baseline")
        agent.memory_size, agent.predict_size, agent.image_size = 8, 24, 224
        agent.is_real, agent.embodiment, agent.device = False, 1, "cpu"
        agent.visualize = False
        agent.reset(1)
        agent.current_scene_name = "easy_0"
        agent.sample_idx_list = [3]
        paths = np.zeros((1,8,24,3), np.float32)
        paths[0,:,:,0] = np.linspace(.1,1,24)[None]
        class Network:
            rtc_enabled = True
            def predict_pointgoal_action_with_guidance(self, *a, **kw):
                values = torch.rand(1,8).numpy()
                idx = np.argsort(-values[0])[:2]
                return paths.copy(), values, paths[:,idx].copy(), None
        agent.navi_former = Network()
        return agent

    def test_recording_preserves_outputs_rng_history_and_episode_reset(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run/"observations").mkdir()
            factory = recording_agent_class(run, {"run_id":"test", "scene":"easy_0", "split":"train"})
            ordinary, recording = self.agent(NavDP_Agent), self.agent(factory)
            for step in range(3):
                args = (np.array([[2.,0,0]]), np.full((1,224,224,3), 10+step, np.uint8),
                        np.ones((1,224,224,1), np.float32), np.zeros((1,3)), np.array([[0.,0,0,1]]))
                feedback = dict(plan_id=step, sim_time=float(step), executed_segments=[[]])
                torch.manual_seed(12+step)
                np.random.seed(12+step)
                with contextlib.redirect_stdout(io.StringIO()):
                    ref = ordinary.step_pointgoal_with_guidance(*args, execution_feedback=feedback)
                rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
                torch.manual_seed(12+step)
                np.random.seed(12+step)
                with contextlib.redirect_stdout(io.StringIO()):
                    actual = recording.step_pointgoal_with_guidance(*args, execution_feedback=feedback)
                for a,b in zip(ref[:3], actual[:3]):
                    np.testing.assert_array_equal(a,b)
                torch.testing.assert_close(torch.get_rng_state(), rng)
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
                record, meta = load_record(run/"observations"/f"obs_0001_{step:06d}.npz")
                validate_observation(record, meta)
                self.assertEqual(meta["step"], step)
                self.assertEqual(meta["sim_time"], float(step))
                if step == 0:
                    np.testing.assert_array_equal(record["rgb_history_steps"], [-1]*7+[0])
                else:
                    self.assertEqual(record["rgb_history_steps"][-2], step-1)
                np.testing.assert_array_equal(record["online_returned_trajectory"], actual[0][0])
            self.assertTrue(meta["stuck"])
            self.assertEqual(meta["guidance_factor"], [0.]*8)
            recording.reset_env(0)
            self.assertEqual(recording._fm_episode, 2)
            self.assertTrue((recording._fm_times == -1).all())
            self.assertEqual(recording._fm_step, 0)
            with self.assertRaises(ValueError):
                recording.step_pointgoal_with_guidance(*args, execution_feedback=None)


if __name__ == "__main__":
    unittest.main()
