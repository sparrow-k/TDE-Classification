"""Demo tests: the demo must use the FINAL checkpoint and the project's own preprocessing.

These guard the three ways a demo can quietly become dishonest:
  * loading a prototype / smoke / scratch checkpoint instead of the final model;
  * re-implementing preprocessing, so the model sees inputs it was not trained on;
  * shipping locked-test objects as demo examples.
"""
import importlib.util
import json
import os
import unittest

import numpy as np
import torch

from tde.config import load_config
from tde.data import build_features, load_mallorn_train, make_splits, to_objects
from tde.evaluate import predict_steps

DEMO_DIR = "demo"
DEMO_PY = os.path.join(DEMO_DIR, "demo.py")
FINAL_RUN = os.path.join("outputs", "final_test", "20260916-172737_final_test")


def load_demo_module():
    """Import demo.py by path, so the test does not depend on the demo folder being importable."""
    spec = importlib.util.spec_from_file_location("demo_module", DEMO_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(os.path.exists(DEMO_PY), "demo.py not found")
class TestDemoUsesFinalModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = load_demo_module()

    def test_checkpoint_is_the_final_primary_model(self):
        """The demo points at the frozen seed-0 checkpoint of the single locked-test run."""
        self.assertEqual(os.path.normpath(os.path.relpath(self.demo.CHECKPOINT, os.getcwd())),
                         os.path.normpath(os.path.join(FINAL_RUN, "models", "frozen_seed0.pt")))
        ckpt = torch.load(self.demo.CHECKPOINT, map_location="cpu", weights_only=False)
        self.assertEqual(ckpt["mode"], "frozen")
        self.assertEqual(ckpt["seed"], 0)

        with open(os.path.join(FINAL_RUN, "summary.json"), encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["protocol"]["primary_model"], "frozen seed 0")
        # trained on the development set only (2,434 objects = fit + early-stopping split)
        self.assertEqual(ckpt["n_fit"] + ckpt["n_early_stop"], summary["development_set"]["n"])

    def test_model_loads_with_the_expected_architecture(self):
        model = self.demo.load_final_model("cpu")
        self.assertFalse(model.encoder.bidirectional, "the demo model must stay causal")
        self.assertEqual(model.encoder.num_layers, 2)
        self.assertEqual(model.encoder.hidden_size, 64)
        self.assertEqual(model.n_outputs, 1, "a binary TDE head is expected")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 43585)

    def test_flux_scale_matches_the_training_configuration(self):
        cfg = load_config("configs/default.yaml")
        self.assertEqual(self.demo.FLUX_SCALE, cfg["data"]["flux_scale"])


@unittest.skipUnless(os.path.exists(DEMO_PY), "demo.py not found")
class TestDemoPreprocessingMatchesPipeline(unittest.TestCase):
    """A bundled CSV must produce exactly the arrays, features and predictions of the real pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.demo = load_demo_module()
        cls.cfg = load_config("configs/default.yaml")
        with open(os.path.join(DEMO_DIR, "examples", "examples.json"), encoding="utf-8") as f:
            cls.index = json.load(f)["examples"]
        lc, meta = load_mallorn_train(cls.cfg["data"]["mallorn_dir"], cls.cfg["data"]["processed_dir"])
        cls.lc, cls.meta = lc, meta

    def project_object(self, object_id):
        meta = self.meta[self.meta["object_id"] == object_id].reset_index(drop=True)
        return to_objects(self.lc[self.lc["object_id"] == object_id], meta)[0]

    def test_csv_reader_reproduces_the_pipeline_arrays(self):
        for entry in self.index:
            with self.subTest(example=entry["file"]):
                from_csv = self.demo.read_lightcurve_csv(os.path.join(DEMO_DIR, "examples", entry["file"]))
                from_pipeline = self.project_object(entry["source_object_id"])
                np.testing.assert_allclose(from_csv["mjd"], from_pipeline["mjd"], atol=1e-3)
                np.testing.assert_allclose(from_csv["flux"], from_pipeline["flux"], atol=1e-3)
                np.testing.assert_allclose(from_csv["flux_err"], from_pipeline["flux_err"], atol=1e-3)
                np.testing.assert_array_equal(from_csv["band"], from_pipeline["band"])

    def test_features_and_predictions_match_the_pipeline(self):
        model = self.demo.load_final_model("cpu")
        scale = self.cfg["data"]["flux_scale"]
        entry = self.index[0]
        from_csv = self.demo.read_lightcurve_csv(os.path.join(DEMO_DIR, "examples", entry["file"]))
        from_pipeline = self.project_object(entry["source_object_id"])

        f_csv = build_features(from_csv["mjd"], from_csv["flux"], from_csv["flux_err"], from_csv["band"], scale)
        f_pipe = build_features(from_pipeline["mjd"], from_pipeline["flux"], from_pipeline["flux_err"],
                                from_pipeline["band"], scale)
        np.testing.assert_allclose(f_csv, f_pipe, atol=1e-4)

        p_csv = predict_steps(model, from_csv, scale, "cpu")
        p_pipe = predict_steps(model, from_pipeline, scale, "cpu")
        np.testing.assert_allclose(p_csv, p_pipe, atol=1e-4)

    def test_predictions_are_causal(self):
        """Truncating a demo curve must not change the predictions of the steps that remain."""
        model = self.demo.load_final_model("cpu")
        obj = self.demo.read_lightcurve_csv(os.path.join(DEMO_DIR, "examples", self.index[0]["file"]))
        full = predict_steps(model, obj, self.cfg["data"]["flux_scale"], "cpu")
        k = len(full) // 2
        prefix = {**obj, "mjd": obj["mjd"][:k], "flux": obj["flux"][:k],
                  "flux_err": obj["flux_err"][:k], "band": obj["band"][:k]}
        np.testing.assert_allclose(predict_steps(model, prefix, self.cfg["data"]["flux_scale"], "cpu"),
                                   full[:k], atol=1e-5)


@unittest.skipUnless(os.path.exists(DEMO_PY), "demo.py not found")
class TestDemoExamplesAreNotTestObjects(unittest.TestCase):
    def test_no_example_comes_from_the_locked_test_set(self):
        cfg = load_config("configs/default.yaml")
        _, meta = load_mallorn_train(cfg["data"]["mallorn_dir"], cfg["data"]["processed_dir"])
        test_ids = set(make_splits(meta, cfg["cv"]["split_seed"], cfg["data"]["n_folds"])["test"])
        with open(os.path.join(DEMO_DIR, "examples", "examples.json"), encoding="utf-8") as f:
            index = json.load(f)["examples"]
        self.assertTrue(index, "no bundled examples found")
        for entry in index:
            self.assertNotIn(entry["source_object_id"], test_ids,
                             f"{entry['file']} is a locked-test object")


if __name__ == "__main__":
    unittest.main()
