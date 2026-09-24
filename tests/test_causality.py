"""Leakage tests: no future observation may influence an earlier prediction."""
import unittest

import numpy as np
import torch

from tde.data import N_FEATURES, build_features, collate, truncate
from tde.evaluate import check_causality, predict_last_step, predict_steps
from tde.model import GRUClassifier


def fake_object(n, seed, label=0):
    rng = np.random.default_rng(seed)
    mjd = np.sort(rng.uniform(60000, 61000, n))
    mjd[5] = mjd[4]  # simultaneous observations in two bands
    return {"object_id": f"obj{seed}", "label": label, "spectype": "X", "mjd": mjd,
            "flux": rng.normal(0, 2, n).astype(np.float32), "flux_err": rng.uniform(0.1, 1, n).astype(np.float32),
            "band": rng.integers(0, 6, n)}


def feats(o):
    return build_features(o["mjd"], o["flux"], o["flux_err"], o["band"])


class TestCausality(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GRUClassifier(N_FEATURES, hidden_size=16, num_layers=2, dropout=0.0).eval()
        self.obj = fake_object(60, seed=1)

    def test_truncate_keeps_only_past(self):
        cutoff = self.obj["mjd"][30]
        pre = truncate(self.obj, cutoff)
        self.assertTrue(np.all(pre["mjd"] <= cutoff))
        self.assertEqual(len(pre["mjd"]), int((self.obj["mjd"] <= cutoff).sum()))
        self.assertTrue(len(pre["flux"]) == len(pre["band"]) == len(pre["mjd"]))

    def test_features_of_prefix_equal_prefix_of_features(self):
        full = feats(self.obj)
        for k in [1, 5, 6, 30, 60]:
            pre = truncate(self.obj, self.obj["mjd"][k - 1])
            n = len(pre["mjd"])
            np.testing.assert_array_equal(feats(pre), full[:n])

    def test_future_changes_do_not_change_past_outputs(self):
        k = 25
        changed = {key: (v.copy() if isinstance(v, np.ndarray) else v) for key, v in self.obj.items()}
        changed["flux"][k:] += 100.0  # a huge future flare
        changed["band"][k:] = 0
        a = predict_steps(self.model, self.obj, 1.0, "cpu")
        b = predict_steps(self.model, changed, 1.0, "cpu")
        np.testing.assert_allclose(a[:k], b[:k], atol=1e-6)
        self.assertGreater(np.abs(a[k:] - b[k:]).max(), 1e-4)  # sanity: the change is visible later

    def test_padding_does_not_change_outputs(self):
        short, long = fake_object(20, seed=2), fake_object(80, seed=3)
        x, mask, _, lengths = collate([(torch.from_numpy(feats(short)), 0), (torch.from_numpy(feats(long)), 1)])
        with torch.no_grad():
            batched = torch.sigmoid(self.model(x))[0, :20].double().numpy()
        alone = predict_steps(self.model, short, 1.0, "cpu")
        np.testing.assert_allclose(batched, alone, atol=1e-6)
        self.assertEqual(mask.sum().item(), 100)

    def test_truncated_prediction_matches_full_track(self):
        track = predict_steps(self.model, self.obj, 1.0, "cpu")
        for k in [1, 10, 40]:
            pre = truncate(self.obj, self.obj["mjd"][k - 1])
            p = predict_last_step(self.model, [pre], 1.0, "cpu")[0]
            self.assertAlmostEqual(p, track[len(pre["mjd"]) - 1], places=5)
        self.assertLess(check_causality(self.model, [fake_object(50, s) for s in range(5)], 1.0, "cpu"), 1e-5)

    def test_empty_prefix_gives_nan(self):
        pre = truncate(self.obj, self.obj["mjd"][0] - 1.0)
        self.assertEqual(len(pre["mjd"]), 0)
        self.assertTrue(np.isnan(predict_last_step(self.model, [pre], 1.0, "cpu")[0]))


if __name__ == "__main__":
    unittest.main()
