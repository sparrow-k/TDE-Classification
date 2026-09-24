"""Real-data loader/split checks and a tiny overfitting test for the training loop."""
import os
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tde.data import N_FEATURES, LightCurveDataset, collate, load_mallorn_train, make_splits, to_objects
from tde.evaluate import compute_metrics, predict_last_step
from tde.model import GRUClassifier
from tde.train import set_seed, train_one_epoch

MALLORN_DIR = "data/mallorn-astronomical-classification-challenge (1)"


@unittest.skipUnless(os.path.isdir(MALLORN_DIR), "MALLORN data not present")
class TestMallornData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lc, cls.meta = load_mallorn_train(MALLORN_DIR, "data/processed")

    def test_schema(self):
        self.assertEqual(len(self.meta), 3043)
        self.assertEqual(int(self.meta["target"].sum()), 148)
        self.assertEqual(set(self.lc["object_id"]), set(self.meta["object_id"]))
        self.assertFalse(self.lc[["mjd", "flux", "flux_err", "band"]].isna().any().any())
        self.assertTrue(self.lc["band"].between(0, 5).all())
        self.assertTrue((self.lc["flux_err"] > 0).all())

    def test_sorted_in_time(self):
        d = self.lc.groupby("object_id")["mjd"].diff().dropna()
        self.assertTrue((d >= 0).all())

    def test_label_is_not_an_input(self):
        obj = to_objects(self.lc, self.meta.head(3))[0]
        self.assertEqual(LightCurveDataset([obj])[0][0].shape[1], N_FEATURES)  # only dt, flux, err, band

    def test_splits_disjoint_grouped_stratified(self):
        s = make_splits(self.meta, seed=42)
        tr, va, te = map(set, (s["train"], s["val"], s["test"]))
        self.assertFalse(tr & va or tr & te or va & te)
        self.assertEqual(len(tr | va | te), len(self.meta))
        m = self.meta.set_index("object_id")
        keys = {name: set(map(tuple, m.loc[list(ids), ["z", "ebv"]].to_numpy())) for name, ids in s.items()}
        self.assertFalse(keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])
        for ids in s.values():
            self.assertGreater(m.loc[list(ids), "target"].mean(), 0.03)


class TestTrainingCanOverfit(unittest.TestCase):
    def test_overfit_tiny_synthetic_set(self):
        """Positives contain a slow flare; the model must separate 32 curves almost perfectly."""
        set_seed(0)
        rng = np.random.default_rng(0)
        objs = []
        for i in range(32):
            n = int(rng.integers(30, 60))
            mjd = np.sort(rng.uniform(0, 300, n))
            label = int(i % 4 == 0)
            flux = rng.normal(0, 0.3, n) + label * 5 * np.exp(-((mjd - 150) / 60) ** 2)
            objs.append({"object_id": str(i), "label": label, "spectype": "X", "mjd": mjd,
                         "flux": flux.astype(np.float32), "flux_err": np.full(n, 0.3, np.float32),
                         "band": rng.integers(0, 6, n)})
        loader = DataLoader(LightCurveDataset(objs), batch_size=16, shuffle=True, collate_fn=collate)
        model = GRUClassifier(N_FEATURES, 32, 1, 0.0)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = [train_one_epoch(model, loader, opt, torch.tensor(3.0), "cpu") for _ in range(60)]
        self.assertLess(losses[-1], 0.5 * losses[0])
        probs = predict_last_step(model, objs, 1.0, "cpu")
        self.assertGreater(compute_metrics([o["label"] for o in objs], probs)["pr_auc"], 0.95)


if __name__ == "__main__":
    unittest.main()
