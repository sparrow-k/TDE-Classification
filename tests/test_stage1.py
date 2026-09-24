"""Stage 1 tests: multi-class head, encoder transfer/freezing, CV folds, PLAsTiCC loader."""
import os
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tde.data import (N_FEATURES, PLASTICC_CODES, PLASTICC_TDE_INDEX, LightCurveDataset, collate,
                      grouped_stratified_folds, load_plasticc_train)
from tde.evaluate import check_causality, predict_steps
from tde.model import GRUClassifier, load_encoder_weights
from tde.train import masked_step_loss, set_seed, train_one_epoch
from tests.test_causality import fake_object

PLASTICC_LC = "data/plasticc_train_lightcurves.csv/plasticc_train_lightcurves.csv"
PLASTICC_META = "data/plasticc_train_metadata.csv/plasticc_train_metadata.csv"


class TestMultiClassModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GRUClassifier(N_FEATURES, 16, 2, 0.0, n_outputs=14, tde_class=PLASTICC_TDE_INDEX).eval()

    def test_multiclass_head_is_causal(self):
        obj = fake_object(60, seed=4)
        changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obj.items()}
        changed["flux"][30:] += 100.0
        np.testing.assert_allclose(predict_steps(self.model, obj, 1.0, "cpu")[:30],
                                   predict_steps(self.model, changed, 1.0, "cpu")[:30], atol=1e-6)
        self.assertLess(check_causality(self.model, [fake_object(50, s) for s in range(5)], 1.0, "cpu"), 1e-5)

    def test_tde_probability(self):
        x = torch.randn(2, 7, N_FEATURES)
        p = self.model.tde_probability(self.model(x))
        self.assertEqual(p.shape, (2, 7))
        self.assertTrue(torch.all((p >= 0) & (p <= 1)))
        binary = GRUClassifier(N_FEATURES, 16, 1, 0.0).eval()
        torch.testing.assert_close(binary.tde_probability(binary(x)), torch.sigmoid(binary(x)))

    def test_multiclass_loss_ignores_padding(self):
        logits = torch.randn(2, 5, 14)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
        labels = torch.tensor([1.0, 3.0])
        a = masked_step_loss(logits, labels, mask, None)
        logits2 = logits.clone()
        logits2[1, 2:] = 50.0
        self.assertAlmostEqual(a.item(), masked_step_loss(logits2, labels, mask, None).item(), places=6)


class TestTransfer(unittest.TestCase):
    def test_load_encoder_weights(self):
        torch.manual_seed(0)
        pre = GRUClassifier(N_FEATURES, 16, 2, 0.0, n_outputs=14, tde_class=1)
        fine = GRUClassifier(N_FEATURES, 16, 2, 0.0)
        load_encoder_weights(fine, pre.state_dict())
        for (n1, p1), (n2, p2) in zip(pre.encoder.named_parameters(), fine.encoder.named_parameters()):
            torch.testing.assert_close(p1, p2)
        self.assertEqual(fine.head[-1].out_features, 1)

    def test_frozen_encoder_does_not_change(self):
        set_seed(0)
        objs = [{**fake_object(40, s), "label": s % 2} for s in range(16)]
        model = GRUClassifier(N_FEATURES, 16, 2, 0.1)
        model.freeze_encoder()
        encoder_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
        loader = DataLoader(LightCurveDataset(objs), batch_size=8, shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
        for _ in range(3):
            train_one_epoch(model, loader, opt, torch.tensor(1.0), "cpu")
        self.assertFalse(model.encoder.training)  # dropout of the frozen encoder stays off
        for k, v in model.encoder.state_dict().items():
            torch.testing.assert_close(v, encoder_before[k])
        self.assertTrue(any(not torch.equal(v, head_before[k]) for k, v in model.head.state_dict().items()))


class TestFolds(unittest.TestCase):
    def test_grouped_stratified_folds(self):
        rng = np.random.default_rng(0)
        y = (rng.random(500) < 0.1).astype(int)
        groups = rng.integers(0, 400, 500)
        folds = grouped_stratified_folds(y, groups, 5, seed=1)
        all_idx = np.concatenate(folds)
        self.assertEqual(sorted(all_idx.tolist()), list(range(500)))
        for i in range(5):
            for j in range(i + 1, 5):
                self.assertFalse(set(groups[folds[i]]) & set(groups[folds[j]]))


@unittest.skipUnless(os.path.exists(PLASTICC_LC), "PLAsTiCC data not present")
class TestPlasticcData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lc, cls.meta = load_plasticc_train(PLASTICC_LC, PLASTICC_META, "data/processed")

    def test_schema_and_labels(self):
        self.assertEqual(len(self.meta), 7848)
        self.assertEqual(int((self.meta["target"] == PLASTICC_TDE_INDEX).sum()), 495)
        self.assertEqual(self.meta["target"].nunique(), len(PLASTICC_CODES))
        self.assertFalse([c for c in self.meta.columns if c.startswith(("true_", "tflux_", "hostgal"))])
        self.assertEqual(list(self.lc.columns), ["object_id", "mjd", "flux", "flux_err", "band"])
        self.assertEqual(set(self.lc["object_id"]), set(self.meta["object_id"]))
        self.assertTrue(self.lc["band"].between(0, 5).all())
        self.assertTrue((self.lc.groupby("object_id")["mjd"].diff().dropna() >= 0).all())


if __name__ == "__main__":
    unittest.main()
