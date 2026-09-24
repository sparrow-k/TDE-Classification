# Usage: python diagnostics/<script>.py outputs/runs/<run_dir>   (uses that run's splits.json / model.pt / predictions)
# Does labelling years of pre-event baseline with the object label dilute learning?
# Modes (inputs are identical and causal in all modes; only per-step TRAINING targets/weights differ):
#   all        : every step gets the object label (Stage 0 default)
#   window     : loss only on steps with mjd >= peak - W
#   prelabel0  : steps with mjd < peak - W get target 0 ("no transient yet", RAPID-style)
import copy, json, os, sys, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np, torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tde.data import N_FEATURES, build_features, estimate_peak_mjd, load_mallorn_train, to_objects
from tde.evaluate import metrics_by_cutoff, predict_at_cutoffs
from tde.model import GRUClassifier
from tde.train import set_seed

RUN = sys.argv[1]  # run directory, e.g. outputs/runs/<timestamp>_stage0_mallorn_gru_seed42
W, EPOCHS, SEED = 100.0, 30, 42
CUT = [-50, -20, -10, 0, 20, 50, 100]
SELECT = ["-20d", "-10d", "+0d", "+20d"]


class StepDataset(Dataset):
    def __init__(self, objs, mode):
        self.items = []
        for o in objs:
            x = torch.from_numpy(build_features(o["mjd"], o["flux"], o["flux_err"], o["band"]))
            early = o["mjd"] < estimate_peak_mjd(o) - W
            y = np.full(len(o["mjd"]), o["label"], np.float32)
            w = np.ones(len(o["mjd"]), np.float32)
            if mode == "window":
                w[early] = 0.0
            elif mode == "prelabel0":
                y[early] = 0.0
            self.items.append((x, torch.from_numpy(y), torch.from_numpy(w)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def coll(batch):
    xs, ys, ws = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs])
    mask = (torch.arange(max(lengths))[None] < lengths[:, None]).float()
    return pad_sequence(xs, True), pad_sequence(ys, True), pad_sequence(ws, True) * mask


lc, meta = load_mallorn_train("data/mallorn-astronomical-classification-challenge (1)", "data/processed")
splits = json.load(open(f"{RUN}/splits.json"))
objs = {o["object_id"]: o for o in to_objects(lc, meta)}
tr, va = [objs[i] for i in splits["train"]], [objs[i] for i in splits["val"]]
pw = torch.tensor((len(tr) - sum(o["label"] for o in tr)) / sum(o["label"] for o in tr), device="cuda")

for mode in ["all", "window", "prelabel0"]:
    set_seed(SEED)
    loader = DataLoader(StepDataset(tr, mode), batch_size=64, shuffle=True, collate_fn=coll,
                        generator=torch.Generator().manual_seed(SEED))
    model = GRUClassifier(N_FEATURES, 64, 2, 0.1).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, best_ep, losses = -1, None, 0, []
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0
        for x, y, w in loader:
            x, y, w = x.cuda(), y.cuda(), w.cuda()
            l = F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pw, reduction="none")
            loss = ((l * w).sum(1) / w.sum(1).clamp(min=1)).mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(x)
        losses.append(tot / len(tr))
        mv = metrics_by_cutoff(predict_at_cutoffs(model, va, CUT, 1.0, "cuda", ["g", "r", "i"], 3.0)).set_index("cutoff")
        s = mv.loc[SELECT, "pr_auc"].mean()
        if s > best:
            best, best_state, best_ep = s, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    mv = metrics_by_cutoff(predict_at_cutoffs(model, va, CUT, 1.0, "cuda", ["g", "r", "i"], 3.0))
    print(f"\n=== mode={mode} best_epoch={best_ep} select={best:.3f} loss first/last={losses[0]:.3f}/{losses[-1]:.3f}")
    print(mv[["cutoff", "pr_auc", "roc_auc", "f1", "f1_best"]].to_string(index=False, float_format="%.3f"))
