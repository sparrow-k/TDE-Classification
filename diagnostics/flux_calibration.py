# Usage: python diagnostics/flux_calibration.py
# Estimates the flux-unit ratio PLAsTiCC / MALLORN from SN Ia peak brightness at matched redshift and from
# per-band noise levels. Expected 27.54 if PLAsTiCC is FLUXCAL (zp 27.5) and MALLORN is microJansky (zp 23.9).
# Uses the PLAsTiCC `true_z` column for this ANALYSIS only; it is never a model input.
import os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, REPO); os.chdir(REPO)
import pandas as pd
from tde.data import load_mallorn_train

lc_m, meta_m = load_mallorn_train("data/mallorn-astronomical-classification-challenge (1)", "data/processed")
meta_p = pd.read_csv("data/plasticc_train_metadata.csv/plasticc_train_metadata.csv", skipinitialspace=True)
lc_p = pd.read_csv("data/plasticc_train_lightcurves.csv/plasticc_train_lightcurves.csv").rename(columns={"passband": "band"})


def peak_table(lc, meta, zcol, ids):
    d = lc[lc.object_id.isin(ids)]
    d = d[(d.band.isin([1, 2, 3])) & (d.flux / d.flux_err >= 5)]
    return d.groupby(["object_id", "band"]).flux.max().unstack().join(meta.set_index("object_id")[zcol].rename("zz"))


pm = peak_table(lc_m, meta_m, "z", meta_m[meta_m.spectype == "SN Ia"].object_id)
pp = peak_table(lc_p, meta_p, "true_z", meta_p[meta_p.target == 90].object_id)
rows = []
for lo, hi in [(0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7)]:
    a, b = pm[(pm.zz >= lo) & (pm.zz < hi)], pp[(pp.zz >= lo) & (pp.zz < hi)]
    rows.append({"z": f"{lo}-{hi}", "n_mallorn": len(a), "n_plasticc": len(b),
                 **{f"ratio_{bn}": b[bi].median() / a[bi].median() for bi, bn in [(1, "g"), (2, "r"), (3, "i")]}})
print("SN Ia median peak-flux ratio PLAsTiCC / MALLORN:\n", pd.DataFrame(rows).round(2).to_string(index=False))
wfd = lc_p[lc_p.object_id.isin(meta_p[meta_p.ddf_bool == 0].object_id)]
print("median flux_err ratio per band (PLAsTiCC WFD / MALLORN):",
      (wfd.groupby("band").flux_err.median() / lc_m.groupby("band").flux_err.median()).round(1).to_dict())
