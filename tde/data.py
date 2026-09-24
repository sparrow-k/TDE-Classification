"""MALLORN loading, causal per-step features, truncation, splits and the PyTorch Dataset.

Leakage rules enforced here:
* Features of observation k depend only on observation k and the time of observation k-1.
* No per-object normalisation (e.g. dividing by the curve's maximum), which would leak the future peak.
* `estimate_peak_mjd` looks at the FULL light curve. It is used only to place evaluation cutoffs,
  never as a model input.
"""
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.nn.utils.rnn import pad_sequence

BANDS = ["u", "g", "r", "i", "z", "y"]
BAND_TO_INDEX = {b: i for i, b in enumerate(BANDS)}
N_FEATURES = 3 + len(BANDS)  # log1p(dt), arcsinh(flux), arcsinh(flux_err), one-hot band


# ----------------------------------------------------------------------------- loading

def load_mallorn_train(mallorn_dir, processed_dir=None):
    """Load the labeled Kaggle MALLORN training set.

    Returns
      lc:   one row per observation: object_id, mjd, flux, flux_err, band (0-5),
            sorted by object, time, band.
      meta: one row per object: object_id, z, ebv, spectype, split, target.
    """
    meta = pd.read_csv(os.path.join(mallorn_dir, "train_log.csv"))
    meta = meta.rename(columns={"Z": "z", "EBV": "ebv", "SpecType": "spectype"})
    meta = meta[["object_id", "z", "ebv", "spectype", "split", "target"]]

    cache = os.path.join(processed_dir, "mallorn_kaggle_train_lc.parquet") if processed_dir else None
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache), meta

    files = sorted(glob.glob(os.path.join(mallorn_dir, "split_*", "train_full_lightcurves.csv")))
    if not files:
        raise FileNotFoundError(f"No train_full_lightcurves.csv found under {mallorn_dir}")
    lc = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lc = lc.rename(columns={"Time (MJD)": "mjd", "Flux": "flux", "Flux_err": "flux_err", "Filter": "filter"})
    lc = lc.dropna(subset=["flux", "flux_err"])  # ~0.2% of rows have NaN flux
    lc["band"] = lc["filter"].map(BAND_TO_INDEX)
    if lc["band"].isna().any():
        raise ValueError(f"Unknown filters: {lc.loc[lc.band.isna(), 'filter'].unique()}")
    lc["band"] = lc["band"].astype("int64")
    lc = lc.drop(columns="filter")
    lc = lc.sort_values(["object_id", "mjd", "band"], kind="mergesort").reset_index(drop=True)

    if cache:
        os.makedirs(processed_dir, exist_ok=True)
        lc.to_parquet(cache, index=False)
    return lc, meta


# PLAsTiCC class codes -> names. The class index used by the model is the position in PLASTICC_CODES.
PLASTICC_CLASSES = {6: "microlens", 15: "TDE", 16: "EB", 42: "SNII", 52: "SNIax", 53: "Mira", 62: "SNIbc",
                    64: "KN", 65: "M-dwarf", 67: "SNIa-91bg", 88: "AGN", 90: "SNIa", 92: "RRL", 95: "SLSN-I"}
PLASTICC_CODES = sorted(PLASTICC_CLASSES)
PLASTICC_TDE_INDEX = PLASTICC_CODES.index(15)


def load_plasticc_train(lc_path, meta_path, processed_dir=None):
    """Load the PLAsTiCC training set in the same format as MALLORN.

    meta: object_id (str), target (class index 0-13), spectype (class name), ddf (deep-drilling field flag).
    All simulation-truth columns (true_*, tflux_*) and host-galaxy columns are dropped, and
    `detected_bool` is not used (MALLORN has no equivalent).
    Flux stays in PLAsTiCC units (FLUXCAL); convert with `flux_scale` when building features.
    """
    raw = pd.read_csv(meta_path, skipinitialspace=True)
    code_to_index = {c: i for i, c in enumerate(PLASTICC_CODES)}
    meta = pd.DataFrame({
        "object_id": raw["object_id"].astype(str),
        "target": raw["target"].map(code_to_index).astype("int64"),
        "spectype": raw["target"].map(PLASTICC_CLASSES),
        "ddf": raw["ddf_bool"].astype(int),
    })

    cache = os.path.join(processed_dir, "plasticc_train_lc.parquet") if processed_dir else None
    if cache and os.path.exists(cache):
        return pd.read_parquet(cache), meta

    lc = pd.read_csv(lc_path)
    lc = lc.rename(columns={"passband": "band"})[["object_id", "mjd", "flux", "flux_err", "band"]]
    lc["object_id"] = lc["object_id"].astype(str)
    lc["band"] = lc["band"].astype("int64")  # PLAsTiCC passbands 0-5 are already u,g,r,i,z,y
    lc = lc.dropna(subset=["flux", "flux_err"])
    lc = lc.sort_values(["object_id", "mjd", "band"], kind="mergesort").reset_index(drop=True)
    if cache:
        os.makedirs(processed_dir, exist_ok=True)
        lc.to_parquet(cache, index=False)
    return lc, meta


def stratified_subset(meta, n_objects, seed):
    """Class-stratified random subset of objects (for smoke tests)."""
    if n_objects is None or n_objects >= len(meta):
        return meta.reset_index(drop=True)
    subset, _ = train_test_split(meta, train_size=n_objects, stratify=meta["target"], random_state=seed)
    return subset.reset_index(drop=True)


def to_objects(lc, meta):
    """Convert tables into a list of per-object dicts of numpy arrays (in `meta` order)."""
    groups = dict(tuple(lc.groupby("object_id", sort=False)))
    objects = []
    for row in meta.itertuples(index=False):
        g = groups[row.object_id]
        objects.append({
            "object_id": row.object_id,
            "label": int(row.target),
            "spectype": row.spectype,
            "mjd": g["mjd"].to_numpy(np.float64),
            "flux": g["flux"].to_numpy(np.float32),
            "flux_err": g["flux_err"].to_numpy(np.float32),
            "band": g["band"].to_numpy(np.int64),
        })
    return objects


# ----------------------------------------------------------------------------- features

def build_features(mjd, flux, flux_err, band, flux_scale=1.0):
    """Per-step input vectors (Δt, flux, flux error, passband), shape (n_obs, N_FEATURES).

    Δt is the time since the previous observation (0 for the first one, and 0 for
    simultaneous observations in different bands). log1p/arcsinh compress the huge
    dynamic range with FIXED transforms, so nothing depends on later observations.
    """
    n = len(mjd)
    feats = np.zeros((n, N_FEATURES), dtype=np.float32)
    if n == 0:
        return feats
    dt = np.diff(mjd, prepend=mjd[0])
    feats[:, 0] = np.log1p(dt)
    feats[:, 1] = np.arcsinh(flux / flux_scale)
    feats[:, 2] = np.arcsinh(flux_err / flux_scale)
    feats[np.arange(n), 3 + band] = 1.0
    return feats


def truncate(obj, cutoff_mjd):
    """Return a copy of the object containing only observations with mjd <= cutoff_mjd.

    Observations are time-sorted, so this is always a prefix of the light curve.
    """
    n_keep = int(np.searchsorted(obj["mjd"], cutoff_mjd, side="right"))
    return {k: (v[:n_keep] if isinstance(v, np.ndarray) else v) for k, v in obj.items()}


def estimate_peak_mjd(obj, bands=("g", "r", "i"), min_snr=3.0):
    """EVALUATION ONLY (uses the full light curve): time of the brightest significant observation.

    Takes the maximum flux among observations in `bands` with flux/flux_err >= min_snr,
    falling back to all observations if none qualify. For AGN the "peak" is just the
    brightest point, but the same rule is applied to every class.
    """
    allowed = np.isin(obj["band"], [BAND_TO_INDEX[b] for b in bands])
    significant = allowed & (obj["flux"] / obj["flux_err"] >= min_snr)
    if not significant.any():
        significant = np.ones(len(obj["mjd"]), dtype=bool)
    i = int(np.argmax(np.where(significant, obj["flux"], -np.inf)))
    return float(obj["mjd"][i])


# ----------------------------------------------------------------------------- splits

def make_splits(meta, seed, n_folds=5):
    """Object-level train/val/test split, stratified by target and grouped by (z, ebv).

    Grouping keeps objects with identical (redshift, extinction) — possible simulated
    siblings of one parent event — inside a single split.
    """
    y = meta["target"].to_numpy()
    groups = meta.groupby(["z", "ebv"]).ngroup().to_numpy()
    ids = meta["object_id"].to_numpy()

    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    trval_idx, test_idx = next(outer.split(ids, y, groups))
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    tr_rel, val_rel = next(inner.split(trval_idx, y[trval_idx], groups[trval_idx]))

    return {
        "train": ids[trval_idx[tr_rel]].tolist(),
        "val": ids[trval_idx[val_rel]].tolist(),
        "test": ids[test_idx].tolist(),
    }


def grouped_stratified_folds(y, groups, n_folds, seed):
    """Split indices 0..len(y)-1 into n_folds held-out index arrays (stratified by y, never splitting a group)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [held for _, held in sgkf.split(np.zeros(len(y)), y, groups)]


# ----------------------------------------------------------------------------- dataset

class LightCurveDataset(torch.utils.data.Dataset):
    """Holds precomputed per-step feature tensors and object labels."""

    def __init__(self, objects, flux_scale=1.0):
        self.labels = [o["label"] for o in objects]
        self.features = [
            torch.from_numpy(build_features(o["mjd"], o["flux"], o["flux_err"], o["band"], flux_scale))
            for o in objects
        ]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def collate(batch):
    """Pad variable-length sequences at the END and build a mask of real observations.

    End-padding is safe for a unidirectional GRU: outputs at real steps never see padding.
    """
    feats, labels = zip(*batch)
    lengths = torch.tensor([len(f) for f in feats], dtype=torch.long)
    x = pad_sequence(feats, batch_first=True)  # (B, T_max, F)
    mask = torch.arange(x.shape[1])[None, :] < lengths[:, None]  # (B, T_max) bool
    return x, mask, torch.tensor(labels, dtype=torch.float32), lengths


# =============================================================================================================
# NEW (post-completion experiment, 2026-09): full-PLAsTiCC pre-training.
#
# Everything above this banner is the original pipeline and is unchanged; nothing above calls anything
# below. This section only ADDS a way to use the full public PLAsTiCC release (training set + unblinded test
# set) for the same 14-class pre-training. See results/full_plasticc_pretraining/.
#
# Row handling is identical to `load_plasticc_train` (keep object_id/mjd/flux/flux_err/passband, drop NaN
# flux, stable-sort each object by (mjd, band)) and features come from the same `build_features`. The only
# new element is storage: ~450 M rows do not fit in memory as per-object feature tensors, so the raw columns
# are written once to flat binary files and memory-mapped.
# =============================================================================================================

FULL_STORE_COLUMNS = {"mjd": np.float64, "flux": np.float32, "flux_err": np.float32, "band": np.int8}


def load_plasticc_test_metadata(meta_path):
    """Unblinded PLAsTiCC test metadata in the same format as `load_plasticc_train`'s `meta`.

    In the public test file the `target` column is blinded (0 for every object); the class is `true_target`
    (in the training file the two columns are identical). Objects whose `true_target` is not one of the 14
    training classes (codes 991-995 exist only in the test set) are returned separately: the 14-class head
    has no output for them, so they cannot be used without changing the pre-training task.

    Returns (meta, excluded_counts {code: n}, excluded_ids set).
    """
    raw = pd.read_csv(meta_path, skipinitialspace=True, usecols=["object_id", "ddf_bool", "true_target"])
    known = raw["true_target"].isin(PLASTICC_CLASSES)
    code_to_index = {c: i for i, c in enumerate(PLASTICC_CODES)}
    kept = raw[known]
    meta = pd.DataFrame({
        "object_id": kept["object_id"].astype(str).to_numpy(),
        "target": kept["true_target"].map(code_to_index).astype("int64").to_numpy(),
        "spectype": kept["true_target"].map(PLASTICC_CLASSES).to_numpy(),
        "ddf": kept["ddf_bool"].astype(int).to_numpy(),
    })
    excluded = raw.loc[~known, "true_target"].value_counts().sort_index()
    return meta, {int(k): int(v) for k, v in excluded.items()}, set(raw.loc[~known, "object_id"].astype(str))


class _BinaryColumnWriter:
    """Appends numpy arrays to one flat binary file per column."""

    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.files = {c: open(os.path.join(out_dir, f"{c}.bin"), "wb") for c in FULL_STORE_COLUMNS}
        self.n_rows = 0

    def write(self, table):
        for c, dtype in FULL_STORE_COLUMNS.items():
            self.files[c].write(np.ascontiguousarray(table[c].to_numpy(dtype)).tobytes())
        self.n_rows += len(table)

    def close(self):
        for f in self.files.values():
            f.close()


def _index_rows(table, start_row):
    """(object_id, start, length) of each contiguous object block in an object-sorted table."""
    ids = table["object_id"].to_numpy()
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    lengths = np.diff(np.r_[starts, len(ids)])
    return pd.DataFrame({"object_id": ids[starts].astype(str), "start": starts + start_row, "length": lengths})


def _write_test_chunk(chunk, writer, test_meta, excluded_ids, index_parts, file_ids, file_name):
    """Apply the `load_plasticc_train` row rules to a chunk of COMPLETE objects and append it to the store."""
    stats = {"kept": 0, "nan": 0, "excluded": 0, "objects": 0}
    chunk = chunk.rename(columns={"passband": "band"})
    chunk["object_id"] = chunk["object_id"].astype(str)
    file_ids.update(chunk["object_id"].unique())
    excluded = chunk["object_id"].isin(excluded_ids)
    stats["excluded"] = int(excluded.sum())
    chunk = chunk[~excluded]
    unknown = set(chunk["object_id"].unique()) - set(test_meta.index)
    if unknown:
        raise ValueError(f"{file_name}: {len(unknown)} objects have no metadata row (e.g. {next(iter(unknown))})")
    n_before = len(chunk)
    chunk = chunk.dropna(subset=["flux", "flux_err"])
    stats["nan"] = n_before - len(chunk)
    chunk = chunk.sort_values(["object_id", "mjd", "band"], kind="mergesort")
    stats["kept"] = len(chunk)
    if len(chunk):
        idx = _index_rows(chunk, writer.n_rows)
        writer.write(chunk)
        index_parts.append(idx.join(test_meta, on="object_id").assign(source="test", file=file_name))
        stats["objects"] = len(idx)
    return stats


def build_plasticc_full_store(train_lc_path, train_meta_path, test_meta_path, test_lc_paths, out_dir,
                              processed_dir=None, chunksize=5_000_000, log=print):
    """Write the full PLAsTiCC release (training set + unblinded test set) as one memory-mappable store.

    Test light-curve files are streamed in chunks, so no file is ever fully in memory. Objects are contiguous
    and id-sorted in the files; the last (possibly incomplete) object of each chunk is carried into the next.
    Returns the manifest (also written to out_dir/manifest.json).
    """
    import json

    writer = _BinaryColumnWriter(out_dir)
    index_parts, per_file = [], []

    # 1) the original training set, through the ORIGINAL loader, so these rows are exactly what Stage 1 used
    lc, meta = load_plasticc_train(train_lc_path, train_meta_path, processed_dir)
    idx = _index_rows(lc, writer.n_rows)  # load_plasticc_train already sorts by (object_id, mjd, band)
    writer.write(lc)
    index_parts.append(idx.merge(meta, on="object_id", how="left", validate="one_to_one")
                       .assign(source="train", file=os.path.basename(train_lc_path)))
    per_file.append({"file": os.path.basename(train_lc_path), "rows": int(len(lc)), "objects_kept": int(len(idx))})
    log(f"train set: {len(idx):,} objects, {len(lc):,} rows")
    del lc

    # 2) the unblinded test set, file by file, chunk by chunk
    test_meta, excluded_counts, excluded_ids = load_plasticc_test_metadata(test_meta_path)
    test_meta = test_meta.set_index("object_id")
    seen_ids = set()
    dtypes = {"object_id": "int64", "mjd": "float64", "passband": "int8", "flux": "float32", "flux_err": "float32"}
    for path in test_lc_paths:
        name = os.path.basename(path)
        totals = {"rows": 0, "kept": 0, "nan": 0, "excluded": 0, "objects": 0}
        carry, file_ids = None, set()
        for chunk in pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, chunksize=chunksize):
            totals["rows"] += len(chunk)
            if carry is not None:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            last_id = chunk["object_id"].iat[-1]
            carry = chunk[chunk["object_id"] == last_id]
            complete = chunk[chunk["object_id"] != last_id]
            for k, v in _write_test_chunk(complete, writer, test_meta, excluded_ids, index_parts, file_ids,
                                          name).items():
                totals[k] += v
        if carry is not None and len(carry):
            for k, v in _write_test_chunk(carry, writer, test_meta, excluded_ids, index_parts, file_ids,
                                          name).items():
                totals[k] += v
        overlap = seen_ids & file_ids
        if overlap:
            raise ValueError(f"{name}: {len(overlap)} objects also appear in an earlier file")
        seen_ids |= file_ids
        per_file.append({"file": name, "rows": totals["rows"], "rows_kept": totals["kept"],
                         "rows_nan_dropped": totals["nan"], "rows_excluded_class": totals["excluded"],
                         "objects_in_file": len(file_ids), "objects_kept": totals["objects"]})
        log(f"{name}: {totals['rows']:,} rows, {len(file_ids):,} objects, {totals['objects']:,} kept")
    writer.close()

    index = pd.concat(index_parts, ignore_index=True)
    index = index.astype({"start": "int64", "length": "int32", "target": "int16", "ddf": "int8"})
    if index["object_id"].duplicated().any():
        raise ValueError("duplicate object ids in the store")
    if (index.groupby("source")["length"].min() <= 0).any():
        raise ValueError("empty light curve in the store")
    index.to_parquet(os.path.join(out_dir, "index.parquet"), index=False)

    missing_lc = set(test_meta.index) - seen_ids
    manifest = {
        "rows": int(writer.n_rows), "objects": int(len(index)),
        "objects_by_source": {k: int(v) for k, v in index["source"].value_counts().items()},
        "class_counts": {k: int(v) for k, v in index["spectype"].value_counts().items()},
        "test_objects_in_metadata_in_14_classes": int(len(test_meta)),
        "test_objects_excluded_non_training_class": excluded_counts,
        "test_metadata_objects_without_light_curve": len(missing_lc),
        "test_light_curve_objects_seen": len(seen_ids),
        "files": per_file, "columns": {c: np.dtype(t).name for c, t in FULL_STORE_COLUMNS.items()},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class PlasticcFullStore:
    """Read-only access to the store written by `build_plasticc_full_store`.

    The memory maps are opened lazily and never pickled, so the store can be handed to DataLoader workers.
    `get(i)` returns the same per-object dict that `to_objects` builds.
    """

    def __init__(self, store_dir):
        self.store_dir = store_dir
        self.index = pd.read_parquet(os.path.join(store_dir, "index.parquet"))
        self._starts = self.index["start"].to_numpy()
        self._lengths = self.index["length"].to_numpy()
        self._targets = self.index["target"].to_numpy().astype(np.int64)
        self._cols = None

    def _open(self):
        if self._cols is None:
            self._cols = {c: np.memmap(os.path.join(self.store_dir, f"{c}.bin"), dtype=t, mode="r")
                          for c, t in FULL_STORE_COLUMNS.items()}
        return self._cols

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cols"] = None
        return state

    def __len__(self):
        return len(self.index)

    def get(self, i):
        cols = self._open()
        s, n = int(self._starts[i]), int(self._lengths[i])
        return {
            "object_id": self.index["object_id"].iat[i], "label": int(self._targets[i]),
            "spectype": self.index["spectype"].iat[i],
            "mjd": np.array(cols["mjd"][s:s + n], dtype=np.float64),
            "flux": np.array(cols["flux"][s:s + n], dtype=np.float32),
            "flux_err": np.array(cols["flux_err"][s:s + n], dtype=np.float32),
            "band": np.array(cols["band"][s:s + n], dtype=np.int64),
        }


class StoreLightCurveDataset(torch.utils.data.Dataset):
    """Like `LightCurveDataset`, but builds each object's features on demand from a `PlasticcFullStore`.

    Same `build_features` and the same (features, label) items, so the original `collate` and `fit` apply.
    """

    def __init__(self, store, indices, flux_scale=1.0):
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.flux_scale = flux_scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        o = self.store.get(int(self.indices[k]))
        feats = build_features(o["mjd"], o["flux"], o["flux_err"], o["band"], self.flux_scale)
        return torch.from_numpy(feats), o["label"]
