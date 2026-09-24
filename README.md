# Dynamic Classification of Tidal Disruption Events

Short demo is available in Demo folder with instructions. 

## Root files

| Path | What it is |
|---|---|
| `PROJECT_PROPOSAL.pdf` | The original project proposal. |
| `requirements.txt` | Python dependencies (PyTorch, pandas, scikit-learn, matplotlib). |
| `run_experiment.py` | Stage 0: GRU trained from scratch on MALLORN only. |
| `pretrain_plasticc.py` | Stage 1a: 14-class pre-training on the PLAsTiCC training set. |
| `run_cv.py` | Stage 1b: 5-fold cross-validation comparing scratch / frozen / fine-tuned transfer. |
| `run_final_test.py` | Stage 2: the evaluation on the locked test set. |
| `pretrain_plasticc_full.py` | Stage 3: same pre-training on the full PLAsTiCC release (`--build-store` first). |
| `run_cv_full_plasticc.py` | Stage 3: repeats the Stage 1b cross-validation with the full-PLAsTiCC encoder. |
| `analyze_full_plasticc.py` | Stage 3: builds the comparison tables and figures from saved outputs. |

## Folders

| Path | What is in it |
|---|---|
| `demo/` | `python demo.py` runs the trained model on five examples of light curves and plots P(TDE) after every observation. |
| `tde/` | Training code: data loading (`data.py`), neural net (`model.py`), training (`train.py`), cutoff evaluation (`evaluate.py`), config (`config.py`). |
| `configs/` | `default.yaml` holds every setting; `full_plasticc_pretraining.yaml` overrides only paths for Stage 3. |
| `tests/` | Unit tests, including the causality/leakage checks. Run `python -m unittest discover -s tests -t .` |
| `diagnostics/` | Analysis scripts: per-class breakdowns, flux calibration. |
| `data/` | The datasets used: the labelled MALLORN training set and the PLAsTiCC training set (does not include full PLAsTiCC since it's too heavy). |
| `outputs/` | Results of the main project: `pretrain/`, `cv/`, `runs/`, and `final_test/` (locked-test metrics, predictions, checkpoints). |
| `results/` | The post-completion full-PLAsTiCC experiment: its pre-training, cross-validation, locked-test run and figures. |

