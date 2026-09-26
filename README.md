# CFDGN: Context-aware Facet Disentanglement Graph Network for Discriminative Bundle Recommendation

This repository contains the CFDGN implementation and the preprocessed Youshu, NetEase, and iFashion datasets. CFDGN builds on a MultiCBR collaborative backbone and adds bundle-specific item roles, member-derived bundle representations, and candidate-conditioned selection over history-derived interests.

## Repository Layout

```text
CFDGN/
├── train.py                 Training, validation-based checkpoint selection,
│                            and test evaluation
├── models/
│   └── CFDGN.py            CFDGN and MultiCBR model implementations
├── utility.py              Dataset loading and ranking evaluation
├── config.yaml             Dataset-specific experimental settings
├── requirements.txt        Python dependencies
└── datasets/
    ├── Youshu/
    ├── NetEase/
    └── iFashion/            Preprocessed benchmark datasets
```

The `log/`, `runs/`, and `checkpoints/` directories are created when training starts.

## Requirements

Install the dependencies from the directory containing `train.py`:

```bash
python -m pip install -r requirements.txt
```

The requirements include PyTorch (at least version 1.9.0), NumPy, SciPy, PyYAML, tqdm, and TensorBoard. A CUDA-capable GPU is recommended for training. The `-g` option selects the visible GPU.

## Data

The three preprocessed datasets are included under `datasets/`. Each dataset directory contains:

| File | Content |
| --- | --- |
| `<dataset>_data_size.txt` | Numbers of users, bundles, and items |
| `bundle_item.txt` | Bundle–item membership relations |
| `user_item.txt` | User–item interactions |
| `user_bundle_train.txt` | Training user–bundle interactions |
| `user_bundle_tune.txt` | Validation user–bundle interactions |
| `user_bundle_test.txt` | Test user–bundle interactions |

The iFashion directory also includes identifier-mapping JSON files. In `config.yaml`, `data_path` defaults to `./datasets`. Run the following commands from the repository's `CFDGN/` directory; the included datasets require no path changes.

## Running CFDGN

```bash
# Youshu
python train.py -g 0 -m CFDGN -d Youshu

# NetEase and iFashion
python train.py -g 0 -m CFDGN -d NetEase
python train.py -g 0 -m CFDGN -d iFashion
```

The default random seed is 2024. To run another seed, add `-s`, for example:

```bash
python train.py -g 0 -m CFDGN -d Youshu -s 2025
```

Dataset-specific model and training settings are defined in `config.yaml`. CFDGN selects a checkpoint using validation Recall and NDCG at 20 and 40, then evaluates the selected checkpoint on the test set once.



## Outputs

Training writes text logs to `log/<dataset>/<model>/`, TensorBoard records to `runs/<dataset>/<model>/`, and model checkpoints and their configurations to `checkpoints/<dataset>/<model>/`. The run names include the seed and experimental settings. Validation and final test metrics are printed and recorded in the run log.
