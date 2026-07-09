# ParetoCL — Reproduction

Reproduction of the **evaluation experiments** in *"Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off"* (AAAI 2025, [arXiv:2503.23390](https://arxiv.org/abs/2503.23390)).


## Results

### Accuracy

| Dataset | Setting | AAA (Paper) | AAA (Ours) | Acc (Paper) | Acc (Ours) |
|---|---|---|---|---|---|
| Seq-CIFAR10 | Online | 70.89 | 64.73 ± 0.26 | 59.95 | 50.33 ± 1.35 |
| Seq-CIFAR10 | Offline | 78.98 | 69.88 ± 1.20 | 69.55 | 52.31 ± 1.45 |
| Seq-CIFAR100 | Online | 33.04 | 29.14 ± 1.04 | 24.45 | 16.93 ± 0.71 |
| Seq-CIFAR100 | Offline | 44.32 | 39.52 ± 1.27 | 28.79 | 22.41 ± 0.83 |
| Seq-TinyImageNet | Online | 31.72 | 16.23 ± 0.34 | 23.09 | 7.18 ± 0.35 |
| Seq-TinyImageNet | Offline | 43.02 | 23.42 ± 1.93 | 28.28 | 9.75 ± 1.27 |

### Training time (Seq-CIFAR10 only)

| Setting | Paper (s) | Ours (s) |
|---|---|---|
| Online | 224.62 | 522.1 ± 1.3 |
| Offline | not reported | 1431.1 ± 1.2 |


---

## Hyperparameters

| Hyperparameter | Value | Location |
|---|---|---|
| Optimizer | SGD (no momentum — see note in `pareto_cl.py`) | `ParetoCL.configure_optimizers` |
| Learning rate | 0.05 | `train.py` default |
| Replay buffer size | 1000 | `train.py` default |
| Training preferences K | 5 | `ParetoCL.TRAIN_K` |
| Inference preferences K | 20 | `ParetoCL.INFER_K` |
| Preference prior | Dirichlet(1,1) | `_sample_dirichlet_preference` |
| Backbone | ResNet-18 (ImageNet-pretrained, 3×3 stem) | `PreferenceConditionedModel` |
| Hypernetwork hidden dim | 100 | `HypernetworkMLP` |
| Offline epochs/task | 5 | `train.py` |
| Online epochs/task | 1 | `train.py --online` |

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv
uv sync                                           # install dependencies
```

All three datasets download automatically on first run from Hugging Face Hub (`uoft-cs/cifar10`, `uoft-cs/cifar100`, `zh-plus/tiny-imagenet`), cached under `./data/hf_datasets`.

---

## Usage

### Table 1 — main results

```bash
LOG_DIR=results/table1_buffer1000 bash scripts/run_train.sh
```

3 seeds × `{cifar10, cifar100, tinyimagenet}` × `{online, offline}`. Aggregated AAA/Acc → `results/table1_buffer1000/visualisation/summary.csv`.

### Table 2 — buffer-size ablation (Seq-CIFAR100, online)

```bash
BUFFER=600  DATASETS=cifar100 SEEDS="0 1 2" scripts/run_train.sh
BUFFER=1000 DATASETS=cifar100 SEEDS="0 1 2" scripts/run_train.sh
BUFFER=1400 DATASETS=cifar100 SEEDS="0 1 2" scripts/run_train.sh
```

### Table 3 — training wall-clock time

Recorded automatically by the Table 1/2 runs above (`train_time_sec` in `seed{N}.json`); aggregated into `visualisation/table3_time.csv`.

### Figure 3(left) — Pareto front

```bash
bash scripts/run_infer_sweep.sh   # after Table 1's run_train.sh (needs its checkpoints)
```

Sweeps 21 preference points per per-task checkpoint → `visualisation/pareto_front_{dataset}_{mode}_aftertask{t}.pdf` (A_old vs A_new). Paper setting: Seq-CIFAR10, offline.

### Figure 3(right) — ParetoCL vs ParetoCL--

```bash
bash scripts/run_paretocl_minus.sh   # after Table 1's run_train.sh (needs its checkpoints)
```

Compares dynamic (entropy-selected) vs static `α=(0.5,0.5)` inference on Seq-CIFAR100/online → `visualisation/figure3_right_cifar100_online.pdf`. Only ParetoCL vs ParetoCL-- is reproduced (other baselines aren't implemented here).

### Quick debug run

```bash
uv run python train.py --debug --dataset cifar10   # 1 epoch, 2 tasks, tiny buffer
```

---

## References

- Lai et al., *Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off*, AAAI 2025. [arXiv:2503.23390](https://arxiv.org/abs/2503.23390)
