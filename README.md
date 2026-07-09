# ParetoCL — Reproduction

Reproduction of the **evaluation experiments** in *"Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off"* (AAAI 2025, [arXiv:2503.23390](https://arxiv.org/abs/2503.23390)).

This repository contains only the code needed to reproduce **Table 1** (main results, ParetoCL row), **Table 2** (replay-buffer-size ablation), **Table 3** (training wall-clock time), and **Figure 3** (preference-conditioned Pareto front / dynamic vs. static preference adaptation). It was extracted and cleaned from a larger, exploratory reproduction codebase that also contained unrelated research extensions (a task-similarity-based preference-grouping scheme and a conditional-VAE hypernetwork variant) — neither of which appears in the paper, so both were removed here.

Baselines shown alongside ParetoCL in the paper's Table 1 (ER, DER++, CLSER, VR-MCL, …) and the MOO-baseline comparison in Table 4 (MGDA, Tchebycheff scalarization) are **not implemented** in this repository; it reproduces the ParetoCL method itself, not the full comparison table.

---

## Overview

Standard continual learning methods fix a single stability-plasticity trade-off (e.g. a constant replay weight λ). This paper reformulates the trade-off as a **multi-objective optimisation** problem and learns a *set* of Pareto-optimal solutions simultaneously, using a preference-conditioned hypernetwork. At inference time the best solution is selected *per sample* via entropy minimisation.

---

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install all dependencies
uv sync
```

All three datasets download automatically on first run from Hugging Face Hub (`uoft-cs/cifar10`, `uoft-cs/cifar100`, `zh-plus/tiny-imagenet`) via the `datasets` library, cached under `./data/hf_datasets`. This replaces torchvision's default sources (`www.cs.toronto.edu` / `cs231n.stanford.edu`) — single, non-CDN academic servers that are extremely slow or unreachable from some networks; HF Hub's CDN is ~10-20x faster in practice. `zh-plus/tiny-imagenet` is an unofficial community mirror; its 200 classes are the same, but the integer class-index-to-class ordering isn't guaranteed to match the original `wnids.txt` order, which only affects *which* classes land in which of the 10 tasks, not the task/class-count structure the paper specifies.

---

## Usage

### Table 1 — main results (5 seeds, paper defaults: buffer=32, lr=0.05)

```bash
bash scripts/run_table1.sh
```

Equivalent to running, for each of `{cifar10, cifar100, tinyimagenet} × {online, offline} × {seed 0..4}`:

```bash
uv run python train.py --dataset cifar100 --buffer_size 32 --lr 0.05 --seed 0 --log_dir results/table1_buffer32
# add --online for the online setting (1 epoch/task); omit for offline (5 epochs/task)
```

Aggregated AAA / final-Acc numbers are written to `results/table1_buffer32/visualisation/summary.csv` by `visualise.py` (run automatically at the end of the script).

### Table 2 — buffer-size ablation (Seq-CIFAR100, online setting)

```bash
BUFFER=600  DATASETS=cifar100 SEEDS="0 1 2" scripts/run_table1.sh
BUFFER=1000 DATASETS=cifar100 SEEDS="0 1 2" scripts/run_table1.sh
BUFFER=1400 DATASETS=cifar100 SEEDS="0 1 2" scripts/run_table1.sh
```

### Table 3 — training wall-clock time

`train.py` times the whole per-run task loop (training + buffer rebalancing + per-task validation) with `src.utils.timer` and writes it to `train_time_sec` in `results/.../seed{N}.json` — no separate script needed, it's recorded automatically by `scripts/run_table1.sh`. `visualise.py` aggregates it across seeds into `results/.../visualisation/table3_time.csv` (mean/std, matching the paper's Table 3 "Time (s)" column). Runs logged before this field existed (or `--debug`/custom `--num_tasks` runs) are skipped with a warning rather than averaged in.

### Figure 3(left) — Pareto front approximated by ParetoCL

Run `scripts/run_table1.sh` first (it saves a checkpoint after every task), then:

```bash
bash scripts/run_infer_sweep.sh
```

This sweeps `α = (α_stability, 1 - α_stability)` over 21 points for every per-task checkpoint. `visualise.py --sweep` (called automatically at the end) turns each stage's sweep into an **A_old vs A_new scatter plot** (`results/.../visualisation/pareto_front_{dataset}_{mode}_aftertask{t}.pdf`), colored by `preference[0]` (α_stability) — this is what Figure 3(left)'s Pareto front is built from. The paper's actual setting is Seq-CIFAR10, offline; stage 1 has no previous tasks (A_old undefined) and is skipped.

`infer.py --sweep` **appends** to any existing `seed*_inference_sweep_aftertask*.json`, so re-running a setting that was already swept would duplicate its entries. Use `SETTINGS` to scope a run to just the setting you need — e.g. to backfill the offline sweep without touching an already-generated online sweep:

```bash
SETTINGS=offline DATASETS=cifar10 bash scripts/run_infer_sweep.sh
```

### Figure 3(right) — ParetoCL vs ParetoCL-- incremental accuracy

The paper compares 5 methods here (ParetoCL, ParetoCL--, ER, DER++, CLSER). Since this repository only implements ParetoCL (see the top-of-README note on unimplemented baselines), this reproduction is limited to **ParetoCL vs ParetoCL--** on Seq-CIFAR100, online setting:

```bash
bash scripts/run_paretocl_minus.sh
```

ParetoCL's dynamic curve (Algorithm 2, entropy-based selection) is already what `train.py` logs as `aa_after` in `seed{N}.json` — no extra run needed. ParetoCL-- uses a static `α=(0.5, 0.5)` at inference instead:

```bash
uv run python infer.py --model-pt <ckpt>.pt --model-config <ckpt>_model_config.json --dataset cifar100 --preference 0.5 0.5
```

`scripts/run_paretocl_minus.sh` runs this against every per-task checkpoint and calls `visualise.py --fig3_right`, which plots both curves to `results/.../visualisation/figure3_right_cifar100_online.pdf`.

### Quick debug run (1 epoch, 2 tasks, tiny buffer)

```bash
uv run python train.py --debug --dataset cifar10
```

---

## Algorithm

### Training — Algorithm 1 (Preference-Conditioned Learning)

For each mini-batch from the current task:

1. Sample **K = 5** preference vectors `α = (α₁, α₂)` from a **Dirichlet(1,1)** prior (uniform over the 2-simplex, so `α₁ + α₂ = 1`).
2. For each `αₖ`:
   - Compute the *plasticity loss* on new-task data: `L_new(θ, αₖ)`
   - Compute the *stability loss* on replay-buffer data: `L_replay(θ, αₖ)`
   - Weighted loss: `ℓₖ = α₁ · L_replay + α₂ · L_new`
3. Backpropagate on the **average** across K samples:

```
L(θ) = (1/K) Σₖ ℓₖ  ≈  E_{α∼p(α)}[α₁·L_replay(θ,α) + α₂·L_new(θ,α)]
```

### Inference — Algorithm 2 (Inference-Time Preference Adaptation)

For each test sample `x`:

1. Sample **K = 20** preference vectors from the same Dirichlet prior.
2. Run a forward pass for each `αₖ`, obtaining logits `f(x; αₖ)`.
3. Compute the Shannon entropy `H(softmax(f(x; αₖ)))` for each `k`.
4. Return the logits with **minimum entropy** (highest prediction confidence):

```
y* = argmin_k  H( softmax( f(x; αₖ) ) )
```

---

## Model Architecture

```
Input image
    │
    ▼
┌─────────────────────────────┐
│  Backbone (feature encoder) │
│  ResNet-18, 3×3 first conv, │
│  no maxpool (CIFAR, 32×32)  │
│  or with maxpool (TinyImageNet, 64×64) │
└─────────────────┬───────────┘
                  │ features h(x)  (512-d)
                  │
          ┌───────┴────────┐
          │  Hypernetwork Ψ│   ← preference α  (2-d)
          │  MLP: 2→100→100→100
          │      →(W, b)   │
          └───────┬────────┘
                  │ generated weights W(α) ∈ ℝ^{C×d}, b(α) ∈ ℝ^C
                  │
                  ▼
         logits = W(α)·h(x) + b(α)
```

The backbone is **shared** across all preferences. Only the final classifier head is preference-conditioned, generated on the fly by the hypernetwork.

### Class-Incremental Head Expansion

The model starts with `classes_per_task` output classes and its head grows at each task boundary. Before `trainer.fit()` for task `t > 0`, `train.py` calls `model.expand_head((task_id + 1) * dm.classes_per_task)`, which replaces the hypernetwork's final `nn.Linear` layer with a wider one, copying existing class weights/biases and randomly initialising the new ones (Kaiming uniform, the PyTorch default).

---

## Replay Buffer

`ReplayBuffer` pre-allocates fixed-size tensors `(buffer_size, *input_shape)`. `update_buffer` in `train.py` rebalances the buffer after each task to hold approximately `buffer_size // n_tasks_so_far` samples per task, preventing any single task from dominating.

---

## Code Structure

```
ParetoCL-reproduction/
├── src/
│   ├── models/
│   │   └── pareto_model.py     # HypernetworkMLP + PreferenceConditionedModel
│   ├── pl_modules/
│   │   └── pareto_cl.py        # Lightning module: Algorithms 1 & 2
│   ├── data/
│   │   └── continual_data.py   # ReplayBuffer, ContinualDataModule (Split CIFAR-10/100/TinyImageNet)
│   └── utils.py                # CLI args, preference-vector generation for infer.py
├── train.py                    # Sequential task-loop training (Table 1 & 2)
├── infer.py                    # Fixed / swept / entropy-based inference (Figure 3)
├── visualise.py                # Aggregates results into Table 1/2 summaries and Figure 3 plots
└── scripts/
    ├── run_table1.sh           # Table 1 & 2 driver
    ├── run_infer_sweep.sh      # Figure 3(left) driver
    └── run_paretocl_minus.sh   # Figure 3(right) driver
```

### Key classes and functions

| Symbol | File | Description |
|---|---|---|
| `HypernetworkMLP` | `pareto_model.py` | 3-hidden-layer MLP mapping `α → (W, b)` for the classifier |
| `PreferenceConditionedModel` | `pareto_model.py` | Backbone + hypernetwork; `torch.bmm` for batched per-sample classification |
| `PreferenceConditionedModel.expand_head(n)` | `pareto_model.py` | Grows the hypernetwork output head to `n` classes; copies old weights |
| `ParetoCL` | `pareto_cl.py` | Lightning module; implements Algorithms 1 & 2 |
| `ParetoCL._sample_dirichlet_preference(K)` | `pareto_cl.py` | Returns `(K, 2)` tensor sampled from `Dirichlet(1,1)` |
| `ParetoCL._infer_with_preference_adaptation(x)` | `pareto_cl.py` | Algorithm 2 — picks min-entropy prediction over K=20 preferences |
| `ReplayBuffer` | `continual_data.py` | Pre-allocated buffer with random-replacement strategy |
| `ContinualDataModule` | `continual_data.py` | Splits CIFAR-10/100/TinyImageNet into tasks; per-task and cumulative val dataloaders |
| `visualise_sweep` | `visualise.py` | Figure 3(left) — A_old vs A_new Pareto-front scatter per stage, colored by preference |
| `visualise_incremental_accuracy` | `visualise.py` | Figure 3(right) — ParetoCL vs ParetoCL-- accuracy-after-each-task curves |
| `update_buffer` | `train.py` | Rebalances buffer after each task |
| `timer` | `utils.py` | Context manager used to time the Table 3 training wall-clock |

---

## Hyperparameters (paper defaults, as used for Table 1)

| Hyperparameter | Value | Location |
|---|---|---|
| Optimizer | SGD (no momentum — see note in `pareto_cl.py`) | `ParetoCL.configure_optimizers` |
| Learning rate | 0.05 | `train.py` default |
| Replay buffer size | 32 | `train.py` default |
| Training preferences K | 5 | `ParetoCL.TRAIN_K` |
| Inference preferences K | 20 | `ParetoCL.INFER_K` |
| Preference prior | Dirichlet(1,1) | `_sample_dirichlet_preference` |
| Backbone | ResNet-18 (ImageNet-pretrained, 3×3 stem) | `PreferenceConditionedModel` |
| Hypernetwork hidden dim | 100 | `HypernetworkMLP` |
| Offline epochs/task | 5 | `train.py` |
| Online epochs/task | 1 | `train.py --online` |

---

## References

- Lai et al., *Pareto Continual Learning: Preference-Conditioned Learning and Adaption for Dynamic Stability-Plasticity Trade-off*, AAAI 2025. [arXiv:2503.23390](https://arxiv.org/abs/2503.23390)
