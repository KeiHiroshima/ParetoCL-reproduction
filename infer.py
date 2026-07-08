import argparse
import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.data.continual_data import ContinualDataModule
from src.models.pareto_model import PreferenceConditionedModel
from src.utils import generate_preferences

DATASET_CONFIG = {
    "cifar10": (5, 2),
    "cifar100": (10, 10),
    "tinyimagenet": (10, 20),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-pt", type=str, required=True, help="Path to model.pt")
    parser.add_argument(
        "--model-config", type=str, required=True, help="Path to model_config.json"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_CONFIG.keys())
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep preference[0] from 0 to 1 (for Figure 3 Pareto front)",
    )
    parser.add_argument(
        "--preference",
        type=float,
        nargs="+",
        default=None,
        help="Fixed preference vector (e.g. [0.5, 0.5])",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def load_model(
    model_pt: str, model_config: str, device: torch.device
) -> tuple[PreferenceConditionedModel, dict]:
    with open(model_config) as f:
        cfg = json.load(f)

    num_classes = (
        DATASET_CONFIG[cfg["dataset"]][1]
        * int(model_pt.split("_aftertask")[-1].split(".")[0])
        if "aftertask" in model_pt
        else cfg["num_classes"]
    )

    model = PreferenceConditionedModel(
        num_classes=num_classes,
        backbone_name=cfg["backbone_name"],
        preference_dim=cfg["preference_dim"],
        hidden_dim=cfg.get("hidden_dim", 100),
    )
    model.load_state_dict(torch.load(model_pt, map_location=device))
    model.to(device)
    model.eval()
    return model, cfg


def infer_fixed(model, loader, preference: torch.Tensor, device: torch.device):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, preference)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return correct / total, logits.argmax(dim=1)[:20].cpu().tolist()


def infer_auto(model, loader, preference_dim: int, device: torch.device):
    """Algorithm 2: pick lowest-entropy preference per sample."""
    INFER_K = 20  # used only when PREFERENCE is None
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            B = x.size(0)

            concentration = torch.ones(INFER_K, preference_dim, device=device)
            preferences = torch.distributions.Dirichlet(
                concentration
            ).sample()  # (K, pref_dim)

            all_logits = []
            for k in range(INFER_K):
                alpha = preferences[k].unsqueeze(0)  # (1, pref_dim)
                all_logits.append(model(x, alpha))  # (B, num_classes)

            all_logits = torch.stack(all_logits, dim=0)  # (K, B, num_classes)
            probs = torch.softmax(all_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-8))).sum(dim=-1)  # (K, B)
            best_k = entropy.argmin(dim=0)  # (B,)
            best_k_exp = best_k.view(1, B, 1).expand(1, B, all_logits.size(-1))
            best_logits = all_logits.gather(0, best_k_exp).squeeze(
                0
            )  # (B, num_classes)

            correct += (best_logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return correct / total


def infer_alltasks(
    model,
    num_tasks,
    task_datasets,
    args,
    alpha: torch.Tensor | None,
    save_path: Path,
    device: torch.device,
    cfg=None,
):
    accs = []
    probs_list = []
    for task_id in range(num_tasks):
        _, test_subset = task_datasets[task_id]
        loader = DataLoader(
            test_subset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )

        if alpha is not None:
            acc, probs = infer_fixed(model, loader, alpha, device)
            probs_list.append(probs)
        else:
            acc = infer_auto(model, loader, cfg["preference_dim"], device)

        accs.append(acc)

    avg_acc = sum(accs) / len(accs)

    entry = {
        "preference": alpha.cpu().tolist() if alpha is not None else None,
        "task_accuracies": accs,
        "average_accuracy": avg_acc,
        "probs": probs_list if alpha is not None else None,
    }

    if save_path.exists():
        data = json.loads(save_path.read_text()) + [entry]
    else:
        data = [entry]
    save_path.write_text(json.dumps(data, indent=4))


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model, cfg = load_model(args.model_pt, args.model_config, device)
    print(
        f"Loaded: {cfg['backbone_name']} | num_classes={cfg['num_classes']} | pref_dim={cfg['preference_dim']}"
    )

    num_tasks, _ = DATASET_CONFIG[args.dataset]
    dm = ContinualDataModule(args.dataset, args.batch_size, num_tasks, buffer_size=0)
    dm.prepare_data()
    dm.setup()

    def _save_path(suffix: str) -> Path:
        if "aftertask" in args.model_pt:
            task_id = int(args.model_pt.split("_aftertask")[-1].split(".")[0])
            return (
                Path(args.model_pt).parent
                / f"seed{args.seed}_inference_{suffix}_aftertask{task_id}.json"
            )
        return Path(args.model_pt).parent / f"seed{args.seed}_inference_{suffix}.json"

    if args.sweep:
        num_samples = 20
        sta_vals = torch.linspace(0, 1, num_samples + 1)
        pla_vals = (1.0 - sta_vals).detach().clone()
        alpha_list = torch.stack([sta_vals, pla_vals], dim=1).to(device)

        save_path = _save_path("sweep")
        for alpha in alpha_list:
            infer_alltasks(
                model, num_tasks, dm.task_datasets, args, alpha, save_path, device
            )

    elif args.preference is not None:
        alpha = torch.tensor([args.preference], dtype=torch.float32, device=device)
        print(f"\nMode: fixed preference α={args.preference}")
        save_path = _save_path("fixed")
        infer_alltasks(
            model, num_tasks, dm.task_datasets, args, alpha, save_path, device
        )

    else:
        save_path = _save_path("fixed")
        preferences = generate_preferences(cfg["preference_dim"])
        for pref in preferences:
            alpha = torch.tensor([pref], dtype=torch.float32, device=device)
            print(f"\nMode: generated preference α={pref}")
            infer_alltasks(
                model, num_tasks, dm.task_datasets, args, alpha, save_path, device
            )


if __name__ == "__main__":
    main()
