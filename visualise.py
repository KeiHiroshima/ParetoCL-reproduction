import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

DATASET_CONFIG = {"cifar10": (5, 2), "cifar100": (10, 10), "tinyimagenet": (10, 20)}

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]


def get_aa_aaa(path_shared, seed_list):
    """Table 1 / Table 2: AAA and final Acc, averaged over seeds."""
    dataset_list = DATASET_CONFIG.keys()

    df = pd.DataFrame(
        columns=["aaa_mean", "aaa_std", "final_aa_mean", "final_aa_std"], index=[]
    )
    for dataset in dataset_list:
        for mode in ["online", "offline"]:
            final_aa_list = []
            aaa_list = []
            for seed in seed_list:
                p = path_shared / dataset / mode / f"seed{seed}.json"
                if not p.exists():
                    print(f"Missing: {p}")
                    continue
                with open(p) as f:
                    results = json.load(f)
                final_aa_list.append(results["final_acc"])
                aaa_list.append(results["aaa"])

            if not final_aa_list:
                continue

            final_aa_mean = sum(final_aa_list) / len(final_aa_list)
            final_aa_std = (
                sum((x - final_aa_mean) ** 2 for x in final_aa_list)
                / len(final_aa_list)
            ) ** 0.5

            aaa_mean = sum(aaa_list) / len(aaa_list)
            aaa_std = (
                sum((x - aaa_mean) ** 2 for x in aaa_list) / len(aaa_list)
            ) ** 0.5

            df.loc[f"{dataset}_{mode}"] = {
                "aaa_mean": round(aaa_mean, 4) * 100,
                "aaa_std": round(aaa_std, 4) * 100,
                "final_aa_mean": round(final_aa_mean, 4) * 100,
                "final_aa_std": round(final_aa_std, 4) * 100,
            }

    print(df)
    df.to_csv(path_shared / "visualisation" / "summary.csv", index=True)


def get_taskwise_aa(path_shared, seed_list):
    """Per-task AA heatmap, averaged over seeds (supplementary detail behind Table 1)."""
    for dataset, (num_tasks, _) in DATASET_CONFIG.items():
        for mode in ["online", "offline"]:
            taskwise_aa_dict = {i: [] for i in range(num_tasks)}
            taskwise_aa_mean_dict = {i: [] for i in range(num_tasks)}
            taskwise_aa_std_dict = {i: [] for i in range(num_tasks)}

            found_any = False
            for seed in seed_list:
                p = path_shared / dataset / mode / f"seed{seed}.json"
                if not p.exists():
                    continue
                found_any = True
                with open(p) as f:
                    results = json.load(f)

                for task_id, aa_list in results["taskwise_accuracies"].items():
                    taskwise_aa_dict[int(task_id)].append(aa_list)

            if not found_any:
                continue

            for i, aa_lists in taskwise_aa_dict.items():
                if not aa_lists:
                    continue
                np_aa_lists = np.array(aa_lists)
                aa_mean = np.mean(np_aa_lists, axis=0)
                aa_std = np.std(np_aa_lists, axis=0)

                taskwise_aa_mean_dict[i] = [round(aa * 100, 4) for aa in aa_mean] + [
                    None
                ] * (num_tasks - len(aa_mean))
                taskwise_aa_std_dict[i] = [round(aa * 100, 4) for aa in aa_std] + [
                    None
                ] * (num_tasks - len(aa_std))

            df_taskwise_aa_mean_dict = pd.DataFrame(taskwise_aa_mean_dict).T
            df_taskwise_aa_std_dict = pd.DataFrame(taskwise_aa_std_dict).T
            df_taskwise_aa_mean_dict.to_csv(
                path_shared
                / "visualisation"
                / f"taskwise_aa_mean_{dataset}_{mode}.csv",
                index=True,
            )
            df_taskwise_aa_std_dict.to_csv(
                path_shared / "visualisation" / f"taskwise_aa_std_{dataset}_{mode}.csv",
                index=True,
            )

            df_annot = df_taskwise_aa_mean_dict.copy().astype(object)
            for i in df_taskwise_aa_mean_dict.index:
                for j in df_taskwise_aa_mean_dict.columns:
                    mean_val = df_taskwise_aa_mean_dict.loc[i, j]
                    std_val = df_taskwise_aa_std_dict.loc[i, j]
                    if mean_val is None or std_val is None:
                        df_annot.loc[i, j] = ""
                    else:
                        df_annot.loc[i, j] = f"{mean_val:.1f}\n±{std_val:.1f}"

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                df_taskwise_aa_mean_dict,
                annot=df_annot,
                fmt="",
                cmap="coolwarm",
                ax=ax,
            )
            ax.set_title(f"{dataset}_{mode} Task-wise AA")
            ax.set_xlabel("Evaluation task", fontsize=12)
            ax.set_ylabel("Training task", fontsize=12)
            plt.tight_layout()
            plt.savefig(
                path_shared
                / "visualisation"
                / f"taskwise_aa_heatmap_{dataset}_{mode}.pdf"
            )
            plt.close()


def _aggregate_seeds(seed_data: list) -> list:
    """Average task_accuracies and average_accuracy across seeds at each preference step."""
    ref = seed_data[0]
    aggregated = []
    for i, entry in enumerate(ref):
        num_tasks = len(entry["task_accuracies"])
        task_accs = np.mean(
            [[s[i]["task_accuracies"][t] for t in range(num_tasks)] for s in seed_data],
            axis=0,
        )
        task_accs_std = np.std(
            [[s[i]["task_accuracies"][t] for t in range(num_tasks)] for s in seed_data],
            axis=0,
        )
        avg_acc = float(np.mean([s[i]["average_accuracy"] for s in seed_data]))
        avg_acc_std = float(np.std([s[i]["average_accuracy"] for s in seed_data]))
        aggregated.append(
            {
                "preference": entry["preference"],
                "task_accuracies": task_accs.tolist(),
                "task_accuracies_std": task_accs_std.tolist(),
                "average_accuracy": avg_acc,
                "average_accuracy_std": avg_acc_std,
            }
        )
    return aggregated


def visualise_sweep(path_shared: Path, seed_list: list) -> None:
    """Figure 3(left): Pareto front approximated by ParetoCL at each training stage."""
    colors = plt.get_cmap("tab10").colors

    for dataset, (num_tasks, _) in DATASET_CONFIG.items():
        for task_id in range(1, num_tasks + 1):
            for mode in ["online", "offline"]:
                seed_data = []
                for seed in seed_list:
                    p = (
                        path_shared
                        / dataset
                        / mode
                        / f"seed{seed}_inference_sweep_aftertask{task_id}.json"
                    )
                    if not p.exists():
                        print(f"Missing: {p}")
                        continue
                    with open(p) as f:
                        seed_data.append(json.load(f))

                if not seed_data:
                    continue

                aggregated = _aggregate_seeds(seed_data)
                pref_x = [entry["preference"][0] for entry in aggregated]

                fig, ax = plt.subplots(figsize=(6, 4))

                for t in range(num_tasks):
                    accs = [entry["task_accuracies"][t] for entry in aggregated]
                    accs_std = [entry["task_accuracies_std"][t] for entry in aggregated]
                    ax.plot(
                        pref_x,
                        accs,
                        color=colors[t % len(colors)],
                        linestyle=LINESTYLES[t % len(LINESTYLES)],
                        marker="o",
                        markersize=3,
                        label=f"task_{t + 1}",
                    )
                    ax.fill_between(
                        pref_x,
                        np.array(accs) - np.array(accs_std),
                        np.array(accs) + np.array(accs_std),
                        color=colors[t % len(colors)],
                        alpha=0.2,
                    )

                avg_accs = [entry["average_accuracy"] for entry in aggregated]
                avg_accs_std = [entry["average_accuracy_std"] for entry in aggregated]
                ax.plot(
                    pref_x,
                    avg_accs,
                    color="black",
                    linestyle="-",
                    linewidth=2,
                    marker="s",
                    markersize=4,
                    label="average_accuracy",
                )
                ax.fill_between(
                    pref_x,
                    np.array(avg_accs) - np.array(avg_accs_std),
                    np.array(avg_accs) + np.array(avg_accs_std),
                    color="black",
                    alpha=0.2,
                )

                ax.set_xlabel("preference[0] (α_stability)")
                ax.set_ylabel("accuracy")
                ax.set_title(f"{dataset} {mode} — preference sweep (after task {task_id})")
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
                plt.tight_layout()

                out_path = (
                    path_shared
                    / "visualisation"
                    / f"sweep_{dataset}_{mode}_aftertask{task_id}.pdf"
                )
                plt.savefig(out_path, bbox_inches="tight")
                plt.close()
                print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path_shared",
        type=str,
        default="results",
        help="Path to the shared results directory.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to aggregate over.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also plot the Figure 3(left) preference sweep (requires infer.py --sweep logs).",
    )
    args = parser.parse_args()
    path_shared = Path(args.path_shared)

    os.makedirs(path_shared / "visualisation", exist_ok=True)

    get_aa_aaa(path_shared, args.seeds)
    get_taskwise_aa(path_shared, args.seeds)

    if args.sweep:
        visualise_sweep(path_shared, args.seeds)
