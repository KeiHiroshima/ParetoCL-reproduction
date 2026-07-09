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


def get_train_time(path_shared, seed_list):
    """Table 3: total training wall-clock time (s), averaged over seeds.

    Reads the "train_time_sec" field written by train.py (total time for the
    task loop: training + buffer rebalancing + per-task validation). Runs
    logged before this field existed are skipped with a warning.
    """
    dataset_list = DATASET_CONFIG.keys()

    df = pd.DataFrame(columns=["time_sec_mean", "time_sec_std"], index=[])
    for dataset in dataset_list:
        for mode in ["online", "offline"]:
            time_list = []
            for seed in seed_list:
                p = path_shared / dataset / mode / f"seed{seed}.json"
                if not p.exists():
                    continue
                with open(p) as f:
                    results = json.load(f)
                if "train_time_sec" not in results:
                    print(
                        f"No train_time_sec in: {p} (re-run with the updated train.py)"
                    )
                    continue
                time_list.append(results["train_time_sec"])

            if not time_list:
                continue

            time_mean = sum(time_list) / len(time_list)
            time_std = (
                sum((x - time_mean) ** 2 for x in time_list) / len(time_list)
            ) ** 0.5

            df.loc[f"{dataset}_{mode}"] = {
                "time_sec_mean": round(time_mean, 1),
                "time_sec_std": round(time_std, 1),
            }

    if not df.empty:
        print(df)
        df.to_csv(path_shared / "visualisation" / "table3_time.csv", index=True)


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
    """Figure 3(left): Pareto front approximated by ParetoCL at each training stage.

    For each per-task checkpoint (stage), scatters A_old (mean accuracy on
    previously-learned tasks, y-axis) against A_new (accuracy on the
    just-learned task, x-axis) for every swept preference vector, colored by
    preference[0] (α_stability). Stage 1 has no previous tasks (A_old is
    undefined) and is skipped, matching the paper's 4-panel layout for
    Seq-CIFAR10 (5 tasks → stages 2..5).
    """
    cmap = plt.get_cmap("coolwarm")

    for dataset, (num_tasks, _) in DATASET_CONFIG.items():
        for task_id in range(2, num_tasks + 1):
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

                pref0 = np.array([entry["preference"][0] for entry in aggregated])
                a_new = np.array(
                    [entry["task_accuracies"][task_id - 1] for entry in aggregated]
                )
                a_old = np.array(
                    [
                        np.mean(entry["task_accuracies"][: task_id - 1])
                        for entry in aggregated
                    ]
                )

                fig, ax = plt.subplots(figsize=(5, 5))
                sc = ax.scatter(
                    a_new * 100,
                    a_old * 100,
                    c=pref0,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                    s=40,
                    edgecolors="black",
                    linewidths=0.3,
                )
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label("preference[0] (α_stability)")

                ax.set_xlabel("A_new (accuracy on current task, %)")
                ax.set_ylabel("A_old (avg. accuracy on previous tasks, %)")
                ax.set_title(f"{dataset} {mode} — Pareto front (after task {task_id})")
                plt.tight_layout()

                out_path = path_shared / "visualisation_inference"
                os.makedirs(out_path, exist_ok=True)
                plt.savefig(
                    out_path / f"pareto_front_{dataset}_{mode}_aftertask{task_id}.pdf",
                    bbox_inches="tight",
                )
                plt.close()
                print(
                    f"Saved: {out_path / f'pareto_front_{dataset}_{mode}_aftertask{task_id}.pdf'}"
                )


def visualise_incremental_accuracy(path_shared: Path, seed_list: list) -> None:
    """Figure 3(right): ParetoCL (dynamic) vs ParetoCL-- (static α=(0.5, 0.5))
    average-accuracy-after-each-task curves, Seq-CIFAR100 online setting.

    ER, DER++, and CLSER are not implemented in this repository and are
    therefore not included as comparison curves — this reproduces only the
    two ParetoCL variants from Figure 3(right).

    ParetoCL-- curves are read from seed{n}_inference_fixed_aftertask{t}.json
    (produced by `infer.py --preference 0.5 0.5` against each per-task
    checkpoint, see scripts/run_paretocl_minus.sh). Note that infer.py
    evaluates against all of the dataset's tasks regardless of how many the
    checkpoint has actually learned, so the JSON's own "average_accuracy"
    field is wrong for non-final stages (it's diluted by never-learned
    tasks); we ignore it and instead recompute the average from
    task_accuracies[:t], which is valid for any t.
    """
    dataset = "cifar100"
    mode = "online"
    num_tasks, _ = DATASET_CONFIG[dataset]
    base = path_shared / dataset / mode

    paretocl_curves = []
    paretocl_minus_curves = []

    for seed in seed_list:
        p = base / f"seed{seed}.json"
        if p.exists():
            with open(p) as f:
                results = json.load(f)
            aa_after = results.get("aa_after")
            if aa_after and len(aa_after) == num_tasks:
                paretocl_curves.append(aa_after)
            else:
                print(f"Incomplete aa_after in: {p}")
        else:
            print(f"Missing: {p}")

        minus_curve = []
        for t in range(1, num_tasks + 1):
            fp = base / f"seed{seed}_inference_fixed_aftertask{t}.json"
            if not fp.exists():
                print(f"Missing: {fp}")
                break
            with open(fp) as f:
                entries = json.load(f)
            task_accs = entries[-1]["task_accuracies"]
            minus_curve.append(float(np.mean(task_accs[:t])))
        if len(minus_curve) == num_tasks:
            paretocl_minus_curves.append(minus_curve)

    if not paretocl_curves and not paretocl_minus_curves:
        print("No data found for Figure 3(right); skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, num_tasks + 1)

    for label, curves, color in [
        ("ParetoCL", paretocl_curves, "tab:blue"),
        ("ParetoCL--", paretocl_minus_curves, "tab:orange"),
    ]:
        if not curves:
            continue
        arr = np.array(curves) * 100
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        ax.plot(x, mean, marker="o", markersize=3, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel("Learned tasks")
    ax.set_ylabel("Accuracy after each task (%)")
    ax.set_title(f"{dataset} {mode} — incremental accuracy")
    ax.set_xticks(x)
    ax.legend()
    plt.tight_layout()

    out_path = path_shared / "visualisation" / f"figure3_right_{dataset}_{mode}.pdf"
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
    parser.add_argument(
        "--fig3_right",
        action="store_true",
        help=(
            "Also plot the Figure 3(right) incremental-accuracy comparison "
            "(ParetoCL vs ParetoCL--, Seq-CIFAR100 online; requires "
            "scripts/run_paretocl_minus.sh logs)."
        ),
    )
    args = parser.parse_args()
    path_shared = Path(args.path_shared)

    os.makedirs(path_shared / "visualisation", exist_ok=True)

    if args.sweep:
        visualise_sweep(path_shared, args.seeds)

    if args.fig3_right:
        visualise_incremental_accuracy(path_shared, args.seeds)

    if not args.sweep and not args.fig3_right:
        get_aa_aaa(path_shared, args.seeds)
        get_taskwise_aa(path_shared, args.seeds)
        get_train_time(path_shared, args.seeds)
