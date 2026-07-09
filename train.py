import json
import os

import pytorch_lightning as pl
import torch
import wandb

from src.data.continual_data import ContinualDataModule
from src.pl_modules.pareto_cl import ParetoCL
from src.utils import parse_args, timer

# Map dataset names to backbone names
BACKBONE_MAP = {
    "cifar100": "resnet18",
    "cifar10": "resnet18",
    "tinyimagenet": "resnet18_tiny",
}

# Dataset → (num_tasks, classes_per_task)
DATASET_CONFIG = {"cifar10": (5, 2), "cifar100": (10, 10), "tinyimagenet": (10, 20)}


def update_buffer(buffer, dataset, task_id):
    """Rebalance the replay buffer after each task.

    Keeps approximately buffer.buffer_size // n_tasks_so_far samples per task.
    """
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=True
    )
    images, targets = next(iter(data_loader))

    total_buffer_size = buffer.buffer_size
    target_count = total_buffer_size // (task_id + 1)

    # Retrieve valid data currently in buffer
    old_images = buffer.images[: buffer.current_size]
    old_targets = buffer.targets[: buffer.current_size]
    old_task_ids = buffer.task_ids[: buffer.current_size]

    new_buffer_images = []
    new_buffer_targets = []
    new_buffer_task_ids = []

    # Downsample old tasks to target_count each
    for t in range(task_id):
        mask = old_task_ids == t
        t_imgs = old_images[mask]
        t_targs = old_targets[mask]
        if len(t_imgs) > target_count:
            t_imgs = t_imgs[:target_count]
            t_targs = t_targs[:target_count]
        new_buffer_images.append(t_imgs)
        new_buffer_targets.append(t_targs)
        new_buffer_task_ids.append(torch.full((len(t_imgs),), t, dtype=torch.long))

    # Add target_count samples from the new task
    n_new = min(len(images), target_count)
    new_buffer_images.append(images[:n_new])
    new_buffer_targets.append(targets[:n_new])
    new_buffer_task_ids.append(torch.full((n_new,), task_id, dtype=torch.long))

    all_images = torch.cat(new_buffer_images)
    all_targets = torch.cat(new_buffer_targets)
    all_task_ids = torch.cat(new_buffer_task_ids)

    n = len(all_images)
    buffer.images[:n] = all_images
    buffer.targets[:n] = all_targets
    buffer.task_ids[:n] = all_task_ids
    buffer.current_size = n

    print(
        f"Buffer updated: size={n}, tasks={torch.unique(buffer.task_ids[:n]).tolist()}"
    )


def main():
    args = parse_args(DATASET_CONFIG)
    pl.seed_everything(args.seed, workers=True)

    if args.online:
        args.epochs = 1

    default_tasks, _ = DATASET_CONFIG[args.dataset]
    num_tasks = args.num_tasks if args.num_tasks is not None else default_tasks

    if args.debug:
        args.epochs = 1
        num_tasks = 2
        args.buffer_size = 10

    # Data
    dm = ContinualDataModule(args.dataset, args.batch_size, num_tasks, args.buffer_size)
    dm.prepare_data()
    dm.setup()

    backbone = BACKBONE_MAP[args.dataset]

    # Model: start with only the first task's classes; head grows each task.
    # The preference vector is always 2-dim: α = (α_stability, α_plasticity).
    model = ParetoCL(
        dm.classes_per_task,
        backbone,
        args.lr,
        dm.buffer,
        preference_dim=2,
    )

    setting = "online" if args.online else "offline"
    os.makedirs(f"{args.log_dir}/{args.dataset}/{setting}", exist_ok=True)
    base_path = f"{args.log_dir}/{args.dataset}/{setting}/seed{args.seed}"
    if os.path.exists(f"{base_path}_model.pt"):
        print(
            f"Model checkpoint already exists at {base_path}_model.pt, skipping training."
        )
        return

    run = wandb.init(
        project="paretocl_reproduction",
        name=f"{args.dataset}_{setting}_{args.log_dir.split('/')[-1]}_seed{args.seed}",
        config={
            "dataset": args.dataset,
            "setting": setting,
            "seed": args.seed,
            "num_tasks": num_tasks,
            "buffer_size": args.buffer_size,
            "lr": args.lr,
        },
    )
    wandb_logger = pl.loggers.WandbLogger(experiment=run)

    print(f"=== ParetoCL | {args.dataset} | {setting} | seed={args.seed} ===")
    print(
        f"    tasks={num_tasks} | epochs/task={args.epochs} | buffer={args.buffer_size} | lr={args.lr}"
    )

    # AAA tracking: aa_after[j] = avg accuracy over tasks 0..j right after training task j
    aa_after = []
    taskwise_accuracies = {}

    # Training Loop
    # Table 3: total wall-clock time for the whole run (training + buffer
    # rebalancing + per-task validation), matching the paper's training-time
    # comparison protocol.
    with timer(
        f"total_training | {args.dataset} | {setting} | seed={args.seed}"
    ) as train_timer:
        for task_id in range(num_tasks):
            print(f"\n=== Task {task_id + 1}/{num_tasks} ===")
            dm.set_task(task_id)
            model.current_task_id = task_id

            # Expand the output head before training on the new task so that the
            # model can predict all classes seen so far (class-incremental setting).
            if task_id > 0:
                model.expand_head((task_id + 1) * dm.classes_per_task)

            model.epoch_offset = (
                task_id * args.epochs
            )  # = total epochs from previous tasks

            trainer = pl.Trainer(
                max_epochs=args.epochs,
                accelerator="auto",
                devices=[args.gpu_id] if torch.cuda.is_available() else None,
                enable_checkpointing=False,
                logger=wandb_logger,
                enable_progress_bar=True,
                gradient_clip_val=1.0,
            )

            trainer.fit(model, datamodule=dm)

            # Update buffer with samples from the just-trained task
            update_buffer(dm.buffer, dm.train_dataset, task_id)

            # Validate on all seen tasks
            val_results = trainer.validate(model, datamodule=dm, verbose=False)

            # Flatten the list of result dicts
            flat_results = {}
            if isinstance(val_results, list):
                for res in val_results:
                    flat_results.update(res)
            else:
                flat_results = val_results

            # Collect per-task accuracies
            accuracies = []
            for i in range(task_id + 1):
                key = f"val_acc_task_{i}"
                if key in flat_results:
                    accuracies.append(flat_results[key])

            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                aa_after.append(avg_acc)
                taskwise_accuracies[task_id] = accuracies
                print(
                    f"After Task {task_id + 1}: AA={avg_acc:.4f} "
                    f"(tasks: {[f'{a:.4f}' for a in accuracies]})"
                )
            else:
                print("Could not compute average accuracy.")

            # Per-task checkpoint, needed to reproduce Figure 3 (Pareto front at each stage).
            torch.save(
                model.model.state_dict(), f"{base_path}_model_aftertask{task_id + 1}.pt"
            )

    # Final metrics
    if aa_after:
        # AAA = average of AA_j across all tasks
        aaa = sum(aa_after) / len(aa_after)
        # Final Acc = AA after the last task
        final_acc = aa_after[-1]
        print(f"\n{'=' * 50}")
        print(f"RESULTS | {args.dataset} | {setting} | seed={args.seed}")
        print(f"  AAA        = {aaa:.4f}  ({aaa * 100:.2f}%)")
        print(f"  Final Acc  = {final_acc:.4f}  ({final_acc * 100:.2f}%)")
        print(f"{'=' * 50}")

    log_data = {
        "dataset": args.dataset,
        "setting": setting,
        "seed": args.seed,
        "taskwise_accuracies": taskwise_accuracies,
        "aa_after": aa_after,
        "aaa": aaa if aa_after else None,
        "final_acc": final_acc if aa_after else None,
        "train_time_sec": train_timer["seconds"],
    }

    with open(f"{base_path}.json", "w") as f:
        json.dump(log_data, f, indent=4)

    torch.save(model.model.state_dict(), f"{base_path}_model.pt")
    model_config = {
        "backbone_name": backbone,
        "num_classes": model.model.num_classes,
        "preference_dim": model.model.preference_dim,
        "hidden_dim": model.model.hidden_dim,
        "dataset": args.dataset,
        "num_tasks": num_tasks,
    }
    with open(f"{base_path}_model_config.json", "w") as f:
        json.dump(model_config, f, indent=4)
    print(f"Model saved to {base_path}_model.pt")


if __name__ == "__main__":
    main()
