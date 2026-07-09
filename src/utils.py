import argparse
import time
from contextlib import contextmanager


@contextmanager
def timer(name: str):
    """Times a block and prints the elapsed duration on exit.

    Yields a dict that is populated with the elapsed seconds (key
    "seconds") once the block completes, so callers can persist the
    measurement (e.g. into a results JSON) rather than only see it printed.

    Usage:
        with timer("training") as t:
            ...
        elapsed = t["seconds"]
    """
    t0 = time.time()
    result = {}
    yield result
    result["seconds"] = time.time() - t0
    print(f"[{name}] done in {result['seconds']:.0f} s")


def generate_preferences(
    preference_dim: int, focus_weight: float = 0.5
) -> list[list[float]]:
    """Generate a list of preference vectors for inference.

    Returns 1 uniform vector plus `preference_dim` task-focused vectors,
    where the focused task receives `focus_weight` and the rest share the remainder equally.
    """
    N = preference_dim
    uniform = [1.0 / N] * N
    other_weight = (1.0 - focus_weight) / (N - 1)
    focused = []
    for i in range(N):
        pref = [other_weight] * N
        pref[i] = focus_weight
        focused.append(pref)
    return [uniform] + focused


def parse_args(DATASET_CONFIG):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="cifar100", choices=list(DATASET_CONFIG.keys())
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Epochs per task (offline=5, online=1)"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Online setting: 1 epoch per task (overrides --epochs)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--buffer_size", type=int, default=1000)
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Override number of tasks (default: dataset-specific)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use (default: 0). Ignored if no GPU is available.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="results",
        help="Directory to save results logs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: fewer epochs, tasks, and smaller buffer for quick runs",
    )

    return parser.parse_args()
