import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]
PROFILE_CSV = BASE_DIR / "benchmark_balanced_networks_profiling.csv"

IMPLEMENTATION_LABELS = {
    "builtin": "Built-in",
    "nestml_exp_lut": "NESTML + exp LUT",
    "nestml_not_exp_lut": "NESTML + no exp LUT"
}


def load_profiling_data():
    """Load profiling data from CSV."""
    if not PROFILE_CSV.exists():
        raise FileNotFoundError(f"Profiling CSV not found: {PROFILE_CSV}")

    with open(PROFILE_CSV, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_profile(path):
    """Load a sample profile JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_experiment_label(row):
    """Determine the experiment label from implementation and LUT setting."""

    implementation = row.get("implementation")
    use_exp_luts = row.get("use_exp_luts", "").lower() == "true"

    if implementation == "builtin":
        return "builtin"

    if implementation == "nestml":
        if use_exp_luts:
            return "nestml_exp_lut"

        return "nestml_not_exp_lut"

    return None


def extract_core_activity(data):
    """Extract core activity values from sample_profile.json."""
    cores = []

    for chip_name, chip_data in data.items():
        if not chip_name.startswith("chip_"):
            continue

        for core_name, core_data in chip_data.items():
            if not core_name.startswith("core_"):
                continue

            if core_data.get("mean_percent_active") is None:
                continue

            cores.append({
                "chip": chip_name,
                "core": core_name,
                "mean_percent_active": core_data["mean_percent_active"]
            })

    return cores


def plot_sum_core_activity(rows):
    """Plot sum of core activity against number of neurons."""

    data = {}

    for row in rows:
        implementation = get_experiment_label(row)

        if implementation not in IMPLEMENTATION_LABELS:
            continue

        try:
            neurons = int(row["N"])
            activity = float(row["sum_core_activity_percent"])
        except (ValueError, TypeError, KeyError):
            continue

        data.setdefault(implementation, []).append((neurons, activity))

    if not data:
        print("No profiling data available for plotting.")
        return

    plt.figure(figsize=(9, 6))

    for implementation, values in data.items():
        values.sort(key=lambda x: x[0])

        neurons = [x[0] for x in values]
        activity = [x[1] for x in values]

        plt.plot(
            neurons,
            activity,
            marker="o",
            label=IMPLEMENTATION_LABELS[implementation]
        )

    plt.xlabel("Number of neurons")
    plt.ylabel("Sum core activity [percentage-points]")
    plt.title("SpiNNaker Core Activity vs Number of Neurons")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = BASE_DIR / "sample_profile_sum_core_activity.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Sum core activity plot saved to: {output_path}")

def plot_core_activity_by_neurons(rows):
    """Plot core activity for different neuron counts."""

    experiments = {}

    for row in rows:
        implementation = get_experiment_label(row)

        if implementation not in IMPLEMENTATION_LABELS:
            continue

        timestamp = row.get("timestamp")

        try:
            neurons = int(row["N"])
        except (ValueError, TypeError, KeyError):
            continue

        profile_path = (
            BASE_DIR
            / "reports_archive"
            / timestamp
            / "sample_profile.json"
        )

        if not profile_path.exists():
            print(f"Profile file not found: {profile_path}")
            continue

        data = load_profile(profile_path)
        cores = extract_core_activity(data)

        experiments.setdefault(implementation, []).append(
            (neurons, cores)
        )

    for implementation, runs in experiments.items():

        core_data = {}

        for neurons, cores in runs:
            for core in cores:
                core_name = f"{core['chip']}:{core['core']}"

                core_data.setdefault(core_name, []).append(
                    (neurons, core["mean_percent_active"])
                )

        if not core_data:
            continue

        core_names = sorted(core_data.keys())

        plt.figure(figsize=(14, 7))

        for x, core_name in enumerate(core_names):

            values = sorted(
                core_data[core_name],
                key=lambda value: value[0]
            )

            for neurons, activity in values:
                plt.scatter(
                    x,
                    activity,
                    s=45
                )

                plt.annotate(
                    f"N={neurons}",
                    (x, activity),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8
                )

        plt.xticks(
            range(len(core_names)),
            core_names,
            rotation=90
        )

        plt.xlabel("Core")
        plt.ylabel("Mean active time [%]")
        plt.title(
            f"{IMPLEMENTATION_LABELS[implementation]} "
            "Core Activity vs Number of Neurons"
        )

        plt.tight_layout()

        output_path = (
            BASE_DIR
            / f"sample_profile_core_activity_vs_N_{implementation}.png"
        )

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Core activity vs N plot saved to: {output_path}")

def main():
    rows = load_profiling_data()

    plot_sum_core_activity(rows)
    plot_core_activity_by_neurons(rows)


if __name__ == "__main__":
    main()