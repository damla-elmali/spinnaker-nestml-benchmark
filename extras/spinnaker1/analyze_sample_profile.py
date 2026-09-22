# -*- coding: utf-8 -*-
#
# analyze_profiling_data.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

"""
Analyze SpiNNaker sample profiling data.

Usage:
    python analyze_profiling_data.py
        Compare the two latest profiles.
        Older profile = built-in, newer profile = NESTML.

    python analyze_profiling_data.py <builtin.json> <nestml.json>
        Compare two explicitly selected profiles.



To use this script, make sure that 

.. code::

   [Reports]
   write_energy_report = True
   write_sample_profile_report = True

in ``.spynnaker.cfg``.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Enable grid lines globally for all plots
plt.rcParams['axes.grid'] = True

# Optional: Set grid aesthetic defaults (style, opacity, line width)
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['grid.linewidth'] = 0.8



REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"    
ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "reports_archive"
PROFILE_NAME = "sample_profile.json"
ENERGY_NAME = "energy_report.rpt"
BENCHMARK_CSV = Path(__file__).resolve().parents[2] / "benchmark_balanced_networks_results.csv"
PROFILE_CSV = Path(__file__).resolve().parents[2] / "benchmark_balanced_networks_profiling.csv"
ENERGY_CSV = Path(__file__).resolve().parents[2] / "benchmark_balanced_networks_energy.csv"


def find_latest_profiles():
    """Find the two newest sample profile files."""
    profiles = sorted(
        REPORT_DIR.rglob(PROFILE_NAME),
        key=lambda path: path.stat().st_mtime
    )

    if len(profiles) < 2:
        raise FileNotFoundError(
            f"At least two {PROFILE_NAME} files are required under {REPORT_DIR}"
        )

    return profiles[-2], profiles[-1]


def load_json(path):
    """Load a sample profile JSON file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_core_activity(data):
    """Extract core activity values from Andrew's JSON format."""
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
                "vertex": core_data.get("vertex"),
                "vertex_slice": core_data.get("vertex_slice"),
                "vertex_label": core_data.get("vertex_label"),
                "min_percent_active": core_data.get("min_percent_active"),
                "max_percent_active": core_data.get("max_percent_active"),
                "mean_percent_active": core_data.get("mean_percent_active")
            })

    return cores

def calculate_sum_core_activity(cores):
    """
    Calculate the sum of mean core activity.

    The values come directly from Andrew's
    sample_profile.json.
    """
    sum_core_activity = 0.0

    for core in cores:
        if core["mean_percent_active"] is None:
            raise ValueError(
                f"Core {core['chip']}:{core['core']} has no mean_percent_active value."
            )

        else:
            sum_core_activity += core["mean_percent_active"]

    return sum_core_activity


# def summarize_activity(cores):
#     """Calculate summary statistics for profiled cores."""
#     values = np.array([core["mean_percent_active"] for core in cores], dtype=float)

#     return {
#         "core_count": len(values),
#         "mean": np.mean(values),
#         "median": np.median(values),
#         "max": np.max(values)
#     }

def archive_run(profile_path, number_of_neurons, use_exp_luts):
    """Archive profiling and energy reports together with run metadata."""

    profile_path = Path(profile_path)
    energy_path = profile_path.parent / ENERGY_NAME

    if not profile_path.exists():
        raise FileNotFoundError(f"Profile file not found: {profile_path}")

    if not energy_path.exists():
        raise FileNotFoundError(f"Energy report not found: {energy_path}")

    timestamp = profile_path.parent.parent.name
    run_dir = ARCHIVE_DIR / f"{timestamp}_N{number_of_neurons}"

    run_dir.mkdir(parents=True, exist_ok=True)

    profile_destination = run_dir / PROFILE_NAME
    energy_destination = run_dir / ENERGY_NAME
    metadata_destination = run_dir / "metadata.json"

    profile_destination.write_bytes(profile_path.read_bytes())
    energy_destination.write_bytes(energy_path.read_bytes())

    metadata = {
        "run_id": run_dir.name,
        "N": number_of_neurons,
        "use_exp_luts": use_exp_luts,
        "source_report_directory": str(profile_path.parent),
        "files": {
            "profile": PROFILE_NAME,
            "energy": ENERGY_NAME
        }
    }

    with open(metadata_destination, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Run archived to: {run_dir}")

def load_benchmark_results():
    """Load benchmark results from the benchmark CSV."""

    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {BENCHMARK_CSV}"
        )

    with open(
        BENCHMARK_CSV,
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        return list(csv.DictReader(f))


def get_latest_neuron_count():
    """
    Get the number of neurons from the latest benchmark result.

    The benchmark CSV already stores N, so it is not extracted
    from sample_profile.json.
    """

    rows = load_benchmark_results()

    if not rows:
        raise ValueError(
            f"No benchmark results found in {BENCHMARK_CSV}"
        )

    latest_row = rows[-1]

    if "N" not in latest_row:
        raise KeyError(
            "Column 'N' was not found in benchmark CSV."
        )

    return int(latest_row["N"])

def get_latest_use_exp_luts_info():
    """
    Get the 'use_exp_luts' information from the latest benchmark result.

    The benchmark CSV already stores 'use_exp_luts', so it is not extracted
    from sample_profile.json.
    """

    rows = load_benchmark_results()

    if not rows:
        raise ValueError(
            f"No benchmark results found in {BENCHMARK_CSV}"
        )

    latest_row = rows[-1]

    if "use_exp_luts" not in latest_row:
        raise KeyError(
            "Column 'use_exp_luts' was not found in benchmark CSV."
        )

    return latest_row["use_exp_luts"].lower() == "true"


def analyze_profiles(builtin_path, nestml_path):
    """Analyze and compare reference and NESTML profiles."""
    builtin = load_json(builtin_path)
    nestml = load_json(nestml_path)

    builtin_cores = extract_core_activity(builtin)
    nestml_cores = extract_core_activity(nestml)

    builtin_sum = calculate_sum_core_activity(builtin_cores)
    nestml_sum = calculate_sum_core_activity(nestml_cores)

    use_exp_luts = get_latest_use_exp_luts_info()
    number_of_neurons = get_latest_neuron_count()

    archive_run(builtin_path, number_of_neurons, None)
    archive_run(nestml_path, number_of_neurons, use_exp_luts)


    # builtin_summary = summarize_activity(builtin_cores)
    # nestml_summary = summarize_activity(nestml_cores)

    # mean_difference = nestml_summary["mean"] - builtin_summary["mean"]
    # median_difference = nestml_summary["median"] - builtin_summary["median"]
    # max_difference = nestml_summary["max"] - builtin_summary["max"]

    # mean_relative_difference = mean_difference / builtin_summary["mean"] * 100 if builtin_summary["mean"] != 0 else np.nan

    print("\n===================================")
    print("SPINNAKER PROFILE COMPARISON")
    print("===================================")

    print(f"Built-in profile: {builtin_path}")
    print(f"NESTML profile:    {nestml_path}")

    print("\n-----------------------------------")
    print("Core activity")
    print("-----------------------------------")

    print(
    f"N={number_of_neurons} | "
    f"Built-in cores={len(builtin_cores)} | "
    f"NESTML cores={len(nestml_cores)} | "
    f"Built-in sum activity={builtin_sum:.4f} | "
    f"NESTML sum activity={nestml_sum:.4f} | "
    f"Difference={nestml_sum - builtin_sum:.4f} percentage-points"
)
    save_profiling_to_csv(number_of_neurons, use_exp_luts, builtin_sum, nestml_sum, builtin_path, nestml_path)

    # print(f"Reference profiled cores: {builtin_summary['core_count']}")
    # print(f"NESTML profiled cores:    {nestml_summary['core_count']}")

    # print(f"\nReference mean activity: {builtin_summary['mean']:.4f}%")
    # print(f"NESTML mean activity:    {nestml_summary['mean']:.4f}%")
    # print(f"Mean difference:          {mean_difference:.4f} percentage points")
    # print(f"Mean relative difference: {mean_relative_difference:.2f}%")

    # print(f"\nReference median activity: {builtin_summary['median']:.4f}%")
    # print(f"NESTML median activity:    {nestml_summary['median']:.4f}%")
    # print(f"Median difference:          {median_difference:.4f} percentage points")

    # print(f"\nReference maximum activity: {builtin_summary['max']:.4f}%")
    # print(f"NESTML maximum activity:    {nestml_summary['max']:.4f}%")
    # print(f"Maximum difference:          {max_difference:.4f} percentage points")

    # plot_core_activity(builtin_cores, builtin_path, "Reference")
    # plot_core_activity(nestml_cores, nestml_path, "NESTML")
    # plot_comparison(builtin_summary, nestml_summary, nestml_path)
    # plot_distribution(builtin_cores, nestml_cores, nestml_path)
    # save_comparison_to_csv(builtin_path, nestml_path, builtin_summary, nestml_summary, mean_difference, mean_relative_difference)




def save_profiling_to_csv(number_of_neurons, use_exp_luts, builtin_sum, nestml_sum, builtin_path, nestml_path):
    """Save profiling summary data for later plotting."""
    rows = [
        {
            "profile": str(builtin_path),
            "implementation": "builtin",
            "use_exp_luts": use_exp_luts,
            "N": number_of_neurons,
            "sum_core_activity_percent": builtin_sum
        },
        {
            "profile": str(nestml_path),
            "implementation": "nestml",
            "use_exp_luts": use_exp_luts,
            "N": number_of_neurons,
            "sum_core_activity_percent": nestml_sum
        }
    ]

    fieldnames = ["profile", "implementation", "use_exp_luts", "N", "sum_core_activity_percent"]
    file_exists = PROFILE_CSV.exists()

    with open(PROFILE_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print(f"Profiling CSV saved to: {PROFILE_CSV}")


def plot_core_activity(cores, profile_path, implementation):
    if not cores:
        print(f"No core profiling data found for {implementation}.")
        return

    labels = [f"{core['chip']}:{core['core']}" for core in cores]
    values = [core["mean_percent_active"] for core in cores]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xlabel("Core")
    plt.ylabel("Mean active time [%]")
    plt.title(f"{implementation} SpiNNaker Core Activity")
    plt.xticks(rotation=90)
    plt.tight_layout()

    output_path = Path(profile_path).parent / f"sample_profile_activity_{implementation.lower()}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"{implementation} core activity plot saved to: {output_path}")

def plot_comparison(reference, nestml, profile_path):
    """Plot summary activity metrics for reference and NESTML."""
    labels = ["Mean", "Median", "Maximum"]
    reference_values = [reference["mean"], reference["median"], reference["max"]]
    nestml_values = [nestml["mean"], nestml["median"], nestml["max"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))
    plt.bar(x - width / 2, reference_values, width, label="Reference")
    plt.bar(x + width / 2, nestml_values, width, label="NESTML")
    plt.xlabel("Core activity metric")
    plt.ylabel("Active time [%]")
    plt.title("Reference vs NESTML Core Activity")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()

    output_path = Path(profile_path).parent / "sample_profile_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


# def plot_distribution(reference_cores, nestml_cores, profile_path):
#     """Plot the distribution of mean core activity."""
#     reference_values = [core["mean_percent_active"] for core in reference_cores]
#     nestml_values = [core["mean_percent_active"] for core in nestml_cores]

#     plt.figure(figsize=(8, 6))
#     plt.boxplot([reference_values, nestml_values], tick_labels=["Reference", "NESTML"])
#     plt.ylabel("Mean active time [%]")
#     plt.title("Distribution of Core Activity")
#     plt.tight_layout()

#     output_path = Path(profile_path).parent / "sample_profile_activity_distribution.png"
#     plt.savefig(output_path, dpi=300)
#     plt.close()

#     print(f"Distribution plot saved to: {output_path}")


# def save_comparison_to_csv(builtin_path, nestml_path, builtin, nestml, mean_difference, mean_relative_difference):
#     """Save the built-in vs NESTML profiling comparison."""
#     csv_path = Path(__file__).resolve().parents[2] / "benchmark_balanced_networks_profiling.csv"

#     row = {
#         "builtin_profile": str(builtin_path),
#         "nestml_profile": str(nestml_path),
#         "builtin_core_count": builtin["core_count"],
#         "nestml_core_count": nestml["core_count"],
#         "builtin_mean_activity_percent": builtin["mean"],
#         "nestml_mean_activity_percent": nestml["mean"],
#         "mean_difference_percentage_points": mean_difference,
#         "mean_relative_difference_percent": mean_relative_difference,
#         "builtin_median_activity_percent": builtin["median"],
#         "nestml_median_activity_percent": nestml["median"],
#         "builtin_max_activity_percent": builtin["max"],
#         "nestml_max_activity_percent": nestml["max"]
#     }

#     file_exists = csv_path.exists()

#     with open(csv_path, "a", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(f, fieldnames=row.keys())

#         if not file_exists:
#             writer.writeheader()

#         writer.writerow(row)

#     print(f"CSV saved to: {csv_path}")


# Energy analysis
ENERGY_FIELDS = {
    "Simulation execution time": "execution_time_s",
    "Simulation execution energy": "execution_energy_J",
    "Simulation execution energy (active chips and cores only)": "execution_energy_active_only_J",
    "Simulation execution energy (ignoring frame power)": "execution_energy_ignoring_frame_J",
    "Mapping time": "mapping_time_s",
    "Mapping energy": "mapping_energy_J",
    "Data Spec time": "data_spec_time_s",
    "Data Spec energy": "data_spec_energy_J",
    "Saving time": "saving_time_s",
    "Saving energy": "saving_energy_J",
    "Other time": "other_time_s",
    "Other energy": "other_energy_J",
    "Total energy": "total_energy_J"
}


def parse_energy_value(value):
    """Convert the numeric part of an energy-report value to float."""
    return float(value.split()[0])


def load_energy_report(profile_path):
    """Parse the energy report belonging to a profile."""
    energy_path = Path(profile_path).parent / ENERGY_NAME

    if not energy_path.exists():
        raise FileNotFoundError(f"Energy report not found: {energy_path}")

    energy = {}

    with open(energy_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line.startswith("Simulation cores used:"):
                parts = line.split()
                energy["cores_used"] = int(parts[3])
                energy["active_cores"] = int(parts[6])
                continue

            for report_field, key in ENERGY_FIELDS.items():
                if line.startswith(f"{report_field}:"):
                    value = line.split(":", 1)[1].strip()
                    energy[key] = parse_energy_value(value)
                    break

    return energy

def analyze_energy(builtin_path, nestml_path):
    """Analyze and compare built-in and NESTML energy reports."""
    builtin = load_energy_report(builtin_path)
    nestml = load_energy_report(nestml_path)

    print_energy_comparison(builtin, nestml)
    save_energy_comparison_to_csv(builtin_path, nestml_path, builtin, nestml)



def print_energy_comparison(builtin, nestml):
    """Print Reference vs NESTML energy results."""
    execution_difference = (
        nestml["execution_energy_J"] -
        builtin["execution_energy_J"]
    )

    execution_relative_difference = (
        abs(execution_difference) /
        builtin["execution_energy_J"] * 100
        if builtin["execution_energy_J"] != 0 else np.nan
    )

    print("\n===================================")
    print("ENERGY COMPARISON")
    print("===================================")

    print(f"Built-in cores used: {builtin['cores_used']}")
    print(f"NESTML cores used:    {nestml['cores_used']}")

    print(f"Built-in active cores: {builtin['active_cores']}")
    print(f"NESTML active cores:    {nestml['active_cores']}")

    print(f"\nBuilt-in execution time: {builtin['execution_time_s']:.6f} s")
    print(f"NESTML execution time:    {nestml['execution_time_s']:.6f} s")

    print(f"\nBuilt-in execution energy: {builtin['execution_energy_J']:.6f} J")
    print(f"NESTML execution energy:    {nestml['execution_energy_J']:.6f} J")
    print(f"Execution energy difference: {execution_difference:.6f} J")
    print(f"Execution energy relative difference: {execution_relative_difference:.2f}%")

    print(
        f"\nBuilt-in active-only execution energy: "
        f"{builtin['execution_energy_active_only_J']:.6f} J"
    )

    print(
        f"NESTML active-only execution energy:    "
        f"{nestml['execution_energy_active_only_J']:.6f} J"
    )

    print(f"\nBuilt-in total energy: {builtin['total_energy_J']:.6f} J")
    print(f"NESTML total energy:    {nestml['total_energy_J']:.6f} J")


def save_energy_comparison_to_csv(
    builtin_path,
    nestml_path,
    builtin,
    nestml
):
    """Save the Reference vs NESTML energy comparison."""
    execution_difference = (
        nestml["execution_energy_J"] -
        builtin["execution_energy_J"]
    )

    execution_relative_difference = (
        abs(execution_difference) /
        builtin["execution_energy_J"] * 100
        if builtin["execution_energy_J"] != 0 else np.nan
    )

    row = {
        "builtin_profile": str(builtin_path),
        "nestml_profile": str(nestml_path),
        "builtin_cores_used": builtin["cores_used"],
        "nestml_cores_used": nestml["cores_used"],
        "builtin_active_cores": builtin["active_cores"],
        "nestml_active_cores": nestml["active_cores"],
        "builtin_execution_time_s": builtin["execution_time_s"],
        "nestml_execution_time_s": nestml["execution_time_s"],
        "builtin_execution_energy_J": builtin["execution_energy_J"],
        "nestml_execution_energy_J": nestml["execution_energy_J"],
        "execution_energy_difference_J": execution_difference,
        "execution_energy_relative_difference_percent": execution_relative_difference,
        "builtin_active_only_energy_J": builtin["execution_energy_active_only_J"],
        "nestml_active_only_energy_J": nestml["execution_energy_active_only_J"],
        "builtin_total_energy_J": builtin["total_energy_J"],
        "nestml_total_energy_J": nestml["total_energy_J"]
    }

    file_exists = ENERGY_CSV.exists()

    with open(ENERGY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

    print(f"Energy CSV saved to: {ENERGY_CSV}")


def analyze_latest_profiles():
    """Analyze the two latest profiles: built-in first, NESTML second."""
    builtin_path, nestml_path = find_latest_profiles()

    analyze_profiles(builtin_path, nestml_path)
    analyze_energy(builtin_path, nestml_path)


def main():
    if len(sys.argv) == 1:
        analyze_latest_profiles()
    elif len(sys.argv) == 3:
        analyze_profiles(sys.argv[1], sys.argv[2])
        analyze_energy(sys.argv[1], sys.argv[2])
    else:
        raise SystemExit(
            "Usage:\n"
            "  python analyze_profiling_data.py\n"
            "  python analyze_profiling_data.py <builtin.json> <nestml.json>"
        )


if __name__ == "__main__":
    main()
