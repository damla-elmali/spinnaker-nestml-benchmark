import csv
import statistics
import matplotlib.pyplot as plt


CSV_FILE = "/users/elmali/nestml/benchmark_balanced_networks_profiling.csv"


def load_data():
    with open(CSV_FILE, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        data = list(reader)

    print(f"Loaded {len(data)} rows.")
    print("Columns:")
    print(reader.fieldnames)

    return data


def analyze_runs(data):

    metrics = [
        "exc_firing_rate",
        "inh_firing_rate",
        "cv",
        "execution_time"
    ]

    groups = {}

    # Group by N, seed and implementation
    for row in data:

        key = (
            int(row["N"]),
            int(row["seed"]),
            row["implementation"],
            row["use_exp_luts"]
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(row)

    results = []

    for key, rows in groups.items():

        N, seed, implementation, use_exp_luts = key

        result = {
            "N": N,
            "seed": seed,
            "implementation": implementation,
            "use_exp_luts": use_exp_luts,
            "number_of_runs": len(rows)
        }

        for metric in metrics:

            values = [
                float(row[metric])
                for row in rows
            ]

            result[f"{metric}_mean"] = statistics.mean(values)

            if len(values) > 1:
                result[f"{metric}_std"] = statistics.stdev(values)
                result[f"{metric}_variance"] = statistics.variance(values)
            else:
                result[f"{metric}_std"] = 0.0
                result[f"{metric}_variance"] = 0.0

        results.append(result)

    return results


def save_run_results(results):

    filename = "benchmark_run_statistics.csv"

    fieldnames = [
        "N",
        "seed",
        "implementation",
        "use_exp_luts",
        "number_of_runs",

        "exc_firing_rate_mean",
        "exc_firing_rate_std",
        "exc_firing_rate_variance",

        "inh_firing_rate_mean",
        "inh_firing_rate_std",
        "inh_firing_rate_variance",

        "cv_mean",
        "cv_std",
        "cv_variance",

        "execution_time_mean",
        "execution_time_std",
        "execution_time_variance"
    ]

    with open(filename, "w", newline="") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    print(f"\nSaved: {filename}")


def print_results(results):

    print("\n=== RUN ANALYSIS ===\n")

    for result in results:

        print(
            f"N={result['N']} | "
            f"seed={result['seed']} | "
            f"{result['implementation']} | "
            f"runs={result['number_of_runs']}"
        )

        print(
            f"  Exc firing rate: "
            f"{result['exc_firing_rate_mean']:.4f} "
            f"+/- {result['exc_firing_rate_std']:.4f}"
        )

        print(
            f"  Inh firing rate: "
            f"{result['inh_firing_rate_mean']:.4f} "
            f"+/- {result['inh_firing_rate_std']:.4f}"
        )

        print(
            f"  CV: "
            f"{result['cv_mean']:.4f} "
            f"+/- {result['cv_std']:.4f}"
        )

        print(
            f"  Execution time: "
            f"{result['execution_time_mean']:.4f} "
            f"+/- {result['execution_time_std']:.4f}"
        )

        print()


def main():

    data = load_data()

    results = analyze_runs(data)

    print_results(results)

    save_run_results(results)


if __name__ == "__main__":
    main()