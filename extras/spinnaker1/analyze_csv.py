import pandas as pd
import os


BASE_DIR = "/users/elmali/nestml"

RESULTS_CSV = os.path.join(
    BASE_DIR,
    "benchmark_balanced_networks_results.csv"
)

PROFILING_CSV = os.path.join(
    BASE_DIR,
    "benchmark_balanced_networks_profiling.csv"
)

ENERGY_CSV = os.path.join(
    BASE_DIR,
    "benchmark_balanced_networks_energy.csv"
)


CONFIG_COLUMNS = [
    "implementation",
    "use_static_synapses",
    "use_exp_luts",
    "N",
    "g",
    "p_conn",
    "rate_ext_input",
    "neurons_per_core",
    "poisson_generators_per_core",
    "t_sim",
]


def get_statistics(values):

    return {
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "variance": values.var(),
        "min": values.min(),
        "max": values.max(),
    }


def analyze_results():

    df = pd.read_csv(RESULTS_CSV)

    rows_per_seed = []
    rows_per_N = []

    # N + seed + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS + ["seed"],
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS + ["seed"],
                group_values
            )
        )

        runs = group["execution_time"]

        row["number_of_runs"] = len(runs)

        stats = get_statistics(runs)

        for name, value in stats.items():
            row[f"execution_time_{name}"] = value

        rows_per_seed.append(row)

    # N + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS,
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS,
                group_values
            )
        )

        runs = group["execution_time"]

        row["number_of_runs"] = len(runs)

        stats = get_statistics(runs)

        for name, value in stats.items():
            row[f"execution_time_{name}"] = value

        rows_per_N.append(row)

    pd.DataFrame(rows_per_seed).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_results_per_seed.csv"
        ),
        index=False
    )

    pd.DataFrame(rows_per_N).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_results_per_N.csv"
        ),
        index=False
    )


def analyze_profiling():

    df = pd.read_csv(PROFILING_CSV)

    metrics = [
        "mean_core_activity_percent",
        "sum_core_activity_percent",
    ]

    rows_per_seed = []
    rows_per_N = []

    # N + seed + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS + ["seed"],
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS + ["seed"],
                group_values
            )
        )

        row["number_of_runs"] = len(group)

        for metric in metrics:

            stats = get_statistics(
                group[metric]
            )

            for name, value in stats.items():
                row[f"{metric}_{name}"] = value

        rows_per_seed.append(row)

    # N + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS,
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS,
                group_values
            )
        )

        row["number_of_runs"] = len(group)

        for metric in metrics:

            stats = get_statistics(
                group[metric]
            )

            for name, value in stats.items():
                row[f"{metric}_{name}"] = value

        rows_per_N.append(row)

    pd.DataFrame(rows_per_seed).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_profiling_per_seed.csv"
        ),
        index=False
    )

    pd.DataFrame(rows_per_N).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_profiling_per_N.csv"
        ),
        index=False
    )


def analyze_energy():

    df = pd.read_csv(ENERGY_CSV)

    metrics = [
        "execution_energy_J",
        "execution_energy_active_only_J",
        "execution_energy_ignoring_frame_J",
        "mapping_time_s",
        "mapping_energy_J",
        "data_spec_time_s",
        "data_spec_energy_J",
        "saving_time_s",
        "saving_energy_J",
        "other_time_s",
        "other_energy_J",
        "total_energy_J",
    ]

    rows_per_seed = []
    rows_per_N = []

    # N + seed + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS + ["seed"],
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS + ["seed"],
                group_values
            )
        )

        row["cores_used"] = group["cores_used"].iloc[0]
        row["active_cores"] = group["active_cores"].iloc[0]

        row["number_of_runs"] = len(group)

        for metric in metrics:

            stats = get_statistics(
                group[metric]
            )

            for name, value in stats.items():
                row[f"{metric}_{name}"] = value

        rows_per_seed.append(row)

    # N + configuration
    for group_values, group in df.groupby(
        CONFIG_COLUMNS,
        dropna=False
    ):

        row = dict(
            zip(
                CONFIG_COLUMNS,
                group_values
            )
        )

        row["cores_used"] = group["cores_used"].iloc[0]
        row["active_cores"] = group["active_cores"].iloc[0]

        row["number_of_runs"] = len(group)

        for metric in metrics:

            stats = get_statistics(
                group[metric]
            )

            for name, value in stats.items():
                row[f"{metric}_{name}"] = value

        rows_per_N.append(row)

    pd.DataFrame(rows_per_seed).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_energy_per_seed.csv"
        ),
        index=False
    )

    pd.DataFrame(rows_per_N).to_csv(
        os.path.join(
            BASE_DIR,
            "benchmark_energy_per_N.csv"
        ),
        index=False
    )


def main():

    analyze_results()
    analyze_profiling()
    analyze_energy()


if __name__ == "__main__":
    main()