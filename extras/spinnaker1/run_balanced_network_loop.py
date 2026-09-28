import os
import subprocess
from datetime import datetime


NESTML_ROOT = "/users/elmali/nestml"

TEST_FILE = os.path.join(
    NESTML_ROOT,
    "tests",
    "spinnaker_tests",
    "test_spinnaker_balanced_network.py"
)

N_NEURONS_VALUES = [256]
SEED_VALUES = [200]
N_RUNS = 1


def prepare_spinnaker_directories():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

    target_dir = os.path.join(
        NESTML_ROOT,
        "spinnaker-target"
    )

    install_dir = os.path.join(
        NESTML_ROOT,
        "spinnaker-install"
    )

    if os.path.exists(target_dir):
        archived_target = f"{target_dir}-{timestamp}"
        os.rename(target_dir, archived_target)
        print(
            f"Archived: {target_dir} -> {archived_target}"
        )

    if os.path.exists(install_dir):
        archived_install = f"{install_dir}-{timestamp}"
        os.rename(install_dir, archived_install)
        print(
            f"Archived: {install_dir} -> {archived_install}"
        )

    os.makedirs(install_dir, exist_ok=True)


def run_test(mode, n_neurons, seed):
    if mode == "comparison":
        test_name = "test_spinnaker_balanced_network"
        use_exp_luts = "true"

    elif mode == "nestml":
        test_name = "test_nestml"
        use_exp_luts = "false"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("\n===================================")
    print(
        f"{mode.upper()} | "
        f"N={n_neurons} | "
        f"SEED={seed} | "
        f"{N_RUNS} RUNS"
    )
    print("===================================")

    failed_runs = []

    for run_number in range(1, N_RUNS + 1):

        print(
            f"\n========== "
            f"{mode.upper()} | "
            f"N={n_neurons} | "
            f"SEED={seed} | "
            f"RUN {run_number}/{N_RUNS} "
            f"==========\n"
        )

        prepare_spinnaker_directories()

        env = os.environ.copy()

        env["USE_EXP_LUTS"] = use_exp_luts
        env["N_NEURONS"] = str(n_neurons)
        env["SEED"] = str(seed)
        env["BENCHMARK_RUN"] = str(run_number)

        command = [
            "pytest",
            "-s",
            "-o", "log_cli=true",
            "-o", "log_cli_level=DEBUG",
            f"{TEST_FILE}::"
            f"TestSpiNNakerBalancedNetwork::"
            f"{test_name}",
        ]

        result = subprocess.run(
            command,
            env=env
        )

        if result.returncode != 0:

            failed_runs.append(run_number)

            print(
                f"\nWARNING: "
                f"{mode.upper()} | "
                f"N={n_neurons} | "
                f"SEED={seed} | "
                f"RUN {run_number}/{N_RUNS} "
                f"FAILED."
            )

            print(
                "Continuing with the next run..."
            )

    print(
        f"\n{mode.upper()} | "
        f"N={n_neurons} | "
        f"SEED={seed} "
        f"FINISHED."
    )

    if failed_runs:
        print(
            f"Failed runs: {failed_runs}"
        )
    else:
        print("All runs passed.")


def main():

    for n_neurons in N_NEURONS_VALUES:

        for seed in SEED_VALUES:

            run_test(
                "comparison",
                n_neurons,
                seed
            )

            run_test(
                "nestml",
                n_neurons,
                seed
            )

    print("\n===================================")
    print("ALL BENCHMARK RUNS COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()