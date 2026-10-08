import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

TIME_STEP = 1.0
TIME_CONSTANT = 20.0
SHIFT = 0

# LUT strategies that we want to compare
LUT_STRATEGIES = ["floor", "round", "ceil"]

# Fractional precision that we want to test
FRACTIONAL_BITS = list(range(4, 17))

# Storage size of one LUT value
STORAGE_BITS = [8, 16, 32]

# Built-in sPyNNaker reference:
# floor + 11 fractional bits + 16-bit storage
BUILTIN_FRACTIONAL_BITS = 11
BUILTIN_STORAGE_BITS = 16
BUILTIN_STRATEGY = "floor"

OUTPUT_DIR = Path("/users/elmali/nestml/extras/lut_benchmark")

CSV_FILE = OUTPUT_DIR / "lut_strategy_benchmark.csv"


# ============================================================
# LUT size
# ============================================================

def calculate_lut_size(
    time_step,
    time_constant,
    shift,
    fractional_bits
):

    fixed_point_one = 1 << fractional_bits

    lambda_value = time_step / float(time_constant)

    size = math.log(fixed_point_one) / lambda_value

    size, extra = divmod(size / (1 << shift), 2)

    size = (int(size) + (extra > 0)) * 2

    return size


def generate_lut(
    time_step,
    time_constant,
    shift,
    fractional_bits,
    storage_bits,
    strategy
):

    # A value scaled with 2^fractional_bits must fit
    # into the selected storage type.
    max_value = (1 << storage_bits) - 1
    fixed_point_one = 1 << fractional_bits

    if fixed_point_one > max_value:
        return None

    # Calculate LUT length
    size = calculate_lut_size(
        time_step,
        time_constant,
        shift,
        fractional_bits
    )

    lambda_value = time_step / float(time_constant)

    indices = np.arange(size)

    exact_values = np.exp((indices << shift) * -lambda_value)

    lut_scaled_values = exact_values * fixed_point_one

    if strategy == "floor":
        lut = np.floor(lut_scaled_values)

    elif strategy == "round":
        lut = np.round(lut_scaled_values)

    elif strategy == "ceil":
        lut = np.ceil(lut_scaled_values)

    # Make sure values fit into selected storage
    lut = np.clip(lut, 0, max_value)

    if storage_bits == 8:
        lut = lut.astype(np.uint8)

    elif storage_bits == 16:
        lut = lut.astype(np.uint16)

    elif storage_bits == 32:
        lut = lut.astype(np.uint32)

    # --------------------------------------------------------
    # Convert LUT values back to floating point
    # --------------------------------------------------------

    lut_values = lut.astype(np.float64) / fixed_point_one

    return exact_values, lut, lut_values


# ============================================================
# Error and memory calculations
# ============================================================

def calculate_metrics(
    exact_values,
    lut,
    lut_values,
    fractional_bits,
    storage_bits
):
    """
    Calculate approximation error and memory usage.
    """

    error = exact_values - lut_values
    absolute_error = np.abs(error)

    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(absolute_error)

    max_absolute_error = np.max(absolute_error)

    # Avoid division by zero for relative error
    epsilon = 1e-15

    relative_error = (
        absolute_error /
        np.maximum(np.abs(exact_values), epsilon)
    )

    mean_relative_error = np.mean(relative_error)
    max_relative_error = np.max(relative_error)

    # LUT payload memory
    memory_bytes = len(lut) * (storage_bits // 8)

    return {
        "fractional_bits": fractional_bits,
        "storage_bits": storage_bits,
        "lut_length": len(lut),

        "mse": mse,
        "rmse": rmse,
        "mae": mae,

        "max_absolute_error": max_absolute_error,
        "mean_relative_error": mean_relative_error,
        "max_relative_error": max_relative_error,

        "memory_bytes": memory_bytes,
        "memory_kb": memory_bytes / 1024.0,
    }


# ============================================================
# CSV
# ============================================================

def save_results(results):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "timestamp",
        "time_step",
        "time_constant",
        "shift",

        "strategy",
        "fractional_bits",
        "storage_bits",

        "lut_length",

        "mse",
        "rmse",
        "mae",

        "max_absolute_error",
        "mean_relative_error",
        "max_relative_error",

        "memory_bytes",
        "memory_kb",

        "is_builtin_reference",
    ]

    file_exists = CSV_FILE.exists()

    with open(
        CSV_FILE,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        for result in results:

            is_builtin = (
                result["strategy"] == BUILTIN_STRATEGY
                and result["fractional_bits"]
                == BUILTIN_FRACTIONAL_BITS
                and result["storage_bits"]
                == BUILTIN_STORAGE_BITS
            )

            writer.writerow({
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),

                "time_step": TIME_STEP,
                "time_constant": TIME_CONSTANT,
                "shift": SHIFT,

                "strategy": result["strategy"],
                "fractional_bits":
                    result["fractional_bits"],
                "storage_bits":
                    result["storage_bits"],

                "lut_length":
                    result["lut_length"],

                "mse":
                    result["mse"],
                "rmse":
                    result["rmse"],
                "mae":
                    result["mae"],

                "max_absolute_error":
                    result["max_absolute_error"],

                "mean_relative_error":
                    result["mean_relative_error"],

                "max_relative_error":
                    result["max_relative_error"],

                "memory_bytes":
                    result["memory_bytes"],

                "memory_kb":
                    result["memory_kb"],

                "is_builtin_reference":
                    is_builtin,
            })


# ============================================================
# Plot helpers
# ============================================================

def get_results_for_storage(results, storage_bits):

    return [
        result
        for result in results
        if result["storage_bits"] == storage_bits
    ]


def plot_mse_vs_fractional_bits(results):

    plt.figure(figsize=(8, 6))

    for storage_bits in STORAGE_BITS:

        storage_results = get_results_for_storage(
            results,
            storage_bits
        )

        for strategy in LUT_STRATEGIES:

            strategy_results = [
                result
                for result in storage_results
                if result["strategy"] == strategy
            ]

            if not strategy_results:
                continue

            bits = [
                result["fractional_bits"]
                for result in strategy_results
            ]

            mse = [
                result["mse"]
                for result in strategy_results
            ]

            plt.plot(
                bits,
                mse,
                marker="o",
                label=f"{strategy}, {storage_bits}-bit"
            )

    # Mark built-in configuration
    builtin = [
        result
        for result in results
        if result["strategy"] == BUILTIN_STRATEGY
        and result["fractional_bits"]
        == BUILTIN_FRACTIONAL_BITS
        and result["storage_bits"]
        == BUILTIN_STORAGE_BITS
    ]

    if builtin:
        plt.scatter(
            [BUILTIN_FRACTIONAL_BITS],
            [builtin[0]["mse"]],
            marker="*",
            s=150,
            label="Built-in reference"
        )

    plt.yscale("log")

    plt.xlabel("Fractional bits")
    plt.ylabel("Mean squared error (MSE)")
    plt.title("LUT approximation error vs fractional bits")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mse_vs_fractional_bits.png",
        dpi=300
    )

    plt.close()


def plot_memory_vs_fractional_bits(results):

    plt.figure(figsize=(8, 6))

    for storage_bits in STORAGE_BITS:

        storage_results = get_results_for_storage(
            results,
            storage_bits
        )

        # Memory does not depend on strategy,
        # so use floor results for the plot.
        strategy_results = [
            result
            for result in storage_results
            if result["strategy"] == "floor"
        ]

        if not strategy_results:
            continue

        bits = [
            result["fractional_bits"]
            for result in strategy_results
        ]

        memory = [
            result["memory_bytes"]
            for result in strategy_results
        ]

        plt.plot(
            bits,
            memory,
            marker="o",
            label=f"{storage_bits}-bit storage"
        )

    plt.xlabel("Fractional bits")
    plt.ylabel("LUT memory (bytes)")
    plt.title("LUT memory consumption vs fractional bits")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "memory_vs_fractional_bits.png",
        dpi=300
    )

    plt.close()


def plot_lut_length_vs_fractional_bits(results):

    plt.figure(figsize=(8, 6))

    # LUT length is independent of strategy and storage.
    # Use floor + 16-bit storage as the reference.
    selected_results = [
        result
        for result in results
        if result["strategy"] == "floor"
        and result["storage_bits"] == 16
    ]

    bits = [
        result["fractional_bits"]
        for result in selected_results
    ]

    lengths = [
        result["lut_length"]
        for result in selected_results
    ]

    plt.plot(
        bits,
        lengths,
        marker="o"
    )

    plt.xlabel("Fractional bits")
    plt.ylabel("LUT length")
    plt.title("LUT length vs fractional bits")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "lut_length_vs_fractional_bits.png",
        dpi=300
    )

    plt.close()


def plot_mse_vs_memory(results):

    plt.figure(figsize=(8, 6))

    for storage_bits in STORAGE_BITS:

        storage_results = get_results_for_storage(
            results,
            storage_bits
        )

        for strategy in LUT_STRATEGIES:

            strategy_results = [
                result
                for result in storage_results
                if result["strategy"] == strategy
            ]

            if not strategy_results:
                continue

            memory = [
                result["memory_bytes"]
                for result in strategy_results
            ]

            mse = [
                result["mse"]
                for result in strategy_results
            ]

            plt.plot(
                memory,
                mse,
                marker="o",
                label=f"{strategy}, {storage_bits}-bit"
            )

    plt.yscale("log")

    plt.xlabel("LUT memory (bytes)")
    plt.ylabel("Mean squared error (MSE)")
    plt.title("LUT memory-error trade-off")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "mse_vs_memory.png",
        dpi=300
    )

    plt.close()


# ============================================================
# Main benchmark
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("==============================================")
    print("          LUT STRATEGY BENCHMARK")
    print("==============================================")

    print(f"time_step     = {TIME_STEP}")
    print(f"time_constant = {TIME_CONSTANT}")
    print(f"shift         = {SHIFT}")

    print()
    print("Strategies:")
    print(LUT_STRATEGIES)

    print()
    print("Fractional bits:")
    print(FRACTIONAL_BITS)

    print()
    print("Storage bits:")
    print(STORAGE_BITS)

    print()
    print(
        "Built-in reference: "
        f"{BUILTIN_STRATEGY}, "
        f"{BUILTIN_FRACTIONAL_BITS} fractional bits, "
        f"{BUILTIN_STORAGE_BITS}-bit storage"
    )

    print()

    results = []

    for strategy in LUT_STRATEGIES:

        for storage_bits in STORAGE_BITS:

            for fractional_bits in FRACTIONAL_BITS:

                # Skip configurations that cannot be represented
                # with the selected storage size.
                if (1 << fractional_bits) > (
                    (1 << storage_bits) - 1
                ):
                    continue

                print(
                    f"Running: "
                    f"{strategy:>5} / "
                    f"{fractional_bits:2d} fractional bits / "
                    f"{storage_bits:2d}-bit storage"
                )

                generated = generate_lut(
                    time_step=TIME_STEP,
                    time_constant=TIME_CONSTANT,
                    shift=SHIFT,
                    fractional_bits=fractional_bits,
                    storage_bits=storage_bits,
                    strategy=strategy
                )

                if generated is None:
                    continue

                (
                    exact_values,
                    lut,
                    lut_values
                ) = generated

                metrics = calculate_metrics(
                    exact_values,
                    lut,
                    lut_values,
                    fractional_bits,
                    storage_bits
                )

                result = {
                    "strategy": strategy,
                    **metrics
                }

                results.append(result)

                print(
                    f"  LUT length : "
                    f"{metrics['lut_length']}"
                )

                print(
                    f"  MSE        : "
                    f"{metrics['mse']:.12e}"
                )

                print(
                    f"  RMSE       : "
                    f"{metrics['rmse']:.12e}"
                )

                print(
                    f"  MAE        : "
                    f"{metrics['mae']:.12e}"
                )

                print(
                    f"  Max error  : "
                    f"{metrics['max_absolute_error']:.12e}"
                )

                print(
                    f"  Memory     : "
                    f"{metrics['memory_bytes']} bytes"
                )

                if (
                    strategy == BUILTIN_STRATEGY
                    and fractional_bits
                    == BUILTIN_FRACTIONAL_BITS
                    and storage_bits
                    == BUILTIN_STORAGE_BITS
                ):
                    print("  *** BUILT-IN REFERENCE ***")

                print()

    # Save all results
    save_results(results)

    # Generate plots
    plot_mse_vs_fractional_bits(results)
    plot_memory_vs_fractional_bits(results)
    plot_lut_length_vs_fractional_bits(results)
    plot_mse_vs_memory(results)

    print("==============================================")
    print("Benchmark completed.")
    print()
    print("CSV:")
    print(f"  {CSV_FILE}")
    print()
    print("Plots:")
    print(
        f"  {OUTPUT_DIR / 'mse_vs_fractional_bits.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'memory_vs_fractional_bits.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'lut_length_vs_fractional_bits.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'mse_vs_memory.png'}"
    )
    print("==============================================")


if __name__ == "__main__":
    main()