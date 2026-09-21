import csv

input_file = "benchmark_balanced_networks_profiling.csv"
output_file = "benchmark_balanced_networks_profiling_test.csv"
results_file = "benchmark_balanced_networks_results.csv"


with open(input_file, "r", newline="", encoding="utf-8") as f:
    profiling_rows = list(csv.DictReader(f))

with open(results_file, "r", newline="", encoding="utf-8") as f:
    results_rows = list(csv.DictReader(f))


for profiling_row, results_row in zip(profiling_rows, results_rows):
    profiling_row["use_exp_luts"] = results_row["use_exp_luts"]


fieldnames = [
    "profile",
    "implementation",
    "use_exp_luts",
    "N",
    "sum_core_activity_percent"
]

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(profiling_rows)

print(f"Test CSV created: {output_file}")