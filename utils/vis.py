import os
from pathlib import Path
import re
import json
import pandas as pd
from glob import glob

# Get the absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAYNT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "test_outputs_saynt2_DTs")


def analyze_saynt_dt_outputs(base_dir=SAYNT_OUTPUT_DIR):
    """
    Analyze decision tree outputs generated from both SAYNT and PAYNT FSCs.
    """
    print(f"Looking for benchmarks in: {base_dir}")
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory does not exist: {base_dir}")
        return pd.DataFrame()

    results = []

    # Find benchmark directories
    benchmark_dirs = sorted(
        [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    )

    print(f"Found {len(benchmark_dirs)} benchmark directories: {benchmark_dirs}")

    for benchmark in benchmark_dirs:
        benchmark_path = os.path.join(base_dir, benchmark)
        print(f"\n=======================================")
        print(f"Processing benchmark: {benchmark}")
        print(f"=======================================")

        # Define a dictionary to store all metrics for this benchmark
        benchmark_data = {
            "Benchmark": benchmark,
            "Observations": 0,
            "Status": "Processed",
        }

        # Read FSC sizes from output.txt
        output_file = os.path.join(benchmark_path, "output.txt")
        if os.path.exists(output_file):
            print(f"Reading FSC sizes from: {output_file}")
            try:
                with open(output_file, "r") as f:
                    content = f.read()

                # Extract STORM/SAYNT FSC size
                storm_patterns = [
                    r"Storm results:\s*[\d\.]+\s*controller size:\s*(\d+)",
                    r"SAYNT FSC has (\d+) nodes",
                    r"storm fsc nodes:?\s*(\d+)",
                    r"controller size:?\s*(\d+)",
                    r"constructed FSC with (\d+) nodes",
                ]

                for pattern in storm_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        benchmark_data["SAYNT FSC Nodes"] = int(matches[-1])
                        print(
                            f"  ✓ Found SAYNT FSC size: {benchmark_data['SAYNT FSC Nodes']} nodes"
                        )
                        break

                # Extract PAYNT FSC size
                paynt_patterns = [
                    r"PAYNT results:[\s\S]*?controller size:?\s*(\d+)",
                    r"PAYNT FSC has (\d+) nodes",
                    r"paynt fsc nodes:?\s*(\d+)",
                    r"Paynt controller size:?\s*(\d+)",
                    r"inductive controller size:?\s*(\d+)",
                ]

                for pattern in paynt_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        benchmark_data["PAYNT FSC Nodes"] = int(matches[-1])
                        print(
                            f"  ✓ Found PAYNT FSC size: {benchmark_data['PAYNT FSC Nodes']} nodes"
                        )
                        break

            except Exception as e:
                print(f"  Error reading {output_file}: {e}")
        else:
            print(f"  ⚠️ Output file not found: {output_file}")

        # Process FSCs for both SAYNT and PAYNT
        for fsc_type in ["SAYNT", "PAYNT"]:
            print(f"\n### Processing {fsc_type} FSC ###")

            # Initialize metrics for this FSC type
            if f"{fsc_type} FSC Nodes" not in benchmark_data:
                benchmark_data[f"{fsc_type} FSC Nodes"] = None
            benchmark_data[f"{fsc_type} Sep DT Inner"] = 0
            benchmark_data[f"{fsc_type} Sep DT Total"] = 0
            benchmark_data[f"{fsc_type} Comb DT Inner"] = 0
            benchmark_data[f"{fsc_type} Comb DT Total"] = 0

            # Direct path to decision tree directories
            fsc_type_dir = os.path.join(benchmark_path, "decision_trees", fsc_type)

            if not os.path.exists(fsc_type_dir):
                print(f"  ⚠️ Directory not found: {fsc_type_dir}")
                continue

            print(f"  ✓ FOUND {fsc_type} directory: {fsc_type_dir}")

            # Process decision tree directories
            dt_combined_dir = os.path.join(fsc_type_dir, "dt_combined")
            dt_separate_dir = os.path.join(fsc_type_dir, "dt_separate")

            # Try alternate directory structures if standard ones not found
            if not (os.path.exists(dt_combined_dir) or os.path.exists(dt_separate_dir)):
                print(f"  ⚠️ Standard DT directories not found, trying alternatives")

                alternative_dt_dirs = [
                    os.path.join(fsc_type_dir, "combined"),
                    os.path.join(fsc_type_dir, "separate"),
                ]

                for alt_dir in alternative_dt_dirs:
                    base_dir_name = os.path.basename(alt_dir)
                    if os.path.exists(alt_dir):
                        print(f"  ✓ Found alternative DT directory: {alt_dir}")
                        if "combined" in base_dir_name:
                            dt_combined_dir = alt_dir
                        elif "separate" in base_dir_name:
                            dt_separate_dir = alt_dir

            print(f"  DT directories status:")
            print(
                f"  - Combined: {'✓ Found' if os.path.exists(dt_combined_dir) else '✗ Not found'} - {dt_combined_dir}"
            )
            print(
                f"  - Separate: {'✓ Found' if os.path.exists(dt_separate_dir) else '✗ Not found'} - {dt_separate_dir}"
            )

            # Try to get observation count from config files
            for dt_dir in [dt_separate_dir, dt_combined_dir]:
                if os.path.exists(dt_dir):
                    config_files = glob(
                        os.path.join(dt_dir, "**", "*_config.json"), recursive=True
                    )
                    if config_files:
                        try:
                            with open(config_files[0], "r") as f:
                                config = json.load(f)
                                if "x_column_names" in config:
                                    benchmark_data["Observations"] = len(
                                        config["x_column_names"]
                                    )
                                    print(
                                        f"  ✓ Found {benchmark_data['Observations']} observations in config"
                                    )
                                    break
                        except Exception as e:
                            print(f"  ⚠️ Error reading config: {e}")

            # Process separate trees directory with detailed error handling
            if os.path.exists(dt_separate_dir):
                print(f"\n  Processing separate DT directory: {dt_separate_dir}")

                # List all JSON files in the directory
                all_json_files = glob(os.path.join(dt_separate_dir, "*.json")) + glob(
                    os.path.join(dt_separate_dir, "**", "*.json")
                )
                print(
                    f"  Found {len(all_json_files)} JSON files in separate DT directory"
                )

                try:
                    sep_inner, sep_total = get_dt_metrics(
                        dt_separate_dir, f"{fsc_type} Separate"
                    )
                    benchmark_data[f"{fsc_type} Sep DT Inner"] = sep_inner
                    benchmark_data[f"{fsc_type} Sep DT Total"] = sep_total
                    print(
                        f"  ✓ {fsc_type} separate DT: {sep_inner} inner nodes, {sep_total} total nodes"
                    )
                except Exception as e:
                    print(f"  ⚠️ Error processing separate DT metrics: {e}")

                # Check if skipped due to large FSC
                if os.path.exists(
                    os.path.join(dt_separate_dir, "skipped_large_fsc.txt")
                ):
                    if benchmark_data["Status"] == "Processed":
                        benchmark_data["Status"] = f"{fsc_type} Sep Skipped"

            # Process combined trees directory with detailed error handling
            if os.path.exists(dt_combined_dir):
                print(f"\n  Processing combined DT directory: {dt_combined_dir}")

                # List all JSON files in the directory
                all_json_files = glob(os.path.join(dt_combined_dir, "*.json")) + glob(
                    os.path.join(dt_combined_dir, "**", "*.json")
                )
                print(
                    f"  Found {len(all_json_files)} JSON files in combined DT directory"
                )

                try:
                    comb_inner, comb_total = get_dt_metrics(
                        dt_combined_dir, f"{fsc_type} Combined"
                    )
                    benchmark_data[f"{fsc_type} Comb DT Inner"] = comb_inner
                    benchmark_data[f"{fsc_type} Comb DT Total"] = comb_total
                    print(
                        f"  ✓ {fsc_type} combined DT: {comb_inner} inner nodes, {comb_total} total nodes"
                    )
                except Exception as e:
                    print(f"  ⚠️ Error processing combined DT metrics: {e}")

                # Check if skipped due to large FSC
                if os.path.exists(
                    os.path.join(dt_combined_dir, "skipped_large_fsc.txt")
                ):
                    if benchmark_data["Status"] == "Processed":
                        benchmark_data["Status"] = f"{fsc_type} Comb Skipped"
                    elif "Sep Skipped" in benchmark_data["Status"]:
                        benchmark_data["Status"] = f"{fsc_type} Both Skipped"

            # Summary of collected metrics
            print(f"\n  === {fsc_type} metrics summary ===")
            summary_fields = [
                f"{fsc_type} FSC Nodes",
                f"{fsc_type} Sep DT Total",
                f"{fsc_type} Comb DT Total",
            ]
            for field in summary_fields:
                value = benchmark_data.get(field, "Not available")
                if pd.isna(value) or value == 0:
                    print(f"  ⚠️ {field}: Missing")
                else:
                    print(f"  ✓ {field}: {value}")

        # Add the benchmark data to results
        results.append(benchmark_data)

    # Create DataFrame and sort by status first, then benchmark name
    df = pd.DataFrame(results).sort_values(["Status", "Benchmark"])

    return df


def get_dt_metrics(dt_dir, type_label=""):
    """Extract decision tree metrics from a directory with detailed logging"""
    inner_nodes = 0
    total_nodes = 0

    print(f"  Getting DT metrics for {type_label}, directory: {dt_dir}")

    # First check for combined_benchmarks.json or benchmark.json
    benchmark_files_found = False
    for bench_filename in ["combined_benchmarks.json", "benchmark.json"]:
        benchmark_file = os.path.join(dt_dir, bench_filename)
        if os.path.exists(benchmark_file):
            benchmark_files_found = True
            print(f"  ✓ Found benchmark file: {benchmark_file}")

            try:
                with open(benchmark_file, "r") as f:
                    benchmark_data = json.load(f)
                    print(
                        f"  ✓ Successfully loaded JSON, keys: {len(benchmark_data.keys())}"
                    )

                    # Debug: Print first few keys to understand structure
                    print(f"  First few keys: {list(benchmark_data.keys())[:3]}")

                    # Check for memory_X keys (old format)
                    memory_keys = [
                        k for k in benchmark_data.keys() if k.startswith("memory_")
                    ]

                    if memory_keys:
                        print(
                            f"  ✓ Found {len(memory_keys)} memory_X keys (old format)"
                        )
                        # OLD FORMAT: benchmark_data[memory_X][tree_id][classifiers][clf][stats]
                        for mem_key in memory_keys:
                            if not isinstance(benchmark_data[mem_key], dict):
                                print(
                                    f"  ⚠️ Memory key {mem_key} content is not a dictionary, skipping"
                                )
                                continue

                            mem_data = benchmark_data[mem_key]
                            for tree_id, tree_info in mem_data.items():
                                if (
                                    isinstance(tree_info, dict)
                                    and "classifiers" in tree_info
                                ):
                                    for clf_name, clf in tree_info[
                                        "classifiers"
                                    ].items():
                                        if "stats" in clf:
                                            stats = clf["stats"]
                                            inner_nodes += stats.get("inner nodes", 0)
                                            total_nodes += stats.get("nodes", 0)
                    else:
                        print(f"  Using flat format (no memory_X keys)")
                        # NEW FORMAT: benchmark_data[tree_id][classifiers][clf][stats]
                        for tree_id, tree_info in benchmark_data.items():
                            if (
                                isinstance(tree_info, dict)
                                and "classifiers" in tree_info
                            ):
                                clfs = tree_info["classifiers"]
                                print(
                                    f"  Tree {tree_id}: Found {len(clfs)} classifiers"
                                )
                                for clf_name, clf in clfs.items():
                                    if "stats" in clf:
                                        stats = clf["stats"]
                                        new_inner = stats.get("inner nodes", 0)
                                        new_total = stats.get("nodes", 0)
                                        inner_nodes += new_inner
                                        total_nodes += new_total
                                        print(
                                            f"  - Classifier {clf_name}: {new_inner} inner nodes, {new_total} total"
                                        )

                print(
                    f"  ✓ From {bench_filename}: found {inner_nodes} inner nodes, {total_nodes} total nodes"
                )
                return inner_nodes, total_nodes
            except Exception as e:
                print(f"  ⚠️ Error reading {benchmark_file}: {str(e)}")

    if not benchmark_files_found:
        print("  ⚠️ No top-level benchmark files found, checking memory directories")

    # If combined files didn't work, check individual memory dirs
    memory_dirs = glob(os.path.join(dt_dir, "memory_*"))
    print(f"  Found {len(memory_dirs)} memory directories")

    for memory_dir in memory_dirs:
        benchmark_file = os.path.join(memory_dir, "benchmark.json")
        if os.path.exists(benchmark_file):
            try:
                print(f"  Processing: {benchmark_file}")
                with open(benchmark_file, "r") as f:
                    benchmark_data = json.load(f)

                    # Process benchmark data
                    tree_count = 0
                    for tree_id, tree_info in benchmark_data.items():
                        tree_count += 1
                        if isinstance(tree_info, dict) and "classifiers" in tree_info:
                            clf_count = 0
                            for clf_name, clf in tree_info["classifiers"].items():
                                clf_count += 1
                                if "stats" in clf:
                                    stats = clf["stats"]
                                    inner_nodes += stats.get("inner nodes", 0)
                                    total_nodes += stats.get("nodes", 0)
                            print(f"    Tree {tree_id}: found {clf_count} classifiers")
                    print(f"    Processed {tree_count} trees")
            except Exception as e:
                print(f"  ⚠️ Error with {benchmark_file}: {e}")

    print(
        f"  ✓ From memory directories: found {inner_nodes} inner nodes, {total_nodes} total nodes"
    )
    return inner_nodes, total_nodes


def generate_saynt_latex_table(df):
    """
    Generate a LaTeX table comparing SAYNT and PAYNT FSC decision tree results
    with separate columns for separate and combined DT metrics
    """
    # Sort dataframe by status first, then by benchmark name
    df_sorted = df.sort_values(["Status", "Benchmark"])

    # Start LaTeX table
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of STORM/PAYNT FSCs and Decision Tree Controllers}\n"
    latex += "\\begin{tabular}{lrrrrrrr}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Benchmark} & \\textbf{Obs.} & \\multicolumn{3}{c}{\\textbf{SAYNT}} & \\multicolumn{3}{c}{\\textbf{PAYNT}} \\\\\n"
    latex += "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\n"
    latex += " & & \\textbf{FSC} & \\textbf{Sep} & \\textbf{Comb} & \\textbf{FSC} & \\textbf{Sep} & \\textbf{Comb} \\\\\n"
    latex += "\\midrule\n"

    # Track the current status group
    current_status = None

    # Add rows
    for _, row in df_sorted.iterrows():
        # Add separator and group header when status changes
        if current_status != row["Status"]:
            if current_status is not None:
                latex += "\\midrule\n"
            current_status = row["Status"]
            latex += f"\\multicolumn{{8}}{{l}}{{\\textbf{{{current_status}}}}}\\\\\n"

        benchmark = row["Benchmark"].replace("_", "\\_")
        observations = (
            str(int(row["Observations"]))
            if pd.notna(row["Observations"]) and row["Observations"] > 0
            else "---"
        )

        # SAYNT metrics
        storm_fsc = (
            str(int(row["SAYNT FSC Nodes"]))
            if pd.notna(row["SAYNT FSC Nodes"]) and row["SAYNT FSC Nodes"] > 0
            else "---"
        )
        storm_sep_dt = (
            str(int(row["SAYNT Sep DT Total"]))
            if pd.notna(row["SAYNT Sep DT Total"]) and row["SAYNT Sep DT Total"] > 0
            else "---"
        )
        storm_comb_dt = (
            str(int(row["SAYNT Comb DT Total"]))
            if pd.notna(row["SAYNT Comb DT Total"]) and row["SAYNT Comb DT Total"] > 0
            else "---"
        )

        # PAYNT metrics
        paynt_fsc = (
            str(int(row["PAYNT FSC Nodes"]))
            if pd.notna(row["PAYNT FSC Nodes"]) and row["PAYNT FSC Nodes"] > 0
            else "---"
        )
        paynt_sep_dt = (
            str(int(row["PAYNT Sep DT Total"]))
            if pd.notna(row["PAYNT Sep DT Total"]) and row["PAYNT Sep DT Total"] > 0
            else "---"
        )
        paynt_comb_dt = (
            str(int(row["PAYNT Comb DT Total"]))
            if pd.notna(row["PAYNT Comb DT Total"]) and row["PAYNT Comb DT Total"] > 0
            else "---"
        )

        latex += f"\\quad {benchmark} & {observations} & {storm_fsc} & {storm_sep_dt} & {storm_comb_dt} & {paynt_fsc} & {paynt_sep_dt} & {paynt_comb_dt} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}"

    return latex


def save_latex_table(latex_content, output_path="saynt_dt_results2.tex"):
    """
    Save LaTeX table content to a file
    """
    with open(output_path, "w") as f:
        # Add LaTeX document structure for standalone compilation
        full_document = (
            "\\documentclass{article}\n"
            "\\usepackage{booktabs}\n"
            "\\usepackage{caption}\n"
            "\\begin{document}\n\n"
            f"{latex_content}\n\n"
            "\\end{document}\n"
        )
        f.write(full_document)

    print(f"LaTeX table saved to: {output_path}")
    return output_path


# Run the analysis
saynt_df = analyze_saynt_dt_outputs()
print(
    saynt_df[
        [
            "Benchmark",
            "SAYNT FSC Nodes",
            "SAYNT Comb DT Total",
            "PAYNT FSC Nodes",
            "PAYNT Comb DT Total",
            "Status",
        ]
    ]
)

# Generate and save LaTeX table
latex_table = generate_saynt_latex_table(saynt_df)
output_file = save_latex_table(latex_table)
print("\nSAYNT LaTeX Table:")
print(latex_table)
