import pandas as pd
import numpy as np
import os
import glob
import re


def latex_escape(text):
    """
    Escape special characters in text for LaTeX compatibility
    """
    if not isinstance(text, str):
        return text

    # Define LaTeX special characters that need escaping
    escape_chars = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }

    # Escape each special character
    for char, escape in escape_chars.items():
        text = text.replace(char, escape)

    return text


def parse_minimization_log(log_file="minimization_2.txt"):
    """
    Parse the minimization log file to extract information about all benchmarks
    """
    results = []

    if not os.path.exists(log_file):
        print(f"Warning: Minimization log file {log_file} not found")
        return results

    with open(log_file, "r") as f:
        content = f.read()

    # Find all benchmark sections in the log
    benchmark_sections = re.findall(
        r"BENCHMARK: ([^\n]+)\nFSC TYPE: ([^\n]+)\nPATH: ([^\n]+)\n(.*?)(?=BENCHMARK:|$)",
        content,
        re.DOTALL,
    )

    for section in benchmark_sections:
        benchmark_name = section[0].strip()
        fsc_type = section[1].strip()
        path = section[2].strip()
        details = section[3]

        if "Error" in details or "Error:" in details:
            continue  # Skip failed minimizations

        # Extract original and minimized node counts
        orig_nodes_match = re.search(r"Original FSC: (\d+) nodes", details)
        min_nodes_match = re.search(r"Minimized FSC: (\d+)", details)
        reduction_match = re.search(r"Size reduction: ([0-9.]+)%", details)

        if orig_nodes_match and min_nodes_match and reduction_match:
            orig_nodes = int(orig_nodes_match.group(1))
            min_nodes = int(min_nodes_match.group(1))
            reduction = float(reduction_match.group(1))

            # Extract benchmark name from path
            path_parts = path.split("/")
            if "decision_trees" in path:
                if "SAYNT" in path:
                    name = path_parts[
                        -4
                    ]  # Format: .../benchmark/decision_trees/SAYNT/...
                else:
                    name = path_parts[
                        -4
                    ]  # Format: .../benchmark/decision_trees/PAYNT/...
            else:
                name = path_parts[-3]  # Format: .../benchmark/SAYNT/...

            results.append(
                {
                    "Benchmark": name,
                    "FSC Type": fsc_type,
                    "Original Nodes": orig_nodes,
                    "Minimized Nodes": min_nodes,
                    "Reduction %": reduction,
                    # These fields will be filled from verification data if available
                    "Original Probability": None,
                    "Minimized Probability": None,
                    "Probability Diff": None,
                    "Original Verification Time (s)": None,
                    "Minimized Verification Time (s)": None,
                    "Speedup": None,
                }
            )

    return results


def generate_latex_table(
    verification_csv=None,
    minimization_log=None,
    output_file="fsc_minimization_table.tex",
):
    """
    Generate a LaTeX table from FSC minimization results,
    combining data from verification CSV and minimization log.

    Parameters:
    -----------
    verification_csv : str, optional
        Path to the CSV file with verification results. If None, use the most recent file.
    minimization_log : str, optional
        Path to the minimization log file. If None, use "minimization_2.txt".
    output_file : str, optional
        Path to save the LaTeX table
    """
    # Parse the minimization log first
    if minimization_log is None:
        minimization_log = "minimization_2.txt"

    minimization_results = parse_minimization_log(minimization_log)

    # Find the verification CSV if not specified
    if verification_csv is None:
        csv_files = glob.glob("verification_*_*.csv") + [
            "verification_summary_intermediate.csv"
        ]
        if not csv_files:
            print("No verification result files found!")
            return

        # Sort by modification time and get the most recent
        csv_files.sort(
            key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0, reverse=True
        )
        verification_csv = csv_files[0]

    # Load the verification data
    print(f"Loading data from {verification_csv}")
    try:
        df_verification = pd.read_csv(verification_csv)

        # Merge verification data with minimization results
        verification_dict = {}
        for _, row in df_verification.iterrows():
            benchmark = row["Benchmark"]

            # Only include entries where both probabilities are available
            has_valid_probs = pd.notnull(
                row.get("Original Probability")
            ) and pd.notnull(row.get("Minimized Probability"))

            if has_valid_probs:
                verification_dict[benchmark] = {
                    "Original Probability": row.get("Original Probability"),
                    "Minimized Probability": row.get("Minimized Probability"),
                    "Probability Diff": row.get("Probability Diff"),
                    "Original Verification Time (s)": row.get(
                        "Original Verification Time (s)"
                    ),
                    "Minimized Verification Time (s)": row.get(
                        "Minimized Verification Time (s)"
                    ),
                    "Speedup": row.get("Speedup"),
                }

        # Update minimization results with verification data
        for result in minimization_results:
            benchmark = result["Benchmark"]
            if benchmark in verification_dict:
                result.update(verification_dict[benchmark])
    except Exception as e:
        print(f"Error loading verification data: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(minimization_results)

    # Filter SAYNT FSCs only (since we're focusing on SAYNT in this visualization)
    df_saynt = df[df["FSC Type"] == "SAYNT"]

    # Sort by reduction percentage (descending)
    sorted_df = df_saynt.sort_values("Reduction %", ascending=False)

    # Start the LaTeX table with document preamble and packages needed
    latex_content = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{array}
\usepackage{siunitx}
\usepackage{caption}

\begin{document}

\begin{table}[htb]
\centering
\caption{SAYNT FSC Minimization Results}
\label{tab:fsc_minimization}
\begin{tabular}{l|r|r|r|r}
\toprule
\textbf{Benchmark} & \textbf{Orig. Nodes} & \textbf{Min. Nodes} & \textbf{Reduction \%} & \textbf{Probability Diff} \\
\midrule
"""

    # Separate entries with and without verification results
    rows_with_verification = []
    rows_without_verification = []

    for _, row in sorted_df.iterrows():
        # Check if we have valid probability comparisons
        has_verification = pd.notnull(row["Original Probability"]) and pd.notnull(
            row["Minimized Probability"]
        )

        if has_verification:
            rows_with_verification.append(row)
        else:
            rows_without_verification.append(row)

    # Add rows with verification results first
    for row in rows_with_verification:
        # Escape benchmark name for LaTeX
        benchmark = latex_escape(str(row["Benchmark"]))

        # Format probability difference with scientific notation if very small
        if pd.notnull(row["Probability Diff"]):
            if row["Probability Diff"] < 1e-10 and row["Probability Diff"] > 0:
                prob_diff = f"${row['Probability Diff']:.2e}$"
            else:
                prob_diff = f"{row['Probability Diff']:.8f}"
                # Remove trailing zeros
                prob_diff = (
                    prob_diff.rstrip("0").rstrip(".") if "." in prob_diff else prob_diff
                )
        else:
            prob_diff = "---"

        # Add the row to the table (with node counts, without speedup)
        latex_content += f"{benchmark} & {int(row['Original Nodes'])} & {int(row['Minimized Nodes'])} & {row['Reduction %']:.1f}\\% & {prob_diff} \\\\\n"

    # Add a separator if both types of rows exist
    if rows_with_verification and rows_without_verification:
        latex_content += r"\midrule" + "\n"

    # Add rows without verification results
    for row in rows_without_verification:
        benchmark = latex_escape(str(row["Benchmark"]))
        latex_content += f"{benchmark} & {int(row['Original Nodes'])} & {int(row['Minimized Nodes'])} & {row['Reduction %']:.1f}\\% & --- \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

    # Write the tables to the output file
    with open(output_file, "w") as f:
        f.write(latex_content)

    print(f"LaTeX document written to {output_file}")
    print("You can compile it directly with: pdflatex " + output_file)

    # Also write just the tables without document preamble for inclusion in other documents
    table_only_file = output_file.replace(".tex", "_tables_only.tex")

    # Extract just the tables without the document preamble
    table_content = latex_content.split(r"\begin{document}")[1].split(
        r"\end{document}"
    )[0]

    with open(table_only_file, "w") as f:
        f.write(table_content)

    print(f"LaTeX tables only (for inclusion) written to {table_only_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total SAYNT benchmarks analyzed: {len(sorted_df)}")
    print(f"With verification results: {len(rows_with_verification)}")

    if rows_with_verification:
        valid_speedups = [
            row["Speedup"]
            for row in rows_with_verification
            if pd.notnull(row["Speedup"]) and row["Speedup"] > 0
        ]

        valid_diffs = [
            row["Probability Diff"]
            for row in rows_with_verification
            if pd.notnull(row["Probability Diff"])
        ]
        if valid_diffs:
            print(f"Maximum probability difference: {max(valid_diffs):.8f}")

    # Find the benchmark with the greatest reduction
    if not sorted_df.empty:
        max_reduction_idx = sorted_df["Reduction %"].idxmax()
        max_reduction = sorted_df.loc[max_reduction_idx]
        print(
            f"\nLargest reduction: {max_reduction['Benchmark']} - {max_reduction['Reduction %']:.2f}%"
        )
        print(
            f"  Original: {int(max_reduction['Original Nodes'])} nodes → Minimized: {int(max_reduction['Minimized Nodes'])} nodes"
        )

    # Find the benchmark with the greatest speedup among verified FSCs
    if rows_with_verification:
        valid_speedups_df = pd.DataFrame(rows_with_verification)
        valid_speedups_df = valid_speedups_df[pd.notnull(valid_speedups_df["Speedup"])]
        if not valid_speedups_df.empty:
            max_speedup_idx = valid_speedups_df["Speedup"].idxmax()
            max_speedup = valid_speedups_df.loc[max_speedup_idx]
            print(
                f"\nLargest speedup: {max_speedup['Benchmark']} - {max_speedup['Speedup']:.2f}x"
            )
            print(
                f"  Original time: {max_speedup.get('Original Verification Time (s)', 0):.2f}s → "
                f"Minimized time: {max_speedup.get('Minimized Verification Time (s)', 0):.2f}s"
            )


if __name__ == "__main__":
    # Generate tables combining both data sources
    generate_latex_table()
