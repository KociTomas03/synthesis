import os
import pickle
import glob
import traceback
import subprocess
import pandas as pd
import re
import time
from datetime import datetime

# Directory where FSC pickle files are stored
base_dir = "test_outputs_saynt2_DTs"


def find_all_saynt_fscs():
    """Find all SAYNT FSC pickle files in the expected locations"""
    pattern = os.path.join(base_dir, "*", "decision_trees", "SAYNT", "fsc.pkl")
    pickle_files = glob.glob(pattern)

    if not pickle_files:
        print(
            "No FSC pickle files found in primary location. Checking alternative location..."
        )
        pattern = os.path.join(base_dir, "*", "SAYNT", "fsc.pkl")
        pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        raise FileNotFoundError(f"No SAYNT FSC pickle files found in {base_dir}")

    # Load all FSCs and sort by number of nodes instead of by name
    fsc_sizes = []
    for path in pickle_files:
        try:
            with open(path, "rb") as f:
                fsc = pickle.load(f)
                fsc_sizes.append((path, fsc.num_nodes))
        except Exception as e:
            print(f"Error loading FSC {path}: {str(e)}")
            fsc_sizes.append((path, float("inf")))  # Put errors at the end

    # Sort by size (number of nodes)
    fsc_sizes.sort(key=lambda x: x[1])

    # Return just the paths, now ordered by size
    return [path for path, _ in fsc_sizes]


def get_minimized_fsc_path(original_path):
    """Get the path to the minimized FSC based on the original FSC path"""
    return original_path.replace(".pkl", "_minimized.pkl")


def get_model_path(fsc_path):
    """
    Extract the model path from the FSC path.
    For PAYNT verification, we only need the directory name without extensions.
    """
    # Extract benchmark name from FSC path
    if "decision_trees" in fsc_path:
        benchmark = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(fsc_path)))
        )
    else:
        benchmark = os.path.basename(os.path.dirname(os.path.dirname(fsc_path)))

    # Default to first location if not found
    return os.path.join("models/archive/cav23-saynt", benchmark)


def run_verification(model_path, original_fsc_path, minimized_fsc_path=None):
    """Run PAYNT verification on the FSCs"""
    results = {}

    print(f"Verifying original FSC: {original_fsc_path}")

    # Build command for original FSC verification
    cmd = [
        "python",
        "paynt.py",
        model_path,
        "--verify-FSC",
        "--import-STORM-fsc",
        original_fsc_path,
    ]

    # Run command and capture output
    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)  # 5 minute timeout
        duration = time.time() - start_time

        # Parse output to extract verification result
        output = result.stdout + result.stderr
        print(output)

        # Extract probability value with regex
        prob_match = re.search(r"Result: (\d+\.\d+)", output)
        if prob_match:
            probability = float(prob_match.group(1))
            results["original_probability"] = probability
        else:
            results["original_probability"] = None

        results["original_duration"] = duration
        results["original_success"] = result.returncode == 0
        results["original_output"] = output

    except Exception as e:
        print(f"Error verifying original FSC: {str(e)}")
        results["original_probability"] = None
        results["original_duration"] = None
        results["original_success"] = False
        results["original_output"] = str(e)

    # Verify minimized FSC if available
    if minimized_fsc_path and os.path.exists(minimized_fsc_path):
        print(f"Verifying minimized FSC: {minimized_fsc_path}")

        cmd = [
            "python",
            "paynt.py",
            model_path,
            "--verify-FSC",
            "--import-STORM-fsc",
            minimized_fsc_path,
        ]

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start_time

            output = result.stdout + result.stderr
            print(output)

            prob_match = re.search(r"Result: (\d+\.\d+)", output)
            if prob_match:
                probability = float(prob_match.group(1))
                results["minimized_probability"] = probability
            else:
                results["minimized_probability"] = None

            results["minimized_duration"] = duration
            results["minimized_success"] = result.returncode == 0
            results["minimized_output"] = output

        except Exception as e:
            print(f"Error verifying minimized FSC: {str(e)}")
            results["minimized_probability"] = None
            results["minimized_duration"] = None
            results["minimized_success"] = False
            results["minimized_output"] = str(e)
    else:
        results["minimized_probability"] = None
        results["minimized_duration"] = None
        results["minimized_success"] = False
        results["minimized_output"] = "No minimized FSC available"

    return results


def load_fscs_and_run_verification():
    """Load FSCs, run verification on them, and store the results"""
    print("Loading SAYNT FSC pairs and running verification...")
    results = []

    try:
        fsc_paths = find_all_saynt_fscs()
        print(f"Found {len(fsc_paths)} SAYNT FSC files to analyze")

        for i, fsc_path in enumerate(fsc_paths):
            try:
                # Get benchmark name from path
                if "decision_trees" in fsc_path:
                    benchmark = os.path.basename(
                        os.path.dirname(os.path.dirname(os.path.dirname(fsc_path)))
                    )
                else:
                    benchmark = os.path.basename(
                        os.path.dirname(os.path.dirname(fsc_path))
                    )

                print(f"\n--- Processing {benchmark} ({i+1}/{len(fsc_paths)}) ---")

                # Load the original FSC for info
                with open(fsc_path, "rb") as f:
                    original_fsc = pickle.load(f)

                minimized_path = get_minimized_fsc_path(fsc_path)
                minimized_fsc = None

                # Load minimized FSC if it exists
                if os.path.exists(minimized_path):
                    try:
                        with open(minimized_path, "rb") as f:
                            minimized_fsc = pickle.load(f)
                    except Exception as e:
                        print(f"Error loading minimized FSC: {str(e)}")

                # Get model path for verification
                model_path = get_model_path(fsc_path)
                print(f"Using model path: {model_path}")

                # Run verification on the FSCs
                verification_results = run_verification(
                    model_path, fsc_path, minimized_path
                )

                # Collect all results
                result_entry = {
                    "Benchmark": benchmark,
                    "Original FSC Path": fsc_path,
                    "Minimized FSC Path": (
                        minimized_path if os.path.exists(minimized_path) else None
                    ),
                    "Original Nodes": original_fsc.num_nodes if original_fsc else 0,
                    "Minimized Nodes": minimized_fsc.num_nodes if minimized_fsc else 0,
                    "Reduction %": (
                        (1 - minimized_fsc.num_nodes / original_fsc.num_nodes) * 100
                        if (original_fsc and minimized_fsc)
                        else 0
                    ),
                    "Original Verification Success": verification_results.get(
                        "original_success", False
                    ),
                    "Original Probability": verification_results.get(
                        "original_probability"
                    ),
                    "Original Verification Time (s)": verification_results.get(
                        "original_duration"
                    ),
                    "Minimized Verification Success": verification_results.get(
                        "minimized_success", False
                    ),
                    "Minimized Probability": verification_results.get(
                        "minimized_probability"
                    ),
                    "Minimized Verification Time (s)": verification_results.get(
                        "minimized_duration"
                    ),
                    "Probability Diff": (
                        abs(
                            verification_results.get("minimized_probability", 0)
                            - verification_results.get("original_probability", 0)
                        )
                        if (
                            verification_results.get("minimized_probability")
                            is not None
                            and verification_results.get("original_probability")
                            is not None
                        )
                        else None
                    ),
                    "Speedup": (
                        verification_results.get("original_duration", 0)
                        / verification_results.get("minimized_duration", 1)
                        if verification_results.get("minimized_duration", 0) > 0
                        else 0
                    ),
                }

                results.append(result_entry)

                # Save intermediate results after each benchmark
                save_results_to_csv(results)

                print(f"Completed verification of {benchmark}")
                print("-" * 80)

            except Exception as e:
                print(f"Error processing {fsc_path}: {str(e)}")
                traceback.print_exc()

        # Final save of all results
        save_results_to_csv(results, is_final=True)
        print("\nVerification complete! Results saved to verification_results.csv")

        return results

    except Exception as e:
        print(f"Error in verification process: {str(e)}")
        traceback.print_exc()

        # Try to save partial results
        if results:
            save_results_to_csv(results)
            print("Saved partial results to verification_results.csv")

        return results


def save_results_to_csv(results, is_final=False):
    """
    Save verification results to CSV files
    If is_final=True, creates timestamped final results
    Otherwise, updates intermediate results files
    """
    if not results:
        print("No results to save")
        return

    # For intermediate results, use fixed filenames that get overwritten
    if not is_final:
        results_file = "verification_results_intermediate.csv"
        summary_file = "verification_summary_intermediate.csv"
    else:
        # For final results, use timestamped filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"verification_results_{timestamp}.csv"
        summary_file = f"verification_summary_{timestamp}.csv"

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)

    # Also save a summary with just the most important columns
    summary_columns = [
        "Benchmark",
        "Original Nodes",
        "Minimized Nodes",
        "Reduction %",
        "Original Probability",
        "Minimized Probability",
        "Probability Diff",
        "Original Verification Time (s)",
        "Minimized Verification Time (s)",
        "Speedup",
    ]

    if all(col in df.columns for col in summary_columns):
        summary_df = df[summary_columns]
        summary_df.to_csv(summary_file, index=False)

    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    results = load_fscs_and_run_verification()
