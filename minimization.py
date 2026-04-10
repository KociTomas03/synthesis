# Thin CLI wrapper — all logic lives in paynt/utils/minimization.py
from paynt.utils.minimization import (
    minimize_fsc_object,
    eliminate_unreachable_states,
    minimize_fsc_internal,
    merge_partitions_with_wildcards,
)

if __name__ == "__main__":
    import os
    import pickle
    import glob
    import time
    import pandas as pd
    import sys
    from datetime import datetime

    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

    print(f"FSC MINIMIZATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    base_dir = "test_outputs_bp"

    pickle_files = []
    pattern = os.path.join(base_dir, "*", "FSCs", "STORM", "fsc.pkl")
    pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print("No FSC pickle files found. Checking alternative locations...")
        for fsc_type in ["STORM", "PAYNT"]:
            pattern = os.path.join(base_dir, "*", "FSCs", fsc_type, "fsc.pkl")
            pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print(f"No minimized FSC pickle files found in {base_dir}")
        exit(1)

    print(f"Found {len(pickle_files)} FSC pickle files")
    pickle_files.sort(key=lambda x: os.path.getsize(x))

    results = []

    for i, pickle_path in enumerate(pickle_files):
        benchmark_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pickle_path))))
        fsc_type = "STORM" if "STORM" in pickle_path else "PAYNT" if "PAYNT" in pickle_path else "Unknown"

        print(f"\n{'='*40}")
        print(f"BENCHMARK: {benchmark_name}")
        print(f"FSC TYPE: {fsc_type}")
        print(f"PATH: {pickle_path}")
        print(f"{'='*40}\n")

        result_entry = {
            "Benchmark": benchmark_name,
            "FSC Type": fsc_type,
            "Original Nodes": 0,
            "Minimized Nodes": 0,
            "Reduction %": 0,
            "Observations": 0,
            "Status": "Failed",
        }

        try:
            start_time = time.time()

            with open(pickle_path, "rb") as f:
                fsc = pickle.load(f)
                if fsc is None:
                    print("Error: Failed to load FSC (None returned)")
                    result_entry["Status"] = "Load Failed"
                    results.append(result_entry)
                    continue

            result_entry["Original Nodes"] = fsc.num_nodes
            result_entry["Observations"] = fsc.num_observations

            print(f"Original FSC: {fsc.num_nodes} nodes, {fsc.num_observations} observations")
            print(f"Deterministic: {fsc.is_deterministic}\n")

            print("Running partition refinement (use_wildcards=False)...")
            minimized_fsc, partitions, initial_state = minimize_fsc_object(fsc, use_wildcards=False)
            print(f"After minimization: {minimized_fsc.num_nodes} nodes")

            print("Removing unreachable states...")
            final_fsc = eliminate_unreachable_states(minimized_fsc, initial_state=initial_state)

            reduction = (1 - final_fsc.num_nodes / fsc.num_nodes) * 100 if fsc.num_nodes else 0
            result_entry["Minimized Nodes"] = final_fsc.num_nodes
            result_entry["Reduction %"] = reduction
            result_entry["Status"] = "Success"

            print(f"Minimization completed in {time.time() - start_time:.2f} seconds")
            print(f"Final FSC: {final_fsc.num_nodes} nodes")
            print(f"Size reduction: {reduction:.2f}%\n")

            output_path = pickle_path.replace("fsc.pkl", "fsc_minimized.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(final_fsc, f)
            print(f"Saved minimized FSC to: {output_path}\n")
            print("-" * 80 + "\n")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            result_entry["Status"] = f"Error: {str(e)[:50]}..."

        results.append(result_entry)

    print("\n\n" + "=" * 80)
    print("MINIMIZATION SUMMARY")
    print("=" * 80 + "\n")

    df = pd.DataFrame(results)
    success_count = len(df[df["Status"] == "Success"])
    print(f"Total FSCs processed: {len(results)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(results) - success_count}\n")

    if success_count > 0:
        successful_df = df[df["Status"] == "Success"]
        print(f"Average size reduction: {successful_df['Reduction %'].mean():.2f}%")
        print(f"Maximum size reduction: {successful_df['Reduction %'].max():.2f}%")
        print(f"Minimum size reduction: {successful_df['Reduction %'].min():.2f}%\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"minimization_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed results saved to {csv_file}")
