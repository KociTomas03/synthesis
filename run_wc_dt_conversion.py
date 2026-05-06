# author: Tomáš Kocí
# file is not wired into the main pipeline and is not discussed within thesis
# This file is solely for prototyping and running offline WC DT conversion on existing benchmark outputs
# which is then explored in future work - currently as the merging is greedy - no structure is to be exploted by DT transformation

"""
Offline WC DT conversion for existing benchmark outputs.

Reads minimized_wc_fsc.pkl and paynt_fsc.pkl from each benchmark directory
under BASE_DIR and runs dtControl on them, appending results to results.json.

Run this script manually after synthesis + minimization are complete.
WC DT conversion is not wired into the main --output-dir pipeline
because runtimes can be very long (up to ~15 min per benchmark on large FSCs);
BASE_DIR and WC_DT_NODE_LIMIT can be overridden at the top of this file.
"""
import os
import sys
import pickle
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from paynt.utils.FSCtoDTConverter import FSCtoDTConverter

WC_DT_NODE_LIMIT = 1000
PAYNT_DT_NODE_LIMIT = 1000
BASE_DIR = "benchmark_results"


def run_conversion(fsc, output_dir, combined_mode, is_storm):
    converter = FSCtoDTConverter(fsc, output_dir=output_dir, is_storm=is_storm, combined_mode=combined_mode)
    t0 = time.perf_counter()
    converter.run_dtcontrol()
    elapsed = round(time.perf_counter() - t0, 4)
    benchmark_path = os.path.join(output_dir, "benchmark.json")
    dt_nodes = FSCtoDTConverter.count_dt_nodes(benchmark_path)
    return elapsed, dt_nodes


def main():
    benchmarks = sorted(os.listdir(BASE_DIR))

    for benchmark in benchmarks:
        dt_dir = os.path.join(BASE_DIR, benchmark, "decision_trees")
        results_path = os.path.join(dt_dir, "results.json")

        if not os.path.exists(results_path):
            print(f"[{benchmark}] No results.json, skipping.")
            continue

        with open(results_path) as f:
            results = json.load(f)

        # --- WC FSC: single combined tree ---
        wc_pkl = os.path.join(dt_dir, "minimized_wc_fsc.pkl")
        if os.path.exists(wc_pkl):
            with open(wc_pkl, "rb") as f:
                wc_fsc = pickle.load(f)
            if wc_fsc.num_nodes > WC_DT_NODE_LIMIT:
                print(f"[{benchmark}] WC FSC has {wc_fsc.num_nodes} nodes (> {WC_DT_NODE_LIMIT}), skipping DT conversion.")
            else:
                try:
                    print(f"[{benchmark}] Running WC combined DT conversion ({wc_fsc.num_nodes} nodes)...")
                    t0_total = time.perf_counter()
                    dt_time, dt_nodes = run_conversion(wc_fsc, os.path.join(dt_dir, "WC"), combined_mode=True, is_storm=True)
                    total_time = round(time.perf_counter() - t0_total, 4)
                    print(f"[{benchmark}]   WC combined — nodes: {dt_nodes} | DT time: {dt_time}s | total: {total_time}s")
                    results["wc_dt_nodes"] = dt_nodes
                    results["wc_dt_conversion_time_s"] = dt_time
                    results["wc_total_time_s"] = total_time
                except Exception as e:
                    print(f"[{benchmark}] WC DT conversion failed: {e}")
                    import traceback; traceback.print_exc()

        # --- PAYNT FSC: two-tree ---
        paynt_pkl = os.path.join(dt_dir, "paynt_fsc.pkl")
        if os.path.exists(paynt_pkl):
            try:
                with open(paynt_pkl, "rb") as f:
                    paynt_fsc = pickle.load(f)
                if paynt_fsc.num_nodes > PAYNT_DT_NODE_LIMIT:
                    print(f"[{benchmark}] PAYNT FSC has {paynt_fsc.num_nodes} nodes (> {PAYNT_DT_NODE_LIMIT}), skipping DT conversion.")
                else:
                    print(f"[{benchmark}] Running PAYNT two-tree DT conversion ({paynt_fsc.num_nodes} nodes)...")
                    dt_time_p, dt_nodes_p = run_conversion(paynt_fsc, os.path.join(dt_dir, "PAYNT_two_tree"), combined_mode=False, is_storm=False)
                    print(f"[{benchmark}]   PAYNT two-tree — nodes: {dt_nodes_p} | time: {dt_time_p}s")
                    results["paynt_dt_two_tree_nodes"] = dt_nodes_p
                    results["paynt_dt_two_tree_time_s"] = dt_time_p
            except Exception as e:
                print(f"[{benchmark}] PAYNT DT conversion failed: {e}")
                import traceback; traceback.print_exc()

        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
