import os
import unittest
import subprocess
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class BenchmarkPipeline(unittest.TestCase):
    # NOTE: most test methods begin with `return` and are intentionally disabled.
    # They invoke full SAYNT synthesis (15+ min per benchmark) or verification
    # and are meant to be run manually, not by an automated test runner.
    # test_saynt_table is the exception — it only reads existing results.json files
    # and can be run at any time.

    @classmethod
    def setUpClass(cls):
        # Setup any necessary paths or configurations
        cls.projects = [
            "models/archive/uai22-pomdp/maze-mo",
            "models/archive/uai22-pomdp/grid-avoid-4-0.1",
            "models/archive/uai22-pomdp/grid-avoid-4-0.1-goal-in-center",
            "models/archive/uai22-pomdp/grid-large-12-4",
            "models/archive/cav23-saynt/4x3-95",
            "models/archive/cav23-saynt/refuel-06",
            "models/archive/cav23-saynt/refuel-08",
            "models/archive/cav23-saynt/query-s2",
            "models/archive/cav23-saynt/network",
        ]
        cls.output_dir = "test_outputs"
        cls.iterative_output_dir = "test_outputs_iterative"
        cls.saynt_output_dir = "benchmark_results"
        cls.memory_constraints = [
            "onestep",
            "bothway",
            "circular",
            "binaryTree",
            "binaryTreeSelfLoop",
            "binaryTreeCyclic",
            "growing",
            "growingMax2",
            "notDecreasing",
            "notDecreasingMax2",
            "notDecreasingCyclic",
            "evenUpOddDown",
            "bothWayCircleSelfLoop",
            "none",
        ]
        cls.fsc_sizes = range(2, 8)

        # Create output directory if it doesn't exist
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)

    # each fsc size given and ran separately - 
    # inspecting the FSC patterns and their availability
    def test_paynt_run(self):
        return
        for project_path in self.projects:
            project_name = os.path.basename(project_path).replace("/", "_")
            for fsc_size in self.fsc_sizes:
                for memory_constraint in self.memory_constraints:
                    # Create the directory structure: problemName/fscSize/memoryConstraint
                    outputFolder = os.path.join(
                        self.output_dir,
                        project_name,
                        f"fsc_size_{fsc_size}",
                        memory_constraint,
                    )
                    output_file = os.path.join(outputFolder, "output.txt")
                    output_dir = os.path.dirname(output_file)

                    # Create the directory structure if it doesn't exist
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Skip the test if the output file already exists
                    if os.path.exists(output_file):
                        print(
                            f"Skipping existing result for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}"
                        )
                        continue

                    print(
                        f"Starting paynt_run test for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}"
                    )
                    # Run the paynt.py script as a separate process and capture its output
                    command = f"python3 paynt.py {project_path} --fsc-memory-size {fsc_size} --generated-fsc-route {outputFolder}/image --memory-constraint {memory_constraint} > {output_file} 2>&1"
                    result = subprocess.run(command, shell=True)

                    # Check the exit code
                    if result.returncode != 0:
                        with open(output_file, "r") as f:
                            print("Output:")
                            print(f.read())

                    self.assertEqual(
                        result.returncode,
                        0,
                        f"Failed to run paynt.py with the specified arguments for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}",
                    )

                    print("Finished paynt_run test")

                    # Print the content of the output file for debugging
                    with open(output_file, "r") as f:
                        print("Output file content:")
                        print(f.read())
    # each benchmark ran with iteratively increasing memory size, whilst being constrained on memory patterns
    def test_paynt_run_iterative(self):
        return
        for project_path in self.projects:
            project_name = os.path.basename(project_path).replace("/", "_")
            for memory_constraint in self.memory_constraints:
                outputFolder = os.path.join(
                    self.iterative_output_dir, project_name, memory_constraint
                )
                output_file = os.path.join(outputFolder, "output.txt")
                output_dir = os.path.dirname(output_file)

                # Create the directory structure if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Skip the test if the output file already exists
                if os.path.exists(output_file):
                    print(
                        f"Skipping existing result for project {project_path}, ITERATIVE, memory_constraint {memory_constraint}"
                    )
                    continue

                print(
                    f"Starting paynt_run test for project {project_path}, ITERATIVE, memory_constraint {memory_constraint}"
                )
                # Run the paynt.py script as a separate process and capture its output
                command = f"python3 paynt.py {project_path} --fsc-synthesis --timeout 900 --generated-fsc-route {outputFolder}/image --memory-constraint {memory_constraint} > {output_file} 2>&1"
                result = subprocess.run(command, shell=True)

                # Check the exit code
                if result.returncode != 0:
                    with open(output_file, "r") as f:
                        print("Output:")
                        print(f.read())

                self.assertEqual(
                    result.returncode,
                    0,
                    f"Failed to run paynt.py with the specified arguments for project {project_path}, ITERATIVE, memory_constraint {memory_constraint}",
                )

                print("Finished paynt_run test")

                # Print the content of the output file for debugging
                with open(output_file, "r") as f:
                    print("Output file content:")
                    print(f.read())

        pass

    # different set of benchmarks - to better reflect the SAYNT evaluation
    # saynt runs with without memory constraints with 15-minute timeout -> results of thesis
    def test_saynt_run(self):
        return
        sayntProjects = [
            "models/archive/cav23-saynt/drone-4-1",
            "models/archive/cav23-saynt/drone-4-2",
            "models/archive/cav23-saynt/maze-alex",
            "models/archive/cav23-saynt/grid-large-10-5",
            "models/archive/cav23-saynt/grid-large-20-5",
            "models/archive/cav23-saynt/network-2-8-20",
            "models/archive/cav23-saynt/network-3-8-20",
            "models/archive/cav23-saynt/refuel-06",
            "models/archive/cav23-saynt/refuel-08",
            "models/archive/cav23-saynt/rocks-12",
            "models/archive/cav23-saynt/rocks-16",
            "models/archive/cav23-saynt/lanes-100-combined-new",
        ]

        for project_path in sayntProjects:
            project_name = os.path.basename(project_path).replace("/", "_")
            memory_constraint = "none"
            outputFolder = os.path.join(self.saynt_output_dir, project_name)
            output_file = os.path.join(outputFolder, "output.txt")
            output_dir = os.path.dirname(output_file)

            # Create the directory structure if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Skip the test if the output file already exists
            # if os.path.exists(output_file):
            #     print(
            #         f"Skipping existing result for project {project_path}, STORM_ITERATIVE, memory_constraint {memory_constraint}"
            #     )
            #     continue

            print(
                f"Starting paynt_run test for project {project_path}, STORM_ITERATIVE"
            )
            # Run the paynt.py script as a separate process and capture its output
            command = (
                f"python3 paynt.py {project_path}"
                f" --fsc-synthesis --storm-pomdp --iterative-storm 900 60 10"
                f" --output-dir {outputFolder}/decision_trees"
                f" --minimize-storm-fsc"
                f" > {output_file} 2>&1"
            )
            result = subprocess.run(command, shell=True)

            # Check the exit code
            if result.returncode != 0:
                with open(output_file, "r") as f:
                    print("Output:")
                    print(f.read())

            self.assertEqual(
                result.returncode,
                0,
                f"Failed to run paynt.py with the specified arguments for project {project_path}, ITERATIVE, memory_constraint {memory_constraint}",
            )

            print("Finished saynt_run test")

            # Print the content of the output file for debugging
            # with open(output_file, "r") as f:
            #     print("Output file content:")
            #     print(f.read())

    # verification of the generated FSCs (policies) - sanity check to ensure PR and WM minimizations
    # do not degrade the quality
    def test_saynt_verify(self):
        return
        import json
        import re

        sayntProjects = [
            "models/archive/cav23-saynt/drone-4-1",
            "models/archive/cav23-saynt/drone-4-2",
            "models/archive/cav23-saynt/maze-alex",
            "models/archive/cav23-saynt/grid-large-10-5",
            "models/archive/cav23-saynt/grid-large-20-5",
            "models/archive/cav23-saynt/network-2-8-20",
            "models/archive/cav23-saynt/network-3-8-20",
            "models/archive/cav23-saynt/refuel-06",
            "models/archive/cav23-saynt/refuel-08",
            "models/archive/cav23-saynt/rocks-12",
            "models/archive/cav23-saynt/rocks-16",
            "models/archive/cav23-saynt/lanes-100-combined-new",
        ]

        fsc_types = [
            ("storm_fsc",       "storm_nodes",        "storm_verified_value"),
            ("minimized_fsc",   "minimized_nodes",     "minimized_verified_value"),
            ("minimized_wc_fsc","minimized_wc_nodes",  "minimized_wc_verified_value"),
        ]

        all_rows = []

        for project_path in sayntProjects:
            project_name = os.path.basename(project_path).replace("/", "_")
            dt_dir = os.path.join(self.saynt_output_dir, project_name, "decision_trees")
            results_path = os.path.join(dt_dir, "results.json")

            if not os.path.exists(results_path):
                print(f"Skipping {project_name}: results.json not found")
                continue

            with open(results_path) as f:
                synth = json.load(f)

            row = {"benchmark": project_name, "storm_value": synth.get("storm_value"), "storm_nodes": synth.get("storm_num_nodes")}

            for fsc_name, nodes_key, value_key in fsc_types:
                pkl_path = os.path.join(dt_dir, f"{fsc_name}.pkl")
                row[nodes_key] = synth.get(
                    {"storm_nodes": "storm_num_nodes",
                     "minimized_nodes": "minimized_num_nodes",
                     "minimized_wc_nodes": "minimized_wc_num_nodes"}[nodes_key]
                )

                if not os.path.exists(pkl_path):
                    print(f"  Skipping {fsc_name}: pickle not found")
                    row[value_key] = None
                    continue

                out_file = os.path.join(dt_dir, f"verify_{fsc_name}.txt")
                cmd = (
                    f"python3 paynt.py {project_path}"
                    f" --fsc-synthesis"
                    f" --verify-fsc {pkl_path}"
                    f" > {out_file} 2>&1"
                )
                r = subprocess.run(cmd, shell=True)
                self.assertEqual(r.returncode, 0, f"Verification failed for {project_name}/{fsc_name}")

                with open(out_file) as f:
                    content = f.read()
                m = re.search(r"Result:\s*([^\s]+)", content)
                raw = m.group(1) if m else None
                try:
                    verified = float(raw) if raw is not None else None
                except ValueError:
                    verified = raw  # keep "inf", "nan", etc. as strings
                row[value_key] = verified
                synth[value_key] = verified
                print(f"  {fsc_name}: verified={verified} ({row[nodes_key]} nodes)")

            # Write verified values back into results.json
            with open(results_path, "w") as f:
                json.dump(synth, f, indent=2)

            print(
                f"{project_name}: storm={row['storm_value']}"
                f" | storm_verified={row.get('storm_verified_value')}"
                f" | minimized_verified={row.get('minimized_verified_value')}"
                f" | wildcard_verified={row.get('minimized_wc_verified_value')}"
            )
            all_rows.append(row)

        summary_path = os.path.join(self.saynt_output_dir, "verification_summary.json")
        with open(summary_path, "w") as f:
            json.dump(all_rows, f, indent=2)
        print(f"\nSummary written to {summary_path}")

    # assembling the final table for the thesis - extracting relevant values from results.json files
    def test_saynt_table(self):
        """
        Reads each benchmark's results.json (which must already contain verified values
        written by test_saynt_verify) and assembles a single table_data.json with all
        columns needed for the thesis table.
        """
        import json

        sayntProjects = [
            "models/archive/cav23-saynt/drone-4-1",
            "models/archive/cav23-saynt/drone-4-2",
            "models/archive/cav23-saynt/maze-alex",
            "models/archive/cav23-saynt/grid-large-10-5",
            "models/archive/cav23-saynt/grid-large-20-5",
            "models/archive/cav23-saynt/network-2-8-20",
            "models/archive/cav23-saynt/network-3-8-20",
            "models/archive/cav23-saynt/refuel-06",
            "models/archive/cav23-saynt/refuel-08",
            "models/archive/cav23-saynt/rocks-12",
            "models/archive/cav23-saynt/rocks-16",
            "models/archive/cav23-saynt/lanes-100-combined-new",
        ]

        rows = []
        for project_path in sayntProjects:
            project_name = os.path.basename(project_path)
            results_path = os.path.join(
                self.saynt_output_dir, project_name, "decision_trees", "results.json"
            )
            if not os.path.exists(results_path):
                print(f"Skipping {project_name}: results.json not found")
                continue

            with open(results_path) as f:
                r = json.load(f)

            rows.append({
                "benchmark":                  project_name,
                # synthesis value
                "storm_value":                r.get("storm_value"),
                # Storm FSC
                "storm_nodes":                r.get("storm_num_nodes"),
                "storm_fsc_size":             r.get("storm_fsc_size"),
                "storm_verified_value":       r.get("storm_verified_value"),
                # PT minimized FSC
                "minimized_nodes":            r.get("minimized_num_nodes"),
                "minimized_fsc_size":         r.get("minimized_fsc_size"),
                "minimized_time_s":           r.get("minimized_time_s"),
                "minimized_verified_value":   r.get("minimized_verified_value"),
                # Wildcard minimized FSC
                "minimized_wc_nodes":         r.get("minimized_wc_num_nodes"),
                "minimized_wc_fsc_size":      r.get("minimized_wc_fsc_size"),
                "minimized_wc_time_s":        r.get("minimized_wc_time_s"),
                "minimized_wc_verified_value":r.get("minimized_wc_verified_value"),
                # WC DT conversion (populated by run_wc_dt_conversion.py)
                "wc_dt_nodes":                r.get("wc_dt_nodes"),
                "wc_dt_conversion_time_s":    r.get("wc_dt_conversion_time_s"),
                "wc_total_time_s":            r.get("wc_total_time_s"),
                # PAYNT FSC
                "paynt_value":                    r.get("paynt_value"),
                "paynt_fsc_size":                 r.get("paynt_fsc_size"),
                "paynt_dt_nodes":                 r.get("paynt_dt_nodes"),
                "paynt_dt_conversion_time_s":     r.get("paynt_dt_conversion_time_s"),
                "paynt_dt_two_tree_nodes":        r.get("paynt_dt_two_tree_nodes"),
                "paynt_dt_two_tree_time_s":       r.get("paynt_dt_two_tree_time_s"),
            })

        table_path = os.path.join(self.saynt_output_dir, "table_data.json")
        with open(table_path, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nTable data written to {table_path}")

    pass


if __name__ == "__main__":
    unittest.main()
