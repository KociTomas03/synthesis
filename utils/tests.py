import os
import unittest
import subprocess
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestPayntRun(unittest.TestCase):

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
        cls.saynt_output_dir = "test_outputs_saynt"
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

    # each fsc size given and ran differently
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

    # TODO: add tests for iterative storm option (15 mins per problem)
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

    # TODO: add SAYNT tests (15 mins per problem)
    def test_saynt_run(self):
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
            if os.path.exists(output_file):
                print(
                    f"Skipping existing result for project {project_path}, STORM_ITERATIVE, memory_constraint {memory_constraint}"
                )
                continue

            print(
                f"Starting paynt_run test for project {project_path}, STORM_ITERATIVE, memory_constraint {memory_constraint}"
            )
            # Run the paynt.py script as a separate process and capture its output
            command = f"python3 paynt.py {project_path} --fsc-synthesis --storm-pomdp --iterative-storm 900 60 10 --generated-fsc-route {outputFolder}/image --memory-constraint {memory_constraint} --export-fsc-storm {outputFolder}/storm --export-fsc-paynt {outputFolder}/paynt --export-generated-dt-fsc {outputFolder}/decisionTree > {output_file} 2>&1"
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
            with open(output_file, "r") as f:
                print("Output file content:")
                print(f.read())

    pass


if __name__ == "__main__":
    unittest.main()
