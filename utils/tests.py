import os
import unittest
import subprocess
import sys

# Add the project directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
            "models/archive/cav23-saynt/refuel-08",
            "models/archive/cav23-saynt/query-s2",
            "models/archive/cav23-saynt/network",
        ]
        cls.output_dir = "test_outputs"
        cls.memory_constraints = ["onestep", "bothway", "circular", "binaryTree", "binaryTreeSelfLoop", "binaryTreeCyclic", "growing", "growingMax2", "notDecreasing", "notDecreasingMax2",
"notDecreasingCyclic", "evenUpOddDown", "bothWayCircleSelfLoop", "none"]
        cls.fsc_sizes = range(2, 8)

        # Create output directory if it doesn't exist
        if not os.path.exists(cls.output_dir):
            os.makedirs(cls.output_dir)

    def test_paynt_run(self):
        for project_path in self.projects:
            project_name = os.path.basename(project_path).replace('/', '_')
            for fsc_size in self.fsc_sizes:
                for memory_constraint in self.memory_constraints:
                    # Create the directory structure: problemName/fscSize/memoryConstraint
                    outputFolder = os.path.join(self.output_dir, project_name, f"fsc_size_{fsc_size}", memory_constraint)
                    output_file = os.path.join(outputFolder, "output.txt")
                    output_dir = os.path.dirname(output_file)
                    
                    # Create the directory structure if it doesn't exist
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Skip the test if the output file already exists
                    if os.path.exists(output_file):
                        print(f"Skipping existing result for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}")
                        continue
                    
                    print(f"Starting paynt_run test for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}")
                    # Run the paynt.py script as a separate process and capture its output
                    command = f'python3 paynt.py {project_path} --fsc-memory-size {fsc_size} --generated-fsc-route {outputFolder}/image --memory-constraint {memory_constraint} > {output_file} 2>&1'
                    result = subprocess.run(command, shell=True)
                    
                    # Check the exit code
                    if result.returncode != 0:
                        with open(output_file, 'r') as f:
                            print("Output:")
                            print(f.read())
                    
                    self.assertEqual(result.returncode, 0, f"Failed to run paynt.py with the specified arguments for project {project_path}, fsc_size {fsc_size}, memory_constraint {memory_constraint}")
                    
                    print("Finished paynt_run test")
                
                    # Print the content of the output file for debugging
                    with open(output_file, 'r') as f:
                        print("Output file content:")
                        print(f.read())

if __name__ == '__main__':
    unittest.main()