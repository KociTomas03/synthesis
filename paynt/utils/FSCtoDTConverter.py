from pathlib import Path
import sys
import pandas as pd
import re
import subprocess
import os
import json
import signal

project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)
from paynt.quotient.pomdp import PomdpQuotient
from paynt.quotient.fsc import FscFactored


class FSCtoDTConverter:
    _ACTION_PATTERN = re.compile(r"A\((.+?),(\d+)\)=(\w+)")
    _MEMORY_PATTERN = re.compile(r"M\((.+?),(\d+)\)=(\d+)")

    def __init__(
        self,
        fsc_input,
        dtcontrol_path="dtcontrol",
        output_dir="decision_trees",
        combined_mode=True,
        is_storm=False,
    ):
        """
        Initializes the converter with FSC input either as a string or FSC object.

        Args:
            fsc_input: FSC output as a string OR FSC object
            dtcontrol_path: Path to dtcontrol executable
            output_dir: Directory to save generated files
            combined_mode: If True, create a single decision tree with two outputs (action and next_memory)
            is_storm: If True, parse as a Storm belief-policy FSC (uses _parse_storm_format
                and storm-specific observation/action/memory label handling); if False,
                parse as a PAYNT FSC object (uses _parse_paynt_format)
        """
        self.output_dir = output_dir
        self.dtcontrol_path = dtcontrol_path
        self.combined_mode = combined_mode
        self.is_storm = is_storm
        self.action_data = []  # To store (observations, m, A) tuples
        self.memory_data = []  # To store (observations, m, M') tuples
        self.observation_keys = set()  # To store all unique observation keys

        # Parse the input based on its type
        if isinstance(fsc_input, FscFactored):
            self.parse_fsc_object(fsc_input)
        else:
            # Assume it's a string
            self.fsc_output = fsc_input.split(", ")
            self.parse_fsc_output()

    def parse_fsc_object(self, fsc):
        """
        Parse FSC object directly instead of string output.

        Args:
            fsc: FSC object instance
        """
        # Process action function: NxZ -> Act
        for node in range(fsc.num_nodes):
            for obs in range(fsc.num_observations):
                # Get action for this node and observation
                if self.is_storm:
                    # Handle Storm FSC format
                    self._handle_storm_fsc_entry(fsc, node, obs)
                elif fsc.is_deterministic:
                    # Handle deterministic PAYNT FSC
                    action = fsc.action_function[node][obs]
                    if fsc.action_labels and action is not None:
                        action = fsc.action_labels[action]
                    if action is not None:
                        # Get observation variables
                        obs_vars = self._get_observation_variables(fsc, obs)
                        self.observation_keys.update(obs_vars.keys())
                        self.action_data.append((obs_vars, node, str(action)))
                else:
                    # Handle stochastic PAYNT FSC
                    action_dict = fsc.action_function[node][obs]
                    if action_dict:  # Check if there are any actions defined
                        # Check if there's just one action with probability 1.0
                        if (
                            len(action_dict) == 1
                            and list(action_dict.values())[0] == 1.0
                        ):
                            # Single deterministic action with p=1.0
                            action = list(action_dict.keys())[0]
                            if fsc.action_labels:
                                action = fsc.action_labels[action]
                            # Get observation variables
                            obs_vars = self._get_observation_variables(fsc, obs)
                            self.observation_keys.update(obs_vars.keys())
                            self.action_data.append((obs_vars, node, str(action)))
                        else:
                            # Multiple actions with probabilities - preserve the distribution as string
                            obs_vars = self._get_observation_variables(fsc, obs)
                            self.observation_keys.update(obs_vars.keys())

                            # Add probability as an observation variable
                            labeled_dict = {}
                            for action_idx, prob in action_dict.items():
                                action = action_idx
                                if fsc.action_labels:
                                    action = fsc.action_labels[action_idx]
                                labeled_dict[str(action)] = prob

                            # Convert to sorted string representation
                            sorted_dict_str = self._dict_to_sorted_str(labeled_dict)
                            self.action_data.append((obs_vars, node, sorted_dict_str))

                # Get next memory node for this node and observation
                if not self.is_storm:  # PAYNT FSC memory update
                    next_node = fsc.update_function[node][obs]
                    if next_node is not None:
                        # Get observation variables
                        obs_vars = self._get_observation_variables(fsc, obs)
                        self.observation_keys.update(obs_vars.keys())

                        # Check if next_node is a dictionary (probabilistic)
                        if isinstance(next_node, dict):
                            # Convert memory distribution to sorted string
                            sorted_dict_str = self._dict_to_sorted_str(next_node)
                            self.memory_data.append((obs_vars, node, sorted_dict_str))
                        else:
                            self.memory_data.append((obs_vars, node, next_node))

    def _handle_storm_fsc_entry(self, fsc, node, obs):
        """
        Handle Storm FSC format for a specific node and observation.

        Args:
            fsc: FSC object
            node: Memory node index
            obs: Observation index
        """
        # Get observation variables
        obs_vars = self._get_observation_variables(fsc, obs)
        self.observation_keys.update(obs_vars.keys())

        try:
            # Try to access action - might be direct index or dict with distribution
            action_value = fsc.action_function[node][obs]

            if action_value is None:
                return

            if isinstance(action_value, dict):
                # Handle probabilistic actions with proper labels
                # First check if it's just a single action with probability 1.0
                if len(action_value) == 1 and list(action_value.values())[0] == 1.0:
                    # Single deterministic action with p=1.0
                    action_idx = list(action_value.keys())[0]
                    action = action_idx
                    if fsc.action_labels and action_idx is not None:
                        action = fsc.action_labels[action_idx]
                    self.action_data.append((obs_vars, node, str(action)))
                else:
                    # Convert dict to use action labels as keys
                    labeled_dict = {}
                    for action_idx, prob in action_value.items():
                        action = action_idx
                        if fsc.action_labels and action_idx is not None:
                            action = fsc.action_labels[action_idx]
                        labeled_dict[str(action)] = prob

                    # Convert to sorted string representation
                    sorted_dict_str = self._dict_to_sorted_str(labeled_dict)
                    self.action_data.append((obs_vars, node, sorted_dict_str))
            else:
                # Handle direct action index
                action = action_value
                if fsc.action_labels and action is not None:
                    action = fsc.action_labels[action]

                self.action_data.append((obs_vars, node, str(action)))

            # Handle memory updates for Storm
            if hasattr(fsc, "update_function") and len(fsc.update_function) > node:
                next_node = fsc.update_function[node][obs]
                if next_node is not None:
                    if isinstance(next_node, dict):
                        # Check for single memory state with probability 1.0
                        if len(next_node) == 1 and list(next_node.values())[0] == 1.0:
                            # Use the single memory state directly
                            next_memory = list(next_node.keys())[0]
                            self.memory_data.append((obs_vars, node, next_memory))
                        else:
                            # Convert memory distribution to sorted string
                            sorted_dict_str = self._dict_to_sorted_str(next_node)
                            self.memory_data.append((obs_vars, node, sorted_dict_str))
                    else:
                        self.memory_data.append((obs_vars, node, next_node))

        except (IndexError, KeyError, TypeError) as e:
            print(f"Error processing Storm FSC at node {node}, obs {obs}: {e}")

    def _dict_to_sorted_str(self, dict_value):
        """Convert a dictionary to a sorted string representation."""
        sorted_items = sorted(dict_value.items(), key=lambda x: str(x[0]))
        return "{" + ", ".join(f"{k}: {v}" for k, v in sorted_items) + "}"

    def _get_observation_variables(self, fsc, obs):
        """Helper to extract observation variables consistently."""
        obs_vars = {}
        if fsc.observation_labels and obs < len(fsc.observation_labels):
            obs_label = fsc.observation_labels[obs]
            if isinstance(obs_label, dict):
                # Already in the right format
                obs_vars = obs_label
            elif isinstance(obs_label, str):
                # Parse from string format using the same logic as FSC output
                var_pattern = re.compile(
                    r"([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)=(\w+)|([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)"
                )
                obs_vars = self._parse_variables(obs_label, var_pattern)
        else:
            # If no labels, create a default observation variable
            obs_vars = {"observation": obs}
        return obs_vars

    def parse_fsc_output(self):
        """
        Parses the FSC output to extract action and memory transitions.
        Each line is wrapped in quotes and contains variable assignments.
        """
        # Different parsing approaches for STORM vs PAYNT
        if self.is_storm:
            self._parse_storm_format()
        else:
            self._parse_paynt_format()

    def _parse_storm_format(self):
        """
        Parse FSC output in STORM format
        """
        # Implement STORM-specific parsing logic
        # This is a placeholder and will need to be adapted to STORM's format
        var_pattern = re.compile(
            r"([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)=(\w+)|([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)"
        )

        for line in self.fsc_output:
            # Remove quotes and whitespace
            line = line.strip(' "')

            # Storm format specific parsing logic here
            # This is just a placeholder - adjust to match Storm's actual format
            action_match = self._ACTION_PATTERN.match(line)
            memory_match = self._MEMORY_PATTERN.match(line)

            if action_match:
                vars_str, m, action = action_match.groups()
                vars_dict = self._parse_variables(vars_str, var_pattern)
                self.observation_keys.update(vars_dict.keys())
                self.action_data.append((vars_dict, int(m), action))

            if memory_match:
                vars_str, m, new_m = memory_match.groups()
                vars_dict = self._parse_variables(vars_str, var_pattern)
                self.observation_keys.update(vars_dict.keys())
                self.memory_data.append((vars_dict, int(m), int(new_m)))

    def _parse_paynt_format(self):
        """
        Parse FSC output in PAYNT format
        """
        # Updated pattern to only match valid variable names (must start with a letter)
        var_pattern = re.compile(
            r"([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)=(\w+)|([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)"
        )

        for line in self.fsc_output:
            # Remove quotes and whitespace
            line = line.strip(' "')

            action_match = self._ACTION_PATTERN.match(line)
            memory_match = self._MEMORY_PATTERN.match(line)

            if action_match:
                vars_str, m, action = action_match.groups()
                vars_dict = self._parse_variables(vars_str, var_pattern)
                self.observation_keys.update(vars_dict.keys())
                self.action_data.append((vars_dict, int(m), action))

            if memory_match:
                vars_str, m, new_m = memory_match.groups()
                vars_dict = self._parse_variables(vars_str, var_pattern)
                self.observation_keys.update(vars_dict.keys())
                self.memory_data.append((vars_dict, int(m), int(new_m)))

    def _parse_variables(self, vars_str, var_pattern):
        """Helper method to parse variables from a string using regex pattern."""
        vars_dict = {}
        for match in var_pattern.findall(vars_str):
            if match[2]:  # Boolean flag
                if match[2].startswith("!"):
                    vars_dict[match[2][1:]] = 0
                else:
                    vars_dict[match[2]] = 1
            else:
                vars_dict[match[0]] = match[1]
        return vars_dict

    def get_action_dataframe(self):
        """
        Returns a DataFrame for action decision trees.
        """
        return self._create_dataframe(
            self.action_data, ["observations", "memory", "action"]
        )

    def get_memory_dataframe(self):
        """
        Returns a DataFrame for memory transition decision trees.
        """
        return self._create_dataframe(
            self.memory_data, ["observations", "memory", "next_memory"]
        )

    def _create_dataframe(self, data, columns):
        """Helper method to create a DataFrame from tuples with observation dictionaries."""
        df = pd.DataFrame(data, columns=columns)
        for key in self.observation_keys:
            df[key] = df["observations"].apply(lambda x: x.get(key, 0))
        df.drop(columns=["observations"], inplace=True)
        return df

    def get_combined_dataframe(self):
        """
        Returns a DataFrame for combined decision trees with two target columns (action and next_memory).
        """
        # Get individual dataframes
        action_df = self.get_action_dataframe()
        memory_df = self.get_memory_dataframe()

        # Merge the two dataframes based on observation variables and memory state
        merge_columns = list(self.observation_keys) + ["memory"]
        combined_df = pd.merge(action_df, memory_df, on=merge_columns, how="outer")
        combined_df["action"].fillna("-", inplace=True)
        combined_df["next_memory"].fillna("-", inplace=True)
        return combined_df

    def save_csv_files(self, memory_value):
        """
        Saves DataFrames as CSV files and their metadata configs for a specific memory value.
        Returns paths to the created files.
        """
        # Create directory structure
        memory_dir = os.path.join(self.output_dir, f"memory_{memory_value}")
        os.makedirs(memory_dir, exist_ok=True)

        action_file = os.path.join(memory_dir, f"{memory_value}_action_data.csv")
        memory_file = os.path.join(memory_dir, f"{memory_value}_memory_data.csv")
        combined_file = os.path.join(memory_dir, f"{memory_value}_combined_data.csv")

        # Filter data for specific memory value
        action_df = self.get_action_dataframe()
        memory_df = self.get_memory_dataframe()
        action_df_filtered = action_df[action_df["memory"] == memory_value]
        memory_df_filtered = memory_df[memory_df["memory"] == memory_value]

        # Process data based on mode
        if self.combined_mode:
            return self._save_combined_data(memory_value, memory_dir, combined_file)
        else:
            return self._save_separate_data(
                memory_value,
                memory_dir,
                action_file,
                memory_file,
                action_df_filtered,
                memory_df_filtered,
            )

    def _save_combined_data(self, memory_value, memory_dir, combined_file):
        """Helper method to save combined action and memory data."""
        combined_df = self.get_combined_dataframe()
        combined_df_filtered = combined_df[combined_df["memory"] == memory_value]

        # Number of input variables is the number of unique observation keys
        num_inputs = len(self.observation_keys)
        x_column_names = sorted(list(self.observation_keys))

        # Ensure consistent column order
        combined_df_filtered = combined_df_filtered[
            ["memory"] + x_column_names + ["action", "next_memory"]
        ]

        # Create mappings for categorical values
        action_values = sorted(combined_df_filtered["action"].unique())
        action_mapping = {val: idx for idx, val in enumerate(action_values)}

        next_memory_values = sorted(
            [str(x) for x in combined_df_filtered["next_memory"].unique()]
        )
        next_memory_mapping = {val: idx for idx, val in enumerate(next_memory_values)}

        # Apply mappings
        combined_df_filtered_mapped = combined_df_filtered.copy()
        combined_df_filtered_mapped["action"] = combined_df_filtered_mapped[
            "action"
        ].map(action_mapping)
        combined_df_filtered_mapped["next_memory"] = (
            combined_df_filtered_mapped["next_memory"]
            .astype(str)
            .map(next_memory_mapping)
        )

        # Create config
        x_category_names = {}
        x_column_types = {"numeric": [], "categorical": []}

        # Process input features
        for idx, col in enumerate(x_column_names):
            unique_values = sorted(str(x) for x in combined_df_filtered[col].unique())
            self._process_column_type(
                idx, col, unique_values, x_column_types, x_category_names
            )

        # Process output columns
        y_column_types = {"numeric": [], "categorical": [0, 1]}
        y_category_names = {"0": action_values, "1": next_memory_values}

        combined_config = {
            "x_column_types": x_column_types,
            "y_column_types": y_column_types,
            "x_column_names": x_column_names,
            "y_column_names": ["action", "next_memory"],
            "x_category_names": x_category_names,
            "y_category_names": y_category_names,
        }

        # Save CSV and config
        self._write_csv_file(
            combined_file,
            combined_df_filtered_mapped,
            num_inputs,
            2,  # 2 output columns
            x_column_names + ["action", "next_memory"],  # Use consistent column order
        )

        combined_config_path = os.path.join(
            memory_dir, f"{memory_value}_combined_data_config.json"
        )
        with open(combined_config_path, "w") as f:
            json.dump(combined_config, f, indent=2)

        return combined_file, None

    def _save_separate_data(
        self,
        memory_value,
        memory_dir,
        action_file,
        memory_file,
        action_df_filtered,
        memory_df_filtered,
    ):
        """Helper method to save separate action and memory data."""
        num_inputs = len(self.observation_keys)
        x_column_names = sorted(list(self.observation_keys))

        # Ensure consistent column order by reordering dataframe columns
        action_df_filtered = action_df_filtered[
            ["memory"] + x_column_names + ["action"]
        ]
        memory_df_filtered = memory_df_filtered[
            ["memory"] + x_column_names + ["next_memory"]
        ]

        # Process action dataframe
        action_values = sorted(action_df_filtered["action"].unique())
        action_mapping = {val: idx for idx, val in enumerate(action_values)}
        action_df_filtered_mapped = action_df_filtered.copy()
        action_df_filtered_mapped["action"] = action_df_filtered_mapped["action"].map(
            action_mapping
        )

        # Create action config
        action_x_category_names = {}
        action_x_column_types = {"numeric": [], "categorical": []}

        for idx, col in enumerate(x_column_names):
            unique_values = sorted(str(x) for x in action_df_filtered[col].unique())
            self._process_column_type(
                idx, col, unique_values, action_x_column_types, action_x_category_names
            )

        action_config = {
            "x_column_types": action_x_column_types,
            "y_column_types": {"numeric": [], "categorical": [0]},
            "x_column_names": x_column_names,
            "y_column_names": ["action"],
            "x_category_names": action_x_category_names,
            "y_category_names": {"0": action_values},
        }

        # Save action data
        self._write_csv_file(
            action_file,
            action_df_filtered_mapped,
            num_inputs,
            1,
            x_column_names + ["action"],  # Use consistent column order
        )

        action_config_path = os.path.join(
            memory_dir, f"{memory_value}_action_data_config.json"
        )
        with open(action_config_path, "w") as f:
            json.dump(action_config, f, indent=2)

        # Process memory dataframe
        next_memory_values = sorted(
            [str(x) for x in memory_df_filtered["next_memory"].unique()]
        )
        next_memory_mapping = {val: idx for idx, val in enumerate(next_memory_values)}
        memory_df_filtered_mapped = memory_df_filtered.copy()
        memory_df_filtered_mapped["next_memory"] = (
            memory_df_filtered_mapped["next_memory"]
            .astype(str)
            .map(next_memory_mapping)
        )

        # Create memory config
        memory_x_category_names = {}
        memory_x_column_types = {"numeric": [], "categorical": []}

        for idx, col in enumerate(x_column_names):
            unique_values = sorted(str(x) for x in memory_df_filtered[col].unique())
            self._process_column_type(
                idx, col, unique_values, memory_x_column_types, memory_x_category_names
            )

        memory_config = {
            "x_column_types": memory_x_column_types,
            "y_column_types": {"numeric": [], "categorical": [0]},
            "x_column_names": x_column_names,
            "y_column_names": ["next_memory"],
            "x_category_names": memory_x_category_names,
            "y_category_names": {"0": next_memory_values},
        }

        # Save memory data - FIX: Use x_column_names instead of list(self.observation_keys)
        self._write_csv_file(
            memory_file,
            memory_df_filtered_mapped,
            num_inputs,
            1,
            x_column_names
            + ["next_memory"],  # Use the same sorted column order as the config
        )

        memory_config_path = os.path.join(
            memory_dir, f"{memory_value}_memory_data_config.json"
        )
        with open(memory_config_path, "w") as f:
            json.dump(memory_config, f, indent=2)

        return action_file, memory_file

    def _process_column_type(
        self, idx, col, unique_values, column_types, category_names
    ):
        """Helper method to determine column type and set category names if applicable."""
        if len(unique_values) <= 2 and all(x in ["0", "1"] for x in unique_values):
            # Binary feature
            column_types["categorical"].append(idx)
            category_names[str(idx)] = ["False", "True"]
        else:
            try:
                # Try to convert all values to float
                [float(x) for x in unique_values]
                column_types["numeric"].append(idx)
            except ValueError:
                # Non-numeric values, treat as categorical
                column_types["categorical"].append(idx)
                category_names[str(idx)] = unique_values

    def _write_csv_file(self, file_path, dataframe, num_inputs, num_outputs, columns):
        """Helper method to write data to a CSV file with the dtcontrol header."""
        with open(file_path, "w") as f:
            f.write("#NON-PERMISSIVE\n")
            f.write(f"#BEGIN {num_inputs} {num_outputs}\n")
            dataframe.drop(columns=["memory"]).to_csv(
                f,
                index=False,
                header=False,
                columns=columns,
            )

    def _process_memory_state(
        self, memory_value, output_dir, shared_benchmark_file=None
    ):
        """
        Process a single memory state, using a shared benchmark file across all memories.

        Args:
            memory_value: Memory state value to process
            output_dir: Output directory for decision trees
            shared_benchmark_file: Path to a shared benchmark file for all memories
        """
        try:
            # Create memory-specific directory
            memory_dir = os.path.join(output_dir, f"memory_{memory_value}")
            os.makedirs(memory_dir, exist_ok=True)

            # Use the shared benchmark file if provided, otherwise create memory-specific one
            benchmark_file = (
                shared_benchmark_file
                if shared_benchmark_file
                else os.path.join(memory_dir, f"benchmark.json")
            )

            if self.combined_mode:
                combined_file, _ = self.save_csv_files(memory_value)
                print(f"\nProcessing combined tree for memory value {memory_value}")
                self._run_dtcontrol_process(combined_file, output_dir, benchmark_file)
            else:
                action_file, memory_file = self.save_csv_files(memory_value)
                print(f"\nProcessing trees for memory value {memory_value}")

                # Action tree
                self._run_dtcontrol_process(action_file, output_dir, benchmark_file)

                # Memory tree
                self._run_dtcontrol_process(memory_file, output_dir, benchmark_file)

            return memory_value, True

        except subprocess.CalledProcessError as e:
            print(f"dtControl failed for memory {memory_value} with error: {e}")
            return memory_value, False

    def run_dtcontrol(self):
        """Runs dtControl on the generated CSV files for each memory value."""
        # Get unique memory values from both action and memory data
        action_df = self.get_action_dataframe()
        memory_df = self.get_memory_dataframe()
        memory_values = sorted(
            set(action_df["memory"].unique()) | set(memory_df["memory"].unique())
        )

        total_states = len(memory_values)
        print(f"Processing {total_states} memory states")

        output_dir = os.path.join(self.output_dir)

        # Create a shared benchmark file at the parent level
        shared_benchmark_file = os.path.join(output_dir, "benchmark.json")

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for memory_value in memory_values:
            self._process_memory_state(
                memory_value,
                output_dir,
                shared_benchmark_file=shared_benchmark_file,
            )

    def _run_dtcontrol_process(self, input_file, output_dir, benchmark_file):
        """Helper method to run dtcontrol with consistent parameters."""
        subprocess.run(
            [
                self.dtcontrol_path,
                "--input",
                input_file,
                "--use-preset",
                "default",
                "--output",
                output_dir,
                "--benchmark-file",
                benchmark_file,
                "--rerun",
            ],
            check=True,
        )

    @staticmethod
    def count_dt_nodes(benchmark_file_path):
        """
        Read a dtcontrol benchmark.json and return the total node count across all trees.
        Returns None if the file does not exist or cannot be parsed.
        """
        import json
        try:
            with open(benchmark_file_path) as f:
                data = json.load(f)
            total = 0
            for entry in data.values():
                total += entry["classifiers"]["default"]["stats"]["nodes"]
            return total
        except Exception:
            return None

    @staticmethod
    def process_existing_outputs(base_dir, combined_mode=False):
        """
        Processes existing FSC outputs in the specified base directory.

        Args:
            base_dir: Base directory containing the FSC outputs
            combined_mode: If True, create combined decision trees
        """
        fsc_files = []

        # Find all FSC files
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == "paynt.fsc" or file == "storm.fsc":
                    fsc_files.append((os.path.join(root, file), file == "storm.fsc"))

        print(f"Found {len(fsc_files)} FSC files to process")

        # Process each FSC file
        for fsc_path, is_storm in fsc_files:
            try:
                with open(fsc_path, "r") as f:
                    fsc_output = f.read()

                # Create output directory (next to the FSC directory)
                parent_dir = os.path.dirname(os.path.dirname(fsc_path))
                output_dir = os.path.join(
                    parent_dir,
                    "decision_trees_combined" if combined_mode else "decision_trees",
                )

                print(
                    f"Processing {'Storm' if is_storm else 'PAYNT'} FSC: {fsc_path} -> {output_dir}"
                )

                converter = FSCtoDTConverter(
                    fsc_output,
                    output_dir=output_dir,
                    combined_mode=combined_mode,
                    is_storm=is_storm,
                )

                converter.run_dtcontrol()

            except Exception as e:
                print(f"Error processing {fsc_path}: {e}")
                import traceback

                traceback.print_exc()
