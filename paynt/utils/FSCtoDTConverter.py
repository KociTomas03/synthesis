import pandas as pd
import re
import subprocess
import os
import json


class FSCtoDTConverter:
    def __init__(
        self,
        fsc_output,
        dtcontrol_path="dtcontrol",
        output_dir="decision_trees",
        combined_mode=False,
    ):
        """
        Initializes the converter with FSC output as a list of strings.

        Args:
            fsc_output: FSC output as a string
            dtcontrol_path: Path to dtcontrol executable
            output_dir: Directory to save generated files
            combined_mode: If True, create a single decision tree with two outputs (action and next_memory)
        """
        self.output_dir = output_dir
        self.fsc_output = fsc_output.split(", ")
        self.dtcontrol_path = dtcontrol_path
        self.combined_mode = combined_mode
        self.action_data = []  # To store (observations, m, A) tuples
        self.memory_data = []  # To store (observations, m, M') tuples
        self.observation_keys = set()  # To store all unique observation keys
        self.parse_fsc_output()

    def parse_fsc_output(self):
        """
        Parses the FSC output to extract action and memory transitions.
        Each line is wrapped in quotes and contains variable assignments.
        """
        action_pattern = re.compile(r"A\((.+?),(\d+)\)=(\w+)")
        memory_pattern = re.compile(r"M\((.+?),(\d+)\)=(\d+)")
        # Updated pattern to only match valid variable names (must start with a letter)
        var_pattern = re.compile(
            r"([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)=(\w+)|([a-zA-Z][\w\d]*|![a-zA-Z][\w\d]*)"
        )

        for line in self.fsc_output:
            # Remove quotes and whitespace
            line = line.strip(' "')

            action_match = action_pattern.match(line)
            memory_match = memory_pattern.match(line)

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
        # return pd.merge(action_df, memory_df, on=merge_columns, how="inner")

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

    def run_dtcontrol(self):
        """
        Runs dtControl on the generated CSV files for each memory value.
        """
        # Get unique memory values from both action and memory data
        action_df = self.get_action_dataframe()
        memory_df = self.get_memory_dataframe()
        memory_values = sorted(
            set(action_df["memory"].unique()) | set(memory_df["memory"].unique())
        )

        benchmark_file = os.path.join(self.output_dir, "benchmark.json")
        output_dir = os.path.join(self.output_dir)

        for memory_value in memory_values:
            try:
                if self.combined_mode:
                    combined_file, _ = self.save_csv_files(memory_value)
                    print(f"\nProcessing combined tree for memory value {memory_value}")
                    self._run_dtcontrol_process(
                        combined_file, output_dir, benchmark_file
                    )
                else:
                    action_file, memory_file = self.save_csv_files(memory_value)
                    print(f"\nProcessing trees for memory value {memory_value}")

                    # Action tree
                    self._run_dtcontrol_process(action_file, output_dir, benchmark_file)

                    # Memory tree
                    self._run_dtcontrol_process(memory_file, output_dir, benchmark_file)

            except subprocess.CalledProcessError as e:
                print(f"dtControl failed for memory {memory_value} with error: {e}")
                continue

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
    def process_existing_outputs(base_dir, combined_mode=False):
        """
        Processes existing FSC outputs in the specified base directory.

        Args:
            base_dir: Base directory containing the FSC outputs
            combined_mode: If True, create combined decision trees
        """
        for root, dirs, files in os.walk(base_dir):
            for file in files:
                if file == "paynt.fsc":
                    with open(os.path.join(root, file), "r") as f:
                        fsc_output = f.read()
                        output_dir = os.path.join(
                            os.path.dirname(root), "decision_trees_combined"
                        )
                        converter = FSCtoDTConverter(
                            fsc_output,
                            output_dir=output_dir,
                            combined_mode=combined_mode,
                        )
                        converter.run_dtcontrol()


if __name__ == "__main__":
    # Example usage:
    # base_dir = "test_silly_tree"
    # test = ""
    # converter = FSCtoDTConverter(test, output_dir=base_dir, combined_mode=False)
    # converter.run_dtcontrol()
    base_dir = "test_outpus_saynt"
    FSCtoDTConverter.process_existing_outputs(base_dir, True)
