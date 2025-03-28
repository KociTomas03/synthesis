import os
import pickle
import glob
import traceback

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

    # Sort by file size
    pickle_files.sort(key=lambda x: os.path.getsize(x))
    return pickle_files


def get_minimized_fsc_path(original_path):
    """Get the path to the minimized FSC based on the original FSC path"""
    return original_path.replace(".pkl", "_minimized.pkl")


def load_fsc_pair(fsc_path):
    """Load an original FSC and its corresponding minimized version"""
    try:
        # Load the original FSC
        print(f"\nLoading original FSC from: {fsc_path}")

        with open(fsc_path, "rb") as f:
            original_fsc = pickle.load(f)

        if original_fsc is None:
            print("Error: Failed to load original FSC (None returned)")
            return None, None

        print(
            f"Original FSC loaded: {original_fsc.num_nodes} nodes, {original_fsc.num_observations} observations"
        )
        print(f"Deterministic: {original_fsc.is_deterministic}")

        # Look for minimized FSC
        pre_minimized_fsc = None
        pre_min_path = get_minimized_fsc_path(fsc_path)

        if os.path.exists(pre_min_path):
            print(f"Loading minimized FSC from: {pre_min_path}")
            try:
                with open(pre_min_path, "rb") as f:
                    pre_minimized_fsc = pickle.load(f)
                print(f"Minimized FSC loaded: {pre_minimized_fsc.num_nodes} nodes")

                # Calculate reduction
                reduction = (
                    1 - pre_minimized_fsc.num_nodes / original_fsc.num_nodes
                ) * 100
                print(f"Size reduction: {reduction:.2f}%")

            except Exception as e:
                print(f"Error loading minimized FSC: {str(e)}")
        else:
            print(f"No minimized FSC found at: {pre_min_path}")

        return original_fsc, pre_minimized_fsc

    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return None, None


class FSCLoader:
    def __init__(self):
        self.fsc_paths = find_all_saynt_fscs()
        self.current_index = 0
        self.fsc_data = {}  # Cache for loaded FSC data

    def get_current_path(self):
        return self.fsc_paths[self.current_index]

    def get_current_basename(self):
        """Get the benchmark name from the current path"""
        path = self.get_current_path()
        # Extract benchmark name from path
        parts = path.split(os.sep)
        if "decision_trees" in parts:
            # Format: base_dir/benchmark/decision_trees/SAYNT/fsc.pkl
            idx = parts.index("decision_trees")
            return parts[idx - 1]
        else:
            # Format: base_dir/benchmark/SAYNT/fsc.pkl
            return os.path.basename(os.path.dirname(os.path.dirname(path)))

    def load_current_pair(self, force_reload=False):
        """Load the current FSC pair"""
        path = self.get_current_path()
        if path not in self.fsc_data or force_reload:
            self.fsc_data[path] = load_fsc_pair(path)
        return self.fsc_data[path]

    def next_pair(self):
        """Move to the next FSC pair"""
        if self.current_index < len(self.fsc_paths) - 1:
            self.current_index += 1
            return self.load_current_pair()
        else:
            print("Already at the last FSC pair")
            return self.load_current_pair()

    def prev_pair(self):
        """Move to the previous FSC pair"""
        if self.current_index > 0:
            self.current_index -= 1
            return self.load_current_pair()
        else:
            print("Already at the first FSC pair")
            return self.load_current_pair()

    def reload_current(self):
        """Reload the current FSC pair"""
        return self.load_current_pair(force_reload=True)


def main():
    print("Loading SAYNT FSC pairs (original and minimized)...")

    try:
        loader = FSCLoader()
        print(f"Found {len(loader.fsc_paths)} SAYNT FSC files to analyze")

        # Load the first pair
        original_fsc, minimized_fsc = loader.load_current_pair()
        benchmark = loader.get_current_basename()

        print(
            f"\n--- Loaded {benchmark} ({loader.current_index+1}/{len(loader.fsc_paths)}) ---"
        )

        # Return the loader and the FSCs - these will be available in the interactive session
        # or when debugging with VSCode's debugger
        return loader, original_fsc, minimized_fsc

    except Exception as e:
        print(f"Error loading FSCs: {str(e)}")
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    loader, original_fsc, minimized_fsc = main()
    while 1:
        original_fsc, minimized_fsc = loader.next_pair()
        continue
