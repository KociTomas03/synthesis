#!/usr/bin/env python3
# filepath: /home/tomas/bakalarka/myFork/synthesis/process_fsc_pickles.py
import os
import pickle
import argparse
import glob
from pathlib import Path
import sys
import re

# Add project root to path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from paynt.utils.FSCtoDTConverter import FSCtoDTConverter


def find_fsc_pickles(base_dir):
    """Find all fsc.pkl files in the directory structure."""
    return glob.glob(os.path.join(base_dir, "**", "fsc.pkl"), recursive=True)


def find_skipped_fscs(base_dir):
    """Find all FSCs that were previously skipped as too large."""
    skipped_files = []
    skipped_marker_files = glob.glob(
        os.path.join(base_dir, "**/skipped_large_fsc.txt"), recursive=True
    )

    for marker_file in skipped_marker_files:
        try:
            with open(marker_file, "r") as f:
                content = f.read()
                # Extract pickle path using regex
                match = re.search(
                    r"Original pickle file: (.+?)$", content, re.MULTILINE
                )
                if match:
                    pickle_path = match.group(1).strip()
                    if os.path.exists(pickle_path):
                        dt_dir = os.path.dirname(marker_file)
                        mode = "combined" if "dt_combined" in dt_dir else "separate"
                        skipped_files.append((pickle_path, mode))
                        print(f"Found skipped FSC: {pickle_path} (mode: {mode})")
        except Exception as e:
            print(f"Error reading marker file {marker_file}: {e}")

    return skipped_files


def process_fsc_pickle(
    pickle_path, combined_mode=True, max_states=None, parallel=False, skip_large=True
):
    """Process a single FSC pickle file."""
    try:
        print(f"Processing FSC pickle: {pickle_path}")

        # Load the FSC from pickle file
        with open(pickle_path, "rb") as f:
            fsc = pickle.load(f)

        # Create output directories
        parent_dir = os.path.dirname(pickle_path)
        output_dir = os.path.join(
            parent_dir, f"dt_{'combined' if combined_mode else 'separate'}"
        )

        # Check if already processed
        if os.path.exists(output_dir):
            # Count memory_X directories to see if this FSC was already processed
            memory_dirs = glob.glob(os.path.join(output_dir, "memory_*"))
            if memory_dirs:
                print(
                    f"Skipping {pickle_path} with {combined_mode=} - already processed ({len(memory_dirs)} memory states found)"
                )
                return True

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        print(
            f"Loaded FSC with {fsc.num_nodes} nodes and {fsc.num_observations} observations"
        )

        # Skip very large FSCs entirely
        if skip_large and fsc.num_nodes > 3000:
            print(f"Skipping {pickle_path} - FSC is too large ({fsc.num_nodes} nodes)")

            # Create a marker file indicating this was intentionally skipped
            with open(os.path.join(output_dir, "skipped_large_fsc.txt"), "w") as f:
                f.write(
                    f"Skipped processing FSC with {fsc.num_nodes} nodes and {fsc.num_observations} observations\n"
                )
                f.write(f"Original pickle file: {pickle_path}\n")
                f.write(f"To process this FSC, run with --process-large flag\n")

            return True

        # Create converter
        converter = FSCtoDTConverter(
            fsc,
            output_dir=output_dir,
            is_storm=True,  # Assume Storm format for pickled FSCs
            combined_mode=combined_mode,
        )

        # Run converter with separate benchmark files for each memory state
        converter.run_dtcontrol(parallel=parallel)

        print(
            f"Successfully processed {pickle_path} with {'combined' if combined_mode else 'separate'} mode"
        )
        return True

    except Exception as e:
        print(
            f"Error processing {pickle_path} with {'combined' if combined_mode else 'separate'} mode: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def process_skipped_fsc_pickle(
    pickle_path, combined_mode=True, max_states=None, parallel=True, skip_large=False
):
    """Process a single previously skipped FSC pickle file."""
    try:
        print(f"Processing previously skipped FSC: {pickle_path}")

        # Load the FSC from pickle file
        with open(pickle_path, "rb") as f:
            fsc = pickle.load(f)

        # Create output directories
        parent_dir = os.path.dirname(os.path.dirname(pickle_path))
        output_dir = os.path.join(
            parent_dir, f"dt_{'combined' if combined_mode else 'separate'}"
        )

        # Remove the skipped marker file if it exists
        skipped_marker = os.path.join(output_dir, "skipped_large_fsc.txt")
        if os.path.exists(skipped_marker):
            os.remove(skipped_marker)

        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        print(
            f"Loaded FSC with {fsc.num_nodes} nodes and {fsc.num_observations} observations"
        )

        # Create converter
        converter = FSCtoDTConverter(
            fsc,
            output_dir=output_dir,
            is_storm=True,  # Assume Storm format for pickled FSCs
            combined_mode=combined_mode,
        )

        # Run converter with separate benchmark files for each memory state

        converter.run_dtcontrol(parallel=parallel)

        print(
            f"Successfully processed {pickle_path} with {'combined' if combined_mode else 'separate'} mode"
        )
        return True

    except Exception as e:
        print(
            f"Error processing {pickle_path} with {'combined' if combined_mode else 'separate'} mode: {e}"
        )
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process FSC pickle files to decision trees"
    )
    parser.add_argument(
        "--base_dir",
        default="test_outputs_saynt2_DTs",
        help="Base directory containing FSC pickles",
    )
    parser.add_argument(
        "--max_states", type=int, help="Maximum memory states to process per FSC"
    )
    parser.add_argument(
        "--sequential", action="store_true", help="Use sequential processing"
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip creating combined decision trees",
    )
    parser.add_argument(
        "--skip-separate",
        action="store_true",
        help="Skip creating separate decision trees",
    )
    parser.add_argument(
        "--process-large",
        action="store_true",
        help="Process large FSCs that would normally be skipped",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force processing even if already processed",
    )
    parser.add_argument(
        "--process-skipped",
        action="store_true",
        help="Process previously skipped large FSCs",
    )

    args = parser.parse_args()

    if False:
        skipped_files = find_skipped_fscs(args.base_dir)
        print(f"Found {len(skipped_files)} previously skipped FSC files")

        if not skipped_files:
            print("No skipped FSCs found. Nothing to do.")
            return

        total_attempts = 0
        success_count = 0

        # Process each previously skipped FSC
        for pickle_path, mode in skipped_files:
            total_attempts += 1
            if process_skipped_fsc_pickle(
                pickle_path,
                combined_mode=(mode == "combined"),
                max_states=args.max_states,
                parallel=not args.sequential,
                skip_large=False,  # Never skip, we're specifically processing skipped files
            ):
                success_count += 1

        print(
            f"Successfully processed {success_count}/{total_attempts} previously skipped FSC files"
        )
    else:
        pickle_files = find_fsc_pickles(args.base_dir)
        print(f"Found {len(pickle_files)} FSC pickle files")

        total_attempts = 0
        success_count = 0

        # Process each FSC with both modes unless skipped
        for pickle_path in pickle_files:
            # Generate combined trees
            if not args.skip_combined:
                total_attempts += 1
                if process_fsc_pickle(
                    pickle_path,
                    combined_mode=True,
                    max_states=args.max_states,
                    parallel=False,
                    skip_large=True,
                ):
                    success_count += 1

            # Generate separate trees
            if not args.skip_separate:
                total_attempts += 1
                if process_fsc_pickle(
                    pickle_path,
                    combined_mode=False,
                    max_states=args.max_states,
                    parallel=False,
                    skip_large=True,
                ):
                    success_count += 1

        print(
            f"Successfully processed {success_count}/{total_attempts} FSC conversions"
        )


if __name__ == "__main__":
    import sys

    main()
