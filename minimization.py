from collections import defaultdict
from paynt.quotient.fsc import FSC


def minimize_fsc_object(fsc):
    """
    Minimizes a Finite State Controller using partition refinement.
    Takes an FSC object and returns a minimized FSC object.

    :param fsc: An instance of the FSC class
    :return: A minimized FSC object
    """
    # Convert the FSC structure to the format needed by the algorithm
    fsc_nodes = list(range(fsc.num_nodes))
    observations = list(range(fsc.num_observations))

    # Create node_transitions dictionary from update_function
    node_transitions = {}
    is_probabilistic_transitions = False

    for node in fsc_nodes:
        node_transitions[node] = {}
        for obs in observations:
            next_node = fsc.update_function[node][obs]
            # Store the transition as-is (whether deterministic or probabilistic)
            # For algorithm comparison, we'll convert as needed
            if isinstance(next_node, dict):
                is_probabilistic_transitions = True
                # Store the full distribution - don't simplify it
                node_transitions[node][
                    obs
                ] = next_node.copy()  # Make a copy to avoid reference issues
            else:
                # For deterministic transitions, store directly
                node_transitions[node][obs] = next_node

    # Create actions dictionary from action_function
    actions = {}
    for node in fsc_nodes:
        actions[node] = {}
        for obs in observations:
            if fsc.is_deterministic:
                actions[node][obs] = fsc.action_function[node][obs]
            else:
                # For non-deterministic FSCs, store the full distribution
                # Don't reduce to just the most likely action
                if isinstance(fsc.action_function[node][obs], dict):
                    actions[node][obs] = fsc.action_function[node][obs].copy()
                else:
                    actions[node][obs] = fsc.action_function[node][obs]

    # Run the minimization algorithm
    min_nodes, min_transitions, min_actions, partitions = minimize_fsc_internal(
        fsc_nodes, observations, node_transitions, actions
    )

    # Create a new minimized FSC
    min_fsc = FSC(len(min_nodes), fsc.num_observations, fsc.is_deterministic)

    # Map the minimized nodes back to integers for the new FSC
    node_name_to_id = {name: i for i, name in enumerate(min_nodes)}

    # Set up the action function
    for i, node_name in enumerate(min_nodes):
        for obs in observations:
            if fsc.is_deterministic:
                min_fsc.action_function[i][obs] = min_actions[node_name][obs]
            else:
                # For non-deterministic FSCs, we need to maintain the distribution
                # We'll copy the distribution from one of the original nodes in this partition
                # Find a representative node from the partition
                if node_name.startswith("n") and node_name[1:].isdigit():
                    # If node names follow our 'nX' format
                    original_node = int(node_name[1:])  # Remove 'n' from 'nX'
                else:
                    # If we're using original node IDs
                    original_node = int(node_name)

                # Make sure original_node is in range
                original_node = min(original_node, fsc.num_nodes - 1)

                if obs < len(fsc.action_function[original_node]):
                    min_fsc.action_function[i][obs] = fsc.action_function[
                        original_node
                    ][obs]

    # Set up the update function
    for i, node_name in enumerate(min_nodes):
        for obs in observations:
            next_node_name = min_transitions[node_name][obs]

            if is_probabilistic_transitions:
                # For probabilistic transitions, we'll merge distributions
                # Get all nodes in this partition
                partition_nodes = partitions[i]

                # For each observation, merge all distributions to the same target partition
                for obs in observations:
                    # Initialize the combined distribution
                    combined_dist = defaultdict(float)
                    total_weight = 0.0

                    # Process each node in the partition
                    for orig_node in partition_nodes:
                        if isinstance(fsc.update_function[orig_node][obs], dict):
                            # For each probability in the node's distribution
                            for next_node, prob in fsc.update_function[orig_node][
                                obs
                            ].items():
                                # Find which partition contains next_node
                                for j, part in enumerate(partitions):
                                    if next_node in part:
                                        # Map to minimized node and add the probability
                                        min_target = node_name_to_id[f"n{j}"]
                                        combined_dist[min_target] += prob
                                        total_weight += prob
                                        break

                    # Normalize if needed
                    if total_weight > 0:
                        for k in combined_dist:
                            combined_dist[k] /= total_weight

                    # Set the distribution in the minimized FSC
                    if combined_dist:
                        min_fsc.update_function[i][obs] = dict(combined_dist)
                    else:
                        # Default if no distribution was found
                        min_fsc.update_function[i][obs] = {0: 1.0}
            else:
                min_fsc.update_function[i][obs] = node_name_to_id[next_node_name]

    # Copy labels
    min_fsc.observation_labels = fsc.observation_labels
    min_fsc.action_labels = fsc.action_labels

    return min_fsc, partitions  # Also return partitions for debugging


def minimize_fsc_internal(fsc_nodes, observations, node_transitions, actions):
    """
    Internal minimization function that works with abstract node and observation lists.

    :param fsc_nodes: List of FSC memory nodes.
    :param observations: List of possible observations.
    :param node_transitions: Dictionary {node: {observation: next_node}}.
    :param actions: Dictionary {node: {observation: action}}.
    :return: Minimized FSC representation (nodes, transitions, actions).
    """
    # Step 1: Initial Partition (group by action behavior)
    partition = {}
    for node in fsc_nodes:
        # Create signatures that account for probabilistic distributions
        try:
            action_signatures = []
            for obs in observations:
                action = actions[node].get(obs)
                if isinstance(action, dict):
                    # For probabilistic actions, sort and convert to hashable format
                    sorted_actions = sorted(action.items())
                    action_signatures.append(tuple(sorted_actions))
                else:
                    action_signatures.append(action)

            # Make a hashable signature
            signature = tuple(action_signatures)
            partition.setdefault(signature, []).append(node)
        except Exception as e:
            print(f"Error creating signature for node {node}: {e}")
            # Fallback: put in its own partition
            partition.setdefault(f"error_{node}", []).append(node)

    # Convert partition to list format
    partitions = list(partition.values())
    stable = False

    # Step 2: Refinement Process
    iteration = 0
    while not stable and iteration < 1000:  # Add iteration limit for safety
        iteration += 1
        stable = True
        new_partitions = []

        for group in partitions:
            split_groups = defaultdict(list)

            for node in group:
                # Create signature based on which partition each successor belongs to
                signature = []
                for obs in observations:
                    next_node_info = node_transitions[node][obs]

                    if isinstance(next_node_info, dict):
                        # For probabilistic transitions, compute distribution to partitions
                        partition_dist = defaultdict(float)
                        for next_node, prob in next_node_info.items():
                            # Find which partition contains next_node
                            for i, part in enumerate(partitions):
                                if next_node in part:
                                    partition_dist[i] += prob
                                    break
                            else:
                                # Node not found in any partition
                                partition_dist[-1] += prob

                        # Convert to sorted tuple for hashing
                        partition_sig = tuple(sorted(partition_dist.items()))
                        signature.append(partition_sig)
                    else:
                        # For deterministic transitions, find the partition
                        next_node = next_node_info
                        partition_idx = -1
                        for i, part in enumerate(partitions):
                            if next_node in part:
                                partition_idx = i
                                break
                        signature.append(partition_idx)

                # Make signature hashable
                signature_tuple = tuple(signature)
                split_groups[signature_tuple].append(node)

            if len(split_groups) > 1:
                stable = False

            new_partitions.extend(split_groups.values())

        partitions = new_partitions
        print(f"Iteration {iteration}: {len(partitions)} partitions")

    # Step 3: Construct Minimized FSC
    minimized_nodes = [f"n{i}" for i in range(len(partitions))]

    # Create mapping from original nodes to minimized nodes
    node_mapping = {}
    for i, group in enumerate(partitions):
        for node in group:
            node_mapping[node] = minimized_nodes[i]

    # Create minimized transitions and actions
    minimized_transitions = {}
    for i, (min_node, group) in enumerate(zip(minimized_nodes, partitions)):
        minimized_transitions[min_node] = {}
        for obs in observations:
            next_node = node_transitions[group[0]][obs]
            # Find the partition that contains next_node
            for j, part in enumerate(partitions):
                if next_node in part:
                    minimized_transitions[min_node][obs] = minimized_nodes[j]
                    break
            else:
                # Default if not found
                minimized_transitions[min_node][obs] = minimized_nodes[0]

    minimized_actions = {
        min_node: {obs: actions[group[0]][obs] for obs in observations}
        for min_node, group in zip(minimized_nodes, partitions)
    }

    # Return the partitions along with the other results
    return minimized_nodes, minimized_transitions, minimized_actions, partitions


if __name__ == "__main__":
    import os
    import pickle
    import glob
    import time
    import pandas as pd
    import sys
    from datetime import datetime

    # Ensure output is flushed immediately (important for redirection)
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

    # Print header for the log
    print(f"FSC MINIMIZATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    # Directory where FSC pickle files are stored
    base_dir = "test_outputs_saynt2_DTs"

    # Find all FSC pickle files
    pickle_files = []
    for fsc_type in ["SAYNT", "PAYNT"]:
        pattern = os.path.join(base_dir, "*", "decision_trees", fsc_type, "fsc.pkl")
        pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print("No FSC pickle files found. Checking alternative locations...")
        # Try alternative locations
        for fsc_type in ["SAYNT", "PAYNT"]:
            pattern = os.path.join(base_dir, "*", fsc_type, "fsc.pkl")
            pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print(f"No FSC pickle files found in {base_dir}")
        exit(1)

    print(f"Found {len(pickle_files)} FSC pickle files")

    # Sort by file size
    pickle_files.sort(key=lambda x: os.path.getsize(x))

    # Prepare results collection
    results = []

    # Process all FSCs
    skipped_count = 0
    for i, pickle_path in enumerate(pickle_files):
        benchmark_name = os.path.basename(os.path.dirname(os.path.dirname(pickle_path)))
        fsc_type = (
            "SAYNT"
            if "SAYNT" in pickle_path
            else "PAYNT" if "PAYNT" in pickle_path else "Unknown"
        )

        # Check if minimized version already exists
        minimized_path = pickle_path.replace(".pkl", "_minimized.pkl")
        if os.path.exists(minimized_path):
            print(f"\n{'='*40}")
            print(f"BENCHMARK: {benchmark_name}")
            print(f"FSC TYPE: {fsc_type}")
            print(f"PATH: {pickle_path}")
            print(f"SKIPPING - Minimized FSC already exists at: {minimized_path}")

            # Load both FSCs to get data for the CSV
            try:
                # Load the original FSC
                with open(pickle_path, "rb") as f:
                    original_fsc = pickle.load(f)

                # Load the minimized FSC
                with open(minimized_path, "rb") as f:
                    minimized_fsc = pickle.load(f)

                # Calculate metrics
                original_nodes = original_fsc.num_nodes
                minimized_nodes = minimized_fsc.num_nodes
                reduction = (1 - minimized_nodes / original_nodes) * 100
                observations = original_fsc.num_observations

                print(
                    f"Original FSC: {original_nodes} nodes, {observations} observations"
                )
                print(f"Minimized FSC: {minimized_nodes} nodes")
                print(f"Size reduction: {reduction:.2f}%")

                # Add to results with actual data
                result_entry = {
                    "Benchmark": benchmark_name,
                    "FSC Type": fsc_type,
                    "Original Nodes": original_nodes,
                    "Minimized Nodes": minimized_nodes,
                    "Reduction %": reduction,
                    "Observations": observations,
                    "Multi-node Partitions": 0,  # We don't have this info without re-minimizing
                    "Status": "Skipped",
                }
            except Exception as e:
                print(f"Error loading FSCs for metrics: {str(e)}")
                # Fallback to empty data if loading fails
                result_entry = {
                    "Benchmark": benchmark_name,
                    "FSC Type": fsc_type,
                    "Original Nodes": 0,
                    "Minimized Nodes": 0,
                    "Reduction %": 0,
                    "Observations": 0,
                    "Multi-node Partitions": 0,
                    "Status": "Skipped (Load Failed)",
                }

            print(f"{'='*40}\n")
            results.append(result_entry)
            skipped_count += 1
            continue

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
            "Multi-node Partitions": 0,
            "Status": "Failed",
        }

        try:
            start_time = time.time()

            # Load the FSC from pickle file
            with open(pickle_path, "rb") as f:
                fsc = pickle.load(f)
                if fsc is None:
                    print("Error: Failed to load FSC (None returned)")
                    result_entry["Status"] = "Load Failed"
                    results.append(result_entry)
                    continue

            result_entry["Original Nodes"] = fsc.num_nodes
            result_entry["Observations"] = fsc.num_observations

            print(
                f"Original FSC: {fsc.num_nodes} nodes, {fsc.num_observations} observations"
            )
            print(f"Deterministic: {fsc.is_deterministic}\n")

            # Minimize the FSC
            print("Minimizing FSC...")
            minimized_fsc, partitions = minimize_fsc_object(fsc)

            # Calculate reduction
            reduction = (1 - minimized_fsc.num_nodes / fsc.num_nodes) * 100
            result_entry["Minimized Nodes"] = minimized_fsc.num_nodes
            result_entry["Reduction %"] = reduction
            result_entry["Status"] = "Success"

            print(f"Minimization completed in {time.time() - start_time:.2f} seconds")
            print(f"Minimized FSC: {minimized_fsc.num_nodes}")
            print(f"Size reduction: {reduction:.2f}%\n")

            # Log detailed partition information
            print("PARTITION DETAILS:")
            print(f"Number of partitions: {len(partitions)}")

            # Find partitions with multiple nodes
            multi_node_partitions = [
                (i, part) for i, part in enumerate(partitions) if len(part) > 1
            ]
            result_entry["Multi-node Partitions"] = len(multi_node_partitions)

            print(f"Found {len(multi_node_partitions)} partitions with multiple nodes:")

            if multi_node_partitions:
                for partition_idx, partition in multi_node_partitions:
                    print(f"  Partition {partition_idx}: {len(partition)} nodes")
                    print(f"    Nodes: {partition}")
            else:
                print(
                    "  No partitions with multiple nodes found - no minimization possible"
                )

            # Save the minimized FSC
            output_path = pickle_path.replace(".pkl", "_minimized.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(minimized_fsc, f)
            print(f"Saved minimized FSC to: {output_path}\n")

            # Separator for readability
            print("-" * 80 + "\n")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            result_entry["Status"] = f"Error: {str(e)[:50]}..."

        # Add result to collection
        results.append(result_entry)

    # Write summary of all results
    print("\n\n" + "=" * 80)
    print("MINIMIZATION SUMMARY")
    print("=" * 80 + "\n")

    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    success_count = len(df[df["Status"] == "Success"])

    print(f"Total FSCs processed: {len(results)}")
    print(f"Successfully minimized: {success_count}")
    print(f"Failed: {len(results) - success_count}\n")

    if success_count > 0:
        successful_df = df[df["Status"] == "Success"]

        avg_reduction = successful_df["Reduction %"].mean()
        max_reduction = successful_df["Reduction %"].max()
        min_reduction = successful_df["Reduction %"].min()

        print(f"Average size reduction: {avg_reduction:.2f}%")
        print(f"Maximum size reduction: {max_reduction:.2f}%")
        print(f"Minimum size reduction: {min_reduction:.2f}%\n")

        # Top 5 most minimized FSCs
        # print("Top 5 most minimized FSCs:")
        # top5 = successful_df.sort_values("Reduction %", ascending=False).head(5)
        # for _, row in top5.iterrows():
        #     print(f"  {row['Benchmark']} ({row['FSC Type']}): {row['Reduction %']:.2f}% reduction " +
        #          f"({row['Original Nodes']} → {row['Minimized Nodes']} nodes)")

        # FSCs with no reduction
        no_reduction = successful_df[successful_df["Reduction %"] == 0]
        if len(no_reduction) > 0:
            print(f"\n{len(no_reduction)} FSCs had no reduction (already minimal):")
            for _, row in no_reduction.iterrows():
                print(
                    f"  {row['Benchmark']} ({row['FSC Type']}): {row['Original Nodes']} nodes"
                )

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"minimization_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to {csv_file}")
