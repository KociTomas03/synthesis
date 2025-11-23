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
        # Get the representative node from the partition
        partition_nodes = partitions[i]
        representative_node = partition_nodes[0]

        for obs in observations:
            min_fsc.action_function[i][obs] = fsc.action_function[representative_node][
                obs
            ]

    # Set up the update function - FIXED to avoid nested loop bug
    for i, node_name in enumerate(min_nodes):
        partition_nodes = partitions[i]

        for current_obs in observations:  # Using current_obs instead of obs
            # Check if all nodes in this partition have None for this observation
            all_none = True
            for orig_node in partition_nodes:
                if fsc.update_function[orig_node][current_obs] is not None:
                    all_none = False
                    break

            if all_none:
                # If all original nodes had None, preserve None
                min_fsc.update_function[i][current_obs] = None
                continue

            if is_probabilistic_transitions:
                # Check if any node in the partition has a non-probabilistic transition
                non_prob_transition = None
                for orig_node in partition_nodes:
                    transition = fsc.update_function[orig_node][current_obs]
                    if not isinstance(transition, dict):
                        non_prob_transition = transition
                        break

                if non_prob_transition is not None:
                    # If any node has a non-probabilistic transition, use it
                    min_fsc.update_function[i][current_obs] = non_prob_transition
                    continue

                # For probabilistic transitions, we need to merge distributions
                combined_dist = defaultdict(float)
                total_weight = 0.0

                # Process each node in the partition
                for orig_node in partition_nodes:
                    for next_node, prob in fsc.update_function[orig_node][
                        current_obs
                    ].items():
                        # Find which partition contains next_node
                        for j, part in enumerate(partitions):
                            if next_node in part:
                                min_target = node_name_to_id[f"n{j}"]
                                combined_dist[min_target] += prob
                                total_weight += prob
                                break

                # Normalize if needed
                if total_weight > 0:
                    for k in combined_dist:
                        combined_dist[k] /= total_weight

                    min_fsc.update_function[i][current_obs] = dict(combined_dist)
                else:
                    # If no valid transition was found, use the original transition
                    min_fsc.update_function[i][current_obs] = fsc.update_function[
                        representative_node
                    ][current_obs]
            else:
                # For deterministic transitions, map to the correct partition
                next_node_name = min_transitions[node_name][current_obs]
                if next_node_name is None:
                    min_fsc.update_function[i][current_obs] = None
                else:
                    min_fsc.update_function[i][current_obs] = node_name_to_id[
                        next_node_name
                    ]

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

    # Merge partitions using wildcards (post-processing)
    partitions = merge_partitions_with_wildcards(
        partitions, node_transitions, actions, observations
    )

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
                minimized_transitions[min_node][obs] = None

    minimized_actions = {
        min_node: {obs: actions[group[0]][obs] for obs in observations}
        for min_node, group in zip(minimized_nodes, partitions)
    }

    # Return the partitions along with the other results
    return minimized_nodes, minimized_transitions, minimized_actions, partitions


def merge_partitions_with_wildcards(
    partitions, node_transitions, actions, observations
):
    merged = [list(part) for part in partitions]
    changed = True
    while changed:
        changed = False
        new_merged = []
        skip = set()
        for i in range(len(merged)):
            if i in skip:
                continue
            merged_this = merged[i]
            for j in range(i + 1, len(merged)):
                if j in skip:
                    continue
                merged_that = merged[j]
                # Check all pairs between merged_this and merged_that
                compatible = True
                for n1 in merged_this:
                    for n2 in merged_that:
                        for obs in observations:
                            # Compare actions (None as wildcard)
                            a1 = actions[n1][obs]
                            a2 = actions[n2][obs]
                            if a1 != a2 and a1 is not None and a2 is not None:
                                compatible = False
                                break
                            # Compare transitions (None as wildcard)
                            t1 = node_transitions[n1][obs]
                            t2 = node_transitions[n2][obs]
                            if t1 != t2 and t1 is not None and t2 is not None:
                                compatible = False
                                break
                        if not compatible:
                            break
                    if not compatible:
                        break
                if compatible:
                    # Merge and mark as changed
                    merged_this += merged_that
                    skip.add(j)
                    changed = True
            new_merged.append(merged_this)
        merged = new_merged
    return merged


def eliminate_unreachable_states(fsc, initial_state=0):
    """
    Removes unreachable states from an FSC object.
    Returns a new FSC object with only reachable states.
    Assumes initial_state is 0 unless specified.
    """
    num_nodes = fsc.num_nodes
    observations = list(range(fsc.num_observations))

    # BFS to find reachable states
    reachable = set()
    queue = [initial_state]
    while queue:
        node = queue.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for obs in observations:
            next_node = fsc.update_function[node][obs]
            if next_node is None:
                continue
            if isinstance(next_node, dict):
                for k in next_node:
                    if k not in reachable:
                        queue.append(k)
            else:
                if next_node not in reachable:
                    queue.append(next_node)

    # Map old indices to new indices
    reachable = sorted(reachable)
    node_mapping = {old: new for new, old in enumerate(reachable)}

    # Create new FSC
    new_fsc = FSC(len(reachable), fsc.num_observations, fsc.is_deterministic)
    for new_idx, old_idx in enumerate(reachable):
        for obs in observations:
            # Actions
            new_fsc.action_function[new_idx][obs] = fsc.action_function[old_idx][obs]
            # Transitions
            next_node = fsc.update_function[old_idx][obs]
            if next_node is None:
                new_fsc.update_function[new_idx][obs] = None
            elif isinstance(next_node, dict):
                new_dist = {}
                for k, v in next_node.items():
                    if k in node_mapping:
                        new_dist[node_mapping[k]] = v
                new_fsc.update_function[new_idx][obs] = new_dist if new_dist else None
            else:
                new_fsc.update_function[new_idx][obs] = node_mapping.get(
                    next_node, None
                )

    # Copy labels
    new_fsc.observation_labels = fsc.observation_labels
    new_fsc.action_labels = fsc.action_labels

    return new_fsc


if __name__ == "__main__":
    import os
    import pickle
    import glob
    import time
    import pandas as pd
    import sys
    from datetime import datetime

    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)

    print(
        f"FSC UNREACHABLE STATE REMOVAL LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("=" * 80 + "\n")

    base_dir = "test_outputs_saynt2_DTs"

    pickle_files = []
    for fsc_type in [
        "SAYNT",
    ]:
        pattern = os.path.join(
            base_dir, "*", "decision_trees", fsc_type, "fsc_minimized.pkl"
        )
        pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print("No minimized FSC pickle files found. Checking alternative locations...")
        for fsc_type in ["SAYNT", "PAYNT"]:
            pattern = os.path.join(base_dir, "*", fsc_type, "fsc_minimized.pkl")
            pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print(f"No minimized FSC pickle files found in {base_dir}")
        exit(1)

    print(f"Found {len(pickle_files)} minimized FSC pickle files")

    pickle_files.sort(key=lambda x: os.path.getsize(x))

    results = []

    for i, pickle_path in enumerate(pickle_files):
        benchmark_name = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(pickle_path)))
        )
        fsc_type = (
            "SAYNT"
            if "SAYNT" in pickle_path
            else "PAYNT" if "PAYNT" in pickle_path else "Unknown"
        )

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

            # Load the minimized FSC from pickle file
            with open(pickle_path, "rb") as f:
                minimized_fsc = pickle.load(f)
                if minimized_fsc is None:
                    print("Error: Failed to load minimized FSC (None returned)")
                    result_entry["Status"] = "Load Failed"
                    results.append(result_entry)
                    continue

            result_entry["Original Nodes"] = minimized_fsc.num_nodes
            result_entry["Observations"] = minimized_fsc.num_observations

            print(
                f"Minimized FSC: {minimized_fsc.num_nodes} nodes, {minimized_fsc.num_observations} observations"
            )
            print(f"Deterministic: {minimized_fsc.is_deterministic}\n")

            # Remove unreachable states
            minimized_fsc_unreachable_removed = eliminate_unreachable_states(
                minimized_fsc
            )

            # Calculate reduction
            if minimized_fsc.num_nodes == 0:
                reduction = 0.0
            else:
                reduction = (
                    1
                    - minimized_fsc_unreachable_removed.num_nodes
                    / minimized_fsc.num_nodes
                ) * 100
            result_entry["Minimized Nodes"] = (
                minimized_fsc_unreachable_removed.num_nodes
            )
            result_entry["Reduction %"] = reduction
            result_entry["Status"] = "Success"

            print(
                f"Unreachable state removal completed in {time.time() - start_time:.2f} seconds"
            )
            print(f"Reachable FSC: {minimized_fsc_unreachable_removed.num_nodes}")
            print(f"Size reduction: {reduction:.2f}%\n")

            # Save the minimized FSC (with unreachable states removed)
            output_path = pickle_path.replace(".pkl", "_unreachable_removed.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(minimized_fsc_unreachable_removed, f)
            print(f"Saved reachable-only FSC to: {output_path}\n")

            print("-" * 80 + "\n")

        except Exception as e:
            print(f"ERROR: {str(e)}")
            import traceback

            traceback.print_exc()
            result_entry["Status"] = f"Error: {str(e)[:50]}..."

        results.append(result_entry)

    print("\n\n" + "=" * 80)
    print("UNREACHABLE REMOVAL SUMMARY")
    print("=" * 80 + "\n")

    df = pd.DataFrame(results)
    success_count = len(df[df["Status"] == "Success"])

    print(f"Total FSCs processed: {len(results)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(results) - success_count}\n")

    if success_count > 0:
        successful_df = df[df["Status"] == "Success"]

        avg_reduction = successful_df["Reduction %"].mean()
        max_reduction = successful_df["Reduction %"].max()
        min_reduction = successful_df["Reduction %"].min()

        print(f"Average size reduction: {avg_reduction:.2f}%")
        print(f"Maximum size reduction: {max_reduction:.2f}%")
        print(f"Minimum size reduction: {min_reduction:.2f}%\n")

        no_reduction = successful_df[successful_df["Reduction %"] == 0]
        if len(no_reduction) > 0:
            print(f"\n{len(no_reduction)} FSCs had no reduction (already minimal):")
            for _, row in no_reduction.iterrows():
                print(
                    f"  {row['Benchmark']} ({row['FSC Type']}): {row['Original Nodes']} nodes"
                )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"unreachable_removal_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to {csv_file}")
