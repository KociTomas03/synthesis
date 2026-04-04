from collections import defaultdict, deque
from paynt.quotient.fsc import FSC


def minimize_fsc_object(fsc, use_wildcards=True):
    """
    Minimizes a Finite State Controller using partition refinement.
    Takes an FSC object and returns a minimized FSC object.

    :param fsc: An instance of the FSC class
    :param use_wildcards: If True, merge partitions using wildcards (None values). Default: True.
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
                node_transitions[node][obs] = next_node.copy()  # Make a copy to avoid reference issues
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
    min_nodes, min_transitions, min_actions, partitions = minimize_fsc_internal(fsc_nodes, observations, node_transitions, actions, use_wildcards=use_wildcards)

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
            min_fsc.action_function[i][obs] = fsc.action_function[representative_node][obs]

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
                    for next_node, prob in fsc.update_function[orig_node][current_obs].items():
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
                    min_fsc.update_function[i][current_obs] = fsc.update_function[representative_node][current_obs]
            else:
                # For deterministic transitions, map to the correct partition
                next_node_name = min_transitions[node_name][current_obs]
                if next_node_name is None:
                    min_fsc.update_function[i][current_obs] = None
                else:
                    min_fsc.update_function[i][current_obs] = node_name_to_id[next_node_name]

    # Copy labels
    min_fsc.observation_labels = fsc.observation_labels
    min_fsc.action_labels = fsc.action_labels

    return min_fsc, partitions  # Also return partitions for debugging


def _build_inverse(fsc_nodes, observations, node_transitions):
    """
    Build the inverse transition map.

    inverse[obs][successor] = set of nodes that can reach successor via obs.
    For probabilistic transitions every node in the support is included.
    None transitions (wildcards) are excluded entirely.
    """
    inverse = defaultdict(lambda: defaultdict(set))
    for node in fsc_nodes:
        for obs in observations:
            successor = node_transitions[node][obs]
            if successor is None:
                continue
            if isinstance(successor, dict):
                for succ_node in successor:
                    inverse[obs][succ_node].add(node)
            else:
                inverse[obs][successor].add(node)
    return inverse


def _initial_partition(fsc_nodes, observations, actions):
    """
    Phase 1: group nodes by their action signature across all observations.
    Probabilistic actions are canonicalised by sorting (action, probability) pairs.
    Returns (blocks, block_of, block_counter).
    """
    sig_to_block = {}
    block_of = {}
    blocks = {}
    block_counter = 0

    for node in fsc_nodes:
        sig_parts = []
        for obs in sorted(observations):
            a = actions[node][obs]
            if isinstance(a, dict):
                a_canonical = tuple(sorted(a.items()))
            else:
                a_canonical = a
            sig_parts.append((obs, a_canonical))
        sig = tuple(sig_parts)

        if sig not in sig_to_block:
            sig_to_block[sig] = block_counter
            blocks[block_counter] = set()
            block_counter += 1
        b = sig_to_block[sig]
        blocks[b].add(node)
        block_of[node] = b

    return blocks, block_of, block_counter


def _initialize_queue(blocks, observations):
    """Phase 2: seed the work queue with all (block_id, obs) pairs."""
    queue = deque()
    for block_id in blocks:
        for obs in observations:
            queue.append((block_id, obs))
    return queue


def _refine(blocks, block_of, block_counter, inverse, node_transitions, observations, queue):
    """
    Phase 3: Paige-Tarjan refinement loop.

    For each (splitter, obs) entry, find all predecessor nodes and compute
    their probability mass into the splitter.  Group predecessors by
    (current_block, mass) and perform a multi-way split.  After a split,
    enqueue all but the largest new sub-block (the standard smaller-half
    strategy, generalised to k sub-blocks).
    """
    while queue:
        splitter_id, obs = queue.popleft()
        splitter = blocks.get(splitter_id)
        if splitter is None:
            continue  # block was already split and removed

        # Collect all unique predecessors of every node in the splitter via obs.
        # Using the inverse map avoids scanning all nodes.
        all_predecessors = set()
        for node in splitter:
            all_predecessors.update(inverse[obs].get(node, set()))

        if not all_predecessors:
            continue

        # Compute the probability mass each predecessor sends into the splitter.
        # Collecting predecessors first (above) prevents double-counting when a
        # node has probabilistic transitions to multiple nodes inside the splitter.
        predecessor_mass = {}
        for pred in all_predecessors:
            succ = node_transitions[pred][obs]
            if isinstance(succ, dict):
                mass = sum(p for s, p in succ.items() if s in splitter)
                if mass > 0:
                    predecessor_mass[pred] = mass
            else:
                # Deterministic: pred is in inverse map so its successor is in splitter.
                predecessor_mass[pred] = 1.0

        if not predecessor_mass:
            continue

        # Group predecessors by (current_block, mass) for a multi-way split.
        # Two nodes can stay together only if they are in the same block AND
        # send identical probability mass into the splitter.
        split_groups = defaultdict(set)
        for node, mass in predecessor_mass.items():
            split_groups[(block_of[node], mass)].add(node)

        affected_block_ids = {block_of[n] for n in predecessor_mass}

        for b in affected_block_ids:
            original_block = blocks[b]

            # Collect sub-groups that ARE predecessors (grouped by mass).
            sub_groups = []
            nodes_covered = set()
            for (block_id, _mass), group in split_groups.items():
                if block_id == b:
                    sub_groups.append(group)
                    nodes_covered.update(group)

            # Nodes in the block that are NOT predecessors form the remainder group.
            remainder = original_block - nodes_covered
            if remainder:
                sub_groups.append(remainder)

            if len(sub_groups) <= 1:
                continue  # no split needed for this block

            # Create a new block for each sub-group and update membership.
            new_block_ids = []
            for group in sub_groups:
                new_b = block_counter
                block_counter += 1
                blocks[new_b] = group
                new_block_ids.append(new_b)
                for node in group:
                    block_of[node] = new_b
            del blocks[b]

            # Enqueue all but the largest sub-block (Paige-Tarjan strategy,
            # generalised to k sub-blocks: skip the single largest one).
            largest = max(new_block_ids, key=lambda bid: len(blocks[bid]))
            for new_b in new_block_ids:
                if new_b != largest:
                    for z in observations:
                        queue.append((new_b, z))

    return blocks, block_of


def _build_quotient(blocks, block_of, observations, node_transitions, actions):
    """
    Phase 4: construct the minimised FSC from block representatives.

    Returns (minimized_nodes, minimized_transitions, minimized_actions, partitions)
    in the format expected by minimize_fsc_object.
    """
    block_ids = sorted(blocks.keys())
    new_id = {b: i for i, b in enumerate(block_ids)}
    minimized_nodes = [f"n{i}" for i in range(len(block_ids))]

    minimized_actions = {}
    minimized_transitions = {}
    partitions = []

    for b in block_ids:
        nodes = blocks[b]
        i = new_id[b]
        node_name = f"n{i}"
        rep = next(iter(nodes))  # one representative per block

        minimized_actions[node_name] = {}
        minimized_transitions[node_name] = {}

        for obs in observations:
            minimized_actions[node_name][obs] = actions[rep][obs]
            succ = node_transitions[rep][obs]
            if succ is None:
                minimized_transitions[node_name][obs] = None
            elif isinstance(succ, dict):
                # Merge probabilities for successors that collapse into the same block.
                new_dist = defaultdict(float)
                for s, p in succ.items():
                    new_dist[new_id[block_of[s]]] += p
                minimized_transitions[node_name][obs] = dict(new_dist)
            else:
                minimized_transitions[node_name][obs] = f"n{new_id[block_of[succ]]}"

        # partitions[i] must be the list of original nodes for block i,
        # so that minimize_fsc_object can index by i and test membership.
        partitions.append(list(nodes))

    return minimized_nodes, minimized_transitions, minimized_actions, partitions


def minimize_fsc_internal(fsc_nodes, observations, node_transitions, actions, use_wildcards=True):
    """
    Internal minimization function that works with abstract node and observation lists.

    :param fsc_nodes: List of FSC memory nodes.
    :param observations: List of possible observations.
    :param node_transitions: Dictionary {node: {observation: next_node}}.
    :param actions: Dictionary {node: {observation: action}}.
    :param use_wildcards: If True, merge partitions using wildcards (None values). Default: True.
    :return: Minimized FSC representation (nodes, transitions, actions).
    """
    if not fsc_nodes:
        return list(fsc_nodes), node_transitions, actions, []

    # Phase 1: group nodes by action signature.
    blocks, block_of, block_counter = _initial_partition(fsc_nodes, observations, actions)

    # Phase 2: build inverse map and seed the work queue.
    inverse = _build_inverse(fsc_nodes, observations, node_transitions)
    queue = _initialize_queue(blocks, observations)

    # Phase 3: Paige-Tarjan refinement.
    blocks, block_of = _refine(
        blocks, block_of, block_counter, inverse, node_transitions, observations, queue
    )

    # Wildcard post-processing (optional): merge blocks whose nodes only differ
    # on observations where one side has None (wildcard) transitions/actions.
    if use_wildcards:
        partitions_list = [list(nodes) for nodes in blocks.values()]
        partitions_list = merge_partitions_with_wildcards(
            partitions_list, node_transitions, actions, observations
        )
        # Rebuild block structures from the merged partitions.
        blocks = {i: set(group) for i, group in enumerate(partitions_list)}
        block_of = {}
        for i, group in enumerate(partitions_list):
            for node in group:
                block_of[node] = i

    # Phase 4: construct and return the minimised FSC.
    return _build_quotient(blocks, block_of, observations, node_transitions, actions)


def merge_partitions_with_wildcards(partitions, node_transitions, actions, observations):
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
                new_fsc.update_function[new_idx][obs] = node_mapping.get(next_node, None)

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

    print(f"FSC MINIMIZATION LOG - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    base_dir = "test_outputs_bp"

    pickle_files = []
    # Look for Storm FSCs
    pattern = os.path.join(base_dir, "*", "FSCs", "STORM", "fsc.pkl")
    pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print("No FSC pickle files found. Checking alternative locations...")
        # Fallback patterns
        for fsc_type in ["STORM", "PAYNT"]:
            pattern = os.path.join(base_dir, "*", "FSCs", fsc_type, "fsc.pkl")
            pickle_files.extend(glob.glob(pattern))

    if not pickle_files:
        print(f"No minimized FSC pickle files found in {base_dir}")
        exit(1)

    print(f"Found {len(pickle_files)} FSC pickle files")

    pickle_files.sort(key=lambda x: os.path.getsize(x))

    results = []

    for i, pickle_path in enumerate(pickle_files):
        # Extract benchmark name from path like test_outputs_bp/slip1/FSCs/STORM/fsc.pkl
        benchmark_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pickle_path))))
        fsc_type = "STORM" if "STORM" in pickle_path else "PAYNT" if "PAYNT" in pickle_path else "Unknown"

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

            print(f"Original FSC: {fsc.num_nodes} nodes, {fsc.num_observations} observations")
            print(f"Deterministic: {fsc.is_deterministic}\n")

            # Minimize without wildcards (for Storm FSCs)
            print("Running partition refinement (use_wildcards=False)...")
            minimized_fsc, partitions = minimize_fsc_object(fsc, use_wildcards=False)
            print(f"After minimization: {minimized_fsc.num_nodes} nodes")

            # Remove unreachable states
            print("Removing unreachable states...")
            final_fsc = eliminate_unreachable_states(minimized_fsc)

            # Calculate reduction
            if fsc.num_nodes == 0:
                reduction = 0.0
            else:
                reduction = (1 - final_fsc.num_nodes / fsc.num_nodes) * 100
            result_entry["Minimized Nodes"] = final_fsc.num_nodes
            result_entry["Reduction %"] = reduction
            result_entry["Status"] = "Success"

            print(f"Minimization completed in {time.time() - start_time:.2f} seconds")
            print(f"Final FSC: {final_fsc.num_nodes} nodes")
            print(f"Size reduction: {reduction:.2f}%\n")

            # Save the minimized FSC
            output_path = pickle_path.replace("fsc.pkl", "fsc_minimized.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(final_fsc, f)
            print(f"Saved minimized FSC to: {output_path}\n")

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
                print(f"  {row['Benchmark']} ({row['FSC Type']}): {row['Original Nodes']} nodes")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"unreachable_removal_results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to {csv_file}")
