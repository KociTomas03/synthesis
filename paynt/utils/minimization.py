# author: Tomáš Kocí
# This module implements Paige-Tarjan partition refinement to minimize finite state controllers (FSCs).
# With initial partitioning based on action signatures inspired by Kanellakis and Smolka
# and an optional second phase that merges partitions with disjoint signatures 

from collections import defaultdict, deque
from paynt.quotient.fsc import FscFactored


def minimize_fsc_object(fsc, use_wildcards=True):
    """
    Minimizes a Finite State Controller using Paige-Tarjan partition refinement.
    Takes an FSC object and returns a minimized FSC object.

    :param fsc: An instance of the FSC class
    :param use_wildcards: If True, merge partitions using wildcards (None values). Default: True.
    :return: A minimized FSC object
    """
    fsc_nodes = list(range(fsc.num_nodes))
    observations = list(range(fsc.num_observations))

    node_transitions = {}
    is_probabilistic_transitions = False

    for node in fsc_nodes:
        node_transitions[node] = {}
        for obs in observations:
            next_node = fsc.update_function[node][obs]
            if isinstance(next_node, dict):
                is_probabilistic_transitions = True
                node_transitions[node][obs] = next_node.copy()
            else:
                node_transitions[node][obs] = next_node

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

    min_nodes, min_transitions, partitions = minimize_fsc_internal(
        fsc_nodes, observations, node_transitions, actions, use_wildcards=use_wildcards
    )

    min_fsc = FscFactored(len(min_nodes), fsc.num_observations, fsc.is_deterministic)
    node_name_to_id = {name: i for i, name in enumerate(min_nodes)}

    for i, node_name in enumerate(min_nodes):
        partition_nodes = partitions[i]
        for obs in observations:
            # Use the first non-None action from any node in the partition.
            # After wildcard merge a partition may contain nodes from different
            # sub-partitions, each covering different observations.
            action = None
            for orig_node in partition_nodes:
                v = fsc.action_function[orig_node][obs]
                if v is not None:
                    action = v
                    break
            min_fsc.action_function[i][obs] = action

    for i, node_name in enumerate(min_nodes):
        partition_nodes = partitions[i]
        for current_obs in observations:
            all_none = all(
                fsc.update_function[orig_node][current_obs] is None
                for orig_node in partition_nodes
            )
            if all_none:
                min_fsc.update_function[i][current_obs] = None
                continue

            if is_probabilistic_transitions:
                non_prob_transition = None
                for orig_node in partition_nodes:
                    transition = fsc.update_function[orig_node][current_obs]
                    if transition is not None and not isinstance(transition, dict):
                        non_prob_transition = transition
                        break

                if non_prob_transition is not None:
                    min_fsc.update_function[i][current_obs] = non_prob_transition
                    continue

                combined_dist = defaultdict(float)
                total_weight = 0.0
                for orig_node in partition_nodes:
                    entry = fsc.update_function[orig_node][current_obs]
                    if entry is None:
                        continue
                    for next_node, prob in entry.items():
                        for j, part in enumerate(partitions):
                            if next_node in part:
                                min_target = node_name_to_id[f"n{j}"]
                                combined_dist[min_target] += prob
                                total_weight += prob
                                break

                if total_weight > 0:
                    for k in combined_dist:
                        combined_dist[k] /= total_weight
                    min_fsc.update_function[i][current_obs] = dict(combined_dist)
                else:
                    min_fsc.update_function[i][current_obs] = fsc.update_function[partition_nodes[0]][current_obs]
            else:
                next_node_name = min_transitions[node_name][current_obs]
                if next_node_name is None:
                    min_fsc.update_function[i][current_obs] = None
                else:
                    min_fsc.update_function[i][current_obs] = node_name_to_id[next_node_name]

    min_fsc.observation_labels = fsc.observation_labels
    min_fsc.action_labels = fsc.action_labels

    # Find which minimized node index corresponds to original node 0 (the initial state).
    # After Paige-Tarjan, block IDs are not guaranteed to preserve this mapping —
    # the initial block may have been split and renumbered, so we must search explicitly.
    initial_partition = next((i for i, part in enumerate(partitions) if 0 in part), None)
    if initial_partition is None:
        raise ValueError("node 0 is not present in any partition; FSC may be empty")

    return min_fsc, partitions, initial_partition


def eliminate_unreachable_states(fsc, initial_state=0):
    """
    Removes unreachable states from an FSC object.
    Returns a new FSC with only reachable states.
    """
    observations = list(range(fsc.num_observations))

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

    reachable = sorted(reachable)
    # Ensure initial_state maps to index 0 — get_induced_dtmc_from_fsc always starts at FSC node 0
    if initial_state in reachable and reachable[0] != initial_state:
        reachable.remove(initial_state)
        reachable = [initial_state] + reachable
    node_mapping = {old: new for new, old in enumerate(reachable)}

    new_fsc = FscFactored(len(reachable), fsc.num_observations, fsc.is_deterministic)
    for new_idx, old_idx in enumerate(reachable):
        for obs in observations:
            new_fsc.action_function[new_idx][obs] = fsc.action_function[old_idx][obs]
            next_node = fsc.update_function[old_idx][obs]
            if next_node is None:
                new_fsc.update_function[new_idx][obs] = None
            elif isinstance(next_node, dict):
                new_dist = {node_mapping[k]: v for k, v in next_node.items() if k in node_mapping}
                new_fsc.update_function[new_idx][obs] = new_dist if new_dist else None
            else:
                new_fsc.update_function[new_idx][obs] = node_mapping.get(next_node, None)

    new_fsc.observation_labels = fsc.observation_labels
    new_fsc.action_labels = fsc.action_labels

    return new_fsc


def minimize_fsc_internal(fsc_nodes, observations, node_transitions, actions, use_wildcards=True):
    if not fsc_nodes:
        return list(fsc_nodes), node_transitions, []

    blocks, block_of, block_counter = _initial_partition(fsc_nodes, observations, actions)
    inverse = _build_inverse(fsc_nodes, observations, node_transitions)
    queue = _initialize_queue(blocks, observations)
    blocks, block_of = _refine(
        blocks, block_of, block_counter, inverse, node_transitions, observations, queue
    )

    if use_wildcards:
        partitions_list = [list(nodes) for nodes in blocks.values()]
        partitions_list = merge_partitions_with_wildcards(
            partitions_list, node_transitions, actions, observations
        )
        blocks = {i: set(group) for i, group in enumerate(partitions_list)}
        block_of = {}
        for i, group in enumerate(partitions_list):
            for node in group:
                block_of[node] = i

    return _build_quotient(blocks, block_of, observations, node_transitions)


def _build_inverse(fsc_nodes, observations, node_transitions):
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
    queue = deque()
    for block_id in blocks:
        for obs in observations:
            queue.append((block_id, obs))
    return queue


def _refine(blocks, block_of, block_counter, inverse, node_transitions, observations, queue):
    while queue:
        splitter_id, obs = queue.popleft()
        splitter = blocks.get(splitter_id)
        if splitter is None:
            continue

        all_predecessors = set()
        for node in splitter:
            all_predecessors.update(inverse[obs].get(node, set()))

        if not all_predecessors:
            continue

        predecessor_mass = {}
        for pred in all_predecessors:
            succ = node_transitions[pred][obs]
            if isinstance(succ, dict):
                mass = sum(p for s, p in succ.items() if s in splitter)
                if mass > 0:
                    predecessor_mass[pred] = mass
            else:
                predecessor_mass[pred] = 1.0

        if not predecessor_mass:
            continue

        split_groups = defaultdict(set)
        for node, mass in predecessor_mass.items():
            split_groups[(block_of[node], mass)].add(node)

        affected_block_ids = {block_of[n] for n in predecessor_mass}

        for b in affected_block_ids:
            original_block = blocks[b]
            sub_groups = []
            nodes_covered = set()
            for (block_id, _), group in split_groups.items():
                if block_id == b:
                    sub_groups.append(group)
                    nodes_covered.update(group)

            remainder = original_block - nodes_covered
            if remainder:
                sub_groups.append(remainder)

            if len(sub_groups) <= 1:
                continue

            new_block_ids = []
            for group in sub_groups:
                new_b = block_counter
                block_counter += 1
                blocks[new_b] = group
                new_block_ids.append(new_b)
                for node in group:
                    block_of[node] = new_b
            del blocks[b]

            # Add all sub-groups to the queue (no largest-skip optimization).
            # The standard PT "skip largest" optimization is only safe when the
            # block being split was already fully processed as a splitter before
            # it was split. Newly-created blocks can be split before their queue
            # entries are consumed, breaking the invariant and leaving nodes
            # incorrectly merged (e.g., {3,4,5} in grid-large-20-5).
            for new_b in new_block_ids:
                for z in observations:
                    queue.append((new_b, z))

    return blocks, block_of


def _build_quotient(blocks, block_of, observations, node_transitions):
    block_ids = sorted(blocks.keys())
    new_id = {b: i for i, b in enumerate(block_ids)}
    minimized_nodes = [f"n{i}" for i in range(len(block_ids))]

    minimized_transitions = {}
    partitions = []

    for b in block_ids:
        nodes = blocks[b]
        i = new_id[b]
        node_name = f"n{i}"
        rep = next(iter(nodes))

        minimized_transitions[node_name] = {}

        for obs in observations:
            succ = node_transitions[rep][obs]
            if succ is None:
                minimized_transitions[node_name][obs] = None
            elif isinstance(succ, dict):
                new_dist = defaultdict(float)
                for s, p in succ.items():
                    new_dist[new_id[block_of[s]]] += p
                minimized_transitions[node_name][obs] = dict(new_dist)
            else:
                minimized_transitions[node_name][obs] = f"n{new_id[block_of[succ]]}"

        partitions.append(list(nodes))

    return minimized_nodes, minimized_transitions, partitions


def merge_partitions_with_wildcards(partitions, node_transitions, actions, observations):
    """
    Merge Paige-Tarjan partitions using wildcard (None) compatibility.

    Algorithm:
      1. Map raw node transitions to partition-index space (fixed, computed once).
      2. Maintain partition_to_block[pi] -> block_idx.  Transitions are resolved
         dynamically through this dict — no renaming of stored signatures needed.
      3. Pass through blocks; for each, find a compatible existing block or open a
         new one.  Update partition_to_block and the block's accumulated signature
         on each merge.
      4. Repeat until stable (merges enabled by prior merges are caught in the next
         pass, since _resolve uses the always-current partition_to_block).
    """
    n_parts = len(partitions)

    # Map every original node to its Paige-Tarjan partition index (fixed).
    node_to_part = {}
    for pi, part in enumerate(partitions):
        for node in part:
            node_to_part[node] = pi

    def _part_trans(pi, obs):
        """Partition-level transition at obs, in partition-index space."""
        rep = partitions[pi][0]
        t = node_transitions[rep][obs]
        if t is None:
            return None
        if isinstance(t, dict):
            mapped = {}
            for node, prob in t.items():
                p = node_to_part[node]
                mapped[p] = mapped.get(p, 0.0) + prob
            return tuple(sorted(mapped.items()))
        return node_to_part[t]

    # partition_to_block[pi] = current block index for partition pi.
    # When blocks merge, only this dict is updated — no signature renaming needed.
    partition_to_block = list(range(n_parts))

    def _resolve(t):
        """Resolve a stored partition-index transition to current block-index space."""
        if t is None:
            return None
        if isinstance(t, tuple):
            mapped = {}
            for p, prob in t:
                b = partition_to_block[p]
                mapped[b] = mapped.get(b, 0.0) + prob
            return tuple(sorted(mapped.items()))
        return partition_to_block[t]

    # Per-block accumulated signatures (action and transition in partition-index space).
    # Transitions are stored as partition indices and resolved on-the-fly via _resolve.
    block_action = [{obs: actions[partitions[pi][0]][obs] for obs in observations}
                    for pi in range(n_parts)]
    block_trans  = [{obs: _part_trans(pi, obs) for obs in observations}
                    for pi in range(n_parts)]
    block_members      = [list(part) for part in partitions]
    block_part_indices = [{pi} for pi in range(n_parts)]

    def _compatible(bi, bj):
        for obs in observations:
            a1, a2 = block_action[bi][obs], block_action[bj][obs]
            if a1 is not None and a2 is not None and a1 != a2:
                return False
            t1, t2 = _resolve(block_trans[bi][obs]), _resolve(block_trans[bj][obs])
            if t1 is not None and t2 is not None and t1 != t2:
                return False
        return True

    # --- Phase 1: greedy linear pass ---
    # Process each partition once (sparsest first) and assign it to the first
    # compatible existing block, or open a new block.  Cost: O(n * k_blocks * |obs|)
    # where k_blocks grows slowly — much cheaper than O(n^2 * |obs|).

    sorted_parts = sorted(range(n_parts), key=lambda pi: sum(
        1 for obs in observations if block_action[pi][obs] is not None
    ))

    # active_blocks holds the indices into block_* arrays that are still live.
    active_blocks = []

    for pi in sorted_parts:
        found = -1
        for bi in active_blocks:
            if _compatible(pi, bi):
                found = bi
                break
        if found >= 0:
            for pj in block_part_indices[pi]:
                partition_to_block[pj] = found
            block_part_indices[found].update(block_part_indices[pi])
            block_members[found].extend(block_members[pi])
            for obs in observations:
                if block_action[found][obs] is None:
                    block_action[found][obs] = block_action[pi][obs]
                if block_trans[found][obs] is None:
                    block_trans[found][obs] = block_trans[pi][obs]
        else:
            active_blocks.append(pi)

    # --- Phase 2: pairwise fixpoint on the (now small) set of blocks ---
    # k_blocks << n_parts, so O(k_blocks^2 * |obs|) is fast.
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(active_blocks):
            bi = active_blocks[i]
            j = i + 1
            while j < len(active_blocks):
                bj = active_blocks[j]
                if _compatible(bi, bj):
                    for pj in block_part_indices[bj]:
                        partition_to_block[pj] = bi
                    block_part_indices[bi].update(block_part_indices[bj])
                    block_members[bi].extend(block_members[bj])
                    for obs in observations:
                        if block_action[bi][obs] is None:
                            block_action[bi][obs] = block_action[bj][obs]
                        if block_trans[bi][obs] is None:
                            block_trans[bi][obs] = block_trans[bj][obs]
                    active_blocks.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return [block_members[bi] for bi in active_blocks]
