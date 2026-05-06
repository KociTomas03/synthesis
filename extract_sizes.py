# author: Tomáš Kocí
# this script extracts the number of states, actions, and observations from benchmark POMDP models used in the thesis

import stormpy
import stormpy.pomdp
import os

# Base path for benchmarks
BASE_PATH = "models/archive/cav23-saynt"

# Benchmarks to process (directory name, display name)
BENCHMARKS = [
    ("grid-large-20-5", "grid-large-20-5"),
    ("grid-large-10-5", "grid-large-10-5"),
    ("maze-alex", "maze-alex"),
    ("refuel-08", "refuel-08"),
    ("refuel-06", "refuel-06"),
    ("rocks-12", "rocks-12"),
    ("rocks-16", "rocks-16"),
    ("lanes-100-combined-new", "lanes-100"),
    ("network-3-8-20", "network-3-8-20"),
    ("network-2-8-20", "network-2-8-20"),
    ("drone-4-2", "drone-4-2"),
    ("drone-4-1", "drone-4-1"),
]


def extract_pomdp_sizes(model_path, props_path):
    """Extract |S|, |A|, |Z| from a POMDP model."""
    prism = stormpy.parse_prism_program(model_path, prism_compat=True)

    # Read property from file
    with open(props_path, 'r') as f:
        prop_string = f.read().strip()

    props = stormpy.parse_properties_for_prism_program(prop_string, prism)
    pomdp = stormpy.build_model(prism, props)
    pomdp = stormpy.pomdp.make_canonic(pomdp)

    num_states = pomdp.nr_states
    num_observations = pomdp.nr_observations

    # Count unique action labels
    action_labels = set()
    for choice in range(pomdp.nr_choices):
        labels = pomdp.choice_labeling.get_labels_of_choice(choice)
        action_labels.update(labels)
    num_actions = len(action_labels)

    return num_states, num_actions, num_observations


def main():
    print(f"{'Benchmark':<20} {'|S|':>10} {'|A|':>6} {'|Z|':>10}")
    print("-" * 50)

    for dir_name, display_name in BENCHMARKS:
        model_path = os.path.join(BASE_PATH, dir_name, "sketch.templ")
        props_path = os.path.join(BASE_PATH, dir_name, "sketch.props")

        try:
            states, actions, obs = extract_pomdp_sizes(model_path, props_path)
            print(f"{display_name:<20} {states:>10,} {actions:>6} {obs:>10,}")
        except Exception as e:
            print(f"{display_name:<20} ERROR: {e}")


if __name__ == "__main__":
    main()
