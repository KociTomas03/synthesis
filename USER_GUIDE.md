# Post-Synthesis Optimization — User Guide

This guide covers the post-synthesis FSC optimization features added to this fork of PAYNT. The standard PAYNT usage and sketching language are documented in [README.md](README.md).

**Implemented by:** Tomáš Kocí

---

## Overview

After SAYNT synthesis completes, two optional flags trigger additional processing of the resulting finite-state controllers (FSCs):

| Flag | What it does |
| --- | --- |
| `--output-dir PATH` | Saves FSC artefacts (JSON/pkl, `results.json`) to the given directory |
| `--dt-conversion` | Runs dtControl DT conversion for F_I (requires `--output-dir`) |
| `--minimize-storm-fsc` | Minimizes F_B (Storm's belief FSC) using Paige-Tarjan bisimulation + wildcard merging |
| `--verify-fsc PATH` | Verifies an FSC stored in a pickle file against the model's property |

`--output-dir`, `--dt-conversion`, and `--minimize-storm-fsc` require a SAYNT run (`--storm-pomdp --iterative-storm ...`).

---

## `--output-dir PATH` — Decision Tree Conversion

Converts the inductively synthesized F_I into a set of decision trees (one per memory node) using [dtControl](https://dtcontrol.model.in.tum.de/).

```shell
python3 -m paynt PROJECT --fsc-synthesis --storm-pomdp --iterative-storm 900 60 10 \
    --output-dir out/maze --dt-conversion
```

### Output structure

```text
out/maze/
├── paynt_fsc.json          # F_I serialized to JSON
├── paynt_fsc.pkl           # F_I serialized as Python pickle
├── storm_fsc.json          # F_B from Storm (JSON)
├── storm_fsc.pkl           # F_B from Storm (pickle)
├── results.json            # FSC sizes, timings, DT node counts
└── PAYNT/
    ├── training/
    │   └── node_x/         # Input CSVs fed to dtControl (one per memory node x)
    ├── default/
    │   └── node_x/         # dtControl decision trees for memory node x
    ├── benchmark.json      # dtControl benchmark summary
    └── benchmark.html      # dtControl benchmark report (HTML)
```

The `default/` directory name is determined by dtControl's preset naming convention.

### How it works

Each memory node of F_I has an action function (observation → action) and a memory update function (observation × action → next memory node). `FSCtoDTConverter` encodes these as labelled data tables and runs dtControl to learn compact decision trees from them. Two trees are produced per memory node: one for the action function and one for the memory update.

---

## `--minimize-storm-fsc` — FSC Minimization

Applies two minimization passes to F_B (Storm's belief-based FSC):

1. **Paige-Tarjan partition refinement** — exact bisimulation that merges behaviourally equivalent memory nodes.
2. **Wildcard merging** — greedy heuristic that further merges nodes by introducing wildcard (don't-care) transitions where the target memory node does not matter for reachable observations.

```shell
python3 -m paynt PROJECT --fsc-synthesis --storm-pomdp --iterative-storm 900 60 10 \
    --minimize-storm-fsc --output-dir out/maze
```

`--minimize-storm-fsc` can be used with or without `--output-dir`. Without it the minimized FSCs are not saved to disk.

### Output files (inside `--output-dir`)

| File | Description |
| --- | --- |
| `storm_fsc.json` / `.pkl` | Raw F_B before minimization |
| `minimized_fsc.json` / `.pkl` | F_B after Paige-Tarjan only |
| `minimized_wc_fsc.json` / `.pkl` | F_B after Paige-Tarjan + wildcard merging |

### `results.json` fields

| Key | Description |
| --- | --- |
| `storm_num_nodes` | Memory nodes in the raw F_B |
| `storm_belief_controller_size` | Belief space size |
| `storm_value` | Best value found by Storm |
| `storm_fsc_size` | Comparable size metric for F_B |
| `minimized_num_nodes` | Nodes after Paige-Tarjan |
| `minimized_time_s` | Time for Paige-Tarjan pass (seconds) |
| `minimized_fsc_size` | Comparable size metric after Paige-Tarjan |
| `minimized_wc_num_nodes` | Nodes after wildcard merging |
| `minimized_wc_time_s` | Time for wildcard pass (seconds) |
| `minimized_wc_fsc_size` | Comparable size metric after wildcard merging |
| `paynt_value` | Best value found by PAYNT |
| `paynt_fsc_size` | Comparable size metric for F_I |
| `paynt_dt_nodes` | Total DT nodes across all memory nodes |
| `paynt_dt_conversion_time_s` | Time for DT conversion (seconds) |

---

## Combined example

Run SAYNT on the maze POMDP, export decision trees for F_I, and minimize F_B:

```shell
python3 -m paynt models/pomdp/maze/slip \
    --fsc-synthesis \
    --storm-pomdp \
    --iterative-storm 900 60 10 \
    --minimize-storm-fsc \
    --output-dir out/maze \
    --dt-conversion
```

---

## `--verify-fsc PATH` — FSC Verification

Loads an FSC from a pickle file and model-checks it against the model's optimality property.

```shell
python3 -m paynt PROJECT --verify-fsc path/to/fsc.pkl
```

This does not run synthesis — it only evaluates the given FSC. Useful for checking the quality of a previously synthesized or minimized FSC.

### Example: verify the minimized WC FSC

```shell
python3 -m paynt models/pomdp/maze/slip --verify-fsc out/maze/minimized_wc_fsc.pkl
```

Output:

```text
Result: 0.9234...
Policy size: 312
```

---

## `run_wc_dt_conversion.py` — Offline WC DT Conversion

DT conversion of the wildcard-merged F_B (`minimized_wc_fsc.pkl`) is **not** run automatically during synthesis because it can take tens of minutes per benchmark on large FSCs. This script runs it offline after synthesis is complete.

### Configuration

At the top of the script, set:

```python
BASE_DIR = "benchmark_results"   # root directory containing benchmark subdirectories
WC_DT_NODE_LIMIT = 1000          # skip WC DT conversion if FSC has more nodes than this
PAYNT_DT_NODE_LIMIT = 1000       # skip PAYNT two-tree DT conversion if FSC exceeds this
```

The node limits exist because dtControl runtimes grow steeply with FSC size — conversion of a wildcard-merged FSC with hundreds of nodes can already take tens of minutes. Benchmarks exceeding the limit are skipped with a printed warning but their `results.json` is still saved.

Each subdirectory under `BASE_DIR` must contain a `decision_trees/results.json` file (produced by `--output-dir`).

```shell
python3 run_wc_dt_conversion.py
```

Results (`wc_dt_nodes`, `wc_dt_conversion_time_s`, `paynt_dt_two_tree_nodes`, ...) are appended to each benchmark's `results.json`.

### Output structure per benchmark

```text
benchmark_results/BENCHMARK/decision_trees/
├── results.json                 # updated with WC DT results
└── WC/
    └── COMBINED/                # combined WC decision tree
```

---

## `utils/benchmark_runner.py` — Benchmark Pipeline

Runs PAYNT with `--output-dir` and `--minimize-storm-fsc` across a set of benchmark models and collects `results.json` from each.

Configure the list of models and timeouts at the top of the file, then run:

```shell
python3 utils/benchmark_runner.py
```
