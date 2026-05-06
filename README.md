# PAYNT — Post-Synthesis Optimization Fork

This is a fork of [PAYNT](https://github.com/randriu/synthesis) (Probabilistic progrAm sYNThesizer) extending it with post-synthesis optimization of finite-state controllers (FSCs) produced by the SAYNT algorithm.

**Author of extensions:** Tomáš Kocí  
**Based on:** PAYNT by Roman Andriushchenko et al. (see [README_upstream.md](README_upstream.md))

---

## Documentation

| Document | Contents |
| --- | --- |
| [INSTALLATION.md](INSTALLATION.md) | Step-by-step installation: Storm, stormpy (with GIL patch), PAYNT |
| [USER_GUIDE.md](USER_GUIDE.md) | Usage guide for all implemented features |
| [README_upstream.md](README_upstream.md) | Original PAYNT documentation: sketching language, synthesis methods, examples |

---

## Implemented Features

### FSC Minimization (`--minimize-storm-fsc`)

Reduces the number of memory nodes in F_B (Storm's belief-based FSC) through two sequential passes:

1. **Paige-Tarjan partition refinement** — exact bisimulation minimization merging behaviourally equivalent memory nodes.
2. **Wildcard merging** — greedy heuristic introducing wildcard (don't-care) transitions to merge nodes that agree on all reachable observations.

Implemented in [`paynt/utils/minimization.py`](paynt/utils/minimization.py).

### FSC → Decision Tree Conversion (`--output-dir PATH`)

Exports F_I (PAYNT's inductively synthesized FSC) as a set of decision trees — one per memory node — using [dtControl](https://dtcontrol.model.in.tum.de/). Produces intermediate CSVs, the decision trees themselves, and a dtControl benchmark report.

Implemented in [`paynt/utils/FSCtoDTConverter.py`](paynt/utils/FSCtoDTConverter.py).

### FSC Verification (`--verify-fsc PATH`)

Loads an FSC from a pickle file and model-checks it against the model's optimality property without running synthesis. Useful for evaluating previously synthesized or minimized FSCs.

Implemented in [`paynt/cli.py`](paynt/cli.py).

### Offline WC DT Conversion (`run_wc_dt_conversion.py`)

DT conversion of the wildcard-minimized F_B (`minimized_wc_fsc.pkl`) is expensive and is therefore run offline after synthesis via this script. Processes a directory of benchmark results and appends DT node counts and timings to each `results.json`. Conversion is skipped automatically if the FSC exceeds a configurable node limit (`WC_DT_NODE_LIMIT = 1000` and `PAYNT_DT_NODE_LIMIT = 1000` by default), since dtControl runtimes grow steeply with FSC size.

### Benchmark Pipeline (`utils/benchmark_runner.py`)

Runs PAYNT with `--output-dir` and `--minimize-storm-fsc` across a set of benchmark models and collects results.

### stormpy GIL Fix

The belief exploration checker in stormpy holds the Python GIL during long C++ calls, causing OOM kills on larger SAYNT benchmarks. A patch adding `py::call_guard<py::gil_scoped_release>()` to the four `check` overloads in `src/pomdp/quantitative_analysis.cpp` resolves this. The patched file is included in [`prerequisites/stormpy/src/pomdp/quantitative_analysis.cpp`](prerequisites/stormpy/src/pomdp/quantitative_analysis.cpp). See [INSTALLATION.md](INSTALLATION.md) for how to apply it.

### Dependency Updates

- `dtcontrol==2.1.15` added as a core dependency (was optional)
- `pandas>=2.0.0,<2.2` pinned (dtcontrol requires `applymap` removed in pandas 2.2)
- `setuptools<80` added for Python 3.12+ `pkg_resources` compatibility

---

## Quick Start

```shell
# Install (see INSTALLATION.md for full developer setup)
pip install .

# Run SAYNT with FSC minimization and DT export
python3 -m paynt models/pomdp/maze/slip \
    --fsc-synthesis --storm-pomdp --iterative-storm 900 60 10 \
    --minimize-storm-fsc --output-dir out/maze --dt-conversion

# Verify a saved FSC
python3 -m paynt models/pomdp/maze/slip --verify-fsc out/maze/minimized_wc_fsc.pkl
```

See [USER_GUIDE.md](USER_GUIDE.md) for the full output structure and description of all flags.
