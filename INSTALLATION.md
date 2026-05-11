# PAYNT Installation Guide

This guide covers a full developer installation of PAYNT including Storm, stormpy, and the required GIL patch for SAYNT.

---

## 1. Prerequisites

You need a C++ build toolchain, CMake, and Python 3.10+. On Ubuntu/Debian:

```shell
sudo apt-get install build-essential cmake libboost-all-dev libgmp-dev \
    libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libeigen3-dev \
    python3-dev python3-venv
```

**Note:** This list may be incomplete depending on your distro and existing packages. If the Storm or stormpy build fails with a missing dependency, install the reported package and re-run cmake/pip. Refer to the [Storm documentation](https://www.stormchecker.org/documentation/obtain-storm/build.html) for a more complete list.

Create and activate a virtual environment:

```shell
python3 -m venv venv && source venv/bin/activate
```

---

## 2. Build Storm

```shell
mkdir prerequisites && cd prerequisites
git clone --branch 1.12.0 --depth 1 https://github.com/moves-rwth/storm.git
mkdir storm/build && cd storm/build
cmake ..
make storm storm-cli storm-pomdp
cd ../..
```

---

## 3. Build stormpy

```shell
cd prerequisites
git clone --branch 1.12.0 --depth 1 https://github.com/moves-rwth/stormpy.git
cp ../patches/stormpy_quantitative_analysis.cpp stormpy/src/pomdp/quantitative_analysis.cpp
cd stormpy
pip install . --config-settings=cmake.define.USE_STORM_DFT=OFF \
              --config-settings=cmake.define.USE_STORM_GSPN=OFF \
              --config-settings=cmake.define.STORM_DIR_HINT=$(pwd)/../storm/build
cd ../..
```

**Note:** The Storm backend used by stormpy must match the one built in step 2. The `cmake.define` flags disable optional Storm components that are not needed for PAYNT.

---

## 4. Install PAYNT

With stormpy already in your environment, install PAYNT without rebuilding Storm:

```shell
pip install -r build-requirements.txt
pip install . --no-build-isolation
```

Or, if you want pip to fetch and build Storm automatically (slower, no GIL patch possible):

```shell
pip install .
```

**Note (Python 3.12+):** `setuptools` is no longer bundled with new virtual environments in Python 3.12+. PAYNT's dependency `dtcontrol` relies on `pkg_resources` from `setuptools`. Installing via `pip install .` adds it automatically via the `setuptools<80` dependency. If you ever see `ModuleNotFoundError: No module named 'pkg_resources'`, run `pip install setuptools` manually.

---

## 5. Verify the installation

```shell
python3 -m paynt --help
```

Run a quick smoke test:

```shell
python3 -m paynt models/archive/cav21-paynt/maze --props hard.props
```

---

## Quick-reference: full command sequence

```shell
# 1. Environment
python3 -m venv venv && source venv/bin/activate

# 2. Storm
mkdir prerequisites && cd prerequisites
git clone --branch 1.12.0 --depth 1 https://github.com/moves-rwth/storm.git
mkdir storm/build && cd storm/build && cmake .. && make storm storm-cli storm-pomdp
cd ../..

# 3 & 4. stormpy (with GIL patch from this repo)
git clone --branch 1.12.0 --depth 1 https://github.com/moves-rwth/stormpy.git
cp ../patches/stormpy_quantitative_analysis.cpp stormpy/src/pomdp/quantitative_analysis.cpp
cd stormpy
pip install . --config-settings=cmake.define.USE_STORM_DFT=OFF \
              --config-settings=cmake.define.USE_STORM_GSPN=OFF \
              --config-settings=cmake.define.STORM_DIR_HINT=$(pwd)/../storm/build
cd ../..

# 5. PAYNT
pip install -r build-requirements.txt
pip install . --no-build-isolation
```
