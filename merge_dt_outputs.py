#!/usr/bin/env python3
# filepath: /home/tomas/bakalarka/myFork/synthesis/merge_dt_outputs.py
import os
import json
import glob
import argparse
from pathlib import Path


def merge_benchmark_files(output_dir):
    """
    Merge multiple benchmark JSON files into a single consolidated file.

    Args:
        output_dir: Directory containing memory_X subdirectories
    """
    print(f"Merging benchmark files in {output_dir}...")

    # Find all memory directories
    memory_dirs = sorted(glob.glob(os.path.join(output_dir, "memory_*")))
    if not memory_dirs:
        print("No memory directories found to merge")
        return False

    # Prepare to merge benchmark files
    all_benchmarks = {}
    for mem_dir in memory_dirs:
        memory_name = os.path.basename(mem_dir)
        # Extract memory ID from memory_X format
        memory_id = (
            memory_name.split("_")[1]
            if len(memory_name.split("_")) > 1
            else memory_name
        )

        benchmark_files = glob.glob(os.path.join(mem_dir, "benchmark*.json"))

        for benchmark_file in benchmark_files:
            try:
                with open(benchmark_file, "r") as f:
                    benchmark_data = json.load(f)

                # Use straightforward naming without nesting
                for key, value in benchmark_data.items():
                    if isinstance(value, dict):
                        # Add memory state info to each entry without wrapping
                        value["memory_state"] = memory_id

                        # Use the original key without appending memory info
                        # This preserves the existing structure while adding the memory state attribute
                        all_benchmarks[key] = value

            except Exception as e:
                print(f"Error processing benchmark {benchmark_file}: {e}")

    # Save combined benchmark file
    if all_benchmarks:
        combined_benchmark_path = os.path.join(output_dir, "combined_benchmarks.json")
        with open(combined_benchmark_path, "w") as f:
            json.dump(all_benchmarks, f, indent=2)
        print(
            f"Created combined benchmark file with {len(all_benchmarks)} entries at {combined_benchmark_path}"
        )
        return True
    else:
        print("No benchmark data found to merge")
        return False


def create_combined_html_viewer(output_dir):
    """
    Create an HTML file that displays all decision trees with tabs.

    Args:
        output_dir: Directory containing memory_X subdirectories
    """
    print(f"Creating combined HTML viewer for {output_dir}...")

    # Find all memory directories and their HTML files
    html_files = []
    memory_dirs = sorted(glob.glob(os.path.join(output_dir, "memory_*")))

    for mem_dir in memory_dirs:
        memory_name = os.path.basename(mem_dir)
        # Extract numeric value from memory_X
        try:
            memory_value = int(memory_name.split("_")[1])
        except (IndexError, ValueError):
            # Fall back to original name if parsing fails
            memory_value = float("inf")  # Puts non-numeric at the end

        # Find all HTML files in this memory directory
        for html_file in glob.glob(os.path.join(mem_dir, "*.html")):
            file_type = os.path.basename(html_file).replace(
                f"{os.path.basename(mem_dir)}_", ""
            )
            html_files.append((memory_name, html_file, file_type, memory_value))

    if not html_files:
        print("No HTML files found to combine")
        return False

    # Sort by numeric memory value first, then by file type
    html_files.sort(key=lambda x: (x[3], x[2]))

    # Create HTML with tabs
    combined_html_path = os.path.join(output_dir, "combined_viewer.html")

    html_header = """<!DOCTYPE html>
<html>
<head>
    <title>Combined Decision Trees</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .tab-container { margin-top: 20px; }
        .tab-buttons { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
        .tab-buttons button {
            background-color: inherit;
            float: left;
            border: none;
            outline: none;
            cursor: pointer;
            padding: 10px 16px;
            transition: 0.3s;
        }
        .tab-buttons button:hover { background-color: #ddd; }
        .tab-buttons button.active { background-color: #ccc; }
        .tab-content { display: none; padding: 20px; border: 1px solid #ccc; border-top: none; }
        .tab-content.active { display: block; }
        iframe { width: 100%; height: 800px; border: none; }
        .search-box { padding: 10px; margin-bottom: 10px; }
        .search-box input { padding: 5px; width: 300px; }
    </style>
    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].classList.remove("active");
            }
            tablinks = document.getElementsByClassName("tab-button");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].classList.remove("active");
            }
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");
        }
        
        function filterTabs() {
            var input = document.getElementById("tabSearch");
            var filter = input.value.toUpperCase();
            var buttons = document.getElementsByClassName("tab-button");
            
            for (var i = 0; i < buttons.length; i++) {
                var buttonText = buttons[i].textContent || buttons[i].innerText;
                if (buttonText.toUpperCase().indexOf(filter) > -1) {
                    buttons[i].style.display = "";
                } else {
                    buttons[i].style.display = "none";
                }
            }
        }
        
        window.onload = function() {
            // Open the first tab by default
            if (document.getElementsByClassName("tab-button").length > 0) {
                document.getElementsByClassName("tab-button")[0].click();
            }
        };
    </script>
</head>
<body>
    <h1>Combined Decision Trees</h1>
    <div class="search-box">
        <input type="text" id="tabSearch" onkeyup="filterTabs()" placeholder="Search for memory states...">
    </div>
    <div class="tab-container">
        <div class="tab-buttons">
"""

    tab_buttons = ""
    tab_content = ""

    for idx, (memory_name, html_path, file_type, _) in enumerate(html_files):
        tab_id = f"tab_{memory_name}_{idx}"
        button_class = " active" if idx == 0 else ""

        # Create buttons for each tab
        tab_buttons += f'            <button class="tab-button{button_class}" onclick="openTab(event, \'{tab_id}\')">{memory_name} - {file_type}</button>\n'

        # Create content div for each tab
        content_class = " active" if idx == 0 else ""

        # Set each HTML file as an iframe in its tab
        rel_path = os.path.relpath(html_path, output_dir)
        tab_content += f"""        <div id="{tab_id}" class="tab-content{content_class}">
            <iframe src="{rel_path}"></iframe>
        </div>
"""

    html_footer = """    </div>
</body>
</html>"""

    with open(combined_html_path, "w") as f:
        f.write(html_header)
        f.write(tab_buttons)
        f.write("        </div>\n")  # Close tab-buttons div
        f.write(tab_content)
        f.write(html_footer)

    print(
        f"Created combined HTML viewer with {len(html_files)} tabs at {combined_html_path}"
    )
    return True


def find_dt_directories(base_dir):
    """Find all decision tree output directories."""
    dt_dirs = []

    # Look for directories containing memory_* subdirectories
    for root, dirs, _ in os.walk(base_dir):
        # Check if this directory has memory_* subdirectories
        memory_dirs = [d for d in dirs if d.startswith("memory_")]
        if memory_dirs:
            dt_dirs.append(root)

    return dt_dirs


def main():
    # Check if no args provided (just script name) - use hardcoded maze-alex path
    if len(sys.argv) == 1:
        print("No arguments provided - using hardcoded path for maze-alex")
        maze_alex_path = os.path.join(
            "test_outputs_saynt_DTs", "network-2-8-20", "decision_trees", "dt_separate"
        )

        # Convert to absolute path if needed
        if not os.path.isabs(maze_alex_path):
            maze_alex_path = os.path.abspath(maze_alex_path)

        print(f"Processing hardcoded directory: {maze_alex_path}")

        if os.path.exists(maze_alex_path):
            merge_benchmark_files(maze_alex_path)
            create_combined_html_viewer(maze_alex_path)
            return
        else:
            print(f"Error: Hardcoded path {maze_alex_path} doesn't exist")
            print("Falling back to regular argument parsing")

    # Regular argument parsing for other cases
    parser = argparse.ArgumentParser(
        description="Merge dtcontrol decision tree outputs"
    )
    parser.add_argument(
        "--base_dir",
        help="Base directory to recursively search for decision tree outputs",
    )
    parser.add_argument(
        "--dir",
        help="Process a single specific directory containing memory_* subdirectories",
    )
    parser.add_argument(
        "--skip-benchmarks", action="store_true", help="Skip merging benchmark files"
    )
    parser.add_argument(
        "--skip-html", action="store_true", help="Skip creating combined HTML viewer"
    )

    args = parser.parse_args()

    if not args.base_dir and not args.dir:
        parser.error("Either --base_dir or --dir must be specified")

    dt_dirs = []

    # If a specific directory is provided, use it directly
    if args.dir:
        # Normalize path to handle both forward and backslashes
        dir_path = Path(args.dir).resolve()
        if os.path.exists(dir_path) and any(
            os.path.isdir(os.path.join(dir_path, d)) and d.startswith("memory_")
            for d in os.listdir(dir_path)
        ):
            dt_dirs = [str(dir_path)]
        else:
            print(f"Error: {args.dir} doesn't contain memory_* subdirectories")
            return

    # If a base directory is provided, search recursively
    if args.base_dir:
        dt_dirs = find_dt_directories(args.base_dir)

    print(f"Found {len(dt_dirs)} decision tree directories")

    for dt_dir in dt_dirs:
        print(f"\nProcessing {dt_dir}")

        if not args.skip_benchmarks:
            merge_benchmark_files(dt_dir)

        if not args.skip_html:
            create_combined_html_viewer(dt_dir)


if __name__ == "__main__":
    import sys

    main()
