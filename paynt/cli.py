import paynt.quotient.mdp_family
import pickle
import time
import json

import stormpy
from paynt.utils.FSCtoDTConverter import FSCtoDTConverter
from paynt.utils.minimization import minimize_fsc_object, eliminate_unreachable_states
from paynt.quotient.fsc import fsc_to_dict
from . import version

import paynt.utils.timer
import paynt.utils.version_check
import paynt.parser.sketch

import paynt.quotient.quotient
import paynt.quotient.pomdp
import paynt.quotient.decpomdp
import paynt.quotient.posmg
import paynt.quotient.storm_pomdp_control

import paynt.synthesizer.synthesizer
import paynt.synthesizer.synthesizer_cegis
import paynt.synthesizer.policy_tree

import paynt.dt

import click
import sys
import os
import cProfile, pstats

import logging

logger = logging.getLogger(__name__)

# Module-level variables that are set by the CLI and accessed by other modules
memory_constraint = None
generated_fsc_route = None
output_dir = None




def setup_logger(log_path=None):
    """Setup routine for logging."""

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # root.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(asctime)s %(threadName)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(message)s")

    handlers = []
    if log_path is not None:
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        handlers.append(fh)
    sh = logging.StreamHandler(sys.stdout)
    handlers.append(sh)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    for h in handlers:
        root.addHandler(h)
    return handlers


@click.command()
@click.argument("project", type=click.Path(exists=True))
@click.option(
    "--sketch",
    default="sketch.templ",
    show_default=True,
    help="name of the sketch file in the project",
)
@click.option(
    "--props",
    default="sketch.props",
    show_default=True,
    help="name of the properties file in the project",
)
@click.option(
    "--relative-error",
    type=click.FLOAT,
    default="0",
    show_default=True,
    help="relative error for optimal synthesis",
)
@click.option("--optimum-threshold", type=click.FLOAT, help="known optimum bound")
@click.option(
    "--precision", type=click.FLOAT, default=1e-4, help="model checking precision"
)
@click.option(
    "--exact",
    is_flag=True,
    default=False,
    help="use exact synthesis (very limited at the moment)",
)
@click.option("--timeout", type=int, help="timeout (s)")
@click.option(
    "--export",
    type=click.Choice(["jani", "drn", "pomdp"]),
    help="export the model to specified format and abort",
)
@click.option(
    "--method",
    type=click.Choice(["onebyone", "ar", "cegis", "hybrid", "ar_multicore"]),
    default="ar",
    show_default=True,
    help="synthesis method",
)
@click.option(
    "--disable-expected-visits",
    is_flag=True,
    default=False,
    help="do not compute expected visits for the splitting heuristic",
)
@click.option(
    "--fsc-synthesis",
    is_flag=True,
    default=False,
    help="enable incremental synthesis of FSCs for a (Dec-)POMDP",
)
@click.option(
    "--fsc-memory-size",
    default=1,
    show_default=True,
    help="implicit memory size for (Dec-)POMDP FSCs",
)
@click.option(
    "--posterior-aware",
    is_flag=True,
    default=False,
    help="unfold MDP taking posterior observation of into account",
)
@click.option(
    "--storm-pomdp",
    is_flag=True,
    default=False,
    help="enable running belief analysis in STorm to enhance FSC synthesis for POMDPs (AR only)",
)
@click.option(
    "--storm-options",
    default="cutoff",
    type=click.Choice(
        [
            "cutoff",
            "clip2",
            "clip4",
            "small",
            "refine",
            "overapp",
            "2mil",
            "5mil",
            "10mil",
            "20mil",
            "30mil",
            "50mil",
        ]
    ),
    show_default=True,
    help="run Storm using pre-defined settings and use the result to enhance PAYNT. Can only be used together with --storm-pomdp flag",
)
@click.option(
    "--iterative-storm",
    nargs=3,
    type=int,
    show_default=True,
    default=None,
    help="runs the iterative PAYNT/Storm integration. Arguments timeout, paynt_timeout, storm_timeout. Can only be used together with --storm-pomdp flag",
)
@click.option(
    "--get-storm-result",
    default=None,
    type=int,
    help="runs PAYNT for given amount of seconds and returns Storm result using FSC at cutoff. If time is 0 returns pure Storm result. Can only be used together with --storm-pomdp flag",
)
@click.option(
    "--prune-storm",
    is_flag=True,
    default=False,
    help="only explore the main family suggested by Storm in each iteration. Can only be used together with --storm-pomdp flag. Can only be used together with --storm-pomdp flag",
)
@click.option(
    "--use-storm-cutoffs",
    is_flag=True,
    default=False,
    help="use storm randomized scheduler cutoffs are used during the prioritization of families. Can only be used together with --storm-pomdp flag. Can only be used together with --storm-pomdp flag",
)
@click.option(
    "--unfold-strategy-storm",
    default="storm",
    type=click.Choice(["storm", "paynt", "cutoff"]),
    show_default=True,
    help="specify memory unfold strategy. Can only be used together with --storm-pomdp flag")

@click.option("--export-synthesis", type=click.Path(), default=None,
    help="base filename to output synthesis result")

@click.option("--mdp-discard-unreachable-choices", is_flag=True, default=False,
    help="if set, unreachable choices will be discarded from the splitting scheduler")

@click.option("--tree-depth", default=0, type=int,
    help="decision tree synthesis: tree depth")
@click.option("--tree-enumeration", is_flag=True, default=False,
    help="decision tree synthesis: if set, all trees of size at most tree_depth will be enumerated")
@click.option("--tree-map-scheduler", type=click.Path(), default=None,
    help="decision tree synthesis: path to a scheduler to be mapped to a decision tree")
@click.option("--add-dont-care-action", is_flag=True, default=False,
    help="decision tree synthesis: # if set, an explicit action executing a random choice of an available action will be added to each state")

@click.option(
    "--constraint-bound", type=click.FLOAT, help="bound for creating constrained POMDP for Cassandra models",
)
@click.option(
    "--export-fsc-storm",
    type=click.Path(),
    default=None,
    help="path to output file for SAYNT belief FSC",
)
@click.option(
    "--export-fsc-paynt",
    type=click.Path(),
    default=None,
    help="path to output file for SAYNT inductive FSC",
)
@click.option(
    "--minimize-storm-fsc",
    is_flag=True,
    default=False,
    help="run Paige-Tarjan minimization on the Storm FSC after synthesis (requires --storm-pomdp)",
)
@click.option(
    "--export-synthesis",
    type=click.Path(),
    default=None,
    help="base filename to output synthesis result",
)
@click.option(
    "--mdp-discard-unreachable-choices",
    is_flag=True,
    default=False,
    help="if set, unreachable choices will be discarded from the splitting scheduler",
)
@click.option(
    "--tree-depth", default=0, type=int, help="decision tree synthesis: tree depth"
)
@click.option(
    "--tree-enumeration",
    is_flag=True,
    default=False,
    help="decision tree synthesis: if set, all trees of size at most tree_depth will be enumerated",
)
@click.option(
    "--tree-map-scheduler",
    type=click.Path(),
    default=None,
    help="decision tree synthesis: path to a scheduler to be mapped to a decision tree",
)
@click.option(
    "--add-dont-care-action",
    is_flag=True,
    default=False,
    help="decision tree synthesis: # if set, an explicit action executing a random choice of an available action will be added to each state",
)
@click.option(
    "--constraint-bound",
    type=click.FLOAT,
    help="bound for creating constrained POMDP for Cassandra models",
)
@click.option(
    "--ce-generator",
    type=click.Choice(["dtmc", "mdp"]),
    default="dtmc",
    show_default=True,
    help="counterexample generator",
)
@click.option("--profiling", is_flag=True, default=False, help="run profiling")
@click.option(
    "--memory-constraint",
    type=click.Choice(
        [
            "none",
            "circular",
            "growing",
            "bothway",
            "onestep",
            "evenUpOddDown",
            "bothWayCircleSelfLoop",
            "notDecreasing",
            "notDecreasingCyclic",
            "growingMax2",
            "notDecreasingMax2",
            "binaryTree",
            "binaryTreeSelfLoop",
            "binaryTreeCyclic",
        ]
    ),
    default="none",
    show_default=True,
    help="maximum number of memory holes to be added to the design space",
)
@click.option(
    "--verify-fsc",
    type=click.Path(),
    default=None,
    help="path to a pickle file containing an FSC to verify",
)
@click.option(
    "--generated-fsc-route",
    type=click.STRING,
    help="Route to save generated FSCs to",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="path to output folder for all synthesis artefacts (FSC pickles, DT results, metrics)",
)
def paynt_run(
    project,
    sketch,
    props,
    relative_error,
    optimum_threshold,
    precision,
    exact,
    timeout,
    export,
    method,
    disable_expected_visits,
    fsc_synthesis,
    fsc_memory_size,
    posterior_aware,
    storm_pomdp,
    iterative_storm,
    get_storm_result,
    storm_options,
    prune_storm,
    use_storm_cutoffs,
    unfold_strategy_storm,
    export_fsc_storm,
    export_fsc_paynt,
    minimize_storm_fsc,
    export_synthesis,
    mdp_discard_unreachable_choices,
    tree_depth,
    tree_enumeration,
    tree_map_scheduler,
    add_dont_care_action,
    constraint_bound,
    ce_generator,
    profiling,
    memory_constraint,
    generated_fsc_route,
    output_dir,
    verify_fsc,
):

    profiler = None
    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()
    paynt.utils.timer.GlobalTimer.start(timeout)

    logger.info("This is Paynt version {}.".format(version()))
    paynt.utils.version_check.check_stormpy_compatibility()

    # set CLI parameters
    paynt.quotient.quotient.Quotient.disable_expected_visits = disable_expected_visits
    paynt.synthesizer.synthesizer.Synthesizer.export_synthesis_filename_base = (
        export_synthesis
    )
    paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS.conflict_generator_type = (
        ce_generator
    )
    paynt.quotient.pomdp.PomdpQuotient.initial_memory_size = fsc_memory_size
    paynt.quotient.pomdp.PomdpQuotient.posterior_aware = posterior_aware
    paynt.quotient.decpomdp.DecPomdpQuotient.initial_memory_size = fsc_memory_size
    paynt.quotient.posmg.PosmgQuotient.initial_memory_size = fsc_memory_size

    paynt.cli.memory_constraint = memory_constraint
    paynt.cli.generated_fsc_route = generated_fsc_route
    paynt.cli.output_dir = output_dir

    paynt.quotient.mdp_family.MdpFamilyQuotient.initial_memory_size = fsc_memory_size

    paynt.synthesizer.policy_tree.SynthesizerPolicyTree.discard_unreachable_choices = mdp_discard_unreachable_choices

    paynt.dt.DtSynthesizer.tree_depth = tree_depth
    paynt.dt.DtSynthesizer.tree_enumeration = tree_enumeration
    paynt.dt.DtSynthesizer.scheduler_path = tree_map_scheduler
    paynt.dt.DtColoredMdpFactory.add_dont_care_action = add_dont_care_action

    storm_control = None
    if storm_pomdp:
        storm_control = paynt.quotient.storm_pomdp_control.StormPOMDPControl()
        storm_control.set_options(
            storm_options,
            get_storm_result,
            iterative_storm,
            use_storm_cutoffs,
            unfold_strategy_storm,
            prune_storm,
            export_fsc_storm,
            export_fsc_paynt,
        )

    sketch_path = os.path.join(project, sketch)
    properties_path = os.path.join(project, props)
    quotient = paynt.parser.sketch.Sketch.load_sketch(
        sketch_path,
        properties_path,
        export,
        relative_error,
        precision,
        constraint_bound,
        exact,
    )
    synthesizer = paynt.synthesizer.synthesizer.Synthesizer.choose_synthesizer(
        quotient, method, fsc_synthesis, storm_control
    )

    if verify_fsc is not None:
        with open(verify_fsc, "rb") as f:
            storm_fsc = pickle.load(f)
            dtmc = quotient.get_induced_dtmc_from_fsc(storm_fsc)
            result = stormpy.model_checking(
                dtmc, quotient.specification.optimality.formula
            )
            print("Result: " + str(result.at(0)))
            print("Policy size: " + str(dtmc.nr_states))
        return

    synthesizer.run(optimum_threshold)

    # Export PAYNT FSCs if requested
    if storm_control is not None and storm_control.export_fsc_paynt is not None:
        if storm_control.latest_paynt_result_fsc is not None:
            storm_control.export_paynt_fsc(storm_control.latest_paynt_result_fsc, "paynt")
            logger.info("Exported PAYNT FSC")
        if storm_control.saynt_fsc is not None:
            storm_control.export_paynt_fsc(storm_control.saynt_fsc, "saynt_combined")
            logger.info("Exported combined SAYNT FSC")

    # Export Storm FSC files (DOT + text) if requested
    if storm_control is not None and storm_control.export_fsc_storm is not None:
        storm_result = storm_control.latest_storm_result
        export_dir = storm_control.export_fsc_storm
        os.makedirs(export_dir, exist_ok=True)

        # 1. Cutoff schedulers
        storm_control.export_storm_cutoff_schedulers(storm_result)

        # 2. Raw Storm belief MC as DOT
        dot_content = storm_result.induced_mc_from_scheduler.to_dot()
        with open(os.path.join(export_dir, "belief_mc.dot"), "w") as f:
            f.write(dot_content)
        logger.info("Exported belief MC DOT")

        # 3. Merged F_B — belief controller converted to FSC (Storm MC + PAYNT FSC nodes)
        try:
            merged_fsc = storm_control.belief_controller_to_fsc(
                storm_result, storm_control.latest_paynt_result_fsc
            )
            with open(os.path.join(export_dir, "merged_fsc.json"), "w") as f:
                json.dump(fsc_to_dict(merged_fsc), f, indent=2)
            with open(os.path.join(export_dir, "merged_fsc.pkl"), "wb") as f:
                pickle.dump(merged_fsc, f)
            logger.info(f"Exported merged F_B ({merged_fsc.num_nodes} nodes)")
        except Exception as e:
            logger.warning(f"Failed to export merged F_B: {e}")

    if storm_control is not None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # Call belief_controller_to_fsc fresh at export time (same as old working export code)
        # so we always get the FSC built from the final synthesis result, not a cached value.
        try:
            storm_fsc = storm_control.belief_controller_to_fsc(
                storm_control.latest_storm_result, storm_control.latest_paynt_result_fsc
            )
        except Exception as e:
            logger.warning(f"Failed to build Storm FSC for export: {e}")
            storm_fsc = None
        paynt_fsc = storm_control.latest_paynt_result_fsc

        # --- Storm FSC metrics ---
        storm_result = storm_control.latest_storm_result
        if storm_fsc is not None:
            results["storm_num_nodes"] = storm_fsc.num_nodes
            results["storm_belief_controller_size"] = storm_control.belief_controller_size
            results["storm_value"] = storm_control.storm_bounds

            if storm_result is not None:
                results["storm_fsc_size"] = storm_control.get_fsc_comparable_size(storm_fsc, storm_result)

            with open(os.path.join(output_dir, "storm_fsc.json"), "w") as f:
                json.dump(fsc_to_dict(storm_fsc), f, indent=2)
            with open(os.path.join(output_dir, "storm_fsc.pkl"), "wb") as f:
                pickle.dump(storm_fsc, f)

            # Minimization of Storm FSC
            if minimize_storm_fsc:
                # --- Paige-Tarjan only (no wildcard merge) ---
                logger.info("Running FSC minimization (no wildcards)...")
                t0 = time.perf_counter()
                minimized_fsc, _, initial_state = minimize_fsc_object(storm_fsc, use_wildcards=False)
                minimized_fsc = eliminate_unreachable_states(minimized_fsc, initial_state=initial_state)
                results["minimized_time_s"] = round(time.perf_counter() - t0, 4)
                results["minimized_num_nodes"] = minimized_fsc.num_nodes
                if storm_result is not None:
                    results["minimized_fsc_size"] = storm_control.get_fsc_comparable_size(minimized_fsc, storm_result)
                logger.info(
                    f"Minimization done: {storm_fsc.num_nodes} -> {minimized_fsc.num_nodes} nodes"
                    f" in {results['minimized_time_s']}s"
                )
                with open(os.path.join(output_dir, "minimized_fsc.json"), "w") as f:
                    json.dump(fsc_to_dict(minimized_fsc), f, indent=2)
                with open(os.path.join(output_dir, "minimized_fsc.pkl"), "wb") as f:
                    pickle.dump(minimized_fsc, f)

                # --- Paige-Tarjan + wildcard merge ---
                logger.info("Running FSC minimization (with wildcard merge)...")
                t0 = time.perf_counter()
                minimized_wc_fsc, _, initial_state_wc = minimize_fsc_object(storm_fsc, use_wildcards=True)
                minimized_wc_fsc = eliminate_unreachable_states(minimized_wc_fsc, initial_state=initial_state_wc)
                results["minimized_wc_time_s"] = round(time.perf_counter() - t0, 4)
                results["minimized_wc_num_nodes"] = minimized_wc_fsc.num_nodes
                if storm_result is not None:
                    results["minimized_wc_fsc_size"] = storm_control.get_fsc_comparable_size(minimized_wc_fsc, storm_result)
                logger.info(
                    f"Wildcard minimization done: {storm_fsc.num_nodes} -> {minimized_wc_fsc.num_nodes} nodes"
                    f" in {results['minimized_wc_time_s']}s"
                )
                with open(os.path.join(output_dir, "minimized_wc_fsc.json"), "w") as f:
                    json.dump(fsc_to_dict(minimized_wc_fsc), f, indent=2)
                with open(os.path.join(output_dir, "minimized_wc_fsc.pkl"), "wb") as f:
                    pickle.dump(minimized_wc_fsc, f)
                # DT conversion for the WC FSC is NOT run here — it can take tens of
                # minutes per benchmark and is done offline via run_wc_dt_conversion.py.
        else:
            logger.warning("Storm FSC is None, skipping Storm metrics")

        # --- PAYNT FSC: DT conversion ---
        if paynt_fsc is not None:
            results["paynt_value"] = storm_control.paynt_bounds
            results["paynt_fsc_size"] = storm_control.paynt_fsc_size

            with open(os.path.join(output_dir, "paynt_fsc.json"), "w") as f:
                json.dump(fsc_to_dict(paynt_fsc), f, indent=2)

            paynt_dt_dir = os.path.join(output_dir, "PAYNT")
            logger.info("Running DT conversion for PAYNT FSC...")
            converter = FSCtoDTConverter(paynt_fsc, output_dir=paynt_dt_dir, is_storm=False)
            t0 = time.perf_counter()
            converter.run_dtcontrol()
            results["paynt_dt_conversion_time_s"] = round(time.perf_counter() - t0, 4)
            benchmark_path = os.path.join(paynt_dt_dir, "benchmark.json")
            results["paynt_dt_nodes"] = FSCtoDTConverter.count_dt_nodes(benchmark_path)
            logger.info(
                f"DT conversion done in {results['paynt_dt_conversion_time_s']}s,"
                f" DT nodes: {results['paynt_dt_nodes']}"
            )
        else:
            logger.warning("PAYNT FSC is None, skipping DT conversion")

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

    if profiling:
        profiler.disable()
        print_profiler_stats(profiler)


def print_profiler_stats(profiler):
    stats = pstats.Stats(profiler)
    NUM_LINES = 10

    logger.debug("cProfiler info:")
    stats.sort_stats("tottime").print_stats(NUM_LINES)

    logger.debug("percentage breakdown:")
    entries = [(key, data[2]) for key, data in stats.stats.items()]
    entries = sorted(entries, key=lambda x: x[1], reverse=True)
    entries = entries[:NUM_LINES]
    for key, data in entries:
        module, line, method = key
        if module == "~":
            callee = method
        else:
            callee = f"{module}:{line}({method})"
        percentage = round(data / stats.total_tt * 100, 1)
        percentage = str(percentage).ljust(4)
        print(f"{percentage} %  {callee}")


def main():
    setup_logger()
    paynt_run()


if __name__ == "__main__":
    main()
