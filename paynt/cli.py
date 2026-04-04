import paynt.quotient.mdp_family
import pickle

import stormpy
from paynt.utils.FSCtoDTConverter import FSCtoDTConverter
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
export_generated_DT_FSC = None


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
    "--verify-FSC",
    is_flag=True,
    default=False,
    help="verify the FSC given by the user",
)
@click.option(
    "--import-STORM-fsc",
    type=click.Path(),
    default=None,
    help="path to the pickle file containing the STORM FSC",
)
@click.option(
    "--generated-fsc-route",
    type=click.STRING,
    help="Route to save generated FSCs to",
)
@click.option(
    "--export-generated-dt-fsc",
    type=click.Path(),
    default=None,
    help="path to folder, where to store generated Decision Tree FSCs",
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
    export_generated_dt_fsc,
    verify_fsc,
    import_storm_fsc,
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
    paynt.cli.export_generated_DT_FSC = export_generated_dt_fsc

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

    if verify_fsc:
        if import_storm_fsc is None:
            print("Please provide STORM FSC file to verify.")
            return

        # with open(import_paynt_fsc, "rb") as f:
        #     paynt_fsc = pickle.load(f)
        with open(import_storm_fsc, "rb") as f:
            storm_fsc = pickle.load(f)
            dtmc = quotient.get_induced_dtmc_from_fsc(storm_fsc)
            result = stormpy.model_checking(
                dtmc, quotient.specification.optimality.formula
            )
            print("Result: " + str(result.at(0)))
        return

    synthesizer.run(optimum_threshold)

    # Export PAYNT FSCs if requested
    if storm_control is not None and storm_control.export_fsc_paynt is not None:
        # Export pure PAYNT FSC if available
        if storm_control.latest_paynt_result_fsc is not None:
            storm_control.export_paynt_fsc(storm_control.latest_paynt_result_fsc, "paynt")
            logger.info("Exported PAYNT FSC")

        # Export combined SAYNT FSC if available
        if storm_control.saynt_fsc is not None:
            storm_control.export_paynt_fsc(storm_control.saynt_fsc, "saynt_combined")
            logger.info("Exported combined SAYNT FSC")

    if paynt.cli.export_generated_DT_FSC:
        if storm_control is None:
            logger.warning("Cannot export FSC: --storm-pomdp flag is required for --export-generated-dt-fsc")
            return
        # Use pre-computed saynt_fsc (combined Storm+PAYNT FSC) if available
        storm_fsc = storm_control.saynt_fsc
        paynt_fsc = storm_control.latest_paynt_result_fsc

        # Create output directories if they don't exist
        os.makedirs(paynt.cli.export_generated_DT_FSC, exist_ok=True)

        # Save Storm FSC in pickle format (binary, compressed)
        if storm_fsc is not None:
            storm_pickle_path = os.path.join(paynt.cli.export_generated_DT_FSC, "STORM", "fsc.pkl")
            os.makedirs(os.path.dirname(storm_pickle_path), exist_ok=True)
            if os.path.exists(storm_pickle_path):
                print(f"FSC file already exists at {storm_pickle_path}, skipping save operation")
            else:
                with open(storm_pickle_path, "wb") as f:
                    pickle.dump(storm_fsc, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved FSC to {storm_pickle_path} (pickle format)")
        elif storm_control.latest_storm_result is not None:
            # FSC conversion failed, but we have the raw Storm result - save it directly
            logger.warning("Storm FSC conversion failed, saving raw Storm result instead")
            storm_pickle_path = os.path.join(paynt.cli.export_generated_DT_FSC, "STORM", "storm_result.pkl")
            os.makedirs(os.path.dirname(storm_pickle_path), exist_ok=True)
            if os.path.exists(storm_pickle_path):
                print(f"Storm result file already exists at {storm_pickle_path}, skipping save operation")
            else:
                try:
                    with open(storm_pickle_path, "wb") as f:
                        pickle.dump(storm_control.latest_storm_result, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved raw Storm result to {storm_pickle_path} (pickle format)")
                except Exception as e:
                    logger.warning(f"Failed to pickle raw Storm result: {e}")
                    # Try saving just the essential data
                    storm_data = {
                        "upper_bound": storm_control.latest_storm_result.upper_bound,
                        "lower_bound": storm_control.latest_storm_result.lower_bound,
                        "belief_mc_dot": storm_control.latest_storm_result.induced_mc_from_scheduler.to_dot(),
                    }
                    with open(storm_pickle_path, "wb") as f:
                        pickle.dump(storm_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    print(f"Saved Storm result data to {storm_pickle_path} (pickle format)")
        else:
            logger.warning("Storm FSC is None and no Storm result available, skipping STORM export")

        # Save PAYNT FSC in pickle format
        if paynt_fsc is not None:
            paynt_pickle_path = os.path.join(paynt.cli.export_generated_DT_FSC, "PAYNT", "fsc.pkl")
            os.makedirs(os.path.dirname(paynt_pickle_path), exist_ok=True)
            if os.path.exists(paynt_pickle_path):
                print(f"FSC file already exists at {paynt_pickle_path}, skipping save operation")
            else:
                with open(paynt_pickle_path, "wb") as f:
                    pickle.dump(paynt_fsc, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved FSC to {paynt_pickle_path} (pickle format)")
        else:
            logger.warning("PAYNT FSC is None, skipping PAYNT export")

        # Save FSC to JSON
        # fsc_json_path = os.path.join(paynt.cli.export_generated_DT_FSC, "fsc.json")
        # with open(fsc_json_path, "w") as f:
        #     f.write(fsc.__str__())
        # print(f"Saved FSC to {fsc_json_path}")

        # separate_converter = FSCtoDTConverter(
        #     fsc,
        #     output_dir=os.path.join(paynt.cli.export_generated_DT_FSC, "separate"),
        #     is_storm=True,
        #     combined_mode=False,
        # )
        # separate_converter.run_dtcontrol()

        # combined_converter = FSCtoDTConverter(
        #     fsc,
        #     output_dir=os.path.join(paynt.cli.export_generated_DT_FSC, "combined"),
        #     is_storm=True,
        #     combined_mode=True,
        # )
        # combined_converter.run_dtcontrol()

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
