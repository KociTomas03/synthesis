import re
import paynt.synthesizer.statistic
import paynt.utils.timer

import logging
logger = logging.getLogger(__name__)


class FamilyEvaluation:
    '''Result associated with a family after its evaluation. '''
    def __init__(self, family, value, sat, policy):
        self.family = family
        self.value = value
        self.sat = sat
        self.policy = policy


class Synthesizer:

    # base filename (i.e. without extension) to export synthesis result
    export_synthesis_filename_base = None

    @staticmethod
    def choose_synthesizer(quotient, method, fsc_synthesis=False, storm_control=None):

        # hiding imports here to avoid mutual top-level imports
        import paynt.quotient.mdp
        import paynt.quotient.pomdp
        import paynt.quotient.decpomdp
        import paynt.quotient.mdp_family
        import paynt.quotient.posmg
        import paynt.synthesizer.synthesizer_onebyone
        import paynt.synthesizer.synthesizer_ar
        import paynt.synthesizer.synthesizer_cegis
        import paynt.synthesizer.synthesizer_hybrid
        import paynt.synthesizer.synthesizer_multicore_ar
        import paynt.synthesizer.synthesizer_pomdp
        import paynt.synthesizer.synthesizer_decpomdp
        import paynt.synthesizer.synthesizer_posmg
        import paynt.synthesizer.policy_tree
        import paynt.synthesizer.decision_tree

        if isinstance(quotient, paynt.quotient.pomdp_family.PomdpFamilyQuotient):
            logger.info("nothing to do with the POMDP sketch, aborting...")
            exit(0)
        if isinstance(quotient, paynt.quotient.mdp.MdpQuotient):
            return paynt.synthesizer.decision_tree.SynthesizerDecisionTree(quotient)
        # FSC synthesis for POMDPs
        if isinstance(quotient, paynt.quotient.pomdp.PomdpQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_pomdp.SynthesizerPomdp(quotient, method, storm_control)
        # FSC synthesis for Dec-POMDPs
        if isinstance(quotient, paynt.quotient.decpomdp.DecPomdpQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_decpomdp.SynthesizerDecPomdp(quotient)
        # Policy Tree synthesis for family of MDPs
        if isinstance(quotient, paynt.quotient.mdp_family.MdpFamilyQuotient):
            if method == "onebyone":
                return paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(quotient)
            else:
                return paynt.synthesizer.policy_tree.SynthesizerPolicyTree(quotient)
        # FSC synthesis for POSMGs
        if isinstance(quotient, paynt.quotient.posmg.PosmgQuotient) and fsc_synthesis:
            return paynt.synthesizer.synthesizer_posmg.SynthesizerPosmg(quotient)

        # synthesis engines
        if method == "onebyone":
            return paynt.synthesizer.synthesizer_onebyone.SynthesizerOneByOne(quotient)
        if method == "ar":
            return paynt.synthesizer.synthesizer_ar.SynthesizerAR(quotient)
        if method == "cegis":
            return paynt.synthesizer.synthesizer_cegis.SynthesizerCEGIS(quotient)
        if method == "hybrid":
            return paynt.synthesizer.synthesizer_hybrid.SynthesizerHybrid(quotient)
        if method == "ar_multicore":
            return paynt.synthesizer.synthesizer_multicore_ar.SynthesizerMultiCoreAR(quotient)
        raise ValueError("invalid method name")


    def __init__(self, quotient):
        self.quotient = quotient
        self.stat = None
        self.synthesis_timer = None
        self.explored = None
        self.best_assignment = None
        self.best_assignment_value = None
        # deciding upon memory constraint
        self.memory_constraintfunc = None
        memory_constraint = paynt.cli.memory_constraint 
        if (memory_constraint is not None):
            if(memory_constraint == "circular"):
                self.memory_constraintfunc = simpleCircle
            elif(memory_constraint == "bothway"):
                self.memory_constraintfunc = bothWayCircle
            elif(memory_constraint == "bothWayCircleSelfLoop"):
                self.memory_constraintfunc = bothWayCircleSelfLoop
            elif(memory_constraint == "growing"):
                self.memory_constraintfunc = growing
            elif (memory_constraint == "notDecreasing"):
                self.memory_constraintfunc = notDecreasing
            elif (memory_constraint == "onestep"):
                self.memory_constraintfunc = oneStep
            elif (memory_constraint == "evenUpOddDown"):
                self.memory_constraintfunc = evenUpOddDown
            elif (memory_constraint == "notDecreasingCyclic"):
                self.memory_constraintfunc = notDecreasingCyclic
            elif (memory_constraint == "growingMax2"):
                self.memory_constraintfunc = growingMax2
            elif (memory_constraint == "notDecreasingMax2"):
                self.memory_constraintfunc = notDecreasingMax2
            elif (memory_constraint == "binaryTree"):
                self.memory_constraintfunc = binaryTree
            elif (memory_constraint == "binaryTreeSelfLoop"):
                self.memory_constraintfunc = binaryTreeSelfLoop
            elif (memory_constraint == "binaryTreeCyclic"):
                self.memory_constraintfunc = binaryTreeCyclic

    @property
    def method_name(self):
        ''' to be overridden '''
        pass

    def time_limit_reached(self):
        if (self.synthesis_timer is not None and self.synthesis_timer.time_limit_reached()) or \
            paynt.utils.timer.GlobalTimer.time_limit_reached():
            logger.info("time limit reached, aborting...")
            return True
        return False

    def memory_limit_reached(self):
        if paynt.utils.timer.GlobalMemoryLimit.limit_reached():
            logger.info("memory limit reached, aborting...")
            return True
        return False

    def resource_limit_reached(self):
        return self.time_limit_reached() or self.memory_limit_reached()

    def set_optimality_threshold(self, optimum_threshold):
        if self.quotient.specification.has_optimality and optimum_threshold is not None:
            self.quotient.specification.optimality.update_optimum(optimum_threshold)
            logger.debug(f"optimality threshold set to {optimum_threshold}")

    def explore(self, family):
        self.explored += family.size

    def evaluate_all(self, family, prop, keep_value_only=False):
        ''' to be overridden '''
        pass

    def export_evaluation_result(self, evaluations, export_filename_base):
        ''' to be overridden '''
        pass

    def evaluate(self, family=None, prop=None, keep_value_only=False, print_stats=True):
        '''
        Evaluate each member of the family wrt the given property.
        :param family if None, then the design space of the quotient will be used
        :param prop if None, then the default property of the quotient will be used
            (assuming single-property specification)
        :param keep_value_only if True, only value will be associated with the family
        :param print_stats if True, synthesis statistic will be printed
        :param export_filename_base base filename used to export the evaluation results
        :returns a list of (family,evaluation) pairs
        '''
        if family is None:
            family = self.quotient.family
        if prop is None:
            prop = self.quotient.get_property()

        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        logger.info("evaluation initiated, design space: {}".format(family.size))
        self.stat.start(family)
        evaluations = self.evaluate_all(family, prop, keep_value_only)
        self.stat.finished_evaluation(evaluations)
        logger.info("evaluation finished")

        if self.export_synthesis_filename_base is not None:
            self.export_evaluation_result(evaluations, self.export_synthesis_filename_base)

        if print_stats:
            self.stat.print()

        return evaluations


    def synthesize_one(self, family):
        ''' to be overridden '''
        pass

    def synthesize(
        self, family=None, optimum_threshold=None, keep_optimum=False, return_all=False, print_stats=True, timeout=None, memory_constraint="none", generated_fsc_route=None
    ):
        '''
        :param family family of assignment to search in
        :param families alternatively, a list of families can be given
        :param optimum_threshold known bound on the optimum value
        :param keep_optimum if True, the optimality specification will not be reset upon finish
        :param return_all if True and the synthesis returns a family, all assignments will be returned instead of an
            arbitrary one
        :param print_stats if True, synthesis stats will be printed upon completion
        :param timeout synthesis time limit, seconds
        '''
        if family is None:
            family = self.quotient.family
        if family.constraint_indices is None:
            family.constraint_indices = list(range(len(self.quotient.specification.constraints)))

        self.set_optimality_threshold(optimum_threshold)
        self.synthesis_timer = paynt.utils.timer.Timer(900) # 15 minutes   
        self.synthesis_timer.start()
        self.stat = paynt.synthesizer.statistic.Statistic(self)
        self.explored = 0
        #consraining the family
        if memory_constraint != "none":
            family = self.get_memory_restrained_family(family)
        self.stat.start(family)
        self.synthesize_one(family)
        if self.best_assignment is not None and self.best_assignment.size > 1 and not return_all:
            self.best_assignment = self.best_assignment.pick_any()
        self.stat.finished_synthesis()
        if self.best_assignment is not None:
            logger.info("printing synthesized assignment below:")
            logger.info(self.best_assignment)

        if self.best_assignment is not None and self.best_assignment.size == 1:
            dtmc = self.quotient.build_assignment(self.best_assignment)
            result = dtmc.check_specification(self.quotient.specification)
            logger.info(f"double-checking specification satisfiability: {result}")
            if(generated_fsc_route is not None):
                self.best_assignment.toGraph(generated_fsc_route)

        if print_stats:
            self.stat.print()

        assignment = self.best_assignment
        if not keep_optimum:
            self.best_assignment = None
            self.best_assignment_value = None
            self.quotient.specification.reset()

        return assignment


    def run(self, optimum_threshold=None, memory_constraint="none", generated_fsc_route=None):
        return self.synthesize(optimum_threshold=optimum_threshold, memory_constraint=memory_constraint, generated_fsc_route=generated_fsc_route)

    def get_memory_restrained_family(self, family):
        restricted_family = family.copy()


        for hole in range(len(restricted_family.hole_to_name)):
            if(get_current_type(restricted_family.hole_to_name[hole]) == "M"):
                if(self.memory_constraintfunc):
                    restricted_family.hole_set_options(hole,self.memory_constraintfunc(restricted_family.hole_to_name[hole], restricted_family.hole_options(hole), paynt.quotient.pomdp.PomdpQuotient.initial_memory_size -1))
        # onto something
        return restricted_family


def get_current_memory(name):
    return int(re.findall(r"[AM]{1,2}\(\[.*\],(\d+)\)", name)[0])

def get_current_observation(name):
    return re.findall(r"[AM]{1,2}\(\[(.*)],\d+\)", name)[-1]

def get_current_type(name):
    return re.findall(r"([AM]{1,2})", name)[0]

def get_current_value(name):
    return re.findall(r".*\)=(.*)", name)


def oneStep(name, option_labels, max_memory):
    current = get_current_memory(name)
    return [i for i, label in enumerate(option_labels) 
            if int(label) == current + 1 or int(label) == current - 1]

def simpleCircle(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) == current + 1]
    if not indices:
        indices = [i for i, label in enumerate(option_labels) if label == "0" or label == 0]
    return indices

def bothWayCircle(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) == current + 1 or int(label) == current - 1]
    if len(indices) == 1:
        if current == 0:
            indices.extend([i for i, label in enumerate(option_labels) 
                          if int(label) == max_memory])
        else:
            indices.extend([i for i, label in enumerate(option_labels) 
                          if label == "0" or label == 0])
    return indices

def bothWayCircleSelfLoop(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) == current + 1 or int(label) == current - 1 or int(label) == current]
    if len(indices) == 2:
        if current == 0:
            indices.extend([i for i, label in enumerate(option_labels) 
                          if int(label) == max_memory])
        else:
            indices.extend([i for i, label in enumerate(option_labels) 
                          if label == "0" or label == 0])
    return indices

def notDecreasingCyclic(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) >= current]
    if current == max_memory:
        indices.extend([i for i, label in enumerate(option_labels) 
                      if label == "0" or label == 0])
    return indices

def growing(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) > current]
    if not indices:
        indices = [i for i, label in enumerate(option_labels) 
                  if int(label) == current]
    return indices

def notDecreasing(name, option_labels, max_memory):
    current = get_current_memory(name)
    return [i for i, label in enumerate(option_labels) 
            if int(label) >= current]

def evenUpOddDown(name, option_labels, max_memory):
    current = get_current_memory(name)
    if current % 2 == 1:
        return [i for i, label in enumerate(option_labels) 
                if int(label) <= current]
    return [i for i, label in enumerate(option_labels) 
            if int(label) >= current]

def growingMax2(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label)-2 == current or int(label)-1 == current]
    if not indices:
        indices = [i for i, label in enumerate(option_labels) 
                  if int(label) == current]
    return indices

def notDecreasingMax2(name, option_labels, max_memory):
    current = get_current_memory(name)
    return [i for i, label in enumerate(option_labels) 
            if int(label) >= current and int(label) <= current+2]

def binaryTree(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) == (2 * current + 1) or int(label) == (2 * current + 2)]
    if not indices:
        indices = [i for i, label in enumerate(option_labels) 
                  if int(label) == current]
    return indices

def binaryTreeSelfLoop(name, option_labels, max_memory):
    current = get_current_memory(name)
    return [i for i, label in enumerate(option_labels) 
            if int(label) == (2 * current + 1) or 
               int(label) == (2 * current + 2) or 
               int(label) == current]

def binaryTreeCyclic(name, option_labels, max_memory):
    current = get_current_memory(name)
    indices = [i for i, label in enumerate(option_labels) 
              if int(label) == (2 * current + 1) or 
                 int(label) == (2 * current + 2)]
    if not indices:
        indices = [i for i, label in enumerate(option_labels) 
                  if label == "0" or label == 0]
    return indices
