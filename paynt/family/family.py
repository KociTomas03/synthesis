import html
import re
import string
import graphviz
import payntbind.synthesis

import paynt.family.smt

import math
import random
import itertools

import logging
logger = logging.getLogger(__name__)


class ParentInfo():
    '''
    Container for stuff to be remembered when splitting an undecided family into subfamilies. Generally used to
    speed-up work with the subfamilies.
    :note it is better to store these things in a separate container instead
        of having a reference to the parent family (that will never be considered again) for memory efficiency.
    '''
    def __init__(self):
        pass
        self.selected_choices = None
        self.constraint_indices = None
        self.refinement_depth = None


class Family:

    def __init__(self, other=None):
        if other is None:
            self.family = payntbind.synthesis.Family()
            self.hole_to_name = []
            self.hole_to_option_labels = []
        else:
            self.family = payntbind.synthesis.Family(other.family)
            self.hole_to_name = other.hole_to_name
            self.hole_to_option_labels = other.hole_to_option_labels

        self.parent_info = None
        self.refinement_depth = 0
        self.constraint_indices = None

        self.selected_choices = None
        self.mdp = None
        self.analysis_result = None
        self.encoding = None

    def add_parent_info(self, parent_info):
        self.parent_info = parent_info
        self.refinement_depth = parent_info.refinement_depth + 1
        self.constraint_indices = parent_info.constraint_indices

    @property
    def num_holes(self):
        return self.family.numHoles()

    def add_hole(self, name, option_labels):
        self.hole_to_name.append(name)
        self.hole_to_option_labels.append(option_labels)
        self.family.addHole(len(option_labels))

    def hole_name(self, hole):
        return self.hole_to_name[hole]

    def hole_options(self, hole):
        return self.family.holeOptions(hole)

    def hole_num_options(self, hole):
        return self.family.holeNumOptions(hole)

    def hole_num_options_total(self, hole):
        return self.family.holeNumOptionsTotal(hole)

    def hole_set_options(self, hole, options):
        self.family.holeSetOptions(hole,options)

    @property
    def size(self):
        return math.prod([self.family.holeNumOptions(hole) for hole in range(self.num_holes)])

    INT_PRINT_MAX_ORDER = 5

    @property
    def size_or_order(self):
        order = int(math.fsum([math.log10(self.family.holeNumOptions(hole)) for hole in range(self.num_holes)]))
        if order <= Family.INT_PRINT_MAX_ORDER:
            return self.size
        return f"1e{order}"

    def hole_options_to_string(self, hole, options):
        name = self.hole_name(hole)
        labels = [str(self.hole_to_option_labels[hole][option]) for option in options]
        if len(labels) == 1:
            return f"{name}={labels[0]}"
        else:
            return name + ": {" + ",".join(labels) + "}"

    def __str__(self):
        hole_strings = []
        for hole in range(self.num_holes):
            options = self.hole_options(hole)
            hole_str = self.hole_options_to_string(hole,options)
            hole_strings.append(hole_str)
        return ", ".join(hole_strings)

    def generate_labels(self):
        """Generate sequential labels a, b, c, ..., z, aa, ab, ..., zz."""
        for length in range(1, 3):  # Adjust the range for longer sequences if needed
            for s in itertools.product(string.ascii_lowercase, repeat=length):
                yield ''.join(s)

    def extract_unique_values(self, arr):
        unique_memory_values = set()
        unique_observations = set()
        unique_actions = set()
        actions = [{"observation": get_current_observation(a), "type": get_current_type(a), "memory": get_current_memory(a), "actions": get_current_value(a)} for a in arr if get_current_type(a) == "A"]
        memory = [{"observation": get_current_observation(a), "type": get_current_type(a), "memory": get_current_memory(a), "next_memory": int(get_current_value(a)[0])} for a in arr if get_current_type(a) == "M"]
        mixed = [{"observation": get_current_observation(a), "type": get_current_type(a), "memory": get_current_memory(a)} for a in arr if get_current_type(a) == "AM"]
        
        for a in actions:
            unique_actions.add(a["actions"][0])
            unique_observations.add(a["observation"])
            matching_records = [m for m in memory if m["observation"] == a["observation"] and a["memory"] == m["memory"]]
            if len(matching_records) == 1:
                a["next_memory"] = matching_records[0]["next_memory"]
                unique_memory_values.add(matching_records[0]["next_memory"])
                unique_memory_values.add(a["memory"])
            else:
                a["next_memory"] = a["memory"]
    
        return unique_memory_values, unique_observations, unique_actions, actions, memory

    def build_edge_dict(self, actions, unique_observations, unique_actions):
        observationLabelGen = self.generate_labels()
        obsToShortDict = {a: next(observationLabelGen) for a in unique_observations}
        
        actionLabelGen = self.generate_labels()
        actionsToShortDict = {a: next(actionLabelGen) for a in unique_actions}
        
        edge_dict = {}
        for a in actions:
            key = (a["memory"], a["next_memory"], a["actions"][0])
            if key not in edge_dict:
                edge_dict[key] = {"observations": set(), "action": actionsToShortDict[a["actions"][0]]}
            edge_dict[key]["observations"].add(obsToShortDict[a["observation"]])
        
        return edge_dict, obsToShortDict, actionsToShortDict

    def generate_legend_rows(self, items_dict):
        return ''.join(
            f'<TR><TD>{html.escape(short)}</TD><TD>{html.escape(obs)}</TD></TR>'
            for obs, short in items_dict.items()
        )

    def generate_legend(self, obsToShortDict, actionsToShortDict):
        legend = (
            '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">'
            '<TR><TD><B>Symbol</B></TD><TD><B>Observation</B></TD></TR>'
            + self.generate_legend_rows(obsToShortDict) +
            '<TR><TD><B>Symbol</B></TD><TD><B>Action</B></TD></TR>' +
            self.generate_legend_rows(actionsToShortDict) +
            '</TABLE>>'
        )
        return legend

    def toGraph(self, route = None):
        txt = self.__str__()
        arr = txt.split(", ")
        
        unique_memory_values, unique_observations, unique_actions, actions, memory = self.extract_unique_values(arr)
        
        dot = graphviz.Digraph(engine='fdp',)
        dot.attr(splines='true',  nodesep='1',  sep='10')
        
        for a in unique_memory_values:
            dot.node(str(a), str(a), shape='ellipse')
        
        edge_dict, obsToShortDict, actionsToShortDict = self.build_edge_dict(actions, unique_observations, unique_actions)
        
        for (memory, next_memory, action), labels in edge_dict.items():
            observation_label = ','.join(sorted(labels["observations"]))
            action_label = labels["action"]
            dot.edge(str(memory), str(next_memory), label=f'{observation_label}/{action_label}')
        
        with dot.subgraph(name='cluster_legend') as legend:
            legend.attr(label='Legend', fontsize='20', fontcolor='blue')
            legend.node('legend', label=self.generate_legend(obsToShortDict, actionsToShortDict), shape='plaintext')
        
        if route is not None:
            dot.save(route)
            dot.render(route, format='pdf')
        # dot.view()
        return

    def copy(self):
        return Family(self)

    def assume_hole_options_copy(self, hole, options):
        '''
        Create a copy and assume suboptions for a given hole.
        @note this does not check whether @options are actually suboptions of this hole.
        '''
        subfamily = self.copy()
        subfamily.hole_set_options(hole,options)
        return subfamily

    def assume_options_copy(self, hole_options):
        '''
        Create a copy and assume suboptions for each hole.
        @note this does not check whether suboptions are actually suboptions of any given hole.
        '''
        subfamily = self.copy()
        for hole,options in enumerate(hole_options):
            subfamily.hole_set_options(hole,options)
        return subfamily

    def split(self, splitter, suboptions):
        return [self.assume_hole_options_copy(splitter,options) for options in suboptions]

    def pick_any(self):
        hole_options = [[self.hole_options(hole)[0]] for hole in range(self.num_holes)]
        return self.assume_options_copy(hole_options)

    def pick_random(self):
        hole_options = [[random.choice(self.hole_options(hole))] for hole in range(self.num_holes)]
        return self.assume_options_copy(hole_options)

    def all_combinations(self):
        '''
        :returns iteratable Cartesian product of hole options
        '''
        all_options = []
        for hole in range(self.num_holes):
            options = self.hole_options(hole)
            all_options.append(options)
        return itertools.product(*all_options)

    def construct_assignment(self, combination):
        ''' Convert hole option combination to a hole assignment. '''
        combination = list(combination)
        suboptions = [[option] for option in combination]
        assignment = self.assume_options_copy(suboptions)
        return assignment

    def collect_parent_info(self, specification):
        pi = ParentInfo()
        pi.selected_choices = self.selected_choices
        pi.refinement_depth = self.refinement_depth
        cr = self.analysis_result.constraints_result
        pi.constraint_indices = cr.undecided_constraints if cr is not None else []
        return pi

    def encode(self, smt_solver):
        if self.encoding is None:
            self.encoding = paynt.family.smt.FamilyEncoding(smt_solver, self)

############################# UTIL FUNCTIONS ####################################
def get_current_memory(name):
    return int(re.findall(r"[AM]{1,2}\(\[.*\],(\d+)\)", name)[0])

def get_current_observation(name):
    return re.findall(r"[AM]{1,2}\(\[(.*)],\d+\)", name)[-1]

def get_current_type(name):
    return re.findall(r"([AM]{1,2})", name)[0]

def get_current_value(name):
    return re.findall(r".*\)=(.*)", name)



############################# MEMORY CONSTRAINT FUNCTIONS ####################################
# def oneStep(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label)+1 == get_current_memory(name) or int(label)-1 == get_current_memory(name)]
#     return option_labels


# def simpleCircle(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label)-1 == get_current_memory(name)]
#             if(len(option_labels) == 0):
#                 option_labels = [str(0)]
#     return option_labels

# def bothWayCircle(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label)+1 == get_current_memory(name) or int(label)-1 == get_current_memory(name)]
#             if(len(option_labels) == 1):
#                 if (get_current_memory(name) == 0):
#                     option_labels.append(str(max_memory))
#                 else:
#                     option_labels.append(str(0))
#     return option_labels

# def bothWayCircleSelfLoop(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label)+1 == get_current_memory(name) or int(label)-1 == get_current_memory(name) or int(label) == get_current_memory(name)]
#             if(len(option_labels) == 2):
#                 if (get_current_memory(name) == 0):
#                     option_labels.append(str(max_memory))
#                 else:
#                     option_labels.append(str(0))
#     return option_labels

# def notDecreasingCyclic(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label) >= get_current_memory(name)]
#             if(get_current_memory(name) == max_memory):
#                 option_labels.append(str(0))
#     return option_labels

# def growing(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label) > get_current_memory(name)]
#             if(len(option_labels) == 0):
#                 option_labels = [str(get_current_memory(name))]
#     return option_labels

# def notDecreasing(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label) >= get_current_memory(name)]
#     return option_labels

# def evenUpOddDown(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             if(get_current_memory(name) % 2 == 1):
#                 option_labels = [label for label in option_labels if int(label) <= get_current_memory(name)]
#             else:
#                 option_labels = [label for label in option_labels if int(label) >= get_current_memory(name)]
#     return option_labels

# def growingMax2(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label)-2 == get_current_memory(name) or int(label)-1 == get_current_memory(name) ]
#             if(len(option_labels) == 0):
#                 option_labels = [str(get_current_memory(name))]
#     return option_labels

# def notDecreasingMax2(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#             option_labels = [label for label in option_labels if int(label) >= get_current_memory(name) and int(label) <= get_current_memory(name)+2]
#     return option_labels


# # TODO: add binary tree mem constraint 
# def binaryTree(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#         option_labels = [label for label in option_labels if int(label) == ( 2 * get_current_memory(name) + 1) or int(label) == (2 * get_current_memory(name) + 2)]
#     if len(option_labels) == 0:
#             option_labels = [str(get_current_memory(name))]
#     return option_labels

# def binaryTreeSelfLoop(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#         option_labels = [label for label in option_labels if int(label) == ( 2 * get_current_memory(name) + 1) or int(label) == (2 * get_current_memory(name) + 2) or int(label) == get_current_memory(name)]
#     return option_labels

# def binaryTreeCyclic(name, option_labels, max_memory):
#     if(get_current_type(name) == "M"):
#         option_labels = [label for label in option_labels if int(label) == ( 2 * get_current_memory(name) + 1) or int(label) == (2 * get_current_memory(name) + 2)]
#         if len(option_labels) == 0:
#             option_labels = [str(0)]
#     return option_labels
    
# TODO: think about dynamic memory constraints (low value -> high variables, high value -> low variables) 
# or colors with high/low entropy(predicting them,....)