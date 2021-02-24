"""
(k, P)-anonymity implementation of the KAPRA algorithm, from Shou et al. 2011,
Supporting Pattern-preserving Anonymization for Time-series Data, 5.3
"""

from loguru import logger

# Custom imports #
from .io import load_dataset
from .k_anonymity import k_anonymity_bottom_up
from .node import Node
from .io import save_anonymized_dataset

def recycle_bad_leaves(p, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value):
    """
    Recycle bad leaves procedure, from Shou et al. 2013,
    Supporting Pattern-Preserving Anonymization for Time-Series Data, 5.3.2

    Parameters
    ----------
    :param p: Int
        P-requirement for (k, P) anonymity

    :param good_leaf_nodes: List
        List of Node objects satisfying the P-requirement for (k, P) anonymity.

    :param bad_leaf_nodes: List
        List of Node objects which don't satisfy the P-requirement for (k, P) anonymity.

    :param suppressed_nodes: List
        Resulting list of suppressed nodes composed by Node objects. Filled during the execution of this procedure.
        A suppressed node is a p-subgroup which has still less than P time series at the end of the recycling procedure, as
        described in Shou et al. 2013.

    :param paa_value: Int
        Number of real numbers used to encode the feature vector representation of each time series to be anonymized.
        For a reference of the encoding procedure, read the following paper:
        Lin, J., Keogh, E., Lonardi, S., & Chiu, B. (2003, June). A symbolic representation of time series, 
        with implications for streaming algorithms. 
        The above paper contains also a reference to the SAX pattern representation mentioned in Shou et al. 2013.
    """       
    # Each Node data structure contains the following information set:
    #   self.level: number of different characters in the SAX encoding for this node
    #   self.paa_value: Number of real numbers used to encode the feature vector representation of each time series to be anonymized
    #   self.pattern_representation: SAX encoding for this p-subgroup
    #   self.size: numbers of time series contained in self.group (see below)
    #   self.label: either bad-leaf, good-leaf or intermediate
    #   self.group: dict of pairs (time series identifier, time series values), which are the contents of a p-subgroup
    
    bad_leaf_nodes_dict = dict()
    # create a dictionary formed by pairs (level, node_list_for_level)
    for node in bad_leaf_nodes:
        if node.level in bad_leaf_nodes_dict.keys():
            bad_leaf_nodes_dict[node.level].append(node)
        else:
            bad_leaf_nodes_dict[node.level] = [node]

    # Total amount of time series contained in bad leaf nodes
    bad_leaf_nodes_size = sum([node.size for node in bad_leaf_nodes])
    
    # TODO: is this if statement strictly necessary?
    if bad_leaf_nodes_size >= p:

        # Initialize the current level with a max operation over the keys of bad_leaf_nodes_dict, which contains
        # associations (level, node_list_for_level)
        current_level = max(bad_leaf_nodes_dict.keys())

        # While loop reported in paper
        # we perform the recycling operation on all the bad leafs having the same level current_level
        while bad_leaf_nodes_size >= p:
            
            # if the current_level is contained in the level list for the anonymized dataset
            if current_level in bad_leaf_nodes_dict.keys():
                merge_dict = dict()
                keys_to_be_removed = list()
                merge = False
                # for each node having the same level current_level
                for current_level_node in bad_leaf_nodes_dict[current_level]:
                    # retrieve the corresponding pattern representation
                    pr_node = current_level_node.pattern_representation
                    # if the pattern representation pr_node has already been set as a key in merge_dict,
                    # then it means that we are about to merge nodes
                    if pr_node in merge_dict.keys():
                        merge = True
                        # append current_level_node to the list of nodes denoted by pattern representation pr_node
                        merge_dict[pr_node].append(current_level_node)
                        # since we have to merge nodes, the corresponding pattern representation must not be removed
                        if pr_node in keys_to_be_removed:
                            keys_to_be_removed.remove(pr_node)
                    else:
                        # if the pattern representation has not been set yet as a key in dictionary merge_dict, 
                        # initialize the corresponding list with the first element current_level_node
                        merge_dict[pr_node] = [current_level_node]
                        # add pr_node to keys_to_be_removed. In case we end up with a single element, we don't merge anything, and
                        # we will end up suppressing that element
                        keys_to_be_removed.append(pr_node)
                
                # if you have more than 1 node associated with current_level, you have to merge
                if merge:
                    # if you have inserted one level inside keys_to_be_removed, it means that you have only one bad leaf
                    # for that level. You have therefore to delete the corresponding association from the dictionary
                    for k in keys_to_be_removed:
                        del merge_dict[k]
                    # for each association (pattern_representation, corresponding_nodes), that is, for each 
                    # set of nodes having the same pattern representation and level (in the previous for loop, 
                    # we have processed only the elements having level current_level)
                    for pr, node_list in merge_dict.items():
                        # create a new dictionary, which will contain all the time series associated with the merged
                        # node
                        group = dict()
                        # for each node having the same pattern representation
                        for node in node_list:
                            # remove the node from the dictionary of bad_leaf_nodes (having associations (level, nodes))
                            bad_leaf_nodes_dict[current_level].remove(node)
                            # insert all the items contained in node.group into the temporary dictionary group
                            # (used to merge the time series of the nodes associated with the same level and pattern representation)
                            group.update(node.group)
                        # check the current level
                        if current_level > 1:
                            level = current_level
                        else:
                            level = 1
                        # create the merged node, with the same level and pattern representation of the merged bad leaf nodes
                        leaf_merge = Node(level=level, pattern_representation=pr,
                            group=group, paa_value=paa_value)

                        # if the size of the merged node is no less than P
                        if leaf_merge.size >= p:
                            # mark the merged node as a good leaf
                            leaf_merge.label = "good-leaf"
                            good_leaf_nodes.append(leaf_merge)
                            # decrease the global bad leaf nodes size by the size of the merged node (this is done
                            # to make the loop work)
                            bad_leaf_nodes_size -= leaf_merge.size
                        else: 
                            # otherwise, the merged node is a bad leaf: add it to the dictionary of bad_leaf_nodes
                            leaf_merge.label = "bad-leaf"
                            # note that the nodes associated with current_level have been removed before, while trying 
                            # to merge them
                            bad_leaf_nodes_dict[current_level].append(leaf_merge)

            # after having marked the merged node, compute the new level temp_level=current_level - 1
            temp_level = current_level-1
            # if there are any nodes associated with the current level which are marked as bad leaf nodes, loop over
            # them 
            # TODO: why do we change the pattern representation AGAIN? 
            # Implementation choice: when we decrease the level, we compute again the pattern representation for a 
            # p-subgroup. This happens when there are bad leaf nodes left associated with the "old" current_level, which could 
            # not be merged into a single good leaf node. This is done to avoid performing too much suppression: if we can
            # represent the time series with a coarser pattern representation, we should try it.
            for node in bad_leaf_nodes_dict[current_level]:
                # if the newly computed level is > 1
                if temp_level > 1:
                    # retrieve all the time series associated with the node 
                    values_group = list(node.group.values())
                    # take the first time series associated with the node
                    data = np.array(values_group[0])
                    # normalize the time series
                    data_znorm = znorm(data)
                    # compress the time series using paa
                    data_paa = paa(data_znorm, paa_value)
                    # encode the compressed time series using sax
                    pr = ts_to_string(data_paa, cuts_for_asize(temp_level))
                else:
                    # if level equal to 1, use the standard encoding reported in the paper (only 'a' paa_value characters)
                    pr = "a"*paa_value
                # assign the new temp level to the node level
                node.level = temp_level
                # assign the newly computed pattern representation to the node
                node.pattern_representation = pr

            
            if current_level > 0:
                # if we haven't inserted the level temp_level yet as a key inside bad_leaf_nodes_dict
                if temp_level not in bad_leaf_nodes_dict.keys():
                    # remove the nodes associated with current_level, and assign them to temp_level
                    bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict.pop(current_level)
                else:
                    # otherwise, concatenate remove the ones of the current level and concatenate them to the ones of
                    # the temp level. 
                    bad_leaf_nodes_dict[temp_level] = bad_leaf_nodes_dict[temp_level] + bad_leaf_nodes_dict.pop(current_level) 
                # at the end of an iteration, decrease the current_level
                current_level -= 1
            else:
                break 
    
    # suppress the remaining bad leaf nodes, by adding them to the list of suppressed nodes
    remaining_bad_leaf_nodes = list(bad_leaf_nodes_dict.values())[0]
    for node in remaining_bad_leaf_nodes:
        suppressed_nodes.append(node)


def KAPRA(k, p, paa_value, data_path):

    """
    k-P anonymity based on work of Shou et al. 2011,
    Supporting Pattern-Preserving Anonymization for Time-Series Data
    implementation of KAPRA approach
    :param k_value:
    :param p_value:
    :param dataset_path:
    :return:
    """
    _, _, QI_time_series, A_s_dict = load_dataset(data_path)

    logger.info('Launching KAPRA (k, P)-anonymity algorithm...')

    # create-tree phase
    # initialize lists containing good nodes and bad nodes
    good_leaf_nodes = list()
    bad_leaf_nodes = list()

    # creation root and start splitting node
    logger.info("Create-tree phase: initialization and start node splitting with entire dataset")
    # initialization of the root node (level=1, coarsest granularity)
    # note that we pass as a group the whole dictionary containing all the time series of table T
    node = Node(level=1, group=QI_time_series, paa_value=paa_value)
    node.start_splitting(p, max_level, good_leaf_nodes, bad_leaf_nodes)
    logger.info("Create-tree phase: finish node splitting")

    # recycle bad-leaves phase
    logger.info("Start recycle bad-leaves phase")
    suppressed_nodes = list()
    # if any bad leaf nodes are left after the create tree procedure, recycle the bad leaves nodes
    if(len(bad_leaf_nodes) > 0):
        Node.recycle_bad_leaves(p, good_leaf_nodes, bad_leaf_nodes, suppressed_nodes, paa_value)
    logger.info("Finish recycle bad-leaves phase")
    # if any bad leaf nodes are left after the recycle bad leaves procedure, add all the suppressed nodes to list 
    # suppressed_nodes_list
    # at the end of the procedure, good_leaf_nodes will contain all the resulting subgroups which have not been
    # suppressed
    suppressed_nodes_list = list()
    # append the time series of all the suppressed nodes to the list suppressed_nodes_list
    for node in suppressed_nodes:
        suppressed_nodes_list.append(node.group) # suppressed nodes!!!
    
    # group formation phase
    # preprocessing
    logger.info("Start group formation phase")
    # dictionary containing pattern representations
    pattern_representation_dict = dict() 

    k_group_list = list()

    k_anonymity_bottom_up(good_leaf_nodes, p, k, pattern_representation_dict, k_group_list)

    logger.info("Finish group formation phase")

    # TODO: Enforce l-diversity

    save_anonymized_dataset(data_path, pattern_representation_dict, k_group_list, A_s_dict)


