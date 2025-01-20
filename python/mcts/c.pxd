#cython: profile=True, language_level=3
cimport numpy as cnp
from mcts cimport c

cnp.import_array()

cdef class NodeToProcess:
    cdef list moves_to_node
    cdef list legal_moves_tuple
    cdef list image
    cdef bint add_exploration_noise

cdef class Node:
    cdef public float P, W
    cdef public unsigned int N, vloss
    cdef public bint to_play
    cdef public dict children
    cdef public dict debug_info

    cdef inline Node get_child(self, int move)

    cdef inline void add_child(self, int move, float p)

    cdef inline bint is_leaf(self)

    cdef inline void apply_vloss(self)

    cdef inline void remove_vloss(self)

cpdef find_best_move(object board, Node root, object trt_func, unsigned int num_mcts_sims, bint debug)

cpdef list gather_nodes_to_process(Node root, object board, unsigned int nodes_to_find, bint debug)

cpdef void process_node(NodeToProcess node_to_process, Node root, cnp.ndarray policy_logits, float value, bint debug)

cpdef cnp.ndarray calculate_search_statistics(Node root, unsigned int num_actions)

cpdef select_best_move(Node node, object rng, float temp)