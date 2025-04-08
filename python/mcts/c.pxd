#cython: profile=False, language_level=3
cimport numpy as cnp
cnp.import_array()

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

cdef class MCTS:
    cdef bint history_flip
    cdef bint root_exploration_noise
    cdef float root_dirichlet_alpha
    cdef float root_exploration_fraction
    cdef float fpu_root
    cdef float fpu_leaf
    cdef float pb_c_init
    cdef float pb_c_factor
    cdef float moves_softmax_temp
    cdef unsigned int pb_c_base
    cdef unsigned int num_vl_searches
    cdef unsigned int num_planes
    cdef unsigned int num_mcts_sampling_moves
    cdef object rng
    cdef dict m_w
    cdef dict m_b

    cpdef expand_root(self, object board, Node root, object trt_func, bint debug)

    cpdef search(self, object board, Node root, object trt_func, bint debug)

    cpdef find_best_move(self, object board, Node root, object trt_func, unsigned int num_sims, float time_limit, bint debug)
    
    cpdef get_history_flip(self)
    
    cpdef select_best_move(self, Node node, float temp)

    cdef void expand_and_evaluate_node(self, Node node, float[:] policy_logits, list legal_moves_tuple, bint debug)

    cdef cnp.ndarray calculate_search_statistics(self, Node root)

