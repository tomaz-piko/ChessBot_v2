cdef class Node:
    cdef public float P, W
    cdef public unsigned int N, vloss
    cdef public bint to_play
    cdef public dict children

    cpdef bint is_leaf(self)

    cpdef void apply_vloss(self)

    cpdef void remove_vloss(self)

cpdef find_best_move(object board, Node root, object trt_func, unsigned int num_mcts_sims)