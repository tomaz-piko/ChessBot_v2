cdef class Node:
    cdef public float P, W
    cdef public unsigned int N, vloss
    cdef public bint to_play
    cdef public dict children

    cdef inline Node get_child(self, int move)

    cdef inline void add_child(self, int move, float p)

    cdef inline bint is_leaf(self)

    cdef inline void apply_vloss(self)

    cdef inline void remove_vloss(self)

cpdef find_best_move(object board, Node root, object trt_func, unsigned int num_mcts_sims)