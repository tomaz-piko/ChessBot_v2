from actionspace import map_w, map_b
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, sqrt

cnp.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef class Node:
    cdef public float P, W
    cdef public unsigned int N, vloss
    cdef public bint to_play
    cdef public dict children

    def __cinit__(self):
        self.P = 0.0
        self.W = 0.0
        self.N = 0
        self.vloss = 0
        self.children = {}
        self.to_play = False

    def __init__(self):
        self.P = 0.0
        self.W = 0.0
        self.N = 0
        self.vloss = 0
        self.children = {}
        self.to_play = False

    def __getitem__(self, move: str):
        return self.children[move]
    
    def __setitem__(self, move: str, node):
        self.children[move] = node

    cpdef is_leaf(self):
        return not self.children
    

cpdef select_child(object node, unsigned int pb_c_base, float pb_c_init, float pb_c_factor, float fpu):
    cdef str move
    cdef str bestmove
    cdef float ucb
    cdef float bestucb = -np.inf
    cdef object child
    cdef object bestchild
    cdef unsigned int cN
    cdef float cQ
    cdef float pN_sqrt = sqrt(node.N)
    cdef float pb_c = log((node.N + pb_c_base + 1) / pb_c_base) * pb_c_factor + pb_c_init

    for move, child in node.children.items():
        cN = child.N
        if cN > 0:
            ucb = (child.W / cN) + pb_c * child.P * pN_sqrt / (cN + 1)
        else:
            ucb = fpu
        if ucb > bestucb:
            bestucb = ucb
            bestmove = move
            bestchild = child
    
    return bestmove, bestchild 

cpdef void evaluate_node(object node, float[:] policy_logits):
    cdef object child
    cdef str move_uci
    cdef float _max
    cdef float expsum
    cdef cnp.ndarray[DTYPE_t, ndim=1] policy_np
    cdef list actions = [map_w[move_uci] if node.to_play else map_b[move_uci] for move_uci in node.children.keys()]
    cdef Py_ssize_t action
    cdef list policy_masked = [policy_logits[action] for action in actions]

    policy_np = np.array(policy_masked)

    _max = policy_np.max()
    expsum = np.sum(np.exp(policy_np - _max))
    policy_np = np.exp(policy_np - (_max + np.log(expsum)))

    cdef DTYPE_t p
    for p, child in zip(policy_np, node.children.values()):
        child.P = p

cpdef expand_node(object node, list legal_moves, bint to_play):
    cdef str move
    node.to_play = to_play
    for move in legal_moves:
        node[move] = Node()

cpdef np.ndarray calculate_search_statistics(object root, unsigned int num_actions):
    cdef cnp.ndarray child_visits = np.zeros(num_actions, dtype=DTYPE)
    cdef str move
    cdef object child
    cdef Py_ssize_t i

    for move, child in root.children.items():
        i = map_w[move] if root.to_play else map_b[move]
        child_visits[i] = child.N
    return child_visits / np.sum(child_visits)

cpdef float value_to_01(float value):
    return (value + 1.0) / 2.0

cpdef float flip_value(float value):
    return 1.0 - value

cpdef void backup(list search_path, float value):
    cdef object node
    for node in reversed(search_path):
        node.N = node.N - node.vloss + 1
        node.W = node.W + node.vloss + value
        node.vloss = 0
        value = flip_value(value)

cpdef void add_vloss(list search_path):
    cdef object node
    for node in reversed(search_path):
        node.N += 1
        node.W -= 1
        node.vloss += 1