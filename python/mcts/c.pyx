#cython: profile=True, language_level=3
from actionspace import map_w, map_b
from configs import selfplayConfig
from libc.math cimport log, sqrt
from rchess import Move
from model import predict_fn
import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

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

    def __getitem__(self, move: int):
        return self.children[move]
    
    def __setitem__(self, move: int, node):
        self.children[move] = node

    cpdef is_leaf(self):
        return not self.children

    cpdef apply_vloss(self):
        self.N += 1
        self.W -= 1
        self.vloss += 1

    cpdef remove_vloss(self):
        self.N -= self.vloss
        self.W += self.vloss
        self.vloss = 0


cpdef find_best_move(object board, object root, object trt_func, unsigned int num_mcts_sims):
    cdef dict config = selfplayConfig
    cdef bint root_exploration_noise = config['root_exploration_noise']
    cdef float root_dirichlet_alpha = config['root_dirichlet_alpha']
    cdef float root_exploration_fraction = config['root_exploration_fraction']
    cdef float fpu_root = config['fpu_root']
    cdef float fpu_leaf = config['fpu_leaf']
    cdef unsigned int pb_c_base = config['pb_c_base']
    cdef float pb_c_init = config['pb_c_init']
    cdef float pb_c_factor = config['pb_c_factor']
    cdef unsigned int num_vl_searches = config['num_vl_searches']
    cdef bint history_flip = config['history_perspective_flip']
    cdef object rng = np.random.default_rng()
    cdef bint tree_reused = True
    cdef unsigned int move_num
    cdef list nodes_to_eval
    cdef list moves_to_node
    cdef list moves_to_nodes
    cdef list histories
    cdef unsigned int failsafe
    cdef unsigned int nodes_found
    cdef object tmp_board
    cdef float fpu
    cdef bint terminal
    cdef bint winner
    cdef cnp.ndarray[DTYPE_t, ndim=2] values
    cdef cnp.ndarray[DTYPE_t, ndim=2] policy_logits
    cdef cnp.ndarray[DTYPE_t, ndim=1] child_visits
    cdef Py_ssize_t idx

    num_mcts_sims = config['num_mcts_sims'] if num_mcts_sims == 0 else num_mcts_sims

    if root is None:
        root = Node()
        tree_reused = False

    if root.is_leaf():
        history, _ = board.history(history_flip)
        _, policy_logits = make_predictions(
                trt_func=trt_func,
                histories=[history],
                batch_size=num_vl_searches
            )
        expand_node(root, board.legal_moves(), board.to_play())
        evaluate_node(root, policy_logits[0])

    if not tree_reused and root_exploration_noise:
        add_exploration_noise(root, rng, root_dirichlet_alpha, root_exploration_fraction)

    while 1:
        if root.N >= num_mcts_sims:
            break
        nodes_to_find = num_vl_searches if root.N + num_vl_searches < num_mcts_sims else num_mcts_sims - root.N
        nodes_found = 0
        nodes_to_eval = []
        moves_to_nodes = []
        histories = []
        failsafe = 0
        while nodes_found < nodes_to_find and failsafe < 4:
            node = root
            tmp_board = board.clone()
            while not node.is_leaf():
                fpu = fpu_root if tmp_board.ply() == board.ply() else fpu_leaf
                move_num, node = select_child(node, pb_c_base=pb_c_base, pb_c_init=pb_c_init, pb_c_factor=pb_c_factor, fpu=fpu)
                node.apply_vloss()
                tmp_board.push_num(move_num)
            
            terminal, winner = tmp_board.terminal()
            if terminal:
                value = 1.0 if winner is not None else 0.0
                moves_to_node = tmp_board.moves_history(tmp_board.ply() - board.ply())
                update(root, moves_to_node, flip_value(value))
                failsafe += 1
                continue

            expand_node(node, tmp_board.legal_moves(), tmp_board.to_play())

            nodes_found += 1
            nodes_to_eval.append(node)
            moves_to_node = tmp_board.moves_history(tmp_board.ply() - board.ply())
            moves_to_nodes.append(moves_to_node)
            history, _ = tmp_board.history(history_flip)
            histories.append(history)

        if nodes_found == 0:
            failsafe += 1
            continue

        values, policy_logits = make_predictions(
                trt_func=trt_func,
                histories=histories,
                batch_size=num_vl_searches
            )
        
        for idx in range(len(nodes_to_eval)):
            evaluate_node(nodes_to_eval[idx], policy_logits[idx])
            update(root, moves_to_nodes[idx], flip_value(value_to_01(values[idx].item())))
        del nodes_to_eval, values, policy_logits, moves_to_nodes, histories

    move_num = select_best_move(root, rng, temp=0)
    child_visits = calculate_search_statistics(root, 1858)
    return move_num, root, child_visits

cdef add_exploration_noise(object node, object rng, float dirichlet_alpha, float exploration_fraction):
    cdef cnp.ndarray noise = rng.dirichlet([dirichlet_alpha] * len(node.children))
    cdef object child
    cdef float n
    noise /= np.sum(noise)
    for n, child in zip(noise, node.children.values()):
        child.P = child.P * (1 - exploration_fraction) + n * exploration_fraction

cdef select_best_move(object node, object rng, float temp):
    cdef list moves = list(node.children.keys())
    cdef cnp.ndarray probs = np.array([node[move].N for move in moves])
    if temp == 0:
        return moves[np.argmax(probs)]
    else:
        probs = np.power(probs, 1.0 / temp)
        probs /= np.sum(probs)
        return rng.choice(moves, p=probs)

cdef make_predictions(object trt_func, list histories, unsigned int batch_size):
    cdef images = np.array(histories, dtype=np.int64)
    if images.shape[0] < batch_size:
        images = np.pad(images, ((0, batch_size - images.shape[0]), (0, 0)), mode='constant')

    values, policy_logits = predict_fn(
        trt_func=trt_func,
        images=images
    )

    return np.array(values, dtype=DTYPE), np.array(policy_logits, dtype=DTYPE)

cdef select_child(object node, unsigned int pb_c_base, float pb_c_init, float pb_c_factor, float fpu):
    cdef unsigned int move
    cdef unsigned int bestmove
    cdef float ucb
    cdef float bestucb = -99.9
    cdef object child
    cdef object bestchild
    cdef unsigned int N = node.N
    cdef unsigned int cN
    cdef float cW
    cdef float cP
    cdef float pN_sqrt = sqrt(N)
    cdef float pb_c = PB_C(N, pb_c_base, pb_c_init, pb_c_factor)

    for move, child in node.children.items():
        cN = child.N
        if cN > 0:
            cW = child.W
            cP = child.P
            ucb = UCB(cN, cW, cP, pN_sqrt, pb_c)
        else:
            ucb = fpu
        if ucb > bestucb:
            bestucb = ucb
            bestmove = move
            bestchild = child
    
    return bestmove, bestchild 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float UCB(unsigned int cN, float cW, float cP, float pN_sqrt, float pb_c):
    return (cW / cN) + pb_c * cP * pN_sqrt / (cN + 1)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float PB_C(unsigned int N, unsigned int pb_c_base, float pb_c_init, float pb_c_factor):
    return log((N + pb_c_base + 1) / pb_c_base) * pb_c_factor + pb_c_init

cdef void evaluate_node(object node, cnp.ndarray[DTYPE_t, ndim=1] policy_logits):
    cdef object child
    cdef float _max
    cdef float expsum
    cdef cnp.ndarray[DTYPE_t, ndim=1] policy_np
    cdef Py_ssize_t action
    cdef dict map = map_w if node.to_play else map_b

    policy_np = np.array([policy_logits[map[Move(move).uci()]] for move in node.children.keys()])
    _max = np.max(policy_np)
    expsum = np.sum(np.exp(policy_np - _max))
    policy_np = np.exp(policy_np - (_max + np.log(expsum)))

    cdef DTYPE_t p
    for p, child in zip(policy_np, node.children.values()):
        child.P = p

cdef void expand_node(object node, list legal_moves, bint to_play):
    cdef object move
    node.to_play = to_play
    for move in legal_moves:
        node[hash(move)] = Node()

cdef cnp.ndarray calculate_search_statistics(object root, unsigned int num_actions):
    cdef cnp.ndarray child_visits = np.zeros(num_actions, dtype=DTYPE)
    cdef unsigned int move
    cdef object child
    cdef object m
    cdef Py_ssize_t i

    for move, child in root.children.items():
        m = Move(move)
        i = map_w[m.uci()] if root.to_play else map_b[m.uci()]
        child_visits[i] = child.N
    return child_visits / np.sum(child_visits)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float value_to_01(float value):
    return (value + 1.0) / 2.0

cdef inline float flip_value(float value):
    return 1.0 - value

cdef void update(object root, list moves_to_leaf, float value):
    root.N += 1
    cdef object node = root
    cdef object move
    for move in moves_to_leaf:
        node = node[move]
    node.remove_vloss()
    node.N += 1
    node.W += flip_value(value)