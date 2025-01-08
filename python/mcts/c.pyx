#cython: profile=True, language_level=3
from actionspace import map_w, map_b
from configs import selfplayConfig
from rchess import Move
from model import predict_fn
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, sqrt, exp

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

cdef dict m_w = map_w
cdef dict m_b = map_b

cdef class Node:
    def __cinit__(self):
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

    cdef inline Node get_child(self, int move):
        return self.children[move]

    cdef inline void add_child(self, int move, float p):
        self.children[move] = Node(p)

    cdef inline bint is_leaf(self):
        return not self.children

    cdef inline void apply_vloss(self):
        self.N += 1
        self.W -= 1
        self.vloss += 1

    cdef inline void remove_vloss(self):
        self.N -= self.vloss
        self.W += self.vloss
        self.vloss = 0

@cython.boundscheck(False)
cpdef find_best_move(object board, Node root, object trt_func, unsigned int num_mcts_sims):
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
    cdef Node node
    cdef unsigned int depth

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
        evaluate_node(root, policy_logits[0], board.legal_moves())

    if not tree_reused and root_exploration_noise:
        add_exploration_noise(root, rng, root_dirichlet_alpha, root_exploration_fraction)

    while 1:
        if root.N >= num_mcts_sims:
            break
        nodes_to_find = num_vl_searches if root.N + num_vl_searches < num_mcts_sims else num_mcts_sims - root.N
        nodes_found = 0
        nodes_to_eval = []
        eval_nodes_legal_moves = []
        moves_to_nodes = []
        histories = []
        failsafe = 0
        while nodes_found < nodes_to_find and failsafe < 4:
            depth = 0
            node = root
            tmp_board = board.clone()
            while not node.is_leaf():
                fpu = fpu_leaf if depth > 0 else fpu_root
                move_num = select_child(node, pb_c_base=pb_c_base, pb_c_init=pb_c_init, pb_c_factor=pb_c_factor, fpu=fpu)
                node = node[move_num]
                node.apply_vloss()
                tmp_board.push_num(move_num)
                depth += 1
            
            terminal, winner = tmp_board.terminal()
            if terminal:
                value = 1.0 if winner is not None else 0.0
                moves_to_node = tmp_board.moves_history(tmp_board.ply() - depth + 1)
                update(root, moves_to_node, flip_value(value))
                failsafe += 1
                continue

            node.to_play = tmp_board.to_play()

            nodes_found += 1
            nodes_to_eval.append(node)
            moves_to_nodes.append(tmp_board.moves_history(tmp_board.ply() - depth + 1))
            eval_nodes_legal_moves.append(tmp_board.legal_moves())
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
            evaluate_node(nodes_to_eval[idx], policy_logits[idx], eval_nodes_legal_moves[idx])
            update(root, moves_to_nodes[idx], flip_value(value_to_01(values[idx].item())))
        del nodes_to_eval, values, policy_logits, moves_to_nodes, histories

    move_num = select_best_move(root, rng, temp=0)
    child_visits = calculate_search_statistics(root, 1858)
    return move_num, root, child_visits

cdef add_exploration_noise(Node node, object rng, float dirichlet_alpha, float exploration_fraction):
    cdef cnp.ndarray noise = rng.dirichlet([dirichlet_alpha] * len(node.children))
    cdef Node child
    cdef float n
    noise /= np.sum(noise)
    for n, child in zip(noise, node.children.values()):
        child.P = child.P * (1 - exploration_fraction) + n * exploration_fraction

cdef select_best_move(Node node, object rng, float temp):
    cdef list moves = list(node.children.keys())
    cdef cnp.ndarray probs = np.array([node[move].N for move in moves])
    if temp == 0:
        return moves[np.argmax(probs)]
    else:
        probs = np.power(probs, 1.0 / temp)
        probs /= np.sum(probs)
        return rng.choice(moves, p=probs)

cdef make_predictions(object trt_func, list histories, unsigned int batch_size):
    cdef cnp.ndarray images = np.array(histories, dtype=np.int64)
    if images.shape[0] < batch_size:
        images = np.pad(images, ((0, batch_size - images.shape[0]), (0, 0)), mode='constant')

    values, policy_logits = predict_fn(
        trt_func=trt_func,
        images=images
    )

    return np.array(values, dtype=DTYPE), np.array(policy_logits, dtype=DTYPE)

cdef unsigned int select_child(Node node, unsigned int pb_c_base, float pb_c_init, float pb_c_factor, float fpu):
    cdef Node child
    cdef unsigned int move
    cdef unsigned int bestmove = 0
    cdef float ucb
    cdef float bestucb = -99.9
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
    
    return bestmove 

@cython.cdivision(True)
cdef inline float UCB(unsigned int cN, float cW, float cP, float pN_sqrt, float pb_c) noexcept:
    return (cW / cN) + pb_c * cP * pN_sqrt / (cN + 1)

@cython.cdivision(True)
cdef inline float PB_C(unsigned int N, unsigned int pb_c_base, float pb_c_init, float pb_c_factor) noexcept:
    return log((N + pb_c_base + 1) / pb_c_base) * pb_c_factor + pb_c_init

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef void evaluate_node(Node node, float[:] policy_logits, list legal_moves):
    cdef dict m = m_w if node.to_play else m_b
    cdef object move_obj
    cdef str move_uci
    cdef Py_ssize_t moves_count = len(legal_moves)
    cdef Py_ssize_t i, p_idx
    cdef cnp.ndarray[DTYPE_t, ndim=1] policy = np.zeros(moves_count, dtype=DTYPE)
    cdef float[:] policy_mw = policy
    cdef float p, logsumexp
    cdef float _max = -99.9
    cdef float expsum = 0.0
    cdef Node child
    
    i = 0
    for i in range(moves_count):
        move_uci = legal_moves[i].uci()
        p_idx = m[move_uci]
        p = policy_logits[p_idx]
        policy_mw[i] = p
        if p > _max:
            _max = p

    i = 0
    for i in range(moves_count):
        expsum += exp(policy_mw[i] - _max)

    i = 0
    logsumexp = log(expsum) + _max
    for i in range(moves_count):
        move_obj = legal_moves[i]
        node.add_child(hash(move_obj), exp(policy[i] - logsumexp))

cdef cnp.ndarray calculate_search_statistics(Node root, unsigned int num_actions):
    cdef cnp.ndarray child_visits = np.zeros(num_actions, dtype=DTYPE)
    cdef unsigned int move_num
    cdef object move
    cdef Node child
    cdef Py_ssize_t i

    for move_num, child in root.children.items():
        move = Move(move_num)
        i = map_w[move.uci()] if root.to_play else map_b[move.uci()]
        child_visits[i] = child.N
    return child_visits / np.sum(child_visits)

@cython.cdivision(True)
cdef inline float value_to_01(float value) noexcept:
    return (value + 1.0) / 2.0

cdef inline float flip_value(float value) noexcept:
    return 1.0 - value

cdef void update(Node root, list moves_to_leaf, float value):
    root.N += 1
    cdef Node node = root
    cdef unsigned int move
    for move in moves_to_leaf:
        node = node[move]
    node.remove_vloss()
    node.N += 1
    node.W += flip_value(value)