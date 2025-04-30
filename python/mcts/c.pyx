#cython: profile=False, language_level=3
from actionspace import map_w, map_b
from rchess import Move
from model import predict_fn
from chess import Board as TbBoard
import chess.syzygy as syzygy
import os
import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport log, sqrt, exp
import time

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

BATCH_DTYPE = np.int64
ctypedef cnp.int64_t BATCH_DTYPE_t

cdef class Node:
    def __cinit__(self, float p):
        self.P = p
        self.W = 0.0
        self.N = 0
        self.vloss = 0
        self.children = {}
        self.to_play = False

        # Debug info
        self.debug_info = {}

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

cdef class MCTS:
    def __cinit__(self, dict cfg):
        self.history_flip = cfg['history_perspective_flip']
        self.root_exploration_noise = cfg['root_exploration_noise']
        self.root_dirichlet_alpha = cfg['root_dirichlet_alpha']
        self.root_exploration_fraction = cfg['root_exploration_fraction']
        self.fpu_root = cfg['fpu_root']
        self.fpu_leaf = cfg['fpu_leaf']
        self.pb_c_init = cfg['pb_c_init']
        self.pb_c_factor = cfg['pb_c_factor']
        self.pb_c_base = cfg['pb_c_base']
        self.moves_softmax_temp = cfg['moves_softmax_temp']
        self.num_vl_searches = cfg['num_vl_searches']
        self.num_mcts_sampling_moves = cfg['num_mcts_sampling_moves']
        self.resignation_threshold = cfg['resignation_threshold']
        self.tablebase_search = cfg['tablebase_search']

        self.num_planes = 109
        self.rng = np.random.default_rng()
        self.m_w = map_w
        self.m_b = map_b

        self.tablebase = syzygy.open_tablebase(os.path.join(cfg['project_dir'], 'syzygy/3-4-5'))

    cpdef get_history_flip(self):
        return self.history_flip

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef expand_root(self, object board, Node root, object trt_func, bint debug):
        if root is None:
            root = Node(0.0) 
        cdef cnp.ndarray[BATCH_DTYPE_t, ndim=2] batch = np.zeros((self.num_vl_searches, self.num_planes), dtype=BATCH_DTYPE)
        cdef long[:, :] batch_mw = batch

        root.to_play = board.to_play()
        history, _ = board.history(self.history_flip)

        for idx in range(self.num_planes):
            batch_mw[0, idx] = history[idx]
        _, policy_logits = make_predictions(
                trt_func=trt_func,
                batch=batch
            )
        self.expand_and_evaluate_node(root, policy_logits[0], board.legal_moves_tuple(), debug)
        return root

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef search(self, object board, Node root, object trt_func, bint debug):
        cdef unsigned int move_num
        cdef list nodes_to_eval = []
        cdef list moves_to_nodes = []
        cdef list moves_to_node = []
        cdef list eval_nodes_legal_moves = []
        cdef unsigned int failsafe = 0
        cdef unsigned int nodes_found = 0
        cdef unsigned int nodes_to_find = self.num_vl_searches
        cdef object tmp_board
        cdef float fpu
        cdef bint terminal, is_drawn
        cdef cnp.ndarray[DTYPE_t, ndim=2] values
        cdef cnp.ndarray[DTYPE_t, ndim=2] policy_logits
        cdef cnp.ndarray[BATCH_DTYPE_t, ndim=2] batch = np.zeros((self.num_vl_searches, self.num_planes), dtype=BATCH_DTYPE)
        cdef long[:, :] batch_mw = batch
        cdef list history
        cdef Py_ssize_t idx
        cdef Node node
        cdef unsigned int depth_to_root

        while nodes_found < nodes_to_find and failsafe < 2:
            depth_to_root = 0
            node = root
            tmp_board = board.clone()
            while not node.is_leaf():
                fpu = self.fpu_leaf if depth_to_root > 0 else self.fpu_root
                move_num = select_child(node, pb_c_base=self.pb_c_base, pb_c_init=self.pb_c_init, pb_c_factor=self.pb_c_factor, fpu=fpu, debug=debug)
                node = node[move_num]
                tmp_board.push_num(move_num)
                depth_to_root += 1
            
            terminal, is_drawn = tmp_board.mid_search_terminal(depth_to_root)
            if terminal:
                # If a position is not drawn, the only other possible outcome is checkmate
                # Because the move has already been played (while traversing the tree), 
                # this node is from the perspective of the checkmated player
                value = 0.5 if is_drawn else 0.0
                update(root, tmp_board.moves_history(depth_to_root), value)
                if debug:
                    node.debug_info["init_value"] = flip_value(value)
                failsafe += 1
                continue

            moves_to_node = tmp_board.moves_history(depth_to_root)
            add_vloss(root, moves_to_node)

            node.to_play = tmp_board.to_play()

            nodes_to_eval.append(node)
            moves_to_nodes.append(moves_to_node)
            eval_nodes_legal_moves.append(tmp_board.legal_moves_tuple())
            history, _ = tmp_board.history(self.history_flip)
            for idx in range(self.num_planes):
                batch_mw[nodes_found, idx] = history[idx]
            nodes_found += 1

        if nodes_found == 0:
            return root

        values, policy_logits = make_predictions(
            trt_func=trt_func,
            batch=batch
        )
        
        for idx in range(nodes_found):
            self.expand_and_evaluate_node(nodes_to_eval[idx], policy_logits[idx], eval_nodes_legal_moves[idx], debug)
            value = value_to_01(values[idx].item())
            update(root, moves_to_nodes[idx], value)
            if debug:
                nodes_to_eval[idx].debug_info["init_value"] = flip_value(value)
        return root

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cpdef find_best_move(self, object board, Node root, object trt_func, unsigned int num_sims, float time_limit, bint debug):
        assert (num_sims > 0) != (time_limit > 0.0), "Only one & at least one, either num_sims or time_limit, must be greater than 0."

        cdef unsigned int move_num
        cdef list nodes_to_eval
        cdef list moves_to_nodes
        cdef list moves_to_node
        cdef list eval_nodes_legal_moves
        cdef unsigned int failsafe
        cdef unsigned int nodes_found
        cdef unsigned int nodes_to_find
        cdef object tmp_board
        cdef float fpu
        cdef bint terminal, is_drawn
        cdef cnp.ndarray[DTYPE_t, ndim=2] values
        cdef cnp.ndarray[DTYPE_t, ndim=2] policy_logits
        cdef cnp.ndarray[DTYPE_t, ndim=1] child_visits
        cdef cnp.ndarray[BATCH_DTYPE_t, ndim=2] batch = np.zeros((self.num_vl_searches, self.num_planes), dtype=BATCH_DTYPE)
        cdef long[:, :] batch_mw = batch
        cdef list history
        cdef Py_ssize_t idx
        cdef Node node
        cdef unsigned int depth_to_root
        cdef double start_time, end_time
        cdef bint time_limit_set = True if time_limit > 0.0 else False


        if root is None or root.is_leaf():
            root = Node(0.0)
            root.to_play = board.to_play()

        if root.is_leaf():
            history, _ = board.history(self.history_flip)
            for idx in range(self.num_planes):
                batch_mw[0, idx] = history[idx]
            _, policy_logits = make_predictions(
                    trt_func=trt_func,
                    batch=batch
                )
            self.expand_and_evaluate_node(root, policy_logits[0], board.legal_moves_tuple(), debug)
            root.N += 1

        if self.root_exploration_noise:
            add_exploration_noise(root, self.rng, self.root_dirichlet_alpha, self.root_exploration_fraction)

        start_time = time.time()

        while 1:
            if time_limit_set:
                if time.time() - start_time > time_limit:
                    break
                nodes_to_find = self.num_vl_searches
            else:
                if root.N >= num_sims:
                    break
                nodes_to_find = self.num_vl_searches if root.N + self.num_vl_searches <= num_sims else num_sims - root.N
            nodes_found = 0
            nodes_to_eval = []
            eval_nodes_legal_moves = []
            moves_to_nodes = []
            failsafe = 0
            while nodes_found < nodes_to_find and failsafe < 2:
                depth_to_root = 0
                node = root
                tmp_board = board.clone()
                while not node.is_leaf():
                    fpu = self.fpu_leaf if depth_to_root > 0 else self.fpu_root
                    move_num = select_child(node, pb_c_base=self.pb_c_base, pb_c_init=self.pb_c_init, pb_c_factor=self.pb_c_factor, fpu=fpu, debug=debug)
                    node = node[move_num]
                    tmp_board.push_num(move_num)
                    depth_to_root += 1
                
                terminal, is_drawn = tmp_board.mid_search_terminal(depth_to_root)
                if terminal:
                    # If a position is not drawn, the only other possible outcome is checkmate
                    # Because the move has already been played (while traversing the tree), 
                    # this node is from the perspective of the checkmated player
                    value = 0.5 if is_drawn else 0.0
                    update(root, tmp_board.moves_history(depth_to_root), value)
                    if debug:
                        node.debug_info["init_value"] = flip_value(value)
                    failsafe += 1
                    continue

                # If tablebase search is enabled, check if the position is in the tablebase
                if self.tablebase_search and tmp_board.pieces_on_board() <= 5:
                    tb_board = TbBoard(tmp_board.fen())
                    wdl = self.tablebase.get_wdl(tb_board)
                    if wdl is not None and not (wdl == 1 or wdl == -1):
                        if wdl == 0:
                            value = 0.5
                        else:
                            to_play = tmp_board.to_play()
                            winner = to_play if wdl > 0 else not to_play
                            value = 1.0 if winner == to_play else 0.0
                        update(root, tmp_board.moves_history(depth_to_root), value)
                        if debug:
                            node.debug_info["init_value"] = flip_value(value)
                        failsafe += 1
                        continue

                moves_to_node = tmp_board.moves_history(depth_to_root)
                add_vloss(root, moves_to_node)

                node.to_play = tmp_board.to_play()

                nodes_to_eval.append(node)
                moves_to_nodes.append(moves_to_node)
                eval_nodes_legal_moves.append(tmp_board.legal_moves_tuple())
                history, _ = tmp_board.history(self.history_flip)
                for idx in range(self.num_planes):
                    batch_mw[nodes_found, idx] = history[idx]
                nodes_found += 1

            if nodes_found == 0:
                failsafe += 1
                continue

            values, policy_logits = make_predictions(
                    trt_func=trt_func,
                    batch=batch
                )
            
            for idx in range(nodes_found):
                self.expand_and_evaluate_node(nodes_to_eval[idx], policy_logits[idx], eval_nodes_legal_moves[idx], debug)
                value = value_to_01(values[idx].item())
                update(root, moves_to_nodes[idx], value)
                if debug:
                    nodes_to_eval[idx].debug_info["init_value"] = flip_value(value)

        end_time = time.time()
        if debug:
            root.debug_info["elapsed_time"] = end_time - start_time
        
        move_num = self.select_best_move(root, temp=self.moves_softmax_temp if board.ply() <= self.num_mcts_sampling_moves else 0.0)
        child_visits = self.calculate_search_statistics(root)
        
        if board.ply() > self.num_mcts_sampling_moves and self.resignation_threshold > 0.0:
            q = root[move_num].W / root[move_num].N if root[move_num].N > 0 else self.fpu_root
            if q < self.resignation_threshold:
                move_num = 0 # Resignation move

        return move_num, root, child_visits

    @cython.nonecheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    cdef void expand_and_evaluate_node(self, Node node, float[:] policy_logits, list legal_moves_tuple, bint debug):
        cdef dict m = self.m_w if node.to_play else self.m_b
        cdef str move_uci
        cdef Py_ssize_t moves_count = len(legal_moves_tuple)
        cdef cnp.ndarray[DTYPE_t, ndim=1] policy = np.zeros(moves_count, dtype=DTYPE)
        cdef float[:] policy_mw = policy
        cdef float p, logsumexp
        cdef float _max = -99.9
        cdef float expsum = 0.0
        cdef Node child
        cdef Py_ssize_t i, p_idx
        
        for i in range(moves_count):
            move_uci = legal_moves_tuple[i][1]
            p_idx = m[move_uci]
            p = policy_logits[p_idx]
            policy_mw[i] = p
            if p > _max:
                _max = p

        for i in range(moves_count):
            expsum += exp(policy_mw[i] - _max)

        logsumexp = log(expsum) + _max
        for i in range(moves_count):
            node.add_child(legal_moves_tuple[i][0], exp(policy_mw[i] - logsumexp))
        if debug:
            for i in range(moves_count):
                child = node.children[legal_moves_tuple[i][0]]
                child.debug_info["move_uci"] = legal_moves_tuple[i][1]
                child.debug_info["move_num"] = legal_moves_tuple[i][0]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef cnp.ndarray calculate_search_statistics(self, Node root):
        cdef cnp.ndarray[DTYPE_t, ndim=1] child_visits = np.zeros(1858, dtype=DTYPE)
        cdef float[:] child_visits_mw = child_visits
        cdef dict m = self.m_w if root.to_play else self.m_b
        cdef unsigned int move_num, cN
        cdef object move_str
        cdef Node child
        cdef Py_ssize_t i
        cdef float _sum = 0.0

        _sum = 0.0
        for move_num, child in root.children.items():
            cN = child.N
            move_str = Move(move_num).uci()
            i = m[move_str]
            child_visits_mw[i] = <float>cN
            _sum += <float>cN
        for i in range(1858):
            child_visits_mw[i] /= _sum
        return child_visits

    cpdef select_best_move(self, Node node, float temp):
        cdef list moves = list(node.children.keys())
        cdef cnp.ndarray probs = np.array([node[move].N for move in moves])
        if temp == 0.0:
            return moves[np.argmax(probs)]
        else:
            probs = np.power(probs, 1.0 / temp)
            probs /= np.sum(probs)
            return self.rng.choice(moves, p=probs)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void add_exploration_noise(Node node, object rng, float dirichlet_alpha, float exploration_fraction):
    cdef cnp.ndarray[DTYPE_t, ndim=1] noise = rng.dirichlet([dirichlet_alpha] * len(node.children)).astype(DTYPE)
    cdef float[:] noise_mw = noise
    cdef unsigned int noise_size = noise.shape[0]
    cdef Node child
    cdef float n
    cdef Py_ssize_t i
    cdef DTYPE_t sum_noise = 0.0
    
    for i in range(noise_size):
        sum_noise += noise_mw[i]

    i = 0
    for child in node.children.values():
        child.P = child.P * (1 - exploration_fraction) + noise_mw[i] * exploration_fraction
        i += 1

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef make_predictions(object trt_func, cnp.ndarray[BATCH_DTYPE_t, ndim=2] batch):
    cdef cnp.ndarray[DTYPE_t, ndim=2] values, policy_logits
    cdef object values_tf, policy_logits_tf

    values_tf, policy_logits_tf = predict_fn(
        trt_func=trt_func,
        images=batch
    )

    values = np.array(values_tf, dtype=DTYPE)
    policy_logits = np.array(policy_logits_tf, dtype=DTYPE)
    return values, policy_logits

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
@cython.initializedcheck(False)
cdef unsigned int select_child(Node node, unsigned int pb_c_base, float pb_c_init, float pb_c_factor, float fpu, bint debug):
    cdef Node child
    cdef unsigned int move
    cdef unsigned int bestmove = 0
    cdef float ucb
    cdef float bestucb = -99.9
    cdef unsigned int N = node.N
    cdef unsigned int cN
    cdef float cQ = 0.0
    cdef float pN_sqrt = sqrt(N)
    cdef float pb_c = PB_C(N, pb_c_base, pb_c_init, pb_c_factor)

    for move, child in node.children.items():
        cN = child.N
        cQ = child.W / cN if cN > 0 else fpu
        ucb = cQ + UCB(cN, child.P, pN_sqrt, pb_c)
        if ucb > bestucb:
            bestucb = ucb
            bestmove = move
        if debug:
            child.debug_info["ucb"] = ucb
    
    return bestmove 

@cython.cdivision(True)
cdef inline float UCB(unsigned int cN, float cP, float pN_sqrt, float pb_c) noexcept:
    return pb_c * cP * (pN_sqrt / (cN + 1))

@cython.cdivision(True)
cdef inline float PB_C(unsigned int pN, unsigned int pb_c_base, float pb_c_init, float pb_c_factor) noexcept:
    return log((pN + pb_c_base + 1) / pb_c_base) * pb_c_factor + pb_c_init

@cython.cdivision(True)
cdef inline float value_to_01(float value) noexcept:
    return (value + 1.0) / 2.0

cdef inline float flip_value(float value) noexcept:
    return 1.0 - value

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void update(Node root, list moves_to_leaf, float value):
    root.N += 1
    cdef Node node = root
    cdef unsigned int move
    cdef int moves_count = len(moves_to_leaf)
    if moves_count % 2 == 1:
        value = flip_value(value)
    for move in moves_to_leaf:
        node = node[move]
        node.remove_vloss()
        node.N += 1
        node.W += value
        value = flip_value(value)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef inline void add_vloss(Node root, list moves_to_leaf):
    cdef Node node = root
    cdef unsigned int move
    for move in moves_to_leaf:
        node = node[move]
        node.apply_vloss()