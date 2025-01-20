#cython: profile=True, language_level=3

import time
from mcts import Node, find_best_move, gather_nodes_to_process, process_node, calculate_search_statistics, select_best_move
from rchess import Board
from chess import Board as TbBoard
from configs import selfplayConfig
from model import predict_fn
import numpy as np
cimport numpy as cnp

cnp.import_array()

DTYPE = np.float32
ctypedef cnp.float32_t DTYPE_t

BATCH_DTYPE = np.int64
ctypedef cnp.int64_t BATCH_DTYPE_t

cdef inline int terminal_value(bint on_turn, bint winner):
    if winner == on_turn:
        return 1
    else:
        return -1

cpdef play_game(object trt_func, object tablebase, unsigned int verbose):
    cdef dict config = selfplayConfig
    cdef bint history_flip = config["history_perspective_flip"]
    cdef unsigned int moves_played = 0
    cdef list images = []
    cdef list statistics = []
    cdef bint terminal
    cdef double start_time, end_time
    cdef str outcome_str = ""

    cdef object board = Board()
    cdef object root = Node(0.0)

    start_time = time.time()
    while 1:
        terminal, winner = board.terminal()
        if terminal:
            break

        move, root, child_visits = find_best_move(board, root, trt_func, 800, False)

        history, _ = board.history(history_flip)
        images.append(history)
        statistics.append(child_visits)

        board.push_num(move)
        root = root[move]

        moves_played += 1

        if tablebase is not None and board.pieces_on_board() <= 5:
            tb_board = TbBoard(board.fen())
            wdl = tablebase.get_wdl(tb_board)
            if wdl is not None and not (wdl == 1 or wdl == -1):
                if wdl == 0:
                    outcome_str = "Tablebase draw"
                    break
                else:
                    winner = board.to_play() if wdl > 0 else not board.to_play()
                    outcome_str = "White wins by tablebase" if winner == True else "Black wins by tablebase"
                break
    end_time = time.time()

    cdef cnp.ndarray images_np = np.array(images)
    cdef cnp.ndarray statistics_np = np.array(statistics)
    cdef cnp.ndarray[cnp.int8_t, ndim=1] terminal_values = np.zeros(moves_played, dtype=np.int8)
    if winner is not None:
        for i in range(moves_played):
            terminal_values[i] = terminal_value(True if i % 2 == 0 else False, winner)        

    if verbose > 1:
        print(board)
    if verbose > 0:
        if outcome_str == "":
            if winner is not None:
                outcome_str = "White wins by checkmate" if winner == True else "Black wins by checkmate"
            else:
                outcome_str = f"Draw by {board.outcome_str()}"
        print(f"{outcome_str} in {moves_played} moves in {end_time - start_time:.2f} seconds")
    return images_np, statistics, terminal_values

cpdef play_n_games(object trt_func, object tablebase, int n_simul_games, int num_games, unsigned int verbose):
    cdef dict config = selfplayConfig
    cdef unsigned int num_mcts_sims = config["num_mcts_sims"]
    cdef unsigned int num_vl_searches = config["num_vl_searches"]
    cdef unsigned int batch_size = num_vl_searches * n_simul_games
    cdef cnp.ndarray[BATCH_DTYPE_t, ndim=2] batch = np.zeros((batch_size, 109), dtype=BATCH_DTYPE)

    cdef list games = []
    cdef list roots = []
    for _ in range(n_simul_games):
        roots.append(Node(0.0))
        games.append(Board())

    cdef int games_finished = 0
    cdef Py_ssize_t i
    cdef list nodes_to_process
    cdef bint finished = False
    while not finished:
        nodes_to_process = []
        for i in range(n_simul_games):
            nodes_to_process.extend(gather_nodes_to_process(roots[i], games[i], num_vl_searches, False))

        if len(nodes_to_process) > 0:
            for i in range(len(nodes_to_process)):
                batch[i] = nodes_to_process[i].image
            values, policy_logits = make_predictions(trt_func, batch)

            for i in range(len(nodes_to_process)):
                process_node(nodes_to_process[i], roots[i], policy_logits[i], values[i], False)

        for i in range(n_simul_games):
            if roots[i].N >= num_mcts_sims:
                move, root, _ = find_best_move(games[i], roots[i], trt_func, 0, False)
                games[i].push_num(move)
                roots[i] = root[move]

                terminal, winner = games[i].terminal()
                if terminal:
                    games_finished += 1
                    if games_finished == num_games:
                        finished = True
                        break
                    games[i] = Board()
                    roots[i] = Node(0.0)


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
    