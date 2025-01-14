#cython: profile=True, language_level=3

import time
from mcts import Node, find_best_move
from rchess import Board
from chess import Board as TbBoard
from configs import selfplayConfig
import numpy as np
cimport numpy as cnp

cnp.import_array()

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
    cdef bint terminal, winner
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
                outcome_str = f"Draw by {board.outcome()}"
        print(f"{outcome_str} in {moves_played} moves in {end_time - start_time:.2f} seconds")
    return images_np, statistics, terminal_values
