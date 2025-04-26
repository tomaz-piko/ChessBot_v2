#cython: profile=True, language_level=3

import time
from mcts import Node
from rchess import Board
from chess import Board as TbBoard
cimport numpy as cnp

cnp.import_array()

cdef inline float terminal_value(bint on_turn, bint winner):
    if winner == on_turn:
        return 1.0
    else:
        return -1.0

cpdef play_game(object mctsSearch, object trt_func, object tablebase, unsigned int verbose):
    cdef unsigned int moves_played = 0
    cdef bint history_flip = mctsSearch.get_history_flip()
    cdef bint to_play
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

        move, root, child_visits = mctsSearch.find_best_move(board, root, trt_func, 800, 0.0, False)

        history, _ = board.history(history_flip)
        images.append(history)
        statistics.append(child_visits)

        if move == 0: # Resignation move
            winner = not board.to_play()
            outcome_str = "White wins by resignation" if winner == True else "Black wins by resignation"
            break

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
                    to_play = board.to_play()
                    winner = to_play if wdl > 0 else not to_play
                    outcome_str = "White wins by tablebase" if winner == True else "Black wins by tablebase"
                break
    end_time = time.time()

    cdef list terminal_values = [0.0] * moves_played
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
    return images, (statistics, terminal_values)
