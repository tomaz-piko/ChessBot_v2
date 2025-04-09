from mcts import Node
from rchess import Board
from rchess import Move

cpdef solve_tests(object mctsSearch, object trt_func, list tests, unsigned int num_mcts_sims, float time_limit):
    cdef Py_ssize_t num_tests = len(tests)
    cdef Py_ssize_t i
    cdef object board
    cdef object root
    cdef unsigned int move_num
    cdef str move_uci
    cdef dict test
    cdef dict move_points
    cdef unsigned int sum_results = 0

    for i in range(num_tests):
        test = tests[i]
        test_results = test["results"]

        board = Board(test['fen'])
        root = Node(0.0)
        move_num, _, _ = mctsSearch.find_best_move(board, root, trt_func, num_mcts_sims, time_limit, False)        

        move_uci = Move(move_num).uci()
        if move_uci in test_results:
            sum_results += test_results[move_uci]
        
    return sum_results