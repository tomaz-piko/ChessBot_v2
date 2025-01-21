from multiprocessing import Manager, Process
from math import ceil
import json
import time
import sys
import re
from datetime import datetime
from rchess import Board, Move
from configs import defaultConfig as config

def _chunk_into_n(lst: list, n: int) -> list:
    """Splits an array into n chunks

    Args:
        lst (list): List to be splitted
        n (int): Number of chunks

    Returns:
        list: List of n chunks (last chunk may be smaller than the rest)
    """
    size = ceil(len(lst) / n)
    return list(
        map(lambda x: lst[x * size:x * size + size],
        list(range(n)))
    )

def _load_mosca_sts() -> list:
    """Parses the STS test suite file and returns a list of tests

    Returns:
        list: List of tests. Each test is a dictionary with the following:
            - fen: FEN string of the position
            - group: Group name (15 motifs in total)
            - results: Dictionary with UCI move as key and score as value (multiple winning moves but one is still best)
    """

    fileR = open(f"{config['project_dir']}/test_suites/STS1-STS15_LAN_v3.epd", "r")
    lines = fileR.readlines()
    fileR.close()

    tests = []
    for line in lines:
        line_info = line.split('; ')
        fen = line_info[0].split(' bm ')[0]
        for info in line_info:
            if 'id' in info:
                id = re.findall(r'"([^"]*)"', info)[0]
        group = id[:-4]
        uci_moves = re.findall(r'"([^"]*)"', line_info[-1])[0].split(' ')
        moves_points = [int(points) for points in re.findall(r'"([^"]*)"', line_info[-2])[0].split(' ')]
        results = {uci_moves[i]: moves_points[i] for i in range(len(uci_moves))}
        test = {"fen": fen, "group": group, "results": results}
        tests.append(test)
    return tests

def _solve_tests(tests: list, results: dict, use_fake_model: bool = False, num_mcts_sims: int = 800, time_limit: float = 1.0):
    """Loads a TRT model and solves a list of tests using MCTS with "time_limit" of seconds per move. Appends the results to the results dictionary

    Args:
        tests (list): List of tests to be solved. Each test is a dictionary with the following:
            - fen: FEN string of the position
            - group: Group name (15 motifs in total)
            - results: Dictionary with UCI move as key and score as value (multiple winning moves but one is still best)
        results (dict): Dictionary to store the results. Each entry is a group name with a list of scores achieved by the model
        model_path (str, optional): Which model from checkpoints to load. Defaults to "latest".
        time_limit (float, optional): Time limit per test / move (solutions are one movers). Defaults to 1.0.
    """
    if use_fake_model:
        from utils import FakeTRTFunc
        trt_func = FakeTRTFunc()
    else:
        import tensorflow as tf
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        from model import load_as_trt_model
        trt_func, _ = load_as_trt_model()
    from configs import selfplayConfig
    from mcts import MCTS
    mcts = MCTS(selfplayConfig)

    for test in tests:
        fen = test["fen"]
        possible_scores = test["results"]
        board = Board(fen)
        try:
            move_num, _, _ = mcts.find_best_move(board, None, trt_func, num_mcts_sims, 0.0, False)
        except:
            print(f"Error on test: {fen}")
            continue
        move = Move(move_num)
        if move.uci() in possible_scores:
            results[test["group"]].append(possible_scores[move.uci()])
        else:
            results[test["group"]].append(0)

def do_strength_test(num_mcts_sims: int = 800, time_limit: float = None, num_actors: int = 1, use_fake_model: bool = False, save_results: bool = False, verbose: int = 0) -> float:
    """Performs a strength test suite (STS) on the model

    Args:
        time_limit (float): Time limit (seconds) per test / move (solutions are one movers).
        num_actors (int): Number of actors to use for parallel processing (multiprocessing).
        model_path (str, optional): Checkpoint path. Defaults to "latest".
        test_suite (str, optional): Name of desired STS to run. Defaults to "mosca".
        save_results (bool, optional): Whether or not to save to sts_results_dir. Defaults to True.

    Returns:
        float: STS Rating of the model (Elo)
    """
    assert num_mcts_sims is None or time_limit is None, "Only one of num_mcts_sims or time_limit can be set"
    final_statistics = {}
    slope = 445.23 # For estimating elo rating: https://github.com/fsmosca/STS-Rating/blob/master/sts_rating.py
    intercept = -242.85
    
    tests = _load_mosca_sts()
    assert len(tests) > 0, "No tests loaded"
    
    chunkz = _chunk_into_n(tests, num_actors)
    assert len(chunkz) == num_actors, "Chunking failed"

    processes = []
    time_start = time.time()

    with Manager() as manager:
        results = manager.dict()
        for test in tests:
            results[test["group"]] = manager.list()

        for i in range(num_actors):
            process = Process(target=_solve_tests, args=(chunkz[i], results, use_fake_model, num_mcts_sims, time_limit))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
        
        total_score = 0
        total_tests = 0
        for group in results:
            group_stats = {
                "score": sum(results[group]),
                "total": len(results[group]),
                "avg": round(sum(results[group]) / len(results[group]), 2)
            }
            total_score += group_stats["score"]
            total_tests += group_stats["total"]
            final_statistics[group] = group_stats
        final_statistics["total"] = total_score
        final_statistics["total_tests"] = total_tests

        stsRating = (slope * total_score / total_tests) + intercept
        final_statistics["stsRating"] = round(stsRating, 2)
    
    time_end = time.time()
    if verbose > 0:
        print(f"{total_tests} tests finished in: {(time_end - time_start):.2f} s")
        print(f"STS Rating: {final_statistics['stsRating']}")
    if save_results:
        save_path = f"{config.sts_results_dir}/{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}.json"
        with open(save_path, "w") as file:
            json.dump(final_statistics, file)
    return final_statistics["stsRating"]

if __name__ == "__main__":    
    do_strength_test(num_mcts_sims=200, 
                    time_limit=None,
                    num_actors=1,
                    use_fake_model=True,
                    save_results=False,
                    verbose=1
                    )