
from .c import solve_tests
from multiprocessing import Manager, Process
from math import ceil
from datetime import datetime
import re
from configs import engineplayConfig
import os


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
            - results: Dictionary with UCI move as key and score as value (multiple winning moves but one is still best)
    """

    fileR = open(os.path.join(engineplayConfig['project_dir'], 'test_suites', 'STS1-STS15_LAN_v3.epd'), "r")
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
        test = {"fen": fen, "results": results}
        tests.append(test)
    return tests

def start_solving(config, total_score, tests: list, model_version: str, use_fake_model: bool = False, num_mcts_sims: int = 0, time_limit: float = 0.0):
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
        trt_func, _ = load_as_trt_model(model_version=model_version)
    from mcts.c import MCTS

    mctsSearch = MCTS(config)
    score = solve_tests(mctsSearch, trt_func, tests, num_mcts_sims=num_mcts_sims, time_limit=time_limit)  
    total_score.value += score
    

def do_strength_test(num_mcts_sims = 0, time_limit = 0.0, num_agents = 1, model_version = "latest", use_fake_model = False):
    assert num_mcts_sims > 0 or time_limit > 0.0, "Only one of num_mcts_sims or time_limit can be set"
    slope = 445.23 # For estimating elo rating: https://github.com/fsmosca/STS-Rating/blob/master/sts_rating.py
    intercept = -242.85
    
    tests = _load_mosca_sts()
    assert len(tests) > 0, "No tests loaded"
    
    chunkz = _chunk_into_n(tests, num_agents)
    assert len(chunkz) == num_agents, "Chunking failed"

    processes = {}
    time_start =  datetime.now()
    formatted_timestart = time_start.strftime("%d/%m %H:%M:%S")
    stsRating = 0

    with Manager() as manager:
        total_score = manager.Value('i', 0)
  
        for i in range(num_agents):
            tests_i = list(chunkz[i])
            p = Process(target=start_solving, args=(engineplayConfig, total_score, tests_i, model_version, use_fake_model, num_mcts_sims, time_limit))
            processes[i] = p
            p.start()

        for p in processes.values():
            p.join()

        stsRating = (slope * total_score.value / len(tests)) + intercept
    
    time_end = datetime.now()
    formatted_timeend = time_end.strftime("%d/%m %H:%M:%S")
    print(f"Strength started at {formatted_timestart} and ended at {formatted_timeend}.")
    time_difference = time_end - time_start
    hours, remainder = divmod(time_difference.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time ({len(tests)} tests): {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
    print(f"STS Rating: {stsRating}")

    return stsRating