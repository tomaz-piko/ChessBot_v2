from .c import play_game
import numpy as np
import chess.syzygy as syzygy
from datetime import datetime
from multiprocessing import Process, Manager
from configs import selfplayConfig
import os

class GamesBuffer:
    def __init__(self, save_path, max_size=1024):
        if save_path[-1] != '/':
            save_path += '/'
        self.save_path = save_path
        self.max_size = max_size
        self.images = []
        self.search_stats = []
        self.terminal_values = []

    def append(self, images, search_stats, terminal_values):
        self.images.extend(images)
        self.search_stats.extend(search_stats)
        self.terminal_values.extend(terminal_values)
        assert len(self.images) == len(self.search_stats) == len(self.terminal_values)
        while len(self.images) >= self.max_size:
            self.flush()

    def flush(self):
        # Save the data to disk
        images_np = np.array(self.images[:self.max_size]).astype(np.int64)
        search_stats_np = np.array(self.search_stats[:self.max_size]).astype(np.float32)
        terminal_values_np = np.array(self.terminal_values[:self.max_size]).astype(np.int64)

        timestamp = datetime.now().strftime('%F_%T.%f')[:-3]
        num_positions = images_np.shape[0]
        file_name = f"{num_positions}_{timestamp}.npz"
        np.savez_compressed(self.save_path + file_name, 
                            images=images_np,
                            search_stats=search_stats_np,
                            terminal_values=terminal_values_np)
        del self.images[:self.max_size], self.search_stats[:self.max_size], self.terminal_values[:self.max_size]

def play_games(config, num_games, games_count, buffer_size=None, use_fake_model=False, verbose=0):
    if use_fake_model:
        from utils import FakeTRTFunc
        trt_func = FakeTRTFunc()
    else:
        import tensorflow as tf
        gpu_devices = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)
        from model import load_as_trt_model
        trt_func, _ = load_as_trt_model(model_version="latest")
    from mcts.c import MCTS

    if buffer_size is None:
        buffer_size = config['buffer_size']
    tablebase = syzygy.open_tablebase(os.path.join(config['project_dir'], 'syzygy/3-4-5'))
    buffer = GamesBuffer(os.path.join(config['project_dir'], 'data', 'selfplay_data'), buffer_size)
    mctsSearch = MCTS(config)

    # Play the games until games_count is reached
    while games_count.value < num_games:
        images, (search_stats, terminal_values) = play_game(mctsSearch, trt_func, tablebase, verbose=verbose)
        games_count.value += 1
        buffer.append(images, search_stats, terminal_values)
        del images, search_stats, terminal_values
    buffer.flush()

def run_selfplay(num_agents=1, num_games=100, buffer_size=1024, use_fake_model=False, verbose=0):
    processes = {}
    time_start =  datetime.now()
    formatted_timestart = time_start.strftime("%d/%m %H:%M:%S")
    with Manager() as manager:
        games_count = manager.Value('i', 0)
        for i in range(num_agents):
            p = Process(target=play_games, args=(selfplayConfig, num_games, games_count, buffer_size, use_fake_model, verbose))
            processes[i] = p
            p.start()

        for p in processes.values():
            p.join()
    time_end = datetime.now()
    formatted_timeend = time_end.strftime("%d/%m %H:%M:%S")
    print(f"Self-play started at {formatted_timestart} and ended at {formatted_timeend}.")
    time_difference = time_end - time_start
    hours, remainder = divmod(time_difference.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")
