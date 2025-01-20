from .c import play_game

from selfplay import play_game
import numpy as np
import chess.syzygy as syzygy
from datetime import datetime
from multiprocessing import Process, Manager
from configs import defaultConfig as config

class GamesBuffer:
    def __init__(self, max_size=1024):
        self.max_size = max_size
        self.images = []
        self.search_stats = []
        self.terminal_values = []

    def append(self, images, search_stats, terminal_values):
        self.images.extend(images)
        self.search_stats.extend(search_stats)
        self.terminal_values.extend(terminal_values)
        assert len(self.images) == len(self.search_stats) == len(self.terminal_values)
        if len(self.images) >= self.max_size:
            self.flush()

    def flush(self):
        # Save the data to disk
        images_np = np.array(self.images[:self.max_size])
        search_stats_np = np.array(self.search_stats[:self.max_size])
        terminal_values_np = np.array(self.terminal_values[:self.max_size])

        timestamp = datetime.now().strftime('%F_%T.%f')[:-3]
        num_positions = images_np.shape[0]
        file_name = f"{num_positions}_{timestamp}.npz"
        np.savez_compressed(file_name, 
                            images=images_np,
                            search_stats=search_stats_np,
                            terminal_values=terminal_values_np)
        self.images = self.images[-self.max_size:]
        self.search_stats = self.search_stats[-self.max_size:]
        self.terminal_values = self.terminal_values[-self.max_size:]

def play_games(num_games, games_count):
    import tensorflow as tf
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    from model import load_as_trt_model

    # Load the model
    trt_func, _ = load_as_trt_model()
    tablebase = syzygy.open_tablebase(f"{config['project_dir']}/syzygy/3-4-5")
    buffer = GamesBuffer()

    # Play the games until games_count is reached
    while games_count.value < num_games:
        images, (search_stats, terminal_values) = play_game(trt_func, tablebase, 0)
        games_count.value += 1
        buffer.append(images, search_stats, terminal_values)
        del images, search_stats, terminal_values

def self_play(num_agents=1, num_games=100):
    processes = {}
    with Manager() as manager:
        games_count = manager.Value('i', 0)
        for i in range(num_agents):
            p = Process(target=play_games, args=(num_games, games_count))
            processes[i] = p
            p.start()

        for p in processes.values():
            p.join()
