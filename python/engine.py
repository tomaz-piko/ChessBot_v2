#!/usr/bin/env python3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
           
import logging
import selectors
import sys
import queue
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from rchess import Board, Move
from mcts import MCTS, Node
from configs import engineplayConfig as config
from model import load_as_trt_model

# Configure logging (disabled by default)
logging.basicConfig(
    filename="engine.log",
    encoding="utf-8",
    level=logging.CRITICAL,  # Default to CRITICAL to suppress logs
    format="%(levelname)s %(asctime)s %(message)s",
)

class UCIEngine:
    def __init__(self):
        self.options = {
            "Move Overhead": {"type": "spin", "default": 0},  # Default value in milliseconds
            "Threads": {"type": "spin", "default": 1},       # Default number of threads
            "Hash": {"type": "spin", "default": 128},        # Default hash size in MB
            "SyzygyPath": {"type": "string", "default": "../syzygy"},  # Default path for Syzygy tablebases
            "UCI_ShowWDL": {"type": "check", "default": False},  # Show Win-Draw-Loss statistics
        }
        self.should_exit = False
        self.board = Board()  # Initialize the chess board
        self.searching = False
        self.stop_event = threading.Event()  # Thread-safe stop signal
        self.search_thread = None  # Thread for the search process

        # Load the TensorRT model
        #logging.info("Loading TensorRT model...")
        #self.trt_model, self.trt_context = load_as_trt_model(model_version="latest")
        #logging.info("TensorRT model loaded successfully.")

        # Initialize MCTS
        self.mcts = MCTS(config)

    def enable_debug(self):
        """
        Enable debug logging.
        """
        global logging
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("Debug logging enabled.")

    def disable_debug(self):
        """
        Disable debug logging.
        """
        global logging
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.info("Debug logging disabled.")

    def handle_command(self, command):
        """
        Handle commands sent from the main process.
        """
        logging.debug(f"Handling command: {command}")
        print(f"Received command: {command}")

        words = command.strip().split()
        if not words:
            logging.warning("Empty command received.")
            return

        command_type = words[0]
        handler = getattr(self, f"handle_{command_type}", self.handle_unknown)
        handler(words)

    def handle_uci(self, words):
        logging.info("Processing 'uci' command.")
        self.send_command("id name chessbot")
        self.send_command("id author YourName")
        for option_name, option_data in self.options.items():
            option_type = option_data["type"]
            default_value = option_data["default"]
            self.send_command(f"option name {option_name} type {option_type} default {default_value}")
        self.send_command("uciok")

    def handle_isready(self, words):
        logging.info("Processing 'isready' command.")
        self.send_command("readyok")

    def handle_position(self, words):
        logging.info("Processing 'position' command.")
        if "startpos" in words:
            self.board = Board()  # Reset to the starting position
            moves_index = words.index("startpos") + 1
        elif "fen" in words:
            fen_index = words.index("fen") + 1
            fen = " ".join(words[fen_index:fen_index + 6])  # FEN strings are 6 parts
            self.board = Board(fen)
            moves_index = fen_index + 6
        else:
            logging.warning("Malformed 'position' command.")
            return

        # Play moves on the board
        if "moves" in words:
            moves_index = words.index("moves") + 1
            for move in words[moves_index:]:
                self.board.push_uci(move)
        logging.debug(f"Board position set to: {self.board.fen()}")

    def handle_go(self, words):
        logging.info("Processing 'go' command.")
        self.searching = True
        self.stop_event.clear()  # Clear the stop signal

        # Start the search in a separate thread
        self.search_thread = threading.Thread(target=self.search, args=(1.0,))
        self.search_thread.start()

    def search(self, search_time):
        """
        Perform MCTS search for the given time or until stopped.
        """
        #start_time = time.time()
        #root = self.mcts.create_root(self.board)

        while True:
            if self.stop_event.is_set():  # Check if stop signal is set
                logging.info("Search stopped.")
                break
            #self.mcts.search(self.board, root, self.trt_model, self.trt_context)
            print("Searching...")

        # Select the best move based on MCTS results
        #best_move = self.mcts.select_best_move(root)
        best_move = None  # Placeholder for the best move selection logic
        if best_move:
            logging.info(f"Best move selected: {best_move}")
            self.send_command(f"bestmove {best_move}")
        else:
            logging.warning("No legal moves available.")
            self.send_command("bestmove (none)")

        self.searching = False

    def handle_stop(self, words):
        logging.info("Processing 'stop' command.")
        if self.searching:
            self.stop_event.set()  # Signal the search thread to stop
            if self.search_thread:
                self.search_thread.join()  # Wait for the search thread to finish
            self.searching = False
        else:
            logging.info("No ongoing search to stop.")

    def handle_quit(self, words):
        logging.info("Processing 'quit' command. Exiting...")
        self.should_exit = True
        self.stop_event.set()  # Ensure any ongoing search is stopped
        if self.search_thread:
            self.search_thread.join()

    def handle_debug(self, words):
        """
        Toggle debug logging.
        """
        if len(words) > 1 and words[1].lower() == "on":
            self.enable_debug()
            self.send_command("info string Debug logging enabled.")
        elif len(words) > 1 and words[1].lower() == "off":
            self.disable_debug()
            self.send_command("info string Debug logging disabled.")
        else:
            self.send_command("info string Debug command requires 'on' or 'off'.")

    def handle_unknown(self, words):
        logging.warning(f"Unknown command received: {' '.join(words)}")

    def send_command(self, command):
        """
        Send a command to the engine.
        """
        logging.debug(f"Sending command: {command}")
        sys.stdout.write(command + "\n")
        sys.stdout.flush()


def producer(q, engine):
    """
    Produce commands to be processed.
    """
    logging.info("Producer started.")

    def read_input(stdin, mask):
        line = stdin.readline()
        if not line:
            logging.warning("No input line received.")
            return None
        else:
            logging.debug(f"Input line read: {line.strip()}")
            return line.strip()

    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ, read_input)

    while not engine.should_exit:
        events = sel.select(timeout=1)
        for key, _ in events:
            callback = key.data
            line = callback(sys.stdin, _)
            if line:
                logging.debug(f"Putting line into queue: {line}")
                q.put(line)

    logging.info("Producer exiting.")


def consumer(q, engine):
    """
    Consume commands from the queue and process them.
    """
    logging.info("Consumer started.")
    while not engine.should_exit:
        try:
            line = q.get(timeout=1)
            if line:
                logging.debug(f"Consumer processing line: {line}")
                engine.handle_command(line)
        except queue.Empty:
            logging.debug("Queue is empty. Waiting for new commands.")
            pass
    logging.info("Consumer exiting.")


if __name__ == "__main__":
    logging.info("Engine started.")
    engine = UCIEngine()
    q = queue.Queue()
    n_consumers = 2

    with ThreadPoolExecutor(max_workers=n_consumers + 1) as executor:
        logging.info("Starting producer and consumer threads.")
        executor.submit(producer, q, engine)
        [executor.submit(consumer, q, engine) for _ in range(n_consumers)]

    logging.info("Engine shutting down.")