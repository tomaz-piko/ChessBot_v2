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
    level=logging.DEBUG,  # Default to INFO to suppress debug logs
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
        self.searching = False
        self.stop_event = threading.Event()  # Thread-safe stop signal
        self.ponderhit_event = threading.Event()
        self.search_thread = None  # Thread for the search process        

        # Load the TensorRT model
        self.trt_func, self._ = load_as_trt_model(model_version="latest")
        # Initialize MCTS with engineplayConfig
        self.mcts = MCTS(config)
        # Initialize the chess board
        self.board = Board()  
        # Create a root node for MCTS (also expands the root node for a cold start of inference model)
        self.root = self.mcts.expand_root(self.board, None, self.trt_func, False)
        self.root = Node(0.0) # Reset the node to avoid using the previous root node
        logging.info("UCIEngine initialized successfully.")

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
        logging.getLogger().setLevel(logging.INFO)
        logging.info("Debug logging disabled.")

    def handle_command(self, command):
        """
        Handle commands sent from the main process.
        """        
        words = command.strip().split()
        if not words:
            logging.warning("Empty command received.")
            return

        command_type = words[0]
        handler = getattr(self, f"handle_{command_type}", self.handle_unknown)
        try:
            handler(words)
        except Exception as e:
            logging.warning(f"Error occured handling: {command_type}, {words} - {e}")
            self.should_exit = True
            self.stop_event.set()  # Ensure any ongoing search is stopped

    def handle_uci(self, words):
        logging.debug("Processing 'uci' command.")
        self.send_command("id name chessbot")
        self.send_command("id author YourName")
        for option_name, option_data in self.options.items():
            option_type = option_data["type"]
            default_value = option_data["default"]
            self.send_command(f"option name {option_name} type {option_type} default {default_value}")
        self.send_command("uciok")

    def handle_setoption(self, words):
        """
        Handle the 'setoption' command to update engine options.
        """
        logging.debug("Processing 'setoption' command.")
        if "name" in words and "value" in words:
            try:
                # Extract the option name and value
                name_index = words.index("name") + 1
                value_index = words.index("value") + 1
                option_name = " ".join(words[name_index:words.index("value")])
                option_value = " ".join(words[value_index:])

                # Check if the option exists
                if option_name in self.options:
                    option_type = self.options[option_name]["type"]
                    if option_type == "spin":
                        self.options[option_name]["default"] = int(option_value)
                    elif option_type == "string":
                        self.options[option_name]["default"] = option_value
                    elif option_type == "check":
                        self.options[option_name]["default"] = option_value.lower() in ["true", "1"]
                    logging.info(f"Set option '{option_name}' to {option_value}")
                else:
                    logging.warning(f"Unknown option '{option_name}'")
            except (ValueError, IndexError) as e:
                logging.warning(f"Malformed 'setoption' command: {words} - {e}")
        else:
            logging.warning(f"Malformed 'setoption' command: {words}")

    def handle_isready(self, words):
        logging.debug("Processing 'isready' command.")
        self.send_command("readyok")

    def handle_ucinewgame(self, words):
        logging.debug("Processing 'ucinewgame' command.")
        del self.board, self.root
        self.board = Board()  # Reset to the starting position
        self.root = Node(0.0)  # Reset the root node

    def handle_position(self, words):
        logging.debug("Processing 'position' command.")
        del self.root, self.board

        self.root = Node(0.0)  # Reset the root node
        if "startpos" in words:
            self.board = Board()  # Reset to the starting position
            moves_index = words.index("startpos") + 1
        elif "fen" in words:
            fen_index = words.index("fen") + 1
            fen = " ".join(words[fen_index:fen_index + 6])  # FEN strings are 6 parts
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

        # ---- Search parameters ----
        # Restrict search to the specified moves
        # searchmoves <move1> <move2> ... 
        if "searchmoves" in words:
            pass
        # Whites time left on clock
        if "wtime" in words:
            pass
        # Blacks time left on clock
        if "btime" in words:
            pass
        # White increment per move in ms if x > 0
        if "winc" in words:
            pass
        # Black increment per move in ms if x > 0
        if "binc" in words:
            pass
        # Moves left to next time control
        if "movestogo" in words:
            pass

        # ---- Search conditions ----
        # Start searching in last move in startpos
        if "ponder" in words:
            self.ponder()
        # Search until given depth
        elif "depth" in words:
            pass
        # Search unitl gives number of nodes reached
        elif "nodes" in words:
            num_nodes_idx = words.index("nodes") + 1
            best_move = self.nodes_search(int(words[num_nodes_idx]))
            self.send_command(f"bestmove {best_move}")
        # Search for mate in x moves
        elif "mate" in words:
            pass
        # Search for exactly x ms
        elif "movetime" in words:
            time_limit_idx = words.index("movetime") + 1
            time_limit = int(words[time_limit_idx]) / 1000.0  # Convert to seconds
            best_move, ponder_move = self.timed_search(time_limit)
            self.send_command(f"bestmove {best_move}")
        # Search until stop command is received
        elif "infinite" in words:
            best_move = self.inifinite_search()
            self.send_command(f"bestmove {best_move}")
        # Engine decides how long to search and what to search for
        else:
            best_move, ponder_move = self.timed_search(5)
            if ponder_move:
                self.send_command(f"bestmove {best_move} ponder {ponder_move}")
            else:
                self.send_command(f"bestmove {best_move}")
        
    def timed_search(self, time_limit):
        self.searching = True
        self.stop_event.clear()
        start_time = time.time()
        if not self.root.children:
            self.mcts.expand_root(self.board, self.root, self.trt_func, False)
        while True:
            if self.stop_event.is_set():
                break
            if time.time() - start_time >= time_limit:
                break
            self.root = self.mcts.search(self.board, self.root, self.trt_func, False)
        self.searching = False
        move_num = self.mcts.select_best_move(self.root, 0.0)
        move_uci = Move(move_num).uci()
        if not self.root.children[move_num].children:
            return move_uci, None
        ponder_move_num = self.mcts.select_best_move(self.root[move_num], 0.0)
        ponder_move_uci = Move(ponder_move_num).uci()
        return move_uci, ponder_move_uci
    
    def nodes_search(self, nodes):
        self.searching = True
        self.stop_event.clear()
        if not self.root.children:
            self.mcts.expand_root(self.board, self.root, self.trt_func, False)
        while 1:
            if self.stop_event.is_set() or self.root.N >= nodes:
                break
            self.root = self.mcts.search(self.board, self.root, self.trt_func, False)
        move_num = self.mcts.select_best_move(self.root, 0.0)
        move_uci = Move(move_num).uci()
        self.searching = False
        return move_uci
    
    def inifinite_search(self):
        self.searching = True
        self.stop_event.clear()
        if not self.root.children:
            self.mcts.expand_root(self.board, self.root, self.trt_func, False)
        while 1:
            if self.stop_event.is_set():
                break
            self.root = self.mcts.search(self.board, self.root, self.trt_func, False)
        move_num = self.mcts.select_best_move(self.root, 0.0)
        move_uci = Move(move_num).uci()
        self.searching = False
        return move_uci

    def ponder(self):
        logging.debug("Pondering...")
        self.searching = True
        self.stop_event.clear()
        self.ponderhit_event.clear()
        if not self.root.children:
            self.mcts.expand_root(self.board, self.root, self.trt_func, False)
        while True:
            if self.stop_event.is_set():
                break
            self.root = self.mcts.search(self.board, self.root, self.trt_func, False)
        if not self.ponderhit_event.is_set():
            logging.debug("Pondering failed. Stopping search.")
            self.searching = False
            self.send_command(f"bestmove none") # dummy move to inform the GUI that pondering failed
            self.ponderhit_event.clear()
            return
        logging.debug(f"Pondering successful. Pondered for {self.root.N} nodes.")
        self.searching = False
        best_move, ponder_move = self.timed_search(2.5)
        if ponder_move:
            self.send_command(f"bestmove {best_move} ponder {ponder_move}")
        else:
            self.send_command(f"bestmove {best_move}")
        return

    def handle_ponderhit(self, words):
        logging.debug("Processing 'ponderhit' command.")
        if self.searching:
            self.ponderhit_event.set()
            self.stop_event.set()
        else:
            logging.debug("No ongoing search to ponderhit.")

    def handle_stop(self, words):
        logging.debug("Processing 'stop' command.")
        if self.searching:
            self.ponderhit_event.clear()
            self.stop_event.set()
        else:
            logging.debug("No ongoing search to stop.")

    def handle_quit(self, words):
        logging.debug("Processing 'quit' command. Exiting...")
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
            return None
        else:
            return line.strip()

    sel = selectors.DefaultSelector()
    sel.register(sys.stdin, selectors.EVENT_READ, read_input)

    while not engine.should_exit:
        events = sel.select(timeout=1)
        for key, _ in events:
            callback = key.data
            line = callback(sys.stdin, _)
            if line:
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