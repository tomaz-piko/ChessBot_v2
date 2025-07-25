#!/usr/bin/env python3
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs
           
import logging
import sys
import queue
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from rchess import Board, Move
from mcts import MCTS, Node
from configs import engineplayConfig as config
from model import load_as_trt_model
from time_control import UniversalTimeControl

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
        self.should_exit = threading.Event()  # Thread-safe signal that tells the engine to shutdown

        self.stop_event = threading.Event()  # Thread-safe signal that tells the engine to stop current search
        self.ponderhit_event = threading.Event() # Thread-safe signal that tells the engine that ponder hit has been received

        
        # Set up time control class
        self.time_control = UniversalTimeControl(
            move_overhead_ms=config["move_overhead_ms"],
        )

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
            self.stop_event.set() 
            self.should_exit.set()

    def handle_uci(self, words):
        self.send_command("id name PikoZero")
        self.send_command("id author TomazPiko")
        for option_name, option_data in self.options.items():
            option_type = option_data["type"]
            default_value = option_data["default"]
            self.send_command(f"option name {option_name} type {option_type} default {default_value}")
        self.send_command("uciok")

    def handle_setoption(self, words):
        """
        Handle the 'setoption' command to update engine options.
        """
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
        self.send_command("readyok")

    def handle_ucinewgame(self, words):
        del self.board, self.root
        self.board = Board()  # Reset to the starting position
        self.root = Node(0.0)  # Reset the root node
        logging.debug("State restored due to 'ucinewgame' command.")

    def handle_position(self, words):
        #self.root = Node(0.0)  # Reset the root node
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
        self.stop_event.clear()  # Reset the stop event
        self.ponderhit_event.clear()  # Reset the ponder hit event
        to_play = self.board.to_play()
        last_move = None
        last_moves = self.board.moves_history(1)
        if len(last_moves) > 0:
            last_move = last_moves[0]
        w_time, b_time, w_inc, b_inc = 0, 0, 0, 0
        # ---- Search parameters ----
        # Restrict search to the specified moves
        # searchmoves <move1> <move2> ... 
        if "searchmoves" in words:
            pass
        # Whites time left on clock
        if "wtime" in words:
            w_time = int(words[words.index("wtime") + 1])
        # Blacks time left on clock
        if "btime" in words:
            b_time = int(words[words.index("btime") + 1])
        # White increment per move in ms if x > 0
        if "winc" in words:
            w_inc = int(words[words.index("winc") + 1])
        # Black increment per move in ms if x > 0
        if "binc" in words:
            b_inc = int(words[words.index("binc") + 1])
        # Moves left to next time control
        if "movestogo" in words:
            pass

        # ---- Search conditions ----
        # Start searching in last move in startpos
        if "ponder" in words:
            remaining_time_ms = w_time if to_play == 1 else b_time
            increment_ms = w_inc if to_play == 1 else b_inc
            self.ponder(last_move, remaining_time_ms=remaining_time_ms, increment_ms=increment_ms)
        # Search until given depth
        elif "depth" in words:
            pass
        # Search unitl gives number of nodes reached
        elif "nodes" in words:
            pass
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
            pass
        # Engine decides how long to search and what to search for
        else:
            self.root = self.root.children.get(last_move, Node(0.0))  # Get the root node for the current position
            logging.debug(f"(Tree reuse) Attempting to reuse tree. Nodes: {self.root.N}.")

            remaining_time_ms = w_time if to_play == 1 else b_time
            increment_ms = w_inc if to_play == 1 else b_inc
            time_limit = self.time_control.get_move_time(
                remaining_time_ms=remaining_time_ms,
                increment_ms=increment_ms,
                has_pondered_ms=0,  # No ponder time yet
                move_num=self.board.ply() // 2 + 1,
            )
            fen = self.board.fen()
            logging.debug(f"(Timed search) Pos: {fen}, Time remaining: {remaining_time_ms / 1000}, Increment: {increment_ms / 1000}, Time for move: {time_limit:.2f} seconds")
            best_move, ponder_move = self.timed_search(time_limit)
            if ponder_move:
                logging.debug(f"(Timed search result) Pos: {fen}, Best move: {best_move}, Ponder move: {ponder_move}")
                self.send_command(f"bestmove {best_move} ponder {ponder_move}")
            else:
                logging.debug(f"(Timed search result) Pos: {fen}, Best move: {best_move}")
                self.send_command(f"bestmove {best_move}")
        
    def timed_search(self, time_limit):
        start_time = time.time()
        if not self.root.children:
            self.mcts.expand_root(self.board, self.root, self.trt_func, False)
        while True:
            if time.time() - start_time >= time_limit:
                break
            if self.stop_event.is_set():
                break
            self.root = self.mcts.search(self.board, self.root, self.trt_func, False)
        move_num = self.mcts.select_best_move(self.root, 0.0)
        move_uci = Move(move_num).uci()

        if not self.root.children[move_num].children:
            self.root = self.root.children[move_num]  # Update root to the best move found
            logging.debug(f"(Tree reuse) Attempting to reuse tree. Nodes: {self.root.N}.")
            return move_uci, None
        ponder_move_num = self.mcts.select_best_move(self.root[move_num], 0.0)
        ponder_move_uci = Move(ponder_move_num).uci()
        self.root = self.root.children[move_num]  # Update root to the best move found
        logging.debug(f"(Tree reuse) Attempting to reuse tree. Nodes: {self.root.N}.")
        return move_uci, ponder_move_uci

    def ponder(self, last_move, remaining_time_ms=0, increment_ms=0):
        ponder_root = self.root.children.get(last_move, Node(0.0))  # Get the root node for pondering
        fen = self.board.fen()
        ponder_success = False
        logging.debug(f"(Pondering) Pos: {fen}")
        time_start = time.time()
        if not ponder_root.children:
            self.mcts.expand_root(self.board, ponder_root, self.trt_func, False)
        while True:
            if self.stop_event.is_set():
                break
            if self.ponderhit_event.is_set():
                ponder_success = True
                break
            ponder_root = self.mcts.search(self.board, ponder_root, self.trt_func, False)
        time_end = time.time()
        if not ponder_success:
            logging.debug("(Pondering) Pondering failed or stopped before completion.")
            self.send_command(f"bestmove none") # dummy move to inform the GUI that pondering failed
            return
        ponder_time_s = time_end - time_start  # Convert to milliseconds
        logging.debug(f"(Pondering). Successfuly pondered for {ponder_time_s:.2f} seconds | {ponder_root.N} nodes.")
        self.root = self.root.children[last_move]  # Update root to the pondered move
        logging.debug(f"(Tree reuse) Attempting to reuse tree. Nodes: {self.root.N}.")
        time_limit = self.time_control.get_move_time(
            remaining_time_ms=remaining_time_ms,
            increment_ms=increment_ms,
            has_pondered_ms=int((time_end - time_start) * 1000),  # Convert to milliseconds
            move_num=self.board.ply() // 2 + 1,
        )
        logging.debug(f"(Timed search) Pos: {fen}, Time remaining: {remaining_time_ms / 1000}s, Increment: {increment_ms / 1000}s, Pondered: {ponder_time_s}s, Time for move: {time_limit:.2f} seconds")
        best_move, ponder_move = self.timed_search(time_limit)
        if ponder_move:
            logging.debug(f"(Timed search result) Pos: {fen}, Best move: {best_move}, Ponder move: {ponder_move}")
            self.send_command(f"bestmove {best_move} ponder {ponder_move}")
        else:
            logging.debug(f"(Timed search result) Pos: {fen}, Best move: {best_move}")
            self.send_command(f"bestmove {best_move}")
        return

    def handle_ponderhit(self, words):
        if self.ponderhit_event.is_set():
            logging.warning("Ponder hit already set elsewhere. Potential risk.")
        self.ponderhit_event.set()

    def handle_stop(self, words):
        if self.stop_event.is_set():
            logging.debug("Stop event already set elsewhere. Potential risk.")
        self.stop_event.set()

    def handle_quit(self, words):
        self.stop_event.set()  # Ensure any ongoing search is stopped
        self.should_exit.set()
        logging.info("Exiting due to 'quit' command.")

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
        sys.stdout.write(command + "\n")
        sys.stdout.flush()


def producer(q):
    """
    Produce commands to be processed.
    """
    logging.info("Producer started.")
    while True:
        line = sys.stdin.readline()
        if not line:
            line = "quit"  # If EOF is reached, send quit command
        line = line.strip()
        if line:
            logging.debug(f"Producer read line: {line}")
            q.put(line)
            if line == "quit":
                break
    logging.info("Producer exiting.")


def consumer(q, engine):
    """
    Consume commands from the queue and process them.
    """
    logging.info("Consumer started.")
    while not engine.should_exit.is_set():
        try:
            line = q.get(timeout=1)
            if line:
                logging.debug(f"Consumer processing line: {line}")
                engine.handle_command(line)
        except queue.Empty:
            pass
    logging.info("Consumer exiting.")


if __name__ == "__main__":
    engine = UCIEngine()
    q = queue.Queue()
    n_consumers = 2

    with ThreadPoolExecutor(max_workers=n_consumers + 1) as executor:
        executor.submit(producer, q)
        [executor.submit(consumer, q, engine) for _ in range(n_consumers)]
