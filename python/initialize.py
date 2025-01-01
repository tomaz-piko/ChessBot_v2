import os
import shutil
from configs import selfplayConfig as config
from rchess import Board
import numpy as np


# Inform the user about the deletion of the existing data
# prompt for confirmation
print("This script will delete the contents of the following directories:")
print("    - ../data/")
prompt = input("Do you want to proceed? (y/n): ")
if prompt.lower() != "y":
    print("Exiting...")
    exit()

# If existing delete & create following directories:
#    - ../data/
#    - ../data/conversion_data/
#    - ../data/logs/
#    - ../data/models/
data_dir = config["data_path"]
if os.path.exists(data_dir):
    shutil.rmtree(data_dir)

os.makedirs(data_dir)
os.makedirs(f"{data_dir}/conversion_data")
os.makedirs(f"{data_dir}/logs")
os.makedirs(f"{data_dir}/models")
print("Data directory has been successfully created.")

# Create conversion data from random histories from games
random_fens = [
    "r3k2r/2p2ppp/2n5/1P1q4/8/3P4/4BPPP/1R2K1NR w Kkq - 2 31",
    "r2q1rk1/1p3pbp/2n1pnp1/pP3P2/1P2N3/P4Q2/5BPP/1R2K2R w KQ - 0 35",
    "r2q1rk1/3bppbp/p1pp2p1/4P3/2P3P1/2N1N1B1/PP3P1P/1RQ1K2R b KQ - 0 36",
    "r3k2r/1bq3pp/p3b3/3nN3/1p1P4/5B2/PP2BPPP/R1Q1K1NR w KQkq - 0 32",
    "r2q1rk1/1pp2pbp/2n1pnp1/p1b1P3/2P5/2N1B3/PP3P1P/1R2K2R b KQ - 0 34",
    "r3kb1r/1pp1ppbp/2n1pnp1/p1b5/4P3/2N1N1B1/PPP1QPPP/1R3RK1 w KQkq - 0 33",
    "r1b1k2r/ppp2p1p/3pn1b1/4P3/2P5/2N2N2/PP2QPPP/1R1R2K1 w KQ - 0 30",
    "r2qkbnr/ppppp1pp/8/8/8/3B4/PPP2PPP/RNBQK1NR w KQkq - 0 32",
    "rnb1kbnr/pppppppp/8/8/2P5/3P4/PPP2PPP/R1BQKBNR b KQkq - 0 34",
    "rnb1kb1r/pppppppp/8/8/2P5/3P4/PPP2PPP/R1BQKBNR w KQkq - 0 36",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/3P4/8/8/PPP1PPPP/RNBQKBNR b KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/4P3/8/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/4P3/PPP1PPPP/RNBQKBNR b KQkq - 1 2",
    "r1bqkbnr/pppppppp/8/8/4P3/8/PPP1P1PP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNBQKBNR b KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/4P3/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 1 2",
    "rnbqkbnr/pppppppp/8/4P3/8/2N5/PPP1PPPP/R1BQKBNR w KQkq - 0 1"
]
histories = []
while len(histories) < config["batch_size"] * 4:
    # Take a random fen
    fen = np.random.choice(random_fens)
    board = Board(fen)
    # play 5 to 15 random moves
    num_moves = np.random.randint(5, 15)
    for _ in range(num_moves):
        move = np.random.choice(board.legal_moves())
        board.push(move)
        terminal, _ = board.terminal()
        if terminal:
            break
        history, _ = board.history(config["history_perspective_flip"])
        histories.append(history)
        if len(histories) == config["batch_size"] * 4:
            break

print(f"Generated {len(histories)} random histories.")
# Save the histories as numpy arrays in zip files
np.savez_compressed(f"{data_dir}/conversion_data/histories.npz", histories=np.array(histories, dtype=np.int64))
print("Histories have been saved.")

from model import generate_model, save_as_trt_model

# Generate the model and save it as a TensorRT model
model = generate_model()
model.save(f"{data_dir}/models/model.keras")
save_as_trt_model(model, precision_mode="INT8")