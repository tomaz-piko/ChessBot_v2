from rchess import Board
import numpy as np
from model import predict_fn
from configs import selfplayConfig
from mcts.c import flip_value, value_to_01, add_vloss, backup, select_child, evaluate_node, expand_node, calculate_search_statistics, Node

""" class Node:
    def __init__(self):
        self.children = {}
        self.vloss = 0
        self.to_play = None

        # Values for MCTS
        self.N = 0
        self.W = 0.0
        self.P = 0.0

    def __getitem__(self, move: str):
        return self.children[move]
    
    def __setitem__(self, move: str, node):
        self.children[move] = node

    def is_leaf(self):
        return not self.children """
    
def find_best_move(board: Board, root: Node, trt_func, num_mcts_sims: int = 0):
    """

    Args:
        board (Board): Current state of the board
        root (Node): Root node of the MCTS tree
        trt_func (trt.InferenceContext): TensorRT Inference Context
        num_mcts_sims (int, optional): Number of MCTS simulations to run. Defaults to config setting.
    """
    config = selfplayConfig
    num_mcts_sims = config['num_mcts_sims'] if num_mcts_sims == 0 else num_mcts_sims
    root_exploration_noise = config['root_exploration_noise']
    root_dirichlet_alpha = config['root_dirichlet_alpha']
    root_exploration_fraction = config['root_exploration_fraction']
    fpu_root = config['fpu_root']
    fpu_leaf = config['fpu_leaf']
    pb_c_base = config['pb_c_base']
    pb_c_init = config['pb_c_init']
    pb_c_factor = config['pb_c_factor']
    num_vl_searches = config['num_vl_searches']
    history_flip = config['history_perspective_flip']

    rng = np.random.default_rng()
    tree_reused = True
    if root is None:
        root = Node()
        tree_reused = False

    if root.is_leaf():
        history, _ = board.history(history_flip)
        _, policy_logits = make_predictions(
                trt_func=trt_func,
                histories=[history],
                batch_size=num_vl_searches
            )
        expand_node(root, board.legal_moves(), board.to_play())
        evaluate_node(root, policy_logits[0])

    if not tree_reused and root_exploration_noise:
        add_exploration_noise(root, rng, root_dirichlet_alpha, root_exploration_fraction)

    while True:
        if root.N >= num_mcts_sims:
            break
        nodes_to_find = num_vl_searches if root.N + num_vl_searches < num_mcts_sims else num_mcts_sims - root.N
        nodes_to_eval = []
        nodes_found = 0
        search_paths = []
        histories = []
        failsafe = 0
        while nodes_found < nodes_to_find and failsafe < 4:
            node = root
            search_path = [node]
            tmp_board = board.clone()
            while not node.is_leaf():
                fpu = fpu_root if len(search_path) == 1 else fpu_leaf
                move, node = select_child(node, pb_c_base=pb_c_base, pb_c_init=pb_c_init, pb_c_factor=pb_c_factor, fpu=fpu)
                search_path.append(node)
                tmp_board.push_uci(move)
            
            terminal, winner = tmp_board.terminal()
            if terminal:
                value = 1.0 if winner is not None else 0.0 # Revisit this
                backup(search_path, flip_value(value))
                failsafe += 1
                continue

            expand_node(node, tmp_board.legal_moves(), tmp_board.to_play())
            add_vloss(search_path)

            nodes_found += 1
            nodes_to_eval.append(node)
            search_paths.append(search_path)
            history, _ = tmp_board.history(history_flip)
            histories.append(history)

        if nodes_found == 0:
            failsafe += 1
            continue

        values, policy_logits = make_predictions(
                trt_func=trt_func,
                histories=histories,
                batch_size=num_vl_searches
            )
        
        for node, search_path, policy_logits, value in zip(nodes_to_eval, search_paths, policy_logits, values):
            evaluate_node(node, policy_logits)
            backup(search_path, flip_value(value_to_01(value.item())))
        del nodes_to_eval, values, policy_logits, search_paths, histories

    move = select_best_move(root, rng, temp=0)
    child_visits = calculate_search_statistics(root, 1858)
    return move, root, child_visits

def add_exploration_noise(node: Node, rng: np.random.Generator, alpha: float, frac: float):
    noise = rng.gamma(alpha, 1, len(node.children))
    noise /= np.sum(noise)
    for n, child in zip(noise, node.children.values()):
        child.P = child.P * (1 - frac) + n * frac

def select_best_move(node: Node, rng: np.random.Generator, temp=0):
    moves = list(node.children.keys())
    visit_counts = np.array([child.N for child in node.children.values()])
    if temp == 0:
        # Greedy selection (select the move with the highest visit count)
        # If more moves have the same visit count, choose the first one
        max_visits = np.max(visit_counts)
        best_moves = np.flatnonzero(visit_counts == max_visits)
        return moves[best_moves[0]]
    else:
        # Use the visit counts as a probability distribution to select a move
        pi = visit_counts ** (1 / temp)
        pi /= np.sum(pi)
        return moves[np.where(rng.multinomial(1, pi) == 1)[0][0]]

def make_predictions(trt_func, histories, batch_size):
    # Each history is a list of 109 integers
    # Convert to numpy array and fill with zeros to make a batch if necessary
    images = np.array(histories, dtype=np.int64)
    if images.shape[0] < batch_size:
        images = np.pad(images, ((0, batch_size - images.shape[0]), (0, 0)), mode='constant')

    values, policy_logits = predict_fn(
        trt_func=trt_func,
        images=images
    )
    return values.numpy(), policy_logits.numpy()