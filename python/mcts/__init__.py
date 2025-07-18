from .c import Node, MCTS
from rchess import Move

def as_str(value):
    if value < 0:
        return f"{value:.6f}"
    else:
        return f" {value:.6f}"

def as_percentage(value):
    str = f"{value*100:.2f}%"
    for _ in range(7-len(str)):
        str = " " + str
    return str

class PV:
    def __init__(self, moves: list = [], score: float = 0.0, visits: int = 0):
        self.moves = moves
        self.score = score
        self.visits = visits

    def __str__(self):
        moves_str = " ".join([move for move in self.moves])
        return f"{self.score:.6f} | {moves_str}"
    
    def __repr__(self):
        return str(self)

def get_pv(root: Node, multi_pv: int = 1, max_depth: int = 5):
    # Get keys from the root node sorted by N
    moves = sorted(root.children.keys(), key=lambda x: root[x].N, reverse=True)
    pvs = []
    pv_moves = []

    for i in range(min(multi_pv, len(moves))):
        move_num = moves[i]
        move_str = Move(move_num).uci()
        pv_moves.append(move_str)
        node = root[move_num]

        score = node.W / node.N if node.N > 0 else 0.0
        visits = node.N

        depth = 0
        while depth < max_depth and node.children:
            move_num = max(node.children, key=lambda x: node[x].N)
            node = node[move_num]
            move_str = Move(move_num).uci()
            pv_moves.append(move_str)
            depth += 1
        
        pvs.append(PV(pv_moves, score, visits))
        pv_moves = []

    return pvs

def debug_search(board, root: Node, multi_pv: int = 1, max_depth: int = 5, limit: tuple = None):
    assert root, "No root node provided."
    sorted_children = sorted(root.children.values(), key=lambda x: x.N, reverse=True)
    print(board)
    time = root.debug_info["elapsed_time"] if "elapsed_time" in root.debug_info else 0.0
    print(f"Elapsed time: {time:.2f}s")
    print(f"Total visits: {root.N}")

    if multi_pv > 0:
        pvs = get_pv(root, multi_pv, max_depth)
        print("-" * 85)
        print("PV:")
        for i, pv in enumerate(pvs):
            print(f"{i+1}. {pv}")

    print("-" * 85)
    print("%-5s | %-6s | %-10s | %-12s | %-12s | %-12s | %-12s" %("Move", "N", "  P. Norm", " V [0, 1]", " Q", " U", " Q+U"))
    print("-" * 85)
    if limit and len(sorted_children) < limit[0] + limit[1]:
        limit = None

    if limit:
        top_children = sorted_children[:limit[0]]
        bottom_children = sorted_children[-limit[1]:]
        for child in top_children:
            move = child.debug_info["move_uci"]
            visits = child.N
            policy = as_percentage(child.P)
            Q = child.W / child.N if child.N > 0 else 0.0
            avg_value = as_str(Q)
            ucb = child.debug_info["ucb"]
            q_plus_ucb = as_str(Q + ucb)
            U = as_str(ucb)
            raw_nn_value = as_str(child.debug_info["init_value"]) if "init_value" in child.debug_info else "N/A"
            print("%-5s | %-6d | %-10s | %-12s | %-12s | %-12s | %-12s" %(move, visits, policy, raw_nn_value, avg_value, U, q_plus_ucb))

        print("%-5s | %-6s | %-10s | %-12s | %-12s | %-12s | %-12s" %('•', '•', '  •', ' •', ' •', ' •', ' •'))
        print("%-5s | %-6s | %-10s | %-12s | %-12s | %-12s | %-12s" %('•', '•', '  •', ' •', ' •', ' •', ' •'))
        print("%-5s | %-6s | %-10s | %-12s | %-12s | %-12s | %-12s" %('•', '•', '  •', ' •', ' •', ' •', ' •'))

        for child in bottom_children:
            move = child.debug_info["move_uci"]
            visits = child.N
            policy = as_percentage(child.P)
            Q = child.W / child.N if child.N > 0 else 0.0
            avg_value = as_str(Q)
            ucb = child.debug_info["ucb"]
            q_plus_ucb = as_str(Q + ucb)
            U = as_str(ucb)
            raw_nn_value = as_str(child.debug_info["init_value"]) if "init_value" in child.debug_info else "N/A"
            print("%-5s | %-6d | %-10s | %-12s | %-12s | %-12s | %-12s" %(move, visits, policy, raw_nn_value, avg_value, U, q_plus_ucb))

    else:
        for child in sorted_children:
            move = child.debug_info["move_uci"]
            visits = child.N
            policy = as_percentage(child.P)
            Q = child.W / child.N if child.N > 0 else 0.0
            avg_value = as_str(Q)
            ucb = child.debug_info["ucb"]
            q_plus_ucb = as_str(Q + ucb)
            U = as_str(ucb)
            raw_nn_value = as_str(child.debug_info["init_value"]) if "init_value" in child.debug_info else "N/A"
            print("%-5s | %-6d | %-10s | %-12s | %-12s | %-12s | %-12s" %(move, visits, policy, raw_nn_value, avg_value, U, q_plus_ucb))
