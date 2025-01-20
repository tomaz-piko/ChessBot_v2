from .c import find_best_move, Node, gather_nodes_to_process, process_node, calculate_search_statistics, select_best_move

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

def debug_search(board, root: Node):
    assert root, "No root node provided."
    sorted_children = sorted(root.children.values(), key=lambda x: x.N, reverse=True)
    best_child = list(sorted_children)[0]
    print(board)
    time = root.debug_info["elapsed_time"] if "elapsed_time" in root.debug_info else 0.0
    print(f"Elapsed time: {time:.2f}s")
    print(f"Visits: {root.N}")
    print(f"Eval: {as_str(best_child.W / best_child.N) if best_child.N > 0 else 0.0}")
    print("%-16s %-10s %-14s %-16s %-16s %-18s %-8s" %("Move", "Visits", "Policy", "Avg. value", "UCB", "Q+U", "Raw NN Value"))
    print("-" * 110)
    for child in sorted_children:
        move = child.debug_info["move_uci"]
        action = child.debug_info["move_num"]
        visits = child.N
        policy = as_percentage(child.P)
        Q = child.W / child.N if child.N > 0 else 0.0
        avg_value = as_str(Q)
        ucb = child.debug_info["ucb"]
        q_plus_ucb = as_str(Q + ucb)
        U = as_str(ucb)
        raw_nn_value = as_str(child.debug_info["init_value"]) if "init_value" in child.debug_info else "N/A"
        print("%-5s (%-6s)   %-10s %-14s %-16s %-16s %-18s %-13s" %(move, action, f"N: {visits}", f"(P: {policy})", f"(Q: {avg_value})", f"(U: {U})", f"(Q+U: {q_plus_ucb})", f"(V: {raw_nn_value})"))
