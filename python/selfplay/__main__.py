import argparse
from selfplay import run_selfplay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run self-play for ChessBot.')
    parser.add_argument('-g', '--games', type=int, default=None, help='Number of games to play (all agents combined) MANDATORY!.')
    parser.add_argument('-a', '--agents', type=int, default=1, help='Number of agents to run in parallel.')
    parser.add_argument('-b', '--buffer_size', type=int, default=None, help='Size of the buffer for storing game data.')
    parser.add_argument('-f', '--use_fake_model', action='store_true', help='Use a fake model for testing.')
    parser.add_argument('-v', '--verbose', type=int, default=0, help='Verbosity level.')

    args = parser.parse_args()
    if args.games is None:
        print("The number of games must be specified.")
        exit()
    if args.agents < 1:
        print("The number of agents must be at least 1.")
        exit()
    if args.buffer_size is not None and args.buffer_size < 1:
        print("The buffer size must be at least 1.")
        exit()
    print("Running self-play with the following arguments:")
    print(f"Number of games: {args.games}")
    print(f"Number of agents: {args.agents}")
    print(f"Buffer size: {args.buffer_size}")
    print(f"Use fake model: {args.use_fake_model}")
    print(f"Verbosity level: {args.verbose}")
    print()

    run_selfplay(num_agents=args.agents, num_games=args.games, buffer_size=args.buffer_size, use_fake_model=args.use_fake_model, verbose=args.verbose)