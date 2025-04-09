import argparse
from strength_test import do_strength_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run strength test for ChessBot.')
    parser.add_argument('-s', '--sims', type=int, default=0, help='Number of MCTS simulations to run. Only one of sims or time can be set.')
    parser.add_argument('-t', '--time', type=float, default=0.0, help='Time limit per move. Only one of sims or time can be set.')
    parser.add_argument('-a', '--agents', type=int, default=1, help='Number of agents to run in parallel.')
    parser.add_argument('-m', '--model', type=str, default="latest", help='Model version to use for testing.')
    parser.add_argument('-f', '--use_fake_model', action='store_true', help='Use a fake model for testing.')
    args = parser.parse_args()

    if args.sims <= 0 and args.time <= 0.0:
        print("One of --sims or --time must be set.")
        exit() 
    if args.sims > 0 and args.time > 0.0:
        print("Only one of --sims or --time can be set.")
        exit()
    print("Running strength test with the following arguments:")
    print(f"Number of MCTS simulations: {args.sims}")
    print(f"Time limit per move: {args.time}")
    print(f"Number of agents: {args.agents}")
    print(f"Use fake model: {args.use_fake_model}")
    print()

    do_strength_test(num_mcts_sims=args.sims, time_limit=args.time, num_agents=args.agents, model_version=args.model, use_fake_model=args.use_fake_model)