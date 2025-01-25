#!/bin/bash
help() {
    echo "Usage: $0 [-s simulations] [-t time] [-a actors] [-v verbose] [-f] [--save]";
    echo "  -s simulations: number of mcts simulations per move";
    echo "  -t time: time per move in seconds";
    echo "     \* At least one of -s or -t is required";
    echo "  -a agents: number of agents to run in parallel";
    echo "  -v verbose: print debug information";
    echo "  -f use fake trt_func instead of loading tensorflow model";
    echo "  -h: display this help";
}

while getopts s:t:a:v:fh flag
do
    case "${flag}" in
        s) sims=${OPTARG};;
        t) time=${OPTARG};;
        a) agents=${OPTARG};;
        v) verbose=${OPTARG};;
        f) use_fake=true;;
        h) help; exit 0;;

    esac
done

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../python
python3 strength_test.py ${sims:+-s $sims} ${agents:+-a $agents} ${time:+-t $time} ${verbose:+-v $verbose} ${use_fake:+-f}