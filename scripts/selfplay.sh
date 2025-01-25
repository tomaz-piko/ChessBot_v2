#!/bin/bash
help() {
    echo "Usage: $0 [-a agents] [-g games] [-b buffer_size] [-f]";
    echo "  -g games: number of games to play (MANDATORY ARG)";
    echo "  -a agents: number of agents to run in parallel (default 1)";
    echo "  -b buffer_size: size of the buffer (default: None & read from config)";
    echo "  -v verbose: print debug information (default: 0)";
    echo "  -f use fake trt_func instead of real model (default: false)";
    echo "  -h: display this help";
}

while getopts g:a:b:v:fh flag
do
    case "${flag}" in
        g) games=${OPTARG};;
        a) agents=${OPTARG};;
        b) buffer_size=${OPTARG};;
        v) verbose=${OPTARG};;
        f) use_fake=true;;
        h) help; exit 0;;

    esac
done

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"
cd ../python
python3 -m selfplay ${games:+-g $games} ${agents:+-a $agents} ${buffer_size:+-b $buffer_size} ${verbose:+-v $verbose} ${use_fake:+-f}