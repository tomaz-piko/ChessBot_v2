#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: chessbot.sh <command> [arguments...]"
    echo "Commands: chessbot.sh list"
    exit 1
fi

# Available commands are [init, setup, test, train, selfplay, strenght_test]
COMMAND="$1"
shift

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Check if the command is valid
if [ "$COMMAND" == "run" ]; then
    #export TF_CPP_MIN_LOG_LEVEL=3
    cd python
    python3 engine.py "$@"
    exit 0
elif [ "$COMMAND" == "lichess-bot" ]; then
    export TF_CPP_MIN_LOG_LEVEL=3
    cd lichess-bot
    python3 lichess-bot.py "$@"
    exit 0
elif [ "$COMMAND" == "init" ]; then
    cd python
    python3 initialize.py "$@"
    exit 0
elif [ "$COMMAND" == "setup" ]; then
    python3 -m pip install -e .
    cd python
    python3 setup.py build_ext --inplace
    exit 0
elif [ "$COMMAND" == "test" ]; then
    python3 -m unittest discover -s python/tests
    exit 0
elif [ "$COMMAND" == "train" ]; then
    cd python
    python3 train.py "$@"
    exit 0
elif [ "$COMMAND" == "selfplay" ]; then
    export TF_CPP_MIN_LOG_LEVEL=3
    cd python
    python3 -m selfplay "$@"
    exit 0
elif [ "$COMMAND" == "strength_test" ]; then
    export TF_CPP_MIN_LOG_LEVEL=3
    cd python
    python3 -m strength_test "$@"
    exit 0
elif [ "$COMMAND" == "list" ]; then
    echo "Available commands are [init, setup, test, selfplay, strength_test]"
    exit 0
elif [ "$COMMAND" == "tensorboard" ]; then
    tensorboard --logdir data/logs
    exit 0
else
    echo "Invalid command: $COMMAND"
    exit 1
fi
