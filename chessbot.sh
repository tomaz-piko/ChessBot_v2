#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: chessbot.sh <command> [arguments...]"
    echo "Commands: chessbot.sh list"
    exit 1
fi

# Available commands are [init, setup, test, selfplay, strenght_test]
COMMAND="$1"
shift

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Check if the command is valid
if [ "$COMMAND" == "init" ]; then
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
elif [ "$COMMAND" == "selfplay" ]; then
    cd python
    python3 -m selfplay "$@"
    exit 0
elif [ "$COMMAND" == "strength_test" ]; then
    cd python
    python3 strength_test.py "$@"
    exit 0
elif [ "$COMMAND" == "list" ]; then
    echo "Available commands are [init, setup, test, selfplay, strength_test]"
    exit 0
else
    echo "Invalid command: $COMMAND"
    exit 1
fi