#!/bin/bash

# Check if at least one argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_name> [arguments...]"
    exit 1
fi

# Extract the script name from the first argument
SCRIPT_NAME="$1"
shift

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

# Call the script with the remaining arguments
"./scripts/${SCRIPT_NAME}.sh" "$@"