#!/bin/bash

# Script to run evaluation for multiple channels
# Usage: ./run_eval_spaces.sh [channels_file] [use_pw]
# Example: ./run_eval_spaces.sh channels_subset.txt FALSE

set -euo pipefail

# Default values
CHANNELS_FILE="${1:-channels_subset.txt}"
USE_PW="${2:-FALSE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if channels file exists
if [[ ! -f "$CHANNELS_FILE" ]]; then
    echo "Error: $CHANNELS_FILE not found"
    exit 1
fi

# Load R environment
source /apps/rocs/init.sh 2024.04
ml CEDAR/2024.04
ml R/release

echo "Running evaluation for channels in: $CHANNELS_FILE"
echo "use_pw: $USE_PW"
echo "----------------------------------------"

# Read channels and run evaluation
while IFS= read -r channel || [[ -n "$channel" ]]; do
    # Skip empty lines
    if [[ -z "$channel" ]]; then
        continue
    fi

    echo ""
    echo ">>> Starting evaluation for channel: $channel"

    # Run R script
    if Rscript "$SCRIPT_DIR/eval_spaces_perchannel.R" "$channel" "$USE_PW"; then
        echo ">>> Successfully completed: $channel"
    else
        echo ">>> ERROR: Failed for channel: $channel"
    fi

    echo "----------------------------------------"

done < "$CHANNELS_FILE"

echo ""
echo "All evaluations complete!"
