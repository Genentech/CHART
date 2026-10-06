#!/bin/bash

# Run test_evalK_onfullPC.R for channels listed in channels_subset.txt

# Set up R environment
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export R_LIBS_USER="/home/leotea/R/x86-64-v4/cedar_r4.5_bioc3.22-release:/apps/rocs/2024.04/CEDAR/x86-64-v4/site-libs/R/cedar_r4.5_bioc3.22-release/2025_11_13/incremental/rlibs/:/apps/rocs/2024.04/CEDAR/x86-64-v4/site-libs/R/cedar_r4.5_bioc3.22-release/2026_02_03/incremental/rlibs/"

module load CEDAR/2024.04
module load R/4.5.0-foss-2024a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/test_evalK_onfullPC.R"
CHANNEL_FILE="${SCRIPT_DIR}/../channels_subset.txt"

# Check if channel file exists
if [ ! -f "$CHANNEL_FILE" ]; then
    echo "ERROR: Channel file not found: $CHANNEL_FILE"
    exit 1
fi

# Optional: set use_pw (default is FALSE)
USE_PW=${1:-FALSE}

echo "Reading channels from: $CHANNEL_FILE"
echo "use_pw: $USE_PW"
echo ""

# Read channels from file and process each
while IFS= read -r channel || [ -n "$channel" ]; do
    # Skip empty lines
    if [ -z "$channel" ]; then
        continue
    fi

    echo "===================="
    echo "Running: $channel"
    echo "===================="
    Rscript "$SCRIPT" "$channel" "$USE_PW"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to process channel $channel"
    else
        echo "SUCCESS: Completed channel $channel"
    fi
    echo ""
done < "$CHANNEL_FILE"

echo "All channels processed!"
