#!/bin/bash

# Script to submit robust archetype analysis jobs for all channels
# Reads channel names from channels file and submits one SLURM array job per channel

set -euo pipefail

# Default parameters
CHANNELS_FILE="${1:-../channels_subset.txt}"
USE_PW="${2:-FALSE}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if channels file exists
if [[ ! -f "$CHANNELS_FILE" ]]; then
    echo "Error: $CHANNELS_FILE not found"
    exit 1
fi

# Check if params_adaptive.txt exists (gamma, knn only - npc auto-detected)
PARAMS_FILE="../params_adaptive.txt"
if [[ ! -f "$PARAMS_FILE" ]]; then
    echo "Error: $PARAMS_FILE not found"
    exit 1
fi

echo "Submitting robust AA jobs for channels in: $CHANNELS_FILE"
echo "use_pw: $USE_PW"
echo "----------------------------------------"

# Read channels and submit jobs
while IFS= read -r channel || [[ -n "$channel" ]]; do
    # Skip empty lines
    if [[ -z "$channel" ]]; then
        continue
    fi

    echo "Submitting job for channel: $channel"

    # Submit the SLURM job with the channel as an argument
    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=AA_${channel}_pw${USE_PW}
#SBATCH --qos=1d
#SBATCH --array=1-4
#SBATCH --time=1-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/AA_${channel}_%A_%a.out
#SBATCH --error=logs/AA_${channel}_%A_%a.err

set -euo pipefail
mkdir -p logs

source /apps/rocs/init.sh 2024.04

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMBOS="$PARAMS_FILE"

# Get the line corresponding to this task id
line=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$COMBOS")

# Parse 2 fields (whitespace-delimited): gamma and knn (npc auto-detected)
read -r p1 p2 <<< "\$line"

ml CEDAR/2024.04
ml R/release

echo "Task \${SLURM_ARRAY_TASK_ID}: gamma=\$p1 knn=\$p2 channel=$channel pw=$USE_PW (npc auto-detected)"

# Run R (npc will be auto-detected from existing supercell files)
Rscript "$SCRIPT_DIR/robust_archetyping.R" "\$p1" "\$p2" "$channel" "$USE_PW"
EOF

done < "$CHANNELS_FILE"

echo "All jobs submitted!"
