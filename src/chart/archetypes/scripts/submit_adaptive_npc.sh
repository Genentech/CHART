#!/bin/bash

# Script to submit adaptive npc metacell jobs for all channels
# Reads channel names from channels.txt and submits one SLURM array job per channel
# npc is adaptively determined per channel using elbow method

set -euo pipefail

CHANNELS_FILE="channels_subset.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if channels file exists
if [[ ! -f "$CHANNELS_FILE" ]]; then
    echo "Error: $CHANNELS_FILE not found"
    exit 1
fi

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

#SBATCH --job-name=metacell_adaptive_${channel}
#SBATCH --qos=1d
#SBATCH --array=1-4
#SBATCH --time=1-00:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/metacell_adaptive_${channel}_%A_%a.out
#SBATCH --error=logs/metacell_adaptive_${channel}_%A_%a.err

set -euo pipefail
mkdir -p logs

source /apps/rocs/init.sh 2024.04

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COMBOS="params_adaptive.txt"

# Get the line corresponding to this task id
line=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$COMBOS")

# Parse 2 fields (whitespace-delimited): gamma and knn
read -r p1 p2 <<< "\$line"

ml CEDAR/2024.04
ml R/release

echo "Task \${SLURM_ARRAY_TASK_ID}: gamma=\$p1 knn=\$p2 channel=$channel (npc will be adaptively selected)"

# Run R with adaptive npc selection
Rscript metacell_adaptive_npc.R "\$p1" "\$p2" "$channel"
EOF

done < "$CHANNELS_FILE"

echo "All jobs submitted!"
