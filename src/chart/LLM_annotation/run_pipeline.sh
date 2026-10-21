#!/bin/bash
# Full archetype annotation pipeline, steps 1-6. See README.md.
#
# Usage:
#   ./run_pipeline.sh                 # every stain enabled in STAIN_CONFIG
#   ./run_pipeline.sh PRPS6 TUBULIN   # named stains
#
# Environment overrides: PYTHON, MODEL, PROVIDER, BASE_PATH, OUT_ROOT.
#
# Requires one of the two providers:
#
#   LiteLLM gateway
#     export LITELLM_API_KEY="sk-..."
#     export LITELLM_BASE_URL="https://<gateway-host>"
#
#   Azure OpenAI
#     export AZURE_OPENAI_API_KEY="..."
#     export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
#     export AZURE_OPENAI_API_VERSION="2025-03-01-preview"        # optional
#     export AZURE_DEPLOYMENT_MAP='{"gpt-5.5":"gpt-5-5-deploy"}'  # optional
#
# PROVIDER=auto picks the gateway when it is configured and falls back to
# Azure; set PROVIDER=gateway or PROVIDER=azure to pin one.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Credentials. ~/.bashrc returns early for non-interactive shells, so sourcing
# this here makes the run work from cron, CI and tool shells too, not just an
# interactive terminal.
for cred in "${LLM_ENV_FILE:-$HOME/.litellm_env}" ./.litellm_env; do
    [ -f "$cred" ] && . "$cred"
done

# ========================
# CONFIGURATION
# ========================

# Interpreter. Set PYTHON=... to point at your own env; the conda env below is
# the one this pipeline was developed in, and python3 is the portable fallback.
DEFAULT_PYTHON=/home/ubuntu/miniforge3/envs/azure-openai/bin/python
[[ -x "$DEFAULT_PYTHON" ]] || DEFAULT_PYTHON="$(command -v python3 || true)"
PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

# Model: gpt-5.5 or gpt-5, both reasoning models; gpt-4o also works and takes
# temperature instead. See MODEL_PROFILES in llm_core.py.
MODEL="${MODEL:-gpt-5.5}"

# Provider: auto | gateway | azure
PROVIDER="${PROVIDER:-${LLM_PROVIDER:-auto}}"

# Where the archetype grid images live: $BASE_PATH/<image_folder_suffix>.
BASE_PATH="${BASE_PATH:-/data/Paula/ePC057/may2026/output2_lines}"

# Where results are written. Defaults to the pipeline folder itself; set
# OUT_ROOT=/path/to/results to keep the code folder clean.
OUT_ROOT="${OUT_ROOT:-.}"

# Completion-token budgets. Reasoning tokens are billed against these, so a
# high-effort step with a small budget returns an empty string. Step 4 runs at
# high effort and needs the largest budget.
MAX_TOKENS_STEP1=8000
MAX_TOKENS_STEP2=8000
MAX_TOKENS_STEP3=8000
MAX_TOKENS_STEP4=16000

# Reasoning effort per step (GPT-5 uses this instead of temperature).
REASONING_STEP1="low"
REASONING_STEP2="low"
REASONING_SUMMARY="medium"
REASONING_NAMES="high"
REASONING_STEP4="high"

# Temperatures are kept for non-GPT-5 models and ignored by GPT-5.
TEMP_STEP1=0.2
TEMP_STEP2=0.7
TEMP_SUMMARY=0.7
TEMP_NAMES=0.2
TEMP_STEP4=0.7

# Step 4 pair selection.
SIMILARITY_THRESHOLD=0.08     # cosine distance below which two summaries count as "too similar"
MAX_PAIRS_PER_ARCHETYPE=0     # 0 = compare every pair under the threshold

# "STAIN"="k:normal_archetype_ids:image_folder_suffix"
declare -A STAIN_CONFIG=(
    ["CALR"]="11:9:CALR"
    ["CAT"]="14:11:CAT-k14"
    ["CTNNB1"]="11:7:CTNNB1"
    ["DAPI"]="15:9:DAPI"
    ["EEA1"]="12:2:EEA1"
    ["ESR1"]="11:10:ESR1"
    ["FAK"]="12:5,11,8:FAK-k12"
    ["FIBRILLARIN"]="11:9:FIBRILLARIN"
    ["G3BP1"]="11:11:G3BP1"
    ["GOLGIN"]="17:3:GOLGIN"
    ["KI67"]="11:1,7:KI67"
    ["LAMP1"]="12:3:LAMP1"
    ["PHALLOIDIN"]="15:10,15:PHALLOIDIN"
    ["PRPS6"]="15:5:PRPS6"
    ["TOM20"]="14:10:TOM20-k14"
    ["TUBULIN"]="15:15:B-TUBULIN"
    ["yH2AX"]="12:11:yH2AX"
)

# ========================
# PREFLIGHT
# ========================

preflight() {
    local failed=0

    if [[ ! -x "$PYTHON" ]]; then
        echo "❌ Python interpreter not found: $PYTHON"
        echo "   Set PYTHON=/path/to/python (needs langchain-openai, sentence-transformers, pillow, scipy, tiktoken)"
        failed=1
    elif ! "$PYTHON" - <<'PY'
import importlib.util as iu
import sys
missing = [m for m in ("langchain_openai", "sentence_transformers", "PIL", "scipy", "tiktoken")
           if iu.find_spec(m) is None]
if missing:
    print("   missing modules: " + ", ".join(missing))
    sys.exit(1)
PY
    then
        echo "❌ $PYTHON is missing required packages"
        failed=1
    fi

    # Credentials: let llm_core resolve the provider so the shell and the
    # Python steps can never disagree about which one is in use.
    local provider_desc
    if provider_desc=$(LLM_PROVIDER="$PROVIDER" MODEL="$MODEL" "$PYTHON" - <<'PY' 2>&1
import os, sys
sys.path.insert(0, ".")
import llm_core
try:
    print(llm_core.check_llm_credentials(None, os.environ["MODEL"]))
except llm_core.PipelineError as exc:
    print(exc)
    sys.exit(1)
PY
    ); then
        echo "🔌 Provider: $provider_desc"
    else
        echo "❌ $provider_desc"
        failed=1
    fi

    if [[ ! -d "$BASE_PATH" ]]; then
        echo "❌ Image base path not found: $BASE_PATH"
        echo "   Set BASE_PATH=/path/to/archetype/images"
        failed=1
    fi

    if ! mkdir -p "$OUT_ROOT" 2>/dev/null; then
        echo "❌ Cannot create output root: $OUT_ROOT"
        failed=1
    fi

    [[ $failed -eq 0 ]] || exit 1
    echo "✅ Preflight OK  (python: $PYTHON)"
}

# ========================
# HELPERS
# ========================

parse_stain_config() {
    local stain=$1
    local config=${STAIN_CONFIG[$stain]:-}

    if [[ -z "$config" ]]; then
        echo "❌ No configuration for stain '$stain'. Available: ${!STAIN_CONFIG[*]}"
        return 1
    fi

    IFS=':' read -r k normal_str suffix <<< "$config"
    IFS=',' read -ra normals <<< "$normal_str"

    CURRENT_K=$k
    CURRENT_NORMALS=("${normals[@]}")
    CURRENT_IMAGE_DIR="${BASE_PATH}/${suffix}"

    if [[ ! -d "$CURRENT_IMAGE_DIR" ]]; then
        echo "❌ Image folder not found for $stain: $CURRENT_IMAGE_DIR"
        return 1
    fi

    echo "📋 $stain: k=$CURRENT_K | normal=${CURRENT_NORMALS[*]} | images=$CURRENT_IMAGE_DIR"
}

# Runs a step, records its duration, and aborts the stain if it fails.
run_step() {
    local label="$1" timing_log="$2"
    shift 2
    local start end elapsed
    start=$(date +%s)
    if ! "$@"; then
        end=$(date +%s)
        printf '%s\t%s\tFAILED after %ss\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label" "$((end - start))" \
            | tee -a "$timing_log"
        echo "❌ Step '$label' failed - stopping this stain."
        return 1
    fi
    end=$(date +%s)
    elapsed=$((end - start))
    printf '%s\t%s\t%ss\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$label" "$elapsed" | tee -a "$timing_log"
}

# ========================
# PIPELINE
# ========================

process_stain() {
    local stain=$1
    echo ""
    echo "======================================"
    echo "🧬 $stain"
    echo "======================================"

    parse_stain_config "$stain" || return 1

    local image_dir="$CURRENT_IMAGE_DIR"
    local annot_dir="${OUT_ROOT}/archetype_annot_${stain}"
    local baseline_dir="${OUT_ROOT}/contrastive_baselines_${stain}"
    local review_dir="${OUT_ROOT}/archetype_reviews_${stain}"
    local contrastive_json="${baseline_dir}/contrastive_baseline_notes.json"
    local dashboard="${OUT_ROOT}/dashboards_from_final/archetype_dashboard_${stain}.html"
    local timing_log="${review_dir}/workflow_timing.log"

    mkdir -p "$annot_dir" "$baseline_dir" "$review_dir" "${OUT_ROOT}/dashboards_from_final"
    printf 'timestamp\tstep\telapsed\n' > "$timing_log"

    echo ""
    echo "🔍 Step 1: annotation"
    run_step "step1_annotation" "$timing_log" \
        "$PYTHON" step1_annotate.py \
            --stain "$stain" \
            --image_folder "$image_dir" \
            --output_folder "$annot_dir" \
            --expected_archetypes "$CURRENT_K" \
            --model "$MODEL" \
            --provider "$PROVIDER" \
            --temperature "$TEMP_STEP1" \
            --reasoning_effort "$REASONING_STEP1" \
            --max_completion_tokens "$MAX_TOKENS_STEP1" \
        || return 1

    echo ""
    echo "🧪 Step 2: contrastive baseline"
    run_step "step2_contrastive_baseline" "$timing_log" \
        "$PYTHON" step2_contrastive_baselines.py \
            --stain "$stain" \
            --image_folder "$image_dir" \
            --output_folder "$baseline_dir" \
            --normal "${CURRENT_NORMALS[@]}" \
            --expected_archetypes "$CURRENT_K" \
            --model "$MODEL" \
            --provider "$PROVIDER" \
            --temperature "$TEMP_STEP2" \
            --reasoning_effort "$REASONING_STEP2" \
            --max_completion_tokens "$MAX_TOKENS_STEP2" \
        || return 1

    # Steps 1-2 validate against the clustering k (counting archetypes that were
    # deliberately excluded by renaming their image). From here on, the number
    # that matters is how many archetypes actually entered the run, which step 2
    # has already checked and written down.
    local effective_k
    effective_k=$("$PYTHON" -c "import json,sys; print(len(json.load(open(sys.argv[1]))['per_archetype_notes']))" \
                  "$contrastive_json") || return 1
    echo "🔢 Archetypes in this run: $effective_k (configured k=$CURRENT_K)"

    echo ""
    echo "🔬 Step 3: synthesis, names, hypothesis"
    run_step "step3_synthesize" "$timing_log" \
        "$PYTHON" step3_synthesize.py \
            --stain "$stain" \
            --report_folder "$annot_dir" \
            --contrastive_json "$contrastive_json" \
            --output_folder "$review_dir" \
            --expected_archetypes "$effective_k" \
            --model "$MODEL" \
            --provider "$PROVIDER" \
            --summary_temp "$TEMP_SUMMARY" \
            --names_temp "$TEMP_NAMES" \
            --summary_reasoning_effort "$REASONING_SUMMARY" \
            --names_reasoning_effort "$REASONING_NAMES" \
            --max_completion_tokens "$MAX_TOKENS_STEP3" \
        || return 1

    echo ""
    echo "🔁 Step 4: contrastive refinement -> final reports"
    run_step "step4_refine" "$timing_log" \
        "$PYTHON" step4_refine.py \
            --stain "$stain" \
            --review_folder "$review_dir" \
            --image_folder "$image_dir" \
            --contrastive_json "$contrastive_json" \
            --expected_archetypes "$effective_k" \
            --model "$MODEL" \
            --provider "$PROVIDER" \
            --temperature "$TEMP_STEP4" \
            --reasoning_effort "$REASONING_STEP4" \
            --max_completion_tokens "$MAX_TOKENS_STEP4" \
            --similarity_threshold "$SIMILARITY_THRESHOLD" \
            --max_pairs_per_archetype "$MAX_PAIRS_PER_ARCHETYPE" \
            --iterations 1 \
        || return 1

    echo ""
    echo "📊 Step 5: dashboard"
    run_step "step5_build_dashboard" "$timing_log" \
        "$PYTHON" step5_build_dashboard.py \
            --stain "$stain" \
            --review_folder "$review_dir" \
            --image_folder "$image_dir" \
            --output "$dashboard" \
            --expected_archetypes "$effective_k" \
        || return 1

    echo ""
    echo "🔎 Step 6: verify"
    run_step "step6_verify" "$timing_log" \
        "$PYTHON" step6_verify_run.py \
            --stain "$stain" \
            --annotation_folder "$annot_dir" \
            --review_folder "$review_dir" \
            --contrastive_json "$contrastive_json" \
            --dashboard "$dashboard" \
            --expected_archetypes "$effective_k" \
        || return 1

    echo ""
    echo "✅ $stain complete"
    echo "   Final reports: ${review_dir}/archetype*_final.md"
    echo "   Dashboard:     $dashboard"
    echo "   Timing:        $timing_log"
    return 0
}

# ========================
# MAIN
# ========================

echo "======================================"
echo "🧬 Archetype annotation pipeline (steps 1-6)"
echo "======================================"
echo "🏠 Working directory: $(pwd)"
echo "📂 Image base:        $BASE_PATH"
echo "🤖 Model:             $MODEL"
echo "🔀 Provider:          $PROVIDER"
echo ""

preflight

if [[ $# -gt 0 ]]; then
    STAINS_TO_PROCESS=("$@")
else
    STAINS_TO_PROCESS=("${!STAIN_CONFIG[@]}")
fi
echo "📌 Stains: ${STAINS_TO_PROCESS[*]}"

successful=()
failed=()
for stain in "${STAINS_TO_PROCESS[@]}"; do
    if process_stain "$stain"; then
        successful+=("$stain")
    else
        failed+=("$stain")
    fi
done

echo ""
echo "======================================"
echo "PIPELINE SUMMARY"
echo "======================================"
if [[ ${#successful[@]} -gt 0 ]]; then
    echo "✅ Succeeded: ${successful[*]}"
    for stain in "${successful[@]}"; do
        echo "   - ${OUT_ROOT}/dashboards_from_final/archetype_dashboard_${stain}.html"
    done
fi
if [[ ${#failed[@]} -gt 0 ]]; then
    echo "❌ Failed: ${failed[*]}"
    exit 1
fi
echo ""
