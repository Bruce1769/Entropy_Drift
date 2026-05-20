#!/usr/bin/env bash
#SBATCH --job-name=aime_e06v13
#SBATCH --partition=gpua800
#SBATCH --account=zhengyongjiang02_phd
#SBATCH --qos=8a800
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --nodelist=gpua800n4
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/cpt/zhaodongwu20/Entropy_Drift/logs/aime_e06v13_%j.out
#SBATCH --error=/gpfs/home/cpt/zhaodongwu20/Entropy_Drift/logs/aime_e06v13_%j.err

set -euo pipefail

ROOT_DIR=/gpfs/home/cpt/zhaodongwu20/Entropy_Drift
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

module load cuda/12.4.1 gcc/11.4.0
export CC=$(which gcc)
export CXX=$(which g++)

source /gpfs/spack/opt/linux-rocky8-icelake/gcc-8.5.0/anaconda3-2022.10-4dp3trddxrrzcg6rozuot7ckgh3zjche/etc/profile.d/conda.sh
conda activate r2r

CONFIG_PATH=config/DeepSeek-R1-Distill-Qwen-1.5B+DeepSeek-R1-Distill-Qwen-32B_multitask_js_router_e0.6.yaml
DATASET=aime
OUTPUT_DIR=output/DeepSeek-1.5B+32B_v13_multitask_js_router_e0.6/aime_retest

echo "Running V13 multitask_js_router (entropy_threshold=0.6) on ${DATASET}"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo

python -u script/evaluate/hf_dataset_sglang.py \
  --dataset "$DATASET" \
  --config-path "$CONFIG_PATH" \
  --use_hybrid \
  --slm_tp_size 1 \
  --llm_tp_size 1 \
  --batch_size 1 \
  --max_new_tokens 20000 \
  --temperature 0.0 \
  --top_p 1.0 \
  --random_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --resume \
  --skip_problem_ids "${SKIP_PROBLEM_IDS:-}"

echo "Done."
