#!/usr/bin/env bash
#SBATCH --job-name=aime32b
#SBATCH --partition=gpua800
#SBATCH --account=zhengyongjiang02_phd
#SBATCH --nodes=1
#SBATCH --qos=8a800
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/home/cpt/zhaodongwu20/Entropy_Drift/logs/aime_32b_pure_%j.out
#SBATCH --error=/gpfs/home/cpt/zhaodongwu20/Entropy_Drift/logs/aime_32b_pure_%j.err

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

OUTPUT_DIR=output/DeepSeek-32B_pure/aime

echo "Running pure 32B on AIME 24-25"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo

python -u script/evaluate/hf_dataset_sglang.py \
  --dataset "aime" \
  --config-path "config/DeepSeek-R1-Distill-Qwen-1.5B+DeepSeek-R1-Distill-Qwen-32B_multitask_js_router_e0.6.yaml" \
  --use_model "reference" \
  --dp_size 1 \
  --tp_size 1 \
  --batch_size 1 \
  --max_new_tokens 20000 \
  --temperature 0.0 \
  --top_p 1.0 \
  --random_seed 42 \
  --output_dir "$OUTPUT_DIR" \
  --resume \
  --skip_problem_ids "${SKIP_PROBLEM_IDS:-}"

echo "Done."
