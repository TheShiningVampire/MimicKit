#!/bin/bash
#SBATCH --job-name=mk-dance_spin
#SBATCH --partition=katef
#SBATCH --qos=katef_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%j.out
# DeepMimic PPO, humanoid dance_spin, headless video every 100 iters -> wandb.
set -euo pipefail
cd /home/vawale/my_things/codes/MimicKit/.worktrees/dance_spin
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /data/user_data/vawale/envs/mimickit
export WANDB_DIR=/data/user_data/vawale/wandb
export WANDB_CACHE_DIR=/data/user_data/vawale/wandb/cache
export WANDB_RUN_GROUP=deepmimic_humanoid
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
RUN=deepmimic_humanoid_dance_spin_$(date +%m%d_%H%M)
OUT=/data/user_data/vawale/MimicKit/output/$RUN
mkdir -p "$OUT"
echo "run=$RUN out=$OUT node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_ppo_args.txt \
  --env_config experiments/dance_spin/deepmimic_humanoid_dance_spin_env.yaml \
  --mode train --visualize false --video true --logger wandb \
  --num_envs 4096 --max_samples 400000000 --save_int_models true \
  --out_dir "$OUT"
