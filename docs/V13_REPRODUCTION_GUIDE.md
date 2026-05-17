# V13 Reproduction Guide

This document provides step-by-step instructions to reproduce the V13 results (53.3% on AIME 2024-2025).

---

## 1. Prerequisites

### Hardware
- 2× NVIDIA GPUs with ≥80GB VRAM each (tested on RTX PRO 6000 Blackwell 96GB)
- ~200GB disk space

### Software
```bash
conda create -n r2r python=3.10
conda activate r2r
pip install torch transformers datasets scikit-learn tqdm numpy
```

### Models (download to local)
- DeepSeek-R1-Distill-Qwen-1.5B (quick model)
- DeepSeek-R1-Distill-Qwen-32B (reference model)

### Clone the code
```bash
git clone https://github.com/Bruce1769/Entropy_Drift.git
cd Entropy_Drift
export PYTHONPATH=$(pwd):$PYTHONPATH
```

---

## 2. Prepare Training Data

The training data consists of 200 problems with teacher-forced 1.5B hidden states + top-100 logits + JS divergence vs 32B.

### 2.1 Data Files Needed

```
datasets/
├── js_entropy_risk_dataset/       # Arrow dataset (2.4G)
│   ├── train/                     # 652K token rows
│   └── validation/                # 106K token rows
├── prefill_1p5b_vs_32b_logits/
│   └── router_repr/               # 200 NPZ files (3.7G)
│       ├── run_0000.npz
│       └── run_0199.npz
├── sampled_200_shard0_0_100__ds32b.jsonl
├── sampled_200_shard1_100_200__ds32b.jsonl
└── sampled_500_queries_4_3_3.jsonl
```

### 2.2 Build Category Mapping

```bash
python -c "
import json, numpy as np
run2cat = {}
for shard in ['sampled_200_shard0_0_100__ds32b.jsonl', 'sampled_200_shard1_100_200__ds32b.jsonl']:
    with open(f'datasets/{shard}') as f:
        for line in f:
            d = json.loads(line)
            run2cat[d['run_index']] = d['category']
np.savez('/tmp/run2cat.npz',
    run_indices=np.array(list(run2cat.keys())), 
    categories=np.array(list(run2cat.values())))
"
```

### 2.3 Dataset Details

- **652,394 training tokens** from 140 runs (200 total, 140 train split)
- Each token row contains:
  - `small_logits[0:100]`: top-100 raw logits from 1.5B
  - `js_value`: Jensen-Shannon divergence between 1.5B and 32B top-100
  - `entropy_value`: Shannon entropy of 1.5B full-vocab distribution
  - `js_high`: original label with tau_js=0.05
  - `run_index`: which problem this token belongs to
  - `token_pos`: position in the response
- `router_repr/run_XXXX.npz`: last_hidden_state (1536-dim) + routing_input_token_ids per token

---

## 3. Training V13

### 3.1 Key Design Decisions

| Decision | V6 (baseline) | V13 (ours) |
|----------|:---:|:---:|
| Architecture | 3 blocks, 3× expansion | 3 blocks, **4× expansion** (r2r-style) |
| Token embedding | Frozen | **Trainable** |
| Weight init | Truncated Normal | **Kaiming Normal** |
| Dropout | 0.15 | **0.3** |
| Training data | All 652K tokens | **Only entropy < 0.6** (402K tokens) |
| Pos:Neg ratio | 1:4 (built-in) | **1:5** (downsampled) |
| Negative sampling | Random | **Category-balanced** (MATH:CODE:QA=4:3:3) |
| Checkpoint selection | max PR-AUC | **max low-entropy recall@0.5** |
| Label | JS > 0.1 | JS > 0.1 |

### 3.2 Launch Training

```bash
python script/train/train_router_multitask_js.py \
  --model_type bottleneck_r2r \
  --bce_pos_weight_power 1.0 \
  --aux_weight 0.0 \
  --dropout 0.3 \
  --js_cls_threshold 0.1 \
  --lowent_downsample 0.6 \
  --lowent_downsample_ratio 5 \
  --lowent_only \
  --lowent_cat_map /tmp/run2cat.npz \
  --out_dir ./run_v13_r2r_lowent_only
```

**Training takes ~1.3 GPU-hours**. Monitors via `grep '^epoch' train.log`.

### 3.3 Expected Metrics

After training, `metrics.json` should show approximately:

```json
{
  "val_pr_auc": 0.61,
  "recall_js_high_given_entropy_lt_0.6_thr_0.5": 0.95,
  "recall_js_high_given_entropy_lt_0.6_thr_best_val_f1": 0.72,
  "best_prob_threshold_on_val": 0.843,
  "best_epoch": 10
}
```

**Note**: PR-AUC (~0.61) is intentionally low because the model was only trained on low-entropy tokens. High-entropy tokens get random predictions, which is fine — they bypass the neural router at inference.

---

## 4. Inference

### 4.1 Create Config YAML

```yaml
{
  "special_tokens": {"think_start": 151648, "think_end": 151649},
  "quick": {
    "model_name": "DeepSeek-R1-Distill-Qwen-1.5B",
    "model_path": "/path/to/DeepSeek-R1-Distill-Qwen-1.5B",
    "param": "1.5", "mem_fraction_static": 0.15
  },
  "reference": {
    "model_name": "DeepSeek-R1-Distill-Qwen-32B",
    "model_path": "/path/to/DeepSeek-R1-Distill-Qwen-32B",
    "param": "32", "mem_fraction_static": 0.8
  },
  "router": {
    "switching_strategy": "multitask_js_router",
    "router_path": "./run_v13_r2r_lowent_only/best.pt",
    "threshold": 0.843,
    "entropy_threshold": 0.6,
    "entropy_topk_k": 100,
    "pretrained_model_name": "/path/to/DeepSeek-R1-Distill-Qwen-1.5B"
  }
}
```

### 4.2 Routing Logic

At each generation step:

```
1. Compute entropy of 1.5B full-vocab logits
2. If entropy >= 0.6 → route to 32B DIRECTLY (skip neural router)
3. If entropy < 0.6 → V13 neural router decides:
   - prob >= 0.843 → route to 32B
   - prob < 0.843 → stay with 1.5B
```

~69% tokens stay with 1.5B, ~31% go to 32B → avg 10.8B params/token.

### 4.3 Run Evaluation

```bash
python script/evaluate/hf_dataset_sglang.py \
  --config-path config/v13_r2r_lowent_only.yaml \
  --dataset aime --dataset_path /path/to/dataset \
  --use_hybrid --max_new_tokens 20000 \
  --output_dir ./eval_v13 \
  --dp_size 1 --tp_size 2
```

---

## 5. Expected Results

| Method | AIME24 | AIME25 | Total | avg_params | speed |
|--------|:------:|:------:|:-----:|:----------:|:-----:|
| V13 | 17/30=57% | 15/30=50% | 32/60=53.3% | 10.8B | 29 tok/s |
| r2r | 18/30=60% | 13/30=43% | 31/60=51.7% | 6.4B | 42 tok/s |

---

## 6. Architecture Details

### RouterBottleneckR2R (V13)

```
Input: concat(last_hidden(1536), token_emb(1536), logits_proj(1536)) = 4608-dim
  ↓
LayerNorm(4608) → Linear(4608, 256)  [bottleneck projection]
  ↓
R2RStyleBlock ×3:  [each: Pre-LN → Linear(256,1024) → GELU → Drop(0.3) → Linear(1024,256) → Drop(0.3) + residual]
  ↓
LayerNorm(256) → Linear(256, 1)  [classification head]
  ↓
Output: sigmoid score ∈ [0,1]
```

### R2RStyleBlock

```python
class R2RStyleBlock(nn.Module):
    def __init__(self, dim=256, expansion_factor=4, dropout=0.3):
        self.ln = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),    # 256 → 1024
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),    # 1024 → 256
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return x + self.mlp(self.ln(x))   # residual
```

**Key differences from V6 (RouterBottleneck3Block):**
- 4× expansion (vs 3×): 256→1024 instead of 256→768
- Kaiming init (vs Truncated Normal)
- Trainable token embeddings (vs frozen)
- Dropout 0.3 (vs 0.15)

---

## 7. Troubleshooting

### Low-entropy recall stuck at 10%
→ Check that `--lowent_only` is set and data loading code has the fix for `le_pos` calculation (should use `(js_high_tr > 0.5) & le` not just `js_high_tr > 0.5`)

### V13 gives empty output on most problems
→ Verify `entropy_threshold` is set to 0.6 in config, not 1.0. threshold=1.0 disables the entropy gate and sends ALL tokens through V13 which was never trained on high-entropy tokens.

### Training crashes with "Tensors must have same dimensions"
→ Check that `PyTorch` version ≥2.0 and all cached `.pyc` files are cleared before training.

### Model loaded as RouterMultiTaskFFN4 instead of RouterBottleneckR2R
→ Ensure `PYTHONPATH` points to Entropy_Drift (not Entropy_JS). The `--model_type bottleneck_r2r` must be saved in the checkpoint args.

---

## 8. Key Files Reference

| File | Purpose |
|------|---------|
| `r2r/models/multitask_js_router.py` | All model classes (V2-V13, R2RStyleBlock) |
| `r2r/utils/switching.py` | MultitaskJSRouterSwitching (inference) |
| `script/train/train_router_multitask_js.py` | Training script with lowent_downsample |
| `config/v13_r2r_lowent_only.yaml` | V13 inference config |
| `run_v13_r2r_lowent_only/best.pt` | V13 checkpoint (902MB) |
