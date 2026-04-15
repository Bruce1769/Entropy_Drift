# Router 与 Entropy / Entropy+JS 验证文档（2026-04-06 更新）

本文基于已完成实验，回答两类问题：

1. `router=1` 的位置是否确实对应高 Entropy / 高 JS
2. 下一步如何严谨验证：在线路由时到底用 `Entropy` 单阶段更好，还是 `Entropy+JS` 二阶段更好

---

## 1. 已完成实验设置

### 1.1 模型与路由器

- SLM: `Qwen/Qwen3-0.6B`
- LLM: `Qwen/Qwen3-32B`
- Router: `Qwen3-0.6B+Qwen3-32B/default_router.pt`
- Router threshold: `0.3564646464646465`（从 router 配置读取）

路由决策定义：

- `router=1`: 该 token 判定为 diverge / 需要切到 reference
- `router=0`: 非 diverge

### 1.2 数据与轨迹口径

- AIME24+25: 60 题
- GPQA: 40 题
- LiveCodeBench: 40 题

统一口径：使用 `Qwen3-0.6B` greedy 轨迹（每题 64 token），在同一轨迹位置上对齐 `0.6B` 与 `32B` logits，计算逐 token 指标。

### 1.3 关键脚本与产物

- 轨迹+logits 运行脚本：`experiments/qwen3_distribution_align/run_experiment.py`
- Router 统计脚本：`experiments/qwen3_distribution_align/scripts/analyze_router_diverge_positions.py`

结果文件：

- AIME: `/remote-home/pxl/experiments/qwen3_distribution_align/runs/aime60_ref06_jscheck/router_diverge_stats/token_level_router_entropy_js.csv`
- GPQA: `/remote-home/pxl/experiments/qwen3_distribution_align/runs/gpqa40_ref06_jscheck/router_diverge_stats/token_level_router_entropy_js.csv`
- LCB: `/remote-home/pxl/experiments/qwen3_distribution_align/runs/lcb40_ref06_jscheck/router_diverge_stats/token_level_router_entropy_js.csv`
- 三数据集合并汇总：`/remote-home/pxl/experiments/qwen3_distribution_align/runs/router_diverge_3datasets_summary.csv`

---

## 2. 已完成实验结论（核心数字）

### 2.1 Router=1 的位置是否高 Entropy / 高 JS

合并三数据集（8960 token）：

- `E[H | router=1] = 1.2718` bits
- `E[H | router=0] = 0.1937` bits
- `E[JS | router=1] = 0.1647`
- `E[JS | router=0] = 0.0257`
- `P(H>0.45 | router=1) = 0.9221`
- `P(JS>0.1 | router=1) = 0.5307`
- `P(JS>0.2 | router=1) = 0.3192`

结论：`router=1` 与高 Entropy / 高 JS 显著相关，这个规律在 AIME / GPQA / LCB 上都成立。

### 2.2 你关心的条件概率（合并）

- `P(JS>0.1 且 H>0.45 | router=1) = 0.5156`（1192/2312）
- `P(router=1 | JS>0.1 且 H>0.45) = 0.8011`（1192/1488）
- `P(router=1 | H>0.45) = 0.6675`（2132/3194）

说明：相比只看 `H>0.45`，加上 `JS>0.1` 后，对 router=1 的定位更“纯”。

### 2.3 与 top1 一致性的关系（合并）

- `P(top1相同 | router=1) = 0.5817`
- `P(top1相同 | JS>0.1 且 H>0.45) = 0.3058`
- `P(top1相同 | JS>0.2 且 H>0.45) = 0.0884`

说明：在高 JS 且高熵区域，SLM/LLM top1 不一致概率明显升高。

### 2.4 分数据集汇总（当前版本）

| dataset | router_1_share | P(H>0.45\|r1) | P(JS>0.1\|r1) | P(JS>0.2\|r1) | P(r1\|H>0.45) | P(r1\|JS>0.1&H>0.45) |
|---|---:|---:|---:|---:|---:|---:|
| AIME60 | 0.1693 | 0.9323 | 0.5231 | 0.3138 | 0.5632 | 0.7243 |
| GPQA40 | 0.2824 | 0.9557 | 0.5740 | 0.3416 | 0.6619 | 0.8100 |
| LCB40 | 0.3668 | 0.8892 | 0.5027 | 0.3056 | 0.7775 | 0.8588 |
| ALL | 0.2580 | 0.9221 | 0.5307 | 0.3192 | 0.6675 | 0.8011 |

> 详细表：`/remote-home/pxl/experiments/qwen3_distribution_align/runs/router_conditional_probs_compact.csv`

---

## 3. 如何验证“Entropy 直接判定” vs “Entropy+JS 二阶段判定”

这里建议直接做**同一评价口径下的 Pareto 对比**，不要只比较单点概率。

### 3.1 待比较策略（必须同时跑）

- `S_entropy`: `route = 1[H >= tau_H]`
- `S_entropy_js`: `route = 1[H >= tau_H and JS >= tau_JS]`
- `S_router`: 当前 neural router（作为参考基线）
- 辅助基线：`always_slm`、`always_llm`

### 3.2 统一评价指标

质量指标（按任务）：

- AIME: accuracy
- GPQA: accuracy
- LCB: pass@1（或现有同口径指标）

成本指标：

- `LLM token ratio`（最终由 LLM 生成的 token 占比）
- `RPC 次数`（SLM->LLM 额外请求次数）
- `RPC 负载`（每次发送 logits 大小）
- 延迟：avg / P95（如果在线服务可量测）

### 3.3 成本差异应如何报告（重点）

若对同一候选集合 `C = {H >= tau_H}`：

- `Entropy` 单阶段：只用 SLM 本地熵，不需要额外 JS 计算
- `Entropy+JS`：仅在候选 `C` 上请求/计算 JS

若 JS 用 full vocab：

- 每次需处理 `V=151936` 维 logits

若 JS 用 top-k（例如 `k=100`）：

- 负载约缩小到 `k/V ≈ 0.000658`（约 1/1519）

因此建议在图里同时给：

- 质量 vs LLM token ratio
- 质量 vs 额外 RPC 负载

这样才能回答“值不值得做二阶段”。

### 3.4 阈值扫描建议（结合当前结果）

- `tau_H`: `[0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.0]`
- `tau_JS`: `[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]`

优先关注：

- `tau_H=0.45`（你当前大量分析的基线）
- `tau_JS=0.10` 与 `tau_JS=0.20`（已验证可明显区分难点）

### 3.5 判定标准（如何下结论）

若出现以下任一情况，可认为 `Entropy+JS` 更合适：

1. 在同等质量下，`LLM token ratio` 更低
2. 在同等成本下，质量更高
3. 在多数据集上更稳定（不是只在单一集有效）

若 `Entropy` 在大多数成本区间与 `Entropy+JS` 几乎重合，且系统实现更简单，可优先 `Entropy`。

---

## 4. 直接可执行的下一步实验清单

1. 先离线扫描（最快）：
   - 基于现有 token-level 文件，生成 `Entropy` 与 `Entropy+JS` 的路由掩码
   - 统计路由比例、top1 一致率、以及与 router 的 agreement
2. 再做在线端到端：
   - 在同一解码配置下跑 `S_entropy` 与 `S_entropy_js`
   - 记录 accuracy/pass@1、LLM token ratio、延迟、RPC 负载
3. 产出一张总 Pareto 图：
   - x 轴成本（LLM token ratio 或 RPC 负载）
   - y 轴质量（AIME/GPQA/LCB）

---

## 5. 当前阶段结论（可用于汇报）

1. `router=1` 与高 Entropy / 高 JS 有稳定正相关，三数据集一致。
2. 仅用 `H>0.45` 能筛出不少 router=1，但纯度一般（`P(router=1|H>0.45)=0.6675`）。
3. 增加 `JS>0.1` 后纯度明显提升（`P(router=1|JS>0.1&H>0.45)=0.8011`）。
4. 高 JS 且高熵位置上，SLM/LLM top1 一致率显著更低，说明二阶段信号确实在捕捉更难位置。
5. 是否“最终应该用 Entropy 还是 Entropy+JS”，要以 Pareto（质量-成本）为准，不建议只看单个条件概率。
