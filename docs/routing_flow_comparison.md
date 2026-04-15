# R2R / entropy / entropy_js 流程对比图

这里把“R2R 方法”按仓库默认运行方式解释为 `neural` 路由器:

- 基础配置 `config/Qwen3-0.6B+Qwen3-8B.yaml` 只提供 `router_path`，未显式指定 `switching_strategy`。
- 评测脚本在未指定策略时会默认回退到 `switching_strategy = "neural"`。

## 流程图

```mermaid
flowchart LR
    A[输入 prompt / 当前 decode 上下文]
    A --> B[SLM 单步前向<br/>产出 next-token logits / hidden_states / quick token]

    B --> N1
    B --> E1
    B --> J1

    subgraph N[R2R 默认方法: neural router]
        direction TB
        N1[抽取 router 特征<br/>logits / hidden_states / token<br/>按训练配置决定]
        N1 --> N2[训练好的 router 前向]
        N2 --> N3[得到 critical_prob / divergent prob]
        N3 --> N4{>= threshold?}
        N4 -- 否 --> N5[接受 SLM token]
        N4 -- 是 --> N6[把当前上下文发给 LLM<br/>继续解 1 个 token]
        N6 --> N7[接受 LLM token]
    end

    subgraph E[entropy]
        direction TB
        E1[仅使用 SLM next-token logits]
        E1 --> E2[计算 entropy]
        E2 --> E3{entropy < entropy_threshold?}
        E3 -- 是 --> E4[接受 SLM token]
        E3 -- 否 --> E5[把当前上下文发给 LLM<br/>继续解 1 个 token]
        E5 --> E6[接受 LLM token]
    end

    subgraph J[entropy_js]
        direction TB
        J1[先读取 SLM next-token logits]
        J1 --> J2[计算 entropy]
        J2 --> J3{entropy >= entropy_threshold?}
        J3 -- 否 --> J4[直接接受 SLM token]
        J3 -- 是 --> J5[保存 quick token + quick logits]
        J5 --> J6[请求 LLM 计算同一步 reference token + full logits]
        J6 --> J7[计算 JS divergence<br/>JS(quick logits, reference logits)]
        J7 --> J8{JS >= js_threshold?}
        J8 -- 否 --> J9[保留 SLM token]
        J8 -- 是 --> J10[切换为 LLM token]
    end

    N5 --> Z[写回最终 token<br/>进入下一个 decode step]
    N7 --> Z
    E4 --> Z
    E6 --> Z
    J4 --> Z
    J9 --> Z
    J10 --> Z
```

## 一句话差异

| 方法 | 决策输入 | 决策阶段 | 是否需要 LLM 的参考分布 | 最终 token 的来源 |
|---|---|---|---|---|
| R2R (`neural`) | SLM 的 `logits / hidden_states / token`，取决于训练好的 router 配置 | 1 阶段 | 不需要 | router 判 quick 则用 SLM；判 reference 则直接用 LLM |
| `entropy` | SLM 的 next-token `logits` | 1 阶段 | 不需要 | 低熵用 SLM；高熵直接用 LLM |
| `entropy_js` | 先看 SLM entropy，再比较 SLM/LLM 的 JS divergence | 2 阶段 | 需要 | 先 entropy 预筛，再由 JS 决定保留 SLM 还是切到 LLM |

## 代码对应

- 默认 `R2R = neural` 的回退逻辑: `script/evaluate/hf_dataset_sglang.py`
- `entropy` 与 `entropy_js` 的实现: `r2r/utils/switching.py`
- SLM 侧先做路由、再决定是否向 LLM 发请求: `r2r/models/sglang_patch/slm_server.py`
- `entropy_js` 中 LLM 返回 reference logits 后的最终 token 决策: `r2r/models/sglang_patch/slm_server.py`
- LLM 侧返回 reference token / reference logits: `r2r/models/sglang_patch/llm_server.py`

## 读图建议

- 如果你想强调“实现最简单”，看 `entropy`。
- 如果你想强调“默认 R2R 的学习式路由”，看 `neural`。
- 如果你想强调“先粗筛再精判”，看 `entropy_js`。
