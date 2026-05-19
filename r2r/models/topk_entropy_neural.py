import torch
import torch.nn as nn
from typing import Optional
from r2r.utils.metrics import compute_topk_entropy
from r2r.utils.switching import ModelSwitchingStrategy
from r2r.utils.dataclass import ModelOutputs
from r2r.models.router import register_model


@register_model
class EntropyTopKRouter(nn.Module):
    """Router: SLM hidden states + token + top-k logits -> critical probability."""

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_dim: int = 1536,
        topk: int = 100,
        bottleneck_dim: int = 256,
        expansion: int = 3,
        num_blocks: int = 3,
        dropout: float = 0.15,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.ln_hidden = nn.LayerNorm(hidden_dim)
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.logits_projection = nn.Linear(topk, hidden_dim)
        fused_dim = hidden_dim * 3
        self.ln_fused = nn.LayerNorm(fused_dim)
        self.input_proj = nn.Linear(fused_dim, bottleneck_dim)

        for i in range(1, num_blocks + 1):
            setattr(self, f'ln{i}', nn.LayerNorm(bottleneck_dim))
            mlp = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim * expansion),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim * expansion, bottleneck_dim),
                nn.Dropout(dropout),
            )
            setattr(self, f'mlp{i}', mlp)

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self.num_blocks = num_blocks

    def forward(self, hidden_states, token, logits):
        h = self.ln_hidden(hidden_states)
        t = self.ln_token(self.token_embeddings(token))
        l = self.logits_projection(logits)
        fused = torch.cat([h, t, l], dim=-1)
        x = self.input_proj(self.ln_fused(fused))
        for i in range(1, self.num_blocks + 1):
            ln = getattr(self, f'ln{i}')
            mlp = getattr(self, f'mlp{i}')
            x = x + mlp(ln(x))
        return self.head_cls(self.ln_out(x))


def load_topk_router(checkpoint_path: str, device: str = "cuda"):
    """Load EntropyTopKRouter from a compatible checkpoint (best.pt or converted)."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check format: raw best.pt or NeuralSwitching-format converted
    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
        args = ckpt.get("args", {})
        threshold = args.get("best_prob_threshold", ckpt.get("best_prob_threshold", 0.5))
    else:
        state_dict = ckpt.get("state_dict", {})
        common_args = ckpt.get("common_args", {})
        init_args = ckpt.get("init_args", {})
        threshold = common_args.get("threshold", 0.5)

    model = EntropyTopKRouter(**(ckpt.get("init_args", {})))
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, threshold


class EntropyTopKNeuralSwitching(ModelSwitchingStrategy):
    """Two-stage routing: top-k entropy gate -> neural router."""

    def __init__(
        self,
        model_path: str,
        entropy_threshold: float = 0.6,
        entropy_topk_k: int = 100,
        neural_threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        use_cuda_graph: bool = True,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_threshold = float(entropy_threshold)
        self.entropy_topk_k = int(entropy_topk_k)
        self.device = device

        self.router_model, loaded_threshold = load_topk_router(model_path, device=device)
        self.neural_threshold = (
            float(neural_threshold) if neural_threshold is not None
            else float(loaded_threshold)
        )

        print(
            f"EntropyTopKNeuralSwitching: topk={self.entropy_topk_k}, "
            f"entropy_threshold={self.entropy_threshold}, "
            f"neural_threshold={self.neural_threshold}"
        )

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        device = next_token_logits.device

        # Synchronize CUDA stream before doing topk on sglang's logits
        if next_token_logits.is_cuda:
            torch.cuda.synchronize(next_token_logits.device)

        entropy_values = compute_topk_entropy(next_token_logits, self.entropy_topk_k)
        self.last_route_scores = entropy_values.detach().cpu()
        self.last_route_metric_name = f"entropy_topk_{self.entropy_topk_k}"

        high_entropy = entropy_values >= self.entropy_threshold
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=device)

        low_indices = (~high_entropy).nonzero(as_tuple=True)[0]
        if len(low_indices) > 0:
            with torch.no_grad():
                hidden = outputs.hidden_states[-1][:, -1, :][low_indices].to(
                    device=self.device, dtype=torch.float32
                )
                token = outputs.token[:, -1][low_indices].to(
                    device=self.device, dtype=torch.long
                )
                topk_logits, _ = torch.topk(
                    next_token_logits[low_indices].to(device=self.device),
                    k=self.entropy_topk_k, dim=-1
                )
                logit = self.router_model(
                    hidden_states=hidden, token=token, logits=topk_logits
                )
                prob = torch.sigmoid(logit).squeeze(-1)
                neural_choices = (prob >= self.neural_threshold).to(torch.int)
                model_choices[low_indices] = neural_choices.to(device=device)

        if high_entropy.any():
            model_choices[high_entropy] = 1

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices
