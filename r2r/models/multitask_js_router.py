"""
FFN router used for JS>0.1 hybrid routing (logits top-k + last hidden + token id).
Shared by training (train_router_multitask_js.py) and MultitaskJSRouterSwitching.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

DEFAULT_PRETRAINED_15B = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-1.5B"


class R2RStyleBlock(nn.Module):
    """Pre-LN block with 4× expansion, GELU, Dropout, and residual connection (r2r style)."""
    def __init__(self, dim, expansion_factor=4, dropout=0.3):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        expand_dim = dim * expansion_factor
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(self.ln(x))


class RouterBottleneckR2R(nn.Module):
    """
    r2r-architecture bottleneck router trained on our JS pipeline.

    concat(hidden, token_emb, logits_proj) -> LayerNorm -> Linear(4608, 256)
      -> Block x3 (r2r-style 4× expansion, Pre-LN, residual)
      -> LayerNorm -> cls_head

    vs V6: 4× expansion (not 3×), Kaiming init, trainable embeddings, dropout 0.3.
    """
    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        bottleneck_dim: int = 256,
        expansion_factor: int = 4,
        dropout: float = 0.3,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.normalize_inputs = normalize_inputs
        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B
        m = AutoModelForCausalLM.from_pretrained(pm, torch_dtype=torch.float32, trust_remote_code=True)
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)
        combined = hidden_states_size + embed_dim + hidden_states_size
        self.ln_fused = nn.LayerNorm(combined)
        self.input_proj = nn.Linear(combined, bottleneck_dim)

        self.blocks = nn.ModuleList([
            R2RStyleBlock(bottleneck_dim, expansion_factor, dropout)
            for _ in range(num_blocks)
        ])

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, logits, hidden_states, token):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        while te.dim() > 2: te = te.squeeze(1)
        while lf.dim() > 2: lf = lf.squeeze(1)
        while hidden_states.dim() > 2: hidden_states = hidden_states.squeeze(1)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        x = torch.cat([hidden_states, te, lf], dim=-1)
        x = self.ln_fused(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.head_cls(x), torch.zeros_like(self.head_cls(x))


class RouterMultiTaskFFN4(nn.Module):
    """
    cat([hidden, token_emb(token_id), logits_proj(logits)]) -> 4-layer FFN -> cls + reg heads.
    """

    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_states_size = hidden_states_size
        self.logits_size = logits_size
        self.normalize_inputs = normalize_inputs
        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B

        m = AutoModelForCausalLM.from_pretrained(
            pm,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)
        combined = hidden_states_size + embed_dim + hidden_states_size
        self.ln_fused = nn.LayerNorm(combined)
        layers: list[nn.Module] = []
        d_in = combined
        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(d_in, ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            d_in = ffn_dim
        self.ffn = nn.Sequential(*layers)
        self.head_cls = nn.Linear(ffn_dim, 1)
        self.head_reg = nn.Linear(ffn_dim, 1)
        self._init_linear_weights()

    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits, hidden_states, token):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        x = torch.cat([hidden_states, te, lf], dim=-1)
        x = self.ln_fused(x)
        h = self.ffn(x)
        return self.head_cls(h), self.head_reg(h)


class RouterBottleneck2Block(nn.Module):
    """
    Lightweight bottleneck router: compress first, then 2 residual Pre-LN blocks
    with 2x expansion, single classification head.

    concat(hidden, token_emb, logits_proj) -> LayerNorm -> Linear(4608, bottleneck)
      -> Block x2: Pre-LN -> Linear(bn, bn*expand) -> GELU -> Dropout -> Linear(bn*expand, bn) + residual
      -> LayerNorm -> cls_head

    ~1.7M params vs ~7.9M for RouterMultiTaskFFN4 (ffn_dim=1024) vs ~4.3M for r2r.
    """

    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        bottleneck_dim: int = 256,
        expansion_factor: int = 2,
        dropout: float = 0.15,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_states_size = hidden_states_size
        self.logits_size = logits_size
        self.normalize_inputs = normalize_inputs
        self.bottleneck_dim = bottleneck_dim
        self.expansion_factor = expansion_factor

        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B
        m = AutoModelForCausalLM.from_pretrained(
            pm, torch_dtype=torch.float32, trust_remote_code=True,
        )
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)

        combined = hidden_states_size + embed_dim + hidden_states_size
        self.ln_fused = nn.LayerNorm(combined)
        self.input_proj = nn.Linear(combined, bottleneck_dim)

        expand_dim = bottleneck_dim * expansion_factor
        # Block 1
        self.ln1 = nn.LayerNorm(bottleneck_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(bottleneck_dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, bottleneck_dim),
            nn.Dropout(dropout),
        )
        # Block 2
        self.ln2 = nn.LayerNorm(bottleneck_dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(bottleneck_dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, bottleneck_dim),
            nn.Dropout(dropout),
        )

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self._init_linear_weights()

    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits, hidden_states, token):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        x = torch.cat([hidden_states, te, lf], dim=-1)
        x = self.ln_fused(x)
        x = self.input_proj(x)           # 4608 -> 256
        # Block 1 with residual
        x = x + self.mlp1(self.ln1(x))
        # Block 2 with residual
        x = x + self.mlp2(self.ln2(x))
        x = self.ln_out(x)
        return self.head_cls(x), torch.zeros_like(self.head_cls(x))  # reg head dummy


class RouterBottleneck3Block(nn.Module):
    """
    Deeper bottleneck router: 3 residual Pre-LN blocks with 3x expansion.

    concat(hidden, token_emb, logits_proj) -> LayerNorm -> Linear(4608, bottleneck)
      -> Block x3: Pre-LN -> Linear(bn, bn*expand) -> GELU -> Dropout -> Linear(bn*expand, bn) + residual
      -> LayerNorm -> cls_head

    ~2.3M FFN params (bottleneck=256, expand=3, dropout=0.15).
    """

    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        bottleneck_dim: int = 256,
        expansion_factor: int = 3,
        dropout: float = 0.15,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_states_size = hidden_states_size
        self.logits_size = logits_size
        self.normalize_inputs = normalize_inputs
        self.bottleneck_dim = bottleneck_dim
        self.expansion_factor = expansion_factor

        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B
        m = AutoModelForCausalLM.from_pretrained(
            pm, torch_dtype=torch.float32, trust_remote_code=True,
        )
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)

        combined = hidden_states_size + embed_dim + hidden_states_size
        self.ln_fused = nn.LayerNorm(combined)
        self.input_proj = nn.Linear(combined, bottleneck_dim)

        expand_dim = bottleneck_dim * expansion_factor
        self.num_blocks = 3
        for i in range(self.num_blocks):
            setattr(self, f"ln{i+1}", nn.LayerNorm(bottleneck_dim))
            setattr(self, f"mlp{i+1}", nn.Sequential(
                nn.Linear(bottleneck_dim, expand_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expand_dim, bottleneck_dim),
                nn.Dropout(dropout),
            ))

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self._init_linear_weights()

    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits, hidden_states, token):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        x = torch.cat([hidden_states, te, lf], dim=-1)
        x = self.ln_fused(x)
        x = self.input_proj(x)
        for i in range(self.num_blocks):
            x = x + getattr(self, f"mlp{i+1}")(getattr(self, f"ln{i+1}")(x))
        x = self.ln_out(x)
        return self.head_cls(x), torch.zeros_like(self.head_cls(x))


class RouterBottleneck3BlockV8(nn.Module):
    """Dual-head: head_cls for JS>0.1, head_div for divergent (JS & low entropy)."""

    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        bottleneck_dim: int = 256,
        expansion_factor: int = 3,
        dropout: float = 0.15,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
    ):
        super().__init__()
        self.hidden_states_size = hidden_states_size
        self.logits_size = logits_size
        self.normalize_inputs = normalize_inputs

        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B
        m = AutoModelForCausalLM.from_pretrained(pm, torch_dtype=torch.float32, trust_remote_code=True)
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)
        combined = hidden_states_size + embed_dim + hidden_states_size
        self.ln_fused = nn.LayerNorm(combined)
        self.input_proj = nn.Linear(combined, bottleneck_dim)

        expand_dim = bottleneck_dim * expansion_factor
        self.num_blocks = 3
        for i in range(self.num_blocks):
            setattr(self, f"ln{i+1}", nn.LayerNorm(bottleneck_dim))
            setattr(self, f"mlp{i+1}", nn.Sequential(
                nn.Linear(bottleneck_dim, expand_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expand_dim, bottleneck_dim),
                nn.Dropout(dropout),
            ))

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self.head_div = nn.Linear(bottleneck_dim, 1)
        self._init_linear_weights()

    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits, hidden_states, token):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        while te.dim() > 2: te = te.squeeze(1)
        while lf.dim() > 2: lf = lf.squeeze(1)
        while hidden_states.dim() > 2: hidden_states = hidden_states.squeeze(1)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        x = torch.cat([hidden_states, te, lf], dim=-1)
        x = self.ln_fused(x)
        x = self.input_proj(x)
        for i in range(self.num_blocks):
            x = x + getattr(self, f"mlp{i+1}")(getattr(self, f"ln{i+1}")(x))
        x = self.ln_out(x)
        return self.head_cls(x), self.head_div(x)


class RouterBottleneck3BlockV7(nn.Module):
    """
    V6 + explicit entropy input + learnable entropy embedding.

    concat(hidden, token_emb, logits_proj, entropy_emb) -> LayerNorm -> Linear(4608+ent_dim, bn)
      -> Block x3: Pre-LN -> Linear(bn, bn*expand) -> GELU -> Dropout -> Linear(bn*expand, bn) + residual
      -> LayerNorm -> cls_head

    Entropy scalar is projected to a small embedding so the network can learn
    non-linear interactions with other features.
    """

    def __init__(
        self,
        hidden_states_size: int = 1536,
        logits_size: int = 100,
        pretrained_model_name: str | None = None,
        bottleneck_dim: int = 256,
        expansion_factor: int = 3,
        dropout: float = 0.15,
        normalize_inputs: bool = True,
        freeze_token_embeddings: bool = False,
        entropy_emb_dim: int = 16,
    ):
        super().__init__()
        self.hidden_states_size = hidden_states_size
        self.logits_size = logits_size
        self.normalize_inputs = normalize_inputs
        self.bottleneck_dim = bottleneck_dim
        self.expansion_factor = expansion_factor
        self.entropy_emb_dim = entropy_emb_dim

        pm = pretrained_model_name or DEFAULT_PRETRAINED_15B
        m = AutoModelForCausalLM.from_pretrained(
            pm, torch_dtype=torch.float32, trust_remote_code=True,
        )
        emb = m.get_input_embeddings()
        embed_dim = emb.embedding_dim
        self.token_embeddings = nn.Embedding(emb.num_embeddings, embed_dim)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(emb.weight)
        if freeze_token_embeddings:
            self.token_embeddings.weight.requires_grad = False
        del m

        if normalize_inputs:
            self.ln_hidden = nn.LayerNorm(hidden_states_size)
            self.ln_token = nn.LayerNorm(embed_dim)
        else:
            self.ln_hidden = None
            self.ln_token = None

        self.logits_projection = nn.Linear(logits_size, hidden_states_size)

        # Entropy embedding: scalar → small vector to learn non-linear interactions
        self.w_entropy = nn.Parameter(torch.empty(1, entropy_emb_dim))
        nn.init.trunc_normal_(self.w_entropy, std=0.02)

        combined = hidden_states_size + embed_dim + hidden_states_size + entropy_emb_dim
        self.ln_fused = nn.LayerNorm(combined)
        self.input_proj = nn.Linear(combined, bottleneck_dim)

        expand_dim = bottleneck_dim * expansion_factor
        self.num_blocks = 3
        for i in range(self.num_blocks):
            setattr(self, f"ln{i+1}", nn.LayerNorm(bottleneck_dim))
            setattr(self, f"mlp{i+1}", nn.Sequential(
                nn.Linear(bottleneck_dim, expand_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expand_dim, bottleneck_dim),
                nn.Dropout(dropout),
            ))

        self.ln_out = nn.LayerNorm(bottleneck_dim)
        self.head_cls = nn.Linear(bottleneck_dim, 1)
        self._init_linear_weights()

    def _init_linear_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    nn.init.trunc_normal_(m.weight, std=0.02)
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, logits, hidden_states, token, entropy=None):
        te = self.token_embeddings(token)
        lf = self.logits_projection(logits)
        # Ensure all tensors are 2D (batch, feat) before concat
        while te.dim() > 2:
            te = te.squeeze(1)
        while lf.dim() > 2:
            lf = lf.squeeze(1)
        while hidden_states.dim() > 2:
            hidden_states = hidden_states.squeeze(1)
        if self.normalize_inputs:
            hidden_states = self.ln_hidden(hidden_states)
            te = self.ln_token(te)
        # Entropy embedding: scalar * learned projection → small vector
        if entropy is None:
            ent_feat = torch.zeros(hidden_states.shape[0], self.entropy_emb_dim,
                                   device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            entropy_flat = entropy.reshape(-1)
            ent_feat = entropy_flat.unsqueeze(-1) * self.w_entropy  # (B,1)*(1,D) → (B,D)
        try:
            x = torch.cat([hidden_states, te, lf, ent_feat], dim=-1)
        except RuntimeError:
            import sys
            print(f"V7 dims: hs={hidden_states.dim()}D sz={hidden_states.shape}, "
                  f"te={te.dim()}D sz={te.shape}, lf={lf.dim()}D sz={lf.shape}, "
                  f"ent={ent_feat.dim()}D sz={ent_feat.shape}", file=sys.stderr)
            raise
        x = self.ln_fused(x)
        x = self.input_proj(x)
        for i in range(self.num_blocks):
            x = x + getattr(self, f"mlp{i+1}")(getattr(self, f"ln{i+1}")(x))
        x = self.ln_out(x)
        return self.head_cls(x), torch.zeros_like(self.head_cls(x))
