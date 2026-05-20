import torch
from typing import List, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass
import json
import os
import sys
import random
import threading
import time
from tqdm import tqdm

from r2r.utils.metrics import (
    compute_logu,
    compute_entropy,
    compute_variance,
    compute_topk_entropy,
    compute_js_divergence,
    compute_sparse_topk_js_divergence,
    extract_topk_logits,
)
from r2r.models.router import load_model
from r2r.utils.dataclass import ModelOutputs

_entropy_lookahead_score_log_lock = threading.Lock()


def append_entropy_lookahead_score_log(path: Optional[str], record: dict) -> None:
    """Append one JSON line per triggered entropy-lookahead judgment (thread-safe)."""
    if not path:
        return
    record.setdefault("ts", time.time())
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with _entropy_lookahead_score_log_lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[EntropyLookahead] score log write failed ({path}): {e}")

@dataclass
class SwitchingState:
    """State information for model switching decisions"""
    last_model: str  # 'quick' or 'reference'
    consecutive_simple_tokens: int = 0
    aleatoric_history: List[float] = None
    critical_history: List[float] = None
    momentum: float = 0.0


@dataclass
class EntropyDriftSequenceState:
    ema_mean: float = 0.0
    n_seen: int = 0
    last_model: str = "quick"
    hold_remaining: int = 0

def _leftmost_argmax_index(vals: list) -> int:
    """Index of the maximum element; ties broken toward the smallest index."""
    if not vals:
        return 0
    m = max(vals)
    for i, v in enumerate(vals):
        if v == m:
            return i
    return 0


class ModelSwitchingStrategy:
    """Base class for model switching strategies"""
    requires_reference_evaluation = False
    requires_reference_logits = False
    reference_distribution_mode = None
    reference_topk_k = None
    reference_decision_mode = None
    async_speculative_validation = False

    def __init__(self, aleatoric_threshold: float = 2.275):
        self.aleatoric_threshold = aleatoric_threshold
        self.last_route_scores = None
        self.last_route_metric_name = None

        # self.entropy_threshold = 0.35
        # self.aleatoric_threshold = 2.250
        # self.epistemic_threshold = 0.0656562983380584
        self.state = SwitchingState(last_model='reference')
    
    def route(self, outputs: ModelOutputs) -> str:
        """Route to appropriate model and update state
        
        Args:
            outputs: Model outputs containing logits for uncertainty computation
            
        Returns:
            str: 'quick' or 'reference' indicating which model to use
        """
        raise NotImplementedError

    def get_reference_candidates(self, outputs: ModelOutputs) -> torch.Tensor:
        """Return the subset that needs reference-model evaluation."""
        return self.route(outputs)

    def get_reference_distribution_request(self) -> dict:
        """Describe what distribution payload should be returned by the reference model."""
        return {
            "mode": self.reference_distribution_mode,
            "topk_k": self.reference_topk_k,
            "decision_mode": self.reference_decision_mode,
            "js_threshold": (
                getattr(self, "js_threshold", None)
                if self.reference_decision_mode is not None
                else None
            ),
        }


class EntropyVarianceJsSwitching(ModelSwitchingStrategy):
    """Entropy gate then JS vs LLM (RPC path in SLMServer). Without RPC: entropy-only routing."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        variance_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        rpc_timeout_s: float = 120.0,
        score_log_path: Optional[str] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        _ = (
            model_path,
            variance_threshold,
            device,
            dtype,
            override_init_args,
            kwargs,
        )
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else 0.45
        self.js_threshold = float(js_threshold) if js_threshold is not None else 0.1
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.score_log_path = score_log_path
        print(
            f"EntropyVarianceJsSwitching: entropy_threshold={self.entropy_threshold}, "
            f"js_threshold={self.js_threshold}, rpc_timeout_s={self.rpc_timeout_s}"
            + (f", score_log_path={self.score_log_path}" if self.score_log_path else "")
        )

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        entropy = compute_entropy(next_token_logits)
        if isinstance(entropy, float):
            entropy = torch.tensor([entropy], device=next_token_logits.device)
        model_choices = (entropy > self.entropy_threshold).to(torch.int)
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyLookaheadSwitching(ModelSwitchingStrategy):
    """Entropy-lookahead routing (full logic in SLMServer)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        lookahead_steps: int = 3,
        score_threshold: float = 2.0,
        rpc_timeout_s: float = 120.0,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        score_log_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else 0.45
        self.lookahead_steps = int(lookahead_steps)
        self.score_threshold = float(score_threshold)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.score_log_path = score_log_path
        print(
            f"EntropyLookaheadSwitching: entropy_threshold={self.entropy_threshold}, "
            f"lookahead_steps={self.lookahead_steps}, score_threshold={self.score_threshold}"
            + (f", score_log_path={self.score_log_path}" if self.score_log_path else "")
        )

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        for i in range(batch_size):
            entropy = compute_entropy(next_token_logits[i : i + 1])
            model_choices[i] = 0 if entropy < self.entropy_threshold else 1
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class SlidingWindowEntropySwitching(ModelSwitchingStrategy):
    """Sliding-window mean entropy (stateful logic in SLMServer)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        window_size: int = 5,
        entropy_sum_threshold: float = 10.0,
        entropy_mean_threshold: Optional[float] = None,
        intervention_mode: str = "replace_first",
        truncate_on_llm_trigger: bool = False,
        rpc_timeout_s: float = 120.0,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        score_log_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        _ = kwargs
        self.window_size = max(1, int(window_size))
        self.entropy_sum_threshold = float(entropy_sum_threshold)
        if entropy_mean_threshold is not None:
            self.entropy_mean_threshold = float(entropy_mean_threshold)
        else:
            self.entropy_mean_threshold = float(entropy_sum_threshold) / float(self.window_size)
        mode = str(intervention_mode).lower()
        aliases = {
            "1": "replace_first",
            "first": "replace_first",
            "replace_first": "replace_first",
            "2": "replace_window",
            "window": "replace_window",
            "replace_window": "replace_window",
            "3": "replace_full_window",
            "full_window": "replace_full_window",
            "replace_full_window": "replace_full_window",
        }
        if mode not in aliases:
            raise ValueError(
                "intervention_mode must be one of "
                "['replace_first', 'replace_window', 'replace_full_window', '1', '2', '3']"
            )
        self.intervention_mode = aliases[mode]
        self.truncate_on_llm_trigger = bool(truncate_on_llm_trigger)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.score_log_path = score_log_path
        print(
            "SlidingWindowEntropySwitching: "
            f"window_size={self.window_size}, "
            f"entropy_mean_threshold={self.entropy_mean_threshold} "
            f"(YAML sum field={self.entropy_sum_threshold}), "
            f"intervention_mode={self.intervention_mode}, "
            f"truncate_on_llm_trigger={self.truncate_on_llm_trigger}"
            + (f", score_log_path={self.score_log_path}" if self.score_log_path else "")
        )

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=outputs.logits.device)
        self.state.last_model = "quick"
        return model_choices


class SlidingWindowEntropyJsSwitching(SlidingWindowEntropySwitching):
    """Sliding-window entropy gate then JS (RPC in SLMServer)."""

    def __init__(
        self,
        js_threshold: float = 0.05,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.js_threshold = float(js_threshold)
        print(
            "SlidingWindowEntropyJsSwitching: "
            f"js_threshold={self.js_threshold}, "
            f"window_size={self.window_size}, "
            f"entropy_mean_threshold={self.entropy_mean_threshold}"
        )


class ImmediateSwitching(ModelSwitchingStrategy):
    
    def __init__(
        self, model_path: Optional[str] = None,
        aleatoric_threshold: Optional[float] = None,
        epistemic_threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = None,
        device: str = "cuda", dtype=torch.float32,
        override_init_args: Optional[dict] = None, **kwargs
    ):
        """Immediate switching based on aleatoric, epistemic, and entropy thresholds.
        Routes to reference model only when ALL provided thresholds are exceeded."""
        super().__init__()  # Initialize parent class to set up self.state
        
        self.aleatoric_threshold = float(aleatoric_threshold) if aleatoric_threshold is not None else None
        self.epistemic_threshold = float(epistemic_threshold) if epistemic_threshold is not None else None
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else None

        active = {k: v for k, v in {
            "aleatoric": self.aleatoric_threshold,
            "epistemic": self.epistemic_threshold,
            "entropy": self.entropy_threshold,
        }.items() if v is not None}
        if not active:
            self.aleatoric_threshold = 2.275
            active["aleatoric"] = self.aleatoric_threshold

        print(f"ImmediateSwitching thresholds: {active}")
    
    def route(self, outputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Routes to reference (1) only when ALL active thresholds are exceeded.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        
        for i in range(batch_size):
            aleatoric, epistemic = compute_logu(next_token_logits[i:i+1])
            entropy = compute_entropy(next_token_logits[i:i+1])
            
            use_reference = True
            if self.aleatoric_threshold is not None and aleatoric < self.aleatoric_threshold:
                use_reference = False
            if self.epistemic_threshold is not None and epistemic < self.epistemic_threshold:
                use_reference = False
            if self.entropy_threshold is not None and entropy < self.entropy_threshold:
                use_reference = False
            
            model_choices[i] = 1 if use_reference else 0
        
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices

class EntropySwitching(ModelSwitchingStrategy):
    
    def __init__(
        self, model_path: Optional[str] = None, entropy_threshold: Optional[float] = None, device: str = "cuda", dtype=torch.float32, override_init_args: Optional[dict] = None, **kwargs
    ):
        """Simple immediate switching based on entropy threshold"""
        super().__init__()  # Initialize parent class to set up self.state
        
        # Use entropy_threshold if provided, otherwise default
        if entropy_threshold is not None:
            self.threshold = float(entropy_threshold)
        else:
            self.threshold = 0.45  # Default entropy threshold
        print(f"Using entropy threshold: {self.threshold}")
    
    def route(self, outputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        
        # Compute entropy for each sample in the batch
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        
        entropy_values = compute_entropy(next_token_logits)
        self.last_route_scores = entropy_values.detach().cpu()
        self.last_route_metric_name = "entropy"

        for i in range(batch_size):
            entropy = entropy_values[i]
            # 0 = quick (low entropy), 1 = reference (high entropy)
            model_choices[i] = 0 if entropy < self.threshold else 1
        
        # Update state based on batch decisions
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyDriftSwitching(ModelSwitchingStrategy):
    """Switch based on entropy drift from a running per-sequence baseline.

    This router tracks an EMA per sequence, resets state when a new sample
    starts, and can apply hysteresis / minimum-hold logic to reduce token-level
    flicker. A confidence filter can further block wasteful escalations when
    the quick model is already highly certain.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        alpha: float = 0.1,
        bias: float = 0.0,
        tau: float = 0.5,
        warmup_steps: int = 10,
        random_seed: Optional[int] = 42,
        hysteresis: float = 0.0,
        hold_tokens: int = 0,
        max_confident_prob: Optional[float] = None,
        min_entropy: Optional[float] = None,
        stochastic: bool = True,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.bias = float(bias)
        self.tau = float(tau)
        self.warmup_steps = int(warmup_steps)
        self.hysteresis = float(hysteresis)
        self.hold_tokens = max(0, int(hold_tokens))
        self.max_confident_prob = (
            float(max_confident_prob) if max_confident_prob is not None else None
        )
        self.min_entropy = float(min_entropy) if min_entropy is not None else None
        self.stochastic = bool(stochastic)
        self.threshold = self.bias
        self._rng = random.Random(random_seed)

        self._ema_mean: float = 0.0
        self._n_seen: int = 0
        self._sequence_states = {}
        self._global_sequence_key = "__entropy_drift_global__"

        print(
            f"EntropyDriftSwitching: alpha={self.alpha}, bias={self.bias}, "
            f"tau={self.tau}, warmup={self.warmup_steps}, hysteresis={self.hysteresis}, "
            f"hold_tokens={self.hold_tokens}, max_confident_prob={self.max_confident_prob}, "
            f"min_entropy={self.min_entropy}, stochastic={self.stochastic}, seed={random_seed}"
        )

    def reset(self):
        self._sequence_states.clear()
        self._ema_mean = 0.0
        self._n_seen = 0
        self.state.last_model = "quick"

    def _resolve_sequence_key(self, outputs, index: int):
        sequence_ids = getattr(outputs, "sequence_ids", None)
        if sequence_ids is None or index >= len(sequence_ids):
            return self._global_sequence_key
        sequence_id = sequence_ids[index]
        return sequence_id if sequence_id is not None else self._global_sequence_key

    def _resolve_position(self, outputs, index: int) -> Optional[int]:
        positions = getattr(outputs, "positions", None)
        if positions is None or index >= len(positions):
            return None
        position = positions[index]
        return int(position) if position is not None else None

    def _get_sequence_state(self, key, position: Optional[int]) -> EntropyDriftSequenceState:
        if position == 0 or key not in self._sequence_states:
            self._sequence_states[key] = EntropyDriftSequenceState()
        return self._sequence_states[key]

    def _passes_secondary_filter(self, logits_row: torch.Tensor, entropy_value: float) -> bool:
        if self.min_entropy is not None and entropy_value < self.min_entropy:
            return False
        if self.max_confident_prob is not None:
            top1_prob = float(torch.softmax(logits_row, dim=-1).max().item())
            if top1_prob > self.max_confident_prob:
                return False
        return True

    def _compute_switch_probability(self, drift: float) -> float:
        return 1.0 / (1.0 + np.exp(-(drift - self.bias) / max(self.tau, 1e-8)))

    def _update_state_ema(self, state: EntropyDriftSequenceState, entropy_value: float):
        if state.n_seen == 0:
            state.ema_mean = entropy_value
        else:
            state.ema_mean = self.alpha * entropy_value + (1.0 - self.alpha) * state.ema_mean
        state.n_seen += 1
        self._ema_mean = state.ema_mean
        self._n_seen = state.n_seen

    def route(self, outputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)

        entropy_values = compute_entropy(next_token_logits)
        drift_values = torch.zeros(batch_size, dtype=torch.float32)

        for i in range(batch_size):
            h_t = float(entropy_values[i].item())
            sequence_key = self._resolve_sequence_key(outputs, i)
            position = self._resolve_position(outputs, i)
            state = self._get_sequence_state(sequence_key, position)
            logits_row = next_token_logits[i]

            if state.n_seen < self.warmup_steps:
                # Warmup: always stay on SLM, just accumulate EMA
                drift_values[i] = 0.0
                model_choices[i] = 0
            else:
                drift = h_t - state.ema_mean
                drift_values[i] = drift
                passes_filter = self._passes_secondary_filter(logits_row, h_t)
                enter_threshold = self.bias
                exit_threshold = self.bias - self.hysteresis

                if self.stochastic:
                    should_sample = passes_filter and drift >= exit_threshold
                    switch_prob = self._compute_switch_probability(drift) if should_sample else 0.0
                    if state.last_model == "reference" and passes_filter and drift >= exit_threshold:
                        choose_reference = True
                    else:
                        choose_reference = should_sample and (self._rng.random() < switch_prob)
                else:
                    if state.last_model == "reference":
                        choose_reference = passes_filter and (
                            state.hold_remaining > 0 or drift >= exit_threshold
                        )
                    else:
                        choose_reference = passes_filter and (drift >= enter_threshold)

                model_choices[i] = 1 if choose_reference else 0

            if int(model_choices[i].item()) == 1:
                if state.last_model != "reference":
                    state.hold_remaining = self.hold_tokens
                elif state.hold_remaining > 0:
                    state.hold_remaining -= 1
                state.last_model = "reference"
            else:
                state.hold_remaining = 0
                state.last_model = "quick"

            self._update_state_ema(state, h_t)

        self.last_route_scores = drift_values.detach().cpu()
        self.last_route_metric_name = "entropy_drift"
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyTopKSwitching(ModelSwitchingStrategy):

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        entropy_topk_k: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        """Immediate switching based on entropy computed over the top-k support."""
        super().__init__()
        self.threshold = float(entropy_threshold) if entropy_threshold is not None else 0.45
        self.entropy_topk_k = int(entropy_topk_k) if entropy_topk_k is not None else 100
        if self.entropy_topk_k <= 0:
            raise ValueError(f"entropy_topk_k must be positive, got {self.entropy_topk_k}")
        print(
            f"Using entropy_topk threshold: {self.threshold}, top-k: {self.entropy_topk_k}"
        )

    def route(self, outputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)

        entropy_values = compute_topk_entropy(next_token_logits, self.entropy_topk_k)
        self.last_route_scores = entropy_values.detach().cpu()
        self.last_route_metric_name = f"entropy_topk_{self.entropy_topk_k}"

        for i in range(batch_size):
            entropy = entropy_values[i]
            model_choices[i] = 0 if entropy < self.threshold else 1

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices

class JSSwitching(ModelSwitchingStrategy):
    """Route every token to reference evaluation and decide with full-vocab JS."""

    requires_reference_evaluation = True
    requires_reference_logits = True
    reference_distribution_mode = "full"
    trace_router_score_from_js = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        js_threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.js_threshold = float(js_threshold) if js_threshold is not None else 0.2
        print(f"Using js threshold: {self.js_threshold}")

    def get_reference_candidates(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        self.last_route_scores = None
        self.last_route_metric_name = "js"
        return torch.ones(batch_size, dtype=torch.int, device=outputs.logits.device)

    def route_with_reference_logits(
        self, quick_logits: torch.Tensor, reference_logits: torch.Tensor
    ) -> torch.Tensor:
        js_divergence = compute_js_divergence(quick_logits, reference_logits)
        self.last_route_scores = js_divergence.detach().cpu()
        self.last_route_metric_name = "js"
        return (js_divergence >= self.js_threshold).to(torch.int)

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        if outputs.reference_logits is None:
            raise ValueError(
                "JSSwitching.route requires outputs.reference_logits for final routing"
            )

        quick_logits = outputs.logits[:, -1, :]
        reference_logits = outputs.reference_logits[:, -1, :]
        model_choices = self.route_with_reference_logits(
            quick_logits, reference_logits
        ).to(device=quick_logits.device, dtype=torch.int)
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices

class JSLLMSwitching(JSSwitching):
    """Route every token to LLM-side full-vocab JS final decision."""

    reference_decision_mode = "llm_full_js"

    def __init__(
        self,
        model_path: Optional[str] = None,
        js_threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            js_threshold=js_threshold,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        print("Using js_llm with LLM-side full-vocab JS final decision")

class JSTopKSparseSwitching(JSSwitching):
    """Route every token using sparse top-k JS divergence."""

    requires_reference_logits = False
    reference_distribution_mode = "topk"

    def __init__(
        self,
        model_path: Optional[str] = None,
        js_threshold: Optional[float] = None,
        js_topk: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            js_threshold=js_threshold,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        self.reference_topk_k = int(js_topk) if js_topk is not None else 64
        print(f"Using js_topk_sparse threshold: {self.js_threshold}, top-k: {self.reference_topk_k}")

    def route_with_reference_topk(
        self,
        quick_topk_indices: torch.Tensor,
        quick_topk_logits: torch.Tensor,
        reference_topk_indices: torch.Tensor,
        reference_topk_logits: torch.Tensor,
    ) -> torch.Tensor:
        js_divergence = compute_sparse_topk_js_divergence(
            logits_p=quick_topk_logits,
            indices_p=quick_topk_indices,
            logits_q=reference_topk_logits,
            indices_q=reference_topk_indices,
        )
        self.last_route_scores = js_divergence.detach().cpu()
        self.last_route_metric_name = "js"
        return (js_divergence >= self.js_threshold).to(torch.int)

    def extract_quick_topk(
        self, quick_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quick_topk_logits, quick_topk_indices = extract_topk_logits(
            quick_logits, self.reference_topk_k
        )
        return quick_topk_indices, quick_topk_logits

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        if outputs.reference_topk_indices is None or outputs.reference_topk_logits is None:
            raise ValueError(
                "JSTopKSparseSwitching.route requires sparse reference top-k outputs"
            )

        quick_logits = outputs.logits[:, -1, :]
        quick_topk_indices, quick_topk_logits = self.extract_quick_topk(quick_logits)
        model_choices = self.route_with_reference_topk(
            quick_topk_indices=quick_topk_indices,
            quick_topk_logits=quick_topk_logits,
            reference_topk_indices=outputs.reference_topk_indices,
            reference_topk_logits=outputs.reference_topk_logits,
        ).to(device=quick_logits.device, dtype=torch.int)
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices

class JSTopKLLMSwitching(JSTopKSparseSwitching):
    """Route every token to LLM-side sparse top-k JS final decision."""

    reference_decision_mode = "llm_sparse_js"

    def __init__(
        self,
        model_path: Optional[str] = None,
        js_threshold: Optional[float] = None,
        js_topk: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            js_threshold=js_threshold,
            js_topk=js_topk,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        print("Using js_topk_llm with LLM-side sparse top-k JS final decision")

class EntropyJSSwitching(ModelSwitchingStrategy):
    """Route high-entropy tokens to reference evaluation, then decide with JS divergence."""

    requires_reference_evaluation = True
    requires_reference_logits = True
    reference_distribution_mode = "full"

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_threshold = (
            float(entropy_threshold) if entropy_threshold is not None else 0.45
        )
        self.js_threshold = float(js_threshold) if js_threshold is not None else 0.2
        print(
            f"Using entropy_js thresholds: entropy={self.entropy_threshold}, js={self.js_threshold}"
        )

    def get_reference_candidates(self, outputs: ModelOutputs) -> torch.Tensor:
        next_token_logits = outputs.logits[:, -1, :]
        entropy = compute_entropy(next_token_logits)
        self.last_route_scores = entropy.detach().cpu()
        self.last_route_metric_name = "entropy"
        return (entropy >= self.entropy_threshold).to(torch.int)

    def route_with_reference_logits(
        self, quick_logits: torch.Tensor, reference_logits: torch.Tensor
    ) -> torch.Tensor:
        js_divergence = compute_js_divergence(quick_logits, reference_logits)
        return (js_divergence >= self.js_threshold).to(torch.int)

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        if outputs.reference_logits is None:
            raise ValueError(
                "EntropyJSSwitching.route requires outputs.reference_logits for final routing"
            )

        batch_size = outputs.logits.shape[0]
        quick_logits = outputs.logits[:, -1, :]
        reference_logits = outputs.reference_logits[:, -1, :]

        candidate_mask = self.get_reference_candidates(outputs).bool()
        model_choices = torch.zeros(
            batch_size, dtype=torch.int, device=quick_logits.device
        )

        if candidate_mask.any():
            candidate_choices = self.route_with_reference_logits(
                quick_logits[candidate_mask], reference_logits[candidate_mask]
            ).to(device=quick_logits.device, dtype=torch.int)
            model_choices[candidate_mask] = candidate_choices

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices

class EntropyJSLLMSwitching(EntropyJSSwitching):
    """Route high-entropy tokens to LLM-side full-logits JS final decision."""

    reference_decision_mode = "llm_full_js"

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            entropy_threshold=entropy_threshold,
            js_threshold=js_threshold,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        print("Using entropy_js_llm with LLM-side full JS final decision")

class EntropyJSTopKSparseSwitching(EntropyJSSwitching):
    """Route high-entropy tokens using sparse top-k reference distributions."""

    requires_reference_logits = False
    reference_distribution_mode = "topk"

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        js_topk: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            entropy_threshold=entropy_threshold,
            js_threshold=js_threshold,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        self.reference_topk_k = int(js_topk) if js_topk is not None else 64
        print(
            f"Using entropy_js_topk_sparse top-k: {self.reference_topk_k}"
        )

    def route_with_reference_topk(
        self,
        quick_topk_indices: torch.Tensor,
        quick_topk_logits: torch.Tensor,
        reference_topk_indices: torch.Tensor,
        reference_topk_logits: torch.Tensor,
    ) -> torch.Tensor:
        js_divergence = compute_sparse_topk_js_divergence(
            logits_p=quick_topk_logits,
            indices_p=quick_topk_indices,
            logits_q=reference_topk_logits,
            indices_q=reference_topk_indices,
        )
        return (js_divergence >= self.js_threshold).to(torch.int)

    def extract_quick_topk(
        self, quick_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quick_topk_logits, quick_topk_indices = extract_topk_logits(
            quick_logits, self.reference_topk_k
        )
        return quick_topk_indices, quick_topk_logits

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        if outputs.reference_topk_indices is None or outputs.reference_topk_logits is None:
            raise ValueError(
                "EntropyJSTopKSparseSwitching.route requires sparse reference top-k outputs"
            )

        batch_size = outputs.logits.shape[0]
        quick_logits = outputs.logits[:, -1, :]
        candidate_mask = self.get_reference_candidates(outputs).bool()
        model_choices = torch.zeros(
            batch_size, dtype=torch.int, device=quick_logits.device
        )

        if candidate_mask.any():
            quick_topk_indices, quick_topk_logits = self.extract_quick_topk(
                quick_logits[candidate_mask]
            )
            candidate_choices = self.route_with_reference_topk(
                quick_topk_indices=quick_topk_indices,
                quick_topk_logits=quick_topk_logits,
                reference_topk_indices=outputs.reference_topk_indices[candidate_mask],
                reference_topk_logits=outputs.reference_topk_logits[candidate_mask],
            ).to(device=quick_logits.device, dtype=torch.int)
            model_choices[candidate_mask] = candidate_choices

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyJSTopKLLMSwitching(EntropyJSTopKSparseSwitching):
    """Route high-entropy tokens to LLM-side sparse top-k JS final decision."""

    reference_decision_mode = "llm_sparse_js"

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        js_topk: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            entropy_threshold=entropy_threshold,
            js_threshold=js_threshold,
            js_topk=js_topk,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        print("Using entropy_js_topk_llm with LLM-side sparse JS final decision")


class EntropyJSTopKAsyncSwitching(EntropyJSTopKSparseSwitching):
    """Route high-entropy tokens to async speculative sparse top-k JS validation."""

    reference_decision_mode = "async_llm_sparse_js"
    async_speculative_validation = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        entropy_threshold: Optional[float] = None,
        js_threshold: Optional[float] = None,
        js_topk: Optional[int] = None,
        device: str = "cuda",
        dtype=torch.float32,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            entropy_threshold=entropy_threshold,
            js_threshold=js_threshold,
            js_topk=js_topk,
            device=device,
            dtype=dtype,
            override_init_args=override_init_args,
            **kwargs,
        )
        print(
            "Using entropy_js_topk_async with async speculative sparse JS validation"
        )

class MomentumSwitching(ModelSwitchingStrategy):
    """Momentum-based switching with asymmetric behavior"""
    def __init__(self, aleatoric_threshold: float = 2.275, 
                 momentum_factor: float = 0.7,
                 quick_to_ref_threshold: float = 0.3,
                 ref_to_quick_threshold: float = 0.7):
        super().__init__(aleatoric_threshold)
        self.momentum_factor = momentum_factor
        self.quick_to_ref_threshold = quick_to_ref_threshold
        self.ref_to_quick_threshold = ref_to_quick_threshold
        self.state.momentum = 0.0
    
    def route(self, outputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        
        for i in range(batch_size):
            aleatoric_uncertainty, _ = compute_logu(next_token_logits[i:i+1])
            is_simple = aleatoric_uncertainty < self.aleatoric_threshold
            
            # Update momentum based on current token
            self.state.momentum = (self.momentum_factor * self.state.momentum + 
                                 (1 - self.momentum_factor) * (1.0 if is_simple else 0.0))
            
            if self.state.last_model == 'quick':
                use_reference = self.state.momentum <= self.quick_to_ref_threshold
            else:
                use_reference = self.state.momentum <= self.ref_to_quick_threshold
            
            model_choices[i] = 1 if use_reference else 0
            self.state.last_model = 'reference' if use_reference else 'quick'
        
        return model_choices

class SingleRollingWindowSwitching(ModelSwitchingStrategy):
    """Rolling window-based switching with asymmetric behavior"""
    def __init__(self, aleatoric_threshold: float = 2.275,
                 epistemic_threshold: float = 0.055,
                 entropy_threshold: float = 0.02,
                 window_size: int = 3,
                 required_simple_ratio: float = 1.0):
        super().__init__(aleatoric_threshold)
        self.epistemic_threshold = epistemic_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        self.state.aleatoric_history = deque(maxlen=window_size)
    
    def route(self, outputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        
        for i in range(batch_size):
            aleatoric_uncertainty, epistemic_uncertainty = compute_logu(next_token_logits[i:i+1])
            entropy = compute_entropy(next_token_logits[i:i+1])
            
            is_simple = aleatoric_uncertainty < self.aleatoric_threshold or entropy < self.entropy_threshold
            
            if self.state.last_model == 'quick':
                use_reference = not is_simple
                if use_reference:
                    self.state.aleatoric_history.clear()
                    self.state.aleatoric_history.append(aleatoric_uncertainty)
            else:
                # Reference model: record history and check average uncertainty
                self.state.aleatoric_history.append(aleatoric_uncertainty)
                if len(self.state.aleatoric_history) == 0:
                    use_reference = True
                else:
                    avg_uncertainty = sum(self.state.aleatoric_history) / len(self.state.aleatoric_history)
                    use_reference = avg_uncertainty >= self.aleatoric_threshold
                
                # Clear history when switching back to quick
                if not use_reference:
                    self.state.aleatoric_history.clear()
            
            model_choices[i] = 1 if use_reference else 0
            self.state.last_model = 'reference' if use_reference else 'quick'
        
        return model_choices

class DuoRollingWindowSwitching(ModelSwitchingStrategy):
    """Rolling window-based switching with separate windows for quick and reference models"""
    def __init__(self, aleatoric_threshold: float = 2.275,
                 window_size: int = 3,
                 required_simple_ratio: float = 1.0):
        super().__init__(aleatoric_threshold)
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        # Initialize separate windows for each model
        self.quick_history = deque(maxlen=window_size)
        self.reference_history = deque(maxlen=window_size)

    def route(self, outputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        
        for i in range(batch_size):
            aleatoric_uncertainty, _ = compute_logu(next_token_logits[i:i+1])
            is_simple = aleatoric_uncertainty < self.aleatoric_threshold

            # Always record uncertainty in the current model's window
            if self.state.last_model == 'quick':
                self.quick_history.append(aleatoric_uncertainty)

                # If current token is complex, switch to reference immediately
                if not is_simple:
                    self.reference_history.clear()  # Clear reference history when switching
                    use_reference = True
                else:
                    # Stay with quick model
                    use_reference = False
            else:  # In reference model
                self.reference_history.append(aleatoric_uncertainty)

                # Only consider switching to quick if we have enough history
                if len(self.reference_history) > 0:
                    # Check average uncertainty in reference window
                    ref_avg = sum(self.reference_history) / len(self.reference_history)
                    if ref_avg < self.aleatoric_threshold:
                        self.quick_history.clear()  # Clear quick history when switching
                        use_reference = False
                    else:
                        use_reference = True
                else:
                    use_reference = True

            model_choices[i] = 1 if use_reference else 0
            self.state.last_model = 'reference' if use_reference else 'quick'
        
        return model_choices


class NeuralSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier"""

    def __init__(
        self, model_path, threshold: Optional[float] = None, device: str = "cuda", dtype=torch.float32, use_cuda_graph=True, override_init_args: Optional[dict] = None
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device, override_init_args=override_init_args)

        # Use saved optimal threshold if available in common_args
        if threshold is None:
            self.threshold = float(model_config["common_args"]["threshold"])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold)
            print(f"Using provided threshold: {self.threshold}")

        # Extract model parameters
        self.init_args = model_config["init_args"]
        self.common_args = model_config["common_args"]
        self.logits_size = self.init_args.get("logits_size", 0)

        # Determine input type from common_args
        self.input_type = self.common_args["input_type"]
        if not isinstance(self.input_type, list):
            self.input_type = [self.input_type]
        self.model_type = model_config["model_type"]

        print(f"Using input types: {self.input_type}")

        # Set model to evaluation mode
        self.model.eval()

        self.use_cuda_graph = use_cuda_graph
        if self.use_cuda_graph:
            self.capture_bs = list(range(16, 0, -1))  # Capture for batch sizes 16 to 1
            self.max_bs = max(self.capture_bs)
            vocab_size = self.model.token_embeddings.num_embeddings
            hidden_states_size = model_config["init_args"]["hidden_states_size"]
            self.model_outputs_buffer = {
                "logits": torch.zeros((self.max_bs, vocab_size), device=self.device, dtype=torch.float32),
                "hidden_states": torch.zeros((self.max_bs, hidden_states_size), device=self.device, dtype=torch.float32),
                "token": torch.zeros((self.max_bs,), device=self.device, dtype=torch.long),
            }
            self.model_choices_buffer = torch.zeros((self.max_bs,), device=self.device, dtype=torch.int)
            with torch.no_grad():
                self.capture()
    
    def capture(self):
        if not self.use_cuda_graph:
            return
        # Capture CUDA graphs for different batch sizes
        self.cuda_graphs = {}
        for bs in tqdm(self.capture_bs):
            # Warm-up
            self.capture_one_batch_size(bs)

            # Capture graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.capture_one_batch_size(bs)
            self.cuda_graphs[bs] = g

    def capture_one_batch_size(self, batch_size: int):
        if "logits" in self.input_type:
            # If the model has a logits_size parameter, use it to get top-k logits
            if self.logits_size > 0:
                top_logits, _ = torch.topk(
                    self.model_outputs_buffer["logits"][:batch_size], k=self.logits_size, dim=-1
                )
            input_logits = top_logits.to(device=self.device, dtype=torch.float32) if self.logits_size > 0 else self.model_outputs_buffer["logits"][:batch_size].to(device=self.device, dtype=torch.float32)
        else:
            input_logits = None
        
        # Process hidden states if needed
        if "hidden_states" in self.input_type:
            input_hidden_states = self.model_outputs_buffer["hidden_states"][:batch_size].to(
                device=self.device, dtype=torch.float32
            )
        else:
            input_hidden_states = None
        
        # Process token IDs if needed
        if "token" in self.input_type:
            input_token = self.model_outputs_buffer["token"][:batch_size].to(
                device=self.device, dtype=torch.long
            )
        else:
            input_token = None
        
        model_output = self.model(logits=input_logits, hidden_states=input_hidden_states, token=input_token)
            
        # Handle different output formats (single output or multi-class)
        if model_output.shape[1] == 1:
            critical_prob = torch.sigmoid(model_output).squeeze(-1)  # [batch_size]
            # Convert probabilities to binary decisions (0 = quick, 1 = reference)
            self.model_choices_buffer[:batch_size].copy_((critical_prob >= self.threshold).to(torch.int))
        else:
            # For multi-class output, consider class 2 as critical (divergent) cases
            # Classes: 0=match, 1=mismatch, 2=divergent
            probabilities = torch.softmax(model_output, dim=1)  # [batch_size, num_classes]
            critical_prob = probabilities[:, 2]  # Get probability of class 2 (divergent)
            self.model_choices_buffer[:batch_size].copy_((critical_prob >= self.threshold).to(torch.int))

    def replay(self, outputs: ModelOutputs):
        batch_size = outputs.logits.shape[0]
        # Prepare inputs based on input_type
        if "logits" in self.input_type:
            self.model_outputs_buffer["logits"][:batch_size].copy_(outputs.logits[:, -1, :].to(device=self.device, dtype=torch.float32))
        
        if "hidden_states" in self.input_type:
            self.model_outputs_buffer["hidden_states"][:batch_size].copy_(outputs.hidden_states[-1][:, -1, :].to(device=self.device, dtype=torch.float32))
        
        if "token" in self.input_type:
            self.model_outputs_buffer["token"][:batch_size].copy_(outputs.token[:, -1].to(device=self.device, dtype=torch.long))
        
        # Replay the captured CUDA graph
        g = self.cuda_graphs[batch_size]
        g.replay()

        return self.model_choices_buffer[:batch_size]

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        
        batch_size = outputs.logits.shape[0]

        with torch.no_grad():
            # Get batch size from outputs
            if self.use_cuda_graph and batch_size in self.capture_bs:
                model_choices = self.replay(outputs)
                # For tracking state, we'll keep the most recent decision for each input
                self.state.last_model = "reference" if model_choices.any().item() else "quick"
                return model_choices

            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # Prepare inputs based on input_type
            inputs = {}
            
            # Process logits if needed
            if "logits" in self.input_type:
                # If the model has a logits_size parameter, use it to get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(
                        next_token_logits, k=self.logits_size, dim=-1
                    )
                    inputs["logits"] = top_logits.to(
                        device=self.device, dtype=torch.float32
                    )  # [batch_size, topk]
                else:
                    # If no logits_size, use all logits
                    inputs["logits"] = next_token_logits.to(
                        device=self.device, dtype=torch.float32
                    )
            
            # Process hidden states if needed
            if "hidden_states" in self.input_type:
                inputs["hidden_states"] = outputs.hidden_states[-1][:, -1, :].to(
                    device=self.device, dtype=torch.float32
                )
            
            # Process token IDs if needed
            if "token" in self.input_type:
                inputs["token"] = outputs.token[:, -1].to(
                    device=self.device, dtype=torch.long
                )

            # Forward pass through the model with appropriate inputs
            model_output = self.model(**inputs)
            
            # Handle different output formats (single output or multi-class)
            if model_output.shape[1] == 1:
                critical_prob = torch.sigmoid(model_output).squeeze(-1)  # [batch_size]
                self.last_route_scores = critical_prob.detach().cpu()
                self.last_route_metric_name = "critical_prob"
                # Convert probabilities to binary decisions (0 = quick, 1 = reference)
                model_choices = (critical_prob >= self.threshold).to(torch.int)
            else:
                # For multi-class output, consider class 2 as critical (divergent) cases
                # Classes: 0=match, 1=mismatch, 2=divergent
                probabilities = torch.softmax(model_output, dim=1)  # [batch_size, num_classes]
                critical_prob = probabilities[:, 2]  # Get probability of class 2 (divergent)
                self.last_route_scores = critical_prob.detach().cpu()
                self.last_route_metric_name = "critical_prob"
                model_choices = (critical_prob >= self.threshold).to(torch.int)
            
            # For tracking state, we'll keep the most recent decision for each input
            self.state.last_model = "reference" if model_choices.any().item() else "quick"
            
            return model_choices


class EntropyNeuralSwitching(ModelSwitchingStrategy):
    """Two-stage hybrid routing on the quick model's last-token distribution.

    1) **Entropy gate (same semantics as ``EntropySwitching``)**:
       if normalized entropy >= ``entropy_threshold`` → route to the reference (LLM).
    2) **Otherwise** run the **neural router** (same as ``NeuralSwitching``):
       if neural critical probability >= neural ``threshold`` → LLM, else stay on SLM.

    Low-entropy positions defer to the trained classifier instead of staying on SLM
    unconditionally.
    """

    def __init__(
        self,
        model_path: str,
        entropy_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        use_cuda_graph: bool = True,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_threshold = (
            float(entropy_threshold) if entropy_threshold is not None else 0.45
        )
        self.neural = NeuralSwitching(
            model_path=model_path,
            threshold=threshold,
            device=device,
            dtype=dtype,
            use_cuda_graph=use_cuda_graph,
            override_init_args=override_init_args,
        )
        print(
            f"EntropyNeuralSwitching: entropy_threshold={self.entropy_threshold} "
            f"(entropy >= → LLM), else neural threshold={self.neural.threshold}"
        )

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        entropy_values = compute_entropy(next_token_logits)
        self.last_route_scores = entropy_values.detach().cpu()
        self.last_route_metric_name = "entropy_then_neural_entropy"

        high_entropy = entropy_values >= self.entropy_threshold
        if bool(high_entropy.all().item()):
            model_choices = torch.ones(
                batch_size, dtype=torch.int, device=next_token_logits.device
            )
        elif bool((~high_entropy).all().item()):
            model_choices = self.neural.route(outputs)
        else:
            neural_choices = self.neural.route(outputs)
            model_choices = torch.where(
                high_entropy,
                torch.ones_like(neural_choices),
                neural_choices,
            )

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyMultitaskNeuralSwitching(ModelSwitchingStrategy):
    """Two-stage routing like ``EntropyNeuralSwitching``, but the neural stage loads
    ``train_router_multitask_js`` checkpoints (``best.pt`` with ``model_state``), not
    ``load_model`` R2R router pickles.

    1) Full-vocab Shannon entropy on the quick model's last-token logits (same as
       ``EntropyNeuralSwitching`` / ``compute_entropy``): if entropy >= threshold → LLM.
    2) Otherwise run ``RouterMultiTaskFFN4`` (top-100 logits, last hidden, last token id),
       same head as ``MultitaskJSRouterSwitching`` without its internal top-k entropy gate.
    """

    def __init__(
        self,
        model_path: str,
        entropy_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        pretrained_model_name: Optional[str] = None,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.entropy_threshold = (
            float(entropy_threshold) if entropy_threshold is not None else 0.45
        )
        from r2r.models.multitask_js_router import RouterMultiTaskFFN4, RouterBottleneck2Block, RouterBottleneck3Block, RouterBottleneck3BlockV7, RouterBottleneck3BlockV8, RouterBottleneckR2R

        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)

        saved = ckpt.get("args") or {}
        oia = override_init_args or {}
        pm = (
            pretrained_model_name
            or oia.get("pretrained_model_name")
            or saved.get("pretrained_model_name")
        )

        self.logits_size = 100
        model_type = saved.get("model_type", "ffn4")
        if model_type == "bottleneck2":
            self.model = RouterBottleneck2Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3":
            self.model = RouterBottleneck3Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v7":
            self.model = RouterBottleneck3BlockV7(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v8":
            self.model = RouterBottleneck3BlockV8(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck_r2r":
            self.model = RouterBottleneckR2R(
                dropout=float(saved.get("dropout", 0.3)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        else:
            self.model = RouterMultiTaskFFN4(
                ffn_dim=int(saved.get("ffn_dim", 768)),
                dropout=float(saved.get("dropout", 0.12)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(device)
        self.model.eval()
        self._is_v7 = isinstance(self.model, RouterBottleneck3BlockV7)
        self._is_v8 = isinstance(self.model, RouterBottleneck3BlockV8)

        if threshold is not None:
            self.threshold = float(threshold)
        elif ckpt.get("best_prob_threshold") is not None:
            self.threshold = float(ckpt["best_prob_threshold"])
        else:
            self.threshold = 0.5

        self.device = device
        self.dtype = dtype
        print(
            f"EntropyMultitaskNeuralSwitching: entropy_threshold={self.entropy_threshold} "
            f"(entropy >= → LLM), multitask_js neural threshold={self.threshold}, "
            f"model_type={model_type}"
        )

    def _multitask_neural_route(self, outputs: ModelOutputs) -> torch.Tensor:
        with torch.no_grad():
            next_token_logits = outputs.logits[:, -1, :]
            top_logits, _ = torch.topk(next_token_logits, k=self.logits_size, dim=-1)
            logits = top_logits.to(device=self.device, dtype=torch.float32)
            hidden_states = outputs.hidden_states[-1][:, -1, :].to(
                device=self.device, dtype=torch.float32
            )
            token_ids = outputs.token[:, -1].to(device=self.device, dtype=torch.long)
            if self._is_v7:
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                cls_logits, _ = self.model(logits, hidden_states, token_ids, entropy=ent)
            elif self._is_v8:
                cls_logits, div_logits = self.model(logits, hidden_states, token_ids)
            else:
                cls_logits, _ = self.model(logits, hidden_states, token_ids)
            critical_prob = torch.sigmoid(cls_logits.squeeze(-1))
            if self._is_v8:
                div_prob = torch.sigmoid(div_logits.squeeze(-1))
                critical_prob = torch.maximum(critical_prob, div_prob)
            self.last_route_scores = critical_prob.detach().cpu()
            self.last_route_metric_name = "critical_prob"
            return (critical_prob >= self.threshold).to(torch.int)

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        entropy_values = compute_entropy(next_token_logits)
        self.last_route_scores = entropy_values.detach().cpu()
        self.last_route_metric_name = "entropy_then_multitask_js_entropy"

        high_entropy = entropy_values >= self.entropy_threshold
        if bool(high_entropy.all().item()):
            model_choices = torch.ones(
                batch_size, dtype=torch.int, device=next_token_logits.device
            )
        elif bool((~high_entropy).all().item()):
            model_choices = self._multitask_neural_route(outputs)
        else:
            neural_choices = self._multitask_neural_route(outputs)
            model_choices = torch.where(
                high_entropy,
                torch.ones_like(neural_choices),
                neural_choices,
            )

        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class EntropyDriftMultitaskJSRouterSwitching(ModelSwitchingStrategy):
    """Two-stage routing: entropy drift gate + V13 multitask JS neural router.

    Stage 1 — Entropy Drift Gate:
      Tracks per-sequence EMA entropy baseline. Computes drift = current_entropy - EMA.
      Drift crossing an adaptive threshold (with stochastic / hysteresis / hold logic)
      immediately escalates to LLM.

    Stage 2 — V13 Multitask JS Router:
      Tokens that survive the drift gate are evaluated by the trained
      ``RouterBottleneckR2R`` (top-100 logits + last hidden + token id).
      If the neural score >= neural threshold → LLM, else stay on SLM.
    """

    def __init__(
        self,
        model_path: str,
        # Entropy drift params
        alpha: float = 0.1,
        bias: float = 0.0,
        tau: float = 0.5,
        warmup_steps: int = 10,
        random_seed: Optional[int] = 42,
        hysteresis: float = 0.0,
        hold_tokens: int = 0,
        max_confident_prob: Optional[float] = None,
        min_entropy: Optional[float] = None,
        stochastic: bool = True,
        entropy_topk_k: int = 100,
        entropy_threshold: Optional[float] = 0.6,
        # Neural router params
        threshold: Optional[float] = None,
        device: str = "cuda",
        dtype=torch.float32,
        pretrained_model_name: Optional[str] = None,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        # --- Drift params ---
        self.alpha = float(alpha)
        self.bias = float(bias)
        self.tau = float(tau)
        self.warmup_steps = int(warmup_steps)
        self.hysteresis = float(hysteresis)
        self.hold_tokens = max(0, int(hold_tokens))
        self.max_confident_prob = (
            float(max_confident_prob) if max_confident_prob is not None else None
        )
        self.min_entropy = float(min_entropy) if min_entropy is not None else None
        self.stochastic = bool(stochastic)
        self.entropy_topk_k = int(entropy_topk_k)
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else None
        self._rng = random.Random(random_seed)
        self._step_counter: int = 0
        self._ema_mean: float = 0.0
        self._n_seen: int = 0
        self._sequence_states: dict = {}
        self._global_sequence_key = "__entropy_drift_v13_global__"

        # --- V13 Multitask JS Router ---
        from r2r.models.multitask_js_router import (
            RouterMultiTaskFFN4,
            RouterBottleneck2Block,
            RouterBottleneck3Block,
            RouterBottleneck3BlockV7,
            RouterBottleneck3BlockV8,
            RouterBottleneckR2R,
        )

        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)

        saved = ckpt.get("args") or {}
        oia = override_init_args or {}
        pm = (
            pretrained_model_name
            or oia.get("pretrained_model_name")
            or saved.get("pretrained_model_name")
        )

        self.logits_size = 100
        model_type = saved.get("model_type", "ffn4")
        if model_type == "bottleneck2":
            self.model = RouterBottleneck2Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3":
            self.model = RouterBottleneck3Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v7":
            self.model = RouterBottleneck3BlockV7(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v8":
            self.model = RouterBottleneck3BlockV8(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck_r2r":
            self.model = RouterBottleneckR2R(
                dropout=float(saved.get("dropout", 0.3)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        else:
            self.model = RouterMultiTaskFFN4(
                ffn_dim=int(saved.get("ffn_dim", 768)),
                dropout=float(saved.get("dropout", 0.12)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(device)
        self.model.eval()
        self._is_v7 = isinstance(self.model, RouterBottleneck3BlockV7)
        self._is_v8 = isinstance(self.model, RouterBottleneck3BlockV8)

        if threshold is not None:
            self.neural_threshold = float(threshold)
        elif ckpt.get("best_prob_threshold") is not None:
            self.neural_threshold = float(ckpt["best_prob_threshold"])
        else:
            self.neural_threshold = 0.5

        self.device = device
        self.dtype = dtype
        self.threshold = self.bias  # expose drift bias as threshold for logging

        print(
            f"EntropyDriftMultitaskJSRouterSwitching: drift_alpha={self.alpha}, "
            f"drift_bias={self.bias}, drift_tau={self.tau}, drift_warmup={self.warmup_steps}, "
            f"hysteresis={self.hysteresis}, hold_tokens={self.hold_tokens}, "
            f"max_confident_prob={self.max_confident_prob}, min_entropy={self.min_entropy}, "
            f"stochastic={self.stochastic}, seed={random_seed}, "
            f"entropy_topk_k={self.entropy_topk_k}, "
            f"ema_entropy_threshold={self.entropy_threshold}, "
            f"neural_threshold={self.neural_threshold}, model_type={model_type}"
        )

    # ---- Drift helper methods ----

    def reset(self):
        self._sequence_states.clear()
        self._ema_mean = 0.0
        self._n_seen = 0
        self.state.last_model = "quick"

    def _resolve_sequence_key(self, outputs, index: int):
        sequence_ids = getattr(outputs, "sequence_ids", None)
        if sequence_ids is None or index >= len(sequence_ids):
            return self._global_sequence_key
        sequence_id = sequence_ids[index]
        return sequence_id if sequence_id is not None else self._global_sequence_key

    def _resolve_position(self, outputs, index: int) -> Optional[int]:
        positions = getattr(outputs, "positions", None)
        if positions is None or index >= len(positions):
            return None
        position = positions[index]
        return int(position) if position is not None else None

    def _get_sequence_state(self, key, position: Optional[int]) -> EntropyDriftSequenceState:
        if position == 0 or key not in self._sequence_states:
            self._sequence_states[key] = EntropyDriftSequenceState()
        return self._sequence_states[key]

    def _passes_secondary_filter(self, logits_row: torch.Tensor, entropy_value: float) -> bool:
        if self.min_entropy is not None and entropy_value < self.min_entropy:
            return False
        if self.max_confident_prob is not None:
            top1_prob = float(torch.softmax(logits_row, dim=-1).max().item())
            if top1_prob > self.max_confident_prob:
                return False
        return True

    def _compute_switch_probability(self, drift: float) -> float:
        return 1.0 / (1.0 + np.exp(-(drift - self.bias) / max(self.tau, 1e-8)))

    def _update_state_ema(self, state: EntropyDriftSequenceState, entropy_value: float):
        if state.n_seen == 0:
            state.ema_mean = entropy_value
        else:
            state.ema_mean = self.alpha * entropy_value + (1.0 - self.alpha) * state.ema_mean
        state.n_seen += 1
        self._ema_mean = state.ema_mean
        self._n_seen = state.n_seen

    # ---- Neural router ----

    def _multitask_neural_route(self, outputs: ModelOutputs) -> torch.Tensor:
        with torch.no_grad():
            next_token_logits = outputs.logits[:, -1, :]
            top_logits, _ = torch.topk(next_token_logits, k=self.logits_size, dim=-1)
            logits = top_logits.to(device=self.device, dtype=torch.float32)
            hidden_states = outputs.hidden_states[-1][:, -1, :].to(
                device=self.device, dtype=torch.float32
            )
            token_ids = outputs.token[:, -1].to(device=self.device, dtype=torch.long)
            if self._is_v7:
                probs = torch.softmax(logits, dim=-1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                cls_logits, _ = self.model(logits, hidden_states, token_ids, entropy=ent)
            elif self._is_v8:
                cls_logits, div_logits = self.model(logits, hidden_states, token_ids)
            else:
                cls_logits, _ = self.model(logits, hidden_states, token_ids)
            critical_prob = torch.sigmoid(cls_logits.squeeze(-1))
            if self._is_v8:
                div_prob = torch.sigmoid(div_logits.squeeze(-1))
                critical_prob = torch.maximum(critical_prob, div_prob)
            self.last_route_scores = critical_prob.detach().cpu()
            self.last_route_metric_name = "entropy_drift_then_multitask_js_prob"
            return (critical_prob >= self.neural_threshold).to(torch.int)

    # ---- Main route ----

    def route(self, outputs) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        next_token_logits = outputs.logits[:, -1, :]
        entropy_values = compute_topk_entropy(next_token_logits, self.entropy_topk_k)
        drift_values = torch.zeros(batch_size, dtype=torch.float32)

        model_choices = torch.zeros(batch_size, dtype=torch.int, device=next_token_logits.device)
        drift_triggered = torch.zeros(batch_size, dtype=torch.bool)

        for i in range(batch_size):
            h_t = float(entropy_values[i].item())
            sequence_key = self._resolve_sequence_key(outputs, i)
            position = self._resolve_position(outputs, i)
            state = self._get_sequence_state(sequence_key, position)
            logits_row = next_token_logits[i]

            if state.n_seen < self.warmup_steps:
                drift_values[i] = 0.0
            else:
                drift = h_t - state.ema_mean
                drift_values[i] = drift
                passes_filter = self._passes_secondary_filter(logits_row, h_t)
                enter_threshold = self.bias
                exit_threshold = self.bias - self.hysteresis

                if self.stochastic:
                    should_sample = passes_filter and drift >= exit_threshold
                    switch_prob = self._compute_switch_probability(drift) if should_sample else 0.0
                    if state.last_model == "reference" and passes_filter and drift >= exit_threshold:
                        choose_reference = True
                    else:
                        choose_reference = should_sample and (self._rng.random() < switch_prob)
                else:
                    if state.last_model == "reference":
                        choose_reference = passes_filter and (
                            state.hold_remaining > 0 or drift >= exit_threshold
                        )
                    else:
                        choose_reference = passes_filter and (drift >= enter_threshold)

                if choose_reference:
                    model_choices[i] = 1
                    drift_triggered[i] = True

            if int(model_choices[i].item()) == 1:
                if state.last_model != "reference":
                    state.hold_remaining = self.hold_tokens
                elif state.hold_remaining > 0:
                    state.hold_remaining -= 1
                state.last_model = "reference"
            else:
                state.hold_remaining = 0
                state.last_model = "quick"

            # Only update EMA if entropy exceeds threshold (or if no threshold set, or during warmup)
            if self.entropy_threshold is None or state.n_seen < self.warmup_steps or h_t >= self.entropy_threshold:
                self._update_state_ema(state, h_t)

            self._step_counter += 1

        # Stage 2: tokens not escalated by drift go through V13 neural router
        if not bool(drift_triggered.all().item()):
            neural_indices = (~drift_triggered).nonzero(as_tuple=True)[0]
            if len(neural_indices) > 0:
                neural_choices = self._multitask_neural_route(outputs)
                for idx in neural_indices:
                    model_choices[idx] = neural_choices[idx]

        self.last_route_scores = drift_values.detach().cpu()
        self.last_route_metric_name = "entropy_drift_v13"
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        return model_choices


class NeuralRollingWindowSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier with rolling window"""

    def __init__(
        self,
        model_path: str = "critical_classifier_0227.pt",
        window_size: int = 3,
        required_simple_ratio: float = 1.0,
        threshold: Optional[float] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        override_init_args: Optional[dict] = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device, override_init_args=override_init_args)

        # Use saved optimal threshold if available in common_args
        if threshold is None:
            self.threshold = float(model_config["common_args"]["threshold"])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold)
            print(f"Using provided threshold: {self.threshold}")

        # Extract model parameters
        self.init_args = model_config["init_args"]
        self.common_args = model_config["common_args"]
        self.logits_size = self.init_args.get("logits_size", 0)

        # Determine input type from common_args
        self.input_type = self.common_args["input_type"]
        self.model_type = model_config["model_type"]

        print(f"Using input type: {self.input_type}")

        # Set window parameters
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        self.critical_history = deque(maxlen=self.window_size)

        # Set model to evaluation mode
        self.model.eval()

    def route(self, outputs) -> str:
        with torch.no_grad():
            # Get top k logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Process logits if needed
            if self.input_type in ["logits", "both"]:
                # If the model has a logits_size parameter, use it to get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(
                        next_token_logits, k=self.logits_size, dim=-1
                    )
                    top_logits = top_logits.to(
                        device=self.device, dtype=torch.float32
                    )  # [batch_size, topk]
                else:
                    # If no logits_size, use all logits
                    top_logits = next_token_logits.to(
                        device=self.device, dtype=torch.float32
                    )
            else:
                top_logits = None

            # Process hidden states if needed
            if self.input_type in ["hidden_states", "both"]:
                last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(
                    device=self.device, dtype=torch.float32
                )
            else:
                last_hidden_state = None

            # Forward pass through the model based on input_type
            if self.input_type == "logits":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(top_logits)
                ).squeeze()
            elif self.input_type == "hidden_states":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(last_hidden_state)
                ).squeeze()
            elif self.input_type == "both":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(top_logits, last_hidden_state)
                ).squeeze()
            else:
                raise ValueError(f"Unsupported input_type: {self.input_type}")

            # Determine if the token is divergent/critical
            is_divergent = critical_prob >= self.threshold

            # Apply rolling window logic
            if self.state.last_model == "quick":
                model_choice = "quick" if not is_divergent else "reference"
                if model_choice == "reference":
                    self.critical_history.clear()
                    self.critical_history.append(critical_prob)
            else:
                # Reference model: record history and check average uncertainty
                self.critical_history.append(critical_prob)
                if len(self.critical_history) == 0:
                    model_choice = "reference"
                else:
                    avg_critical_prob = sum(self.critical_history) / len(self.critical_history)
                    model_choice = "quick" if avg_critical_prob < self.threshold else "reference"

                # Clear history when switching back to quick
                if model_choice == "quick":
                    self.critical_history.clear()

            self.state.last_model = model_choice
            return model_choice

class NeuralMultiInputSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier"""
    def __init__(self, model_path: str = 'critical_classifier_multi_input_0304.pt',
                 neural_window_size: int = 3,
                 threshold: Optional[float] = None,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32,
                 override_init_args: Optional[str] = None,
                 **kwargs):
        super().__init__()
        
        raise NotImplementedError

        self.device = device
        self.dtype = dtype
        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device, override_init_args=override_init_args)
        
        # Use saved optimal threshold if available in common_args
        if 'threshold' in model_config['common_args'] and model_config['common_args']['threshold'] is not None and threshold is None:
            self.threshold = float(model_config['common_args']['threshold'])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold) if threshold is not None else 0.5
            print(f"Using provided threshold: {self.threshold}")
        
        # Extract model parameters
        self.init_args = model_config['init_args']
        self.common_args = model_config['common_args']
        self.logits_size = self.init_args.get('logits_size', 0)
        self.hidden_states_size = self.init_args.get('hidden_states_size', 0)
        
        # Determine input type from common_args or model type
        self.input_type = self.common_args.get('input_type', 'logits')
        self.model_type = model_config['model_type']
        
        # For backward compatibility, also check model type
        if self.input_type == 'logits' and 'HiddenStates' in self.model_type:
            self.input_type = 'hidden_states'
        elif self.input_type == 'logits' and 'LogitsHiddenStates' in self.model_type:
            self.input_type = 'both'
        
        print(f"Using input type: {self.input_type}")
        
        # Get neural window size from model config or use provided value
        self.neural_window_size = self.init_args.get('neural_window_size', neural_window_size)
        
        # Initialize queues for storing token information
        self.output_logits_queue = deque(maxlen=self.neural_window_size)
        self.output_hidden_states_queue = deque(maxlen=self.neural_window_size)
        
        # Set model to evaluation mode
        self.model.eval()

    def route(self, outputs) -> str:
        with torch.no_grad():
            # Get the last token's logits and hidden states
            batch_size, seq_len, vocab_size = outputs.logits.size()
            
            if seq_len != 1:  # If the current output length is not 1, reset queues (prefill stage)
                self.output_logits_queue.clear()
                self.output_hidden_states_queue.clear()
            
            # Process logits if needed
            if self.input_type in ['logits', 'both']:
                # If logits_size is specified, get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(outputs.logits[:, -1:, :], 
                                              k=self.logits_size // self.neural_window_size, 
                                              dim=-1)  # [batch_size, 1, topk]
                else:
                    # Otherwise use all logits
                    top_logits = outputs.logits[:, -1:, :]
                
                # Add to queue
                self.output_logits_queue.append(top_logits)
            
            # Process hidden states if needed
            if self.input_type in ['hidden_states', 'both']:
                last_hidden_states = outputs.hidden_states[-1][:, -1:, :].to(device=self.device, dtype=torch.float32)
                # Add to queue
                self.output_hidden_states_queue.append(last_hidden_states)
            
            # If we don't have enough tokens yet, default to reference model
            if (self.input_type in ['logits', 'both'] and len(self.output_logits_queue) < self.neural_window_size) or \
               (self.input_type in ['hidden_states', 'both'] and len(self.output_hidden_states_queue) < self.neural_window_size):
                self.state.last_model = 'reference'
                return 'reference'
            
            # Prepare inputs based on model type and input_type
            if 'Multi' in self.model_type:
                # For multi-input models, concatenate the window of tokens
                if self.input_type in ['logits', 'both']:
                    logits_tensor = torch.cat(list(self.output_logits_queue), dim=1)
                    # Reshape for multi-logits models
                    logits_tensor = logits_tensor.view(batch_size, -1)  # Flatten to [batch_size, neural_window_size * topk]
                else:
                    logits_tensor = None
                
                if self.input_type in ['hidden_states', 'both']:
                    hidden_states_tensor = torch.cat(list(self.output_hidden_states_queue), dim=1)
                    # Reshape for multi-hidden-states models
                    hidden_states_tensor = hidden_states_tensor.view(batch_size, -1)  # Flatten to [batch_size, neural_window_size * hidden_size]
                else:
                    hidden_states_tensor = None
            else:
                # For single-token models, just use the latest token
                if self.input_type in ['logits', 'both']:
                    logits_tensor = self.output_logits_queue[-1].squeeze(1)  # [batch_size, topk]
                else:
                    logits_tensor = None
                
                if self.input_type in ['hidden_states', 'both']:
                    hidden_states_tensor = self.output_hidden_states_queue[-1].squeeze(1)  # [batch_size, hidden_size]
                else:
                    hidden_states_tensor = None
            
            # Apply softmax normalization to logits if needed
            if logits_tensor is not None and hasattr(self.model, 'normalize_input') and getattr(self.model, 'normalize_input', False):
                if 'Multi' in self.model_type and 'Logits' in self.model_type:
                    # For MultiLogitsClassifier, reshape to apply softmax correctly
                    batch_size = logits_tensor.shape[0]
                    single_logit_size = logits_tensor.shape[1] // self.neural_window_size
                    reshaped_logits = logits_tensor.view(batch_size, self.neural_window_size, single_logit_size)
                    normalized_logits = torch.nn.functional.softmax(reshaped_logits, dim=-1)
                    logits_tensor = normalized_logits.reshape(batch_size, -1)
                else:
                    logits_tensor = torch.nn.functional.softmax(logits_tensor, dim=-1)
            
            # Forward pass through the model based on input_type
            if self.input_type == 'logits':
                critical_prob = torch.nn.functional.sigmoid(self.model(logits_tensor)).squeeze()
            elif self.input_type == 'hidden_states':
                critical_prob = torch.nn.functional.sigmoid(self.model(hidden_states_tensor)).squeeze()
            elif self.input_type == 'both':
                critical_prob = torch.nn.functional.sigmoid(self.model(logits_tensor, hidden_states_tensor)).squeeze()
            else:
                raise ValueError(f"Unsupported input_type: {self.input_type}")
            
            # Determine if the token is simple or complex
            is_simple = (critical_prob < self.threshold).item()
            
            model_choice = 'quick' if is_simple else 'reference'
            self.state.last_model = model_choice
            return model_choice

class RandomSwitching(ModelSwitchingStrategy):
    """Random switching strategy that selects reference model with a given probability"""
    
    def __init__(self, reference_prob: float = 0.5, random_seed: Optional[int] = 42):
        """Initialize random switching strategy
        
        Args:
            reference_prob: Probability of selecting the reference model (0.0 to 1.0)
            random_seed: Optional random seed for reproducibility
        """
        super().__init__()
        self.reference_prob = reference_prob
        
        # Set random seed
        random.seed(random_seed)
            
        print(f"Initialized RandomSwitching with reference_prob={reference_prob}, random_seed={random_seed}")
    
    def route(self, outputs) -> torch.Tensor:
        """Randomly select between quick and reference models
        
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        # Get batch size from outputs
        batch_size = outputs.logits.size(0)
        
        # Generate random values for each item in batch
        rand_vals = torch.rand(batch_size, device=outputs.logits.device)
        
        # Convert to binary decisions (0 = quick, 1 = reference)
        model_choices = (rand_vals < self.reference_prob).to(torch.int)
        
        # Update state with most recent decision
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        
        return model_choices

class MultitaskJSRouterSwitching(ModelSwitchingStrategy):
    """Hybrid router from train_router_multitask_js best.pt (RouterMultiTaskFFN4).
    Uses top-100 logits, last-layer hidden, and last token id.
    Entropy gate: tokens with entropy > entropy_threshold skip the router and go to reference.
    """

    def __init__(
        self,
        model_path: str,
        threshold: Optional[float] = None,
        entropy_threshold: Optional[float] = 0.6,
        entropy_topk_k: int = 100,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        use_cuda_graph: bool = False,
        pretrained_model_name: Optional[str] = None,
        override_init_args: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.entropy_threshold = float(entropy_threshold) if entropy_threshold is not None else None
        self.entropy_topk_k = int(entropy_topk_k)

        from r2r.models.multitask_js_router import RouterMultiTaskFFN4, RouterBottleneck2Block, RouterBottleneck3Block, RouterBottleneck3BlockV7, RouterBottleneck3BlockV8, RouterBottleneckR2R

        try:
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(model_path, map_location=device)

        saved = ckpt.get("args") or {}
        pm = pretrained_model_name or saved.get("pretrained_model_name", None)

        self.logits_size = 100
        model_type = saved.get("model_type", "ffn4")
        if model_type == "bottleneck2":
            self.model = RouterBottleneck2Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3":
            self.model = RouterBottleneck3Block(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v7":
            self.model = RouterBottleneck3BlockV7(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck3_v8":
            self.model = RouterBottleneck3BlockV8(
                dropout=float(saved.get("dropout", 0.15)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        elif model_type == "bottleneck_r2r":
            self.model = RouterBottleneckR2R(
                dropout=float(saved.get("dropout", 0.3)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        else:
            self.model = RouterMultiTaskFFN4(
                ffn_dim=int(saved.get("ffn_dim", 768)),
                dropout=float(saved.get("dropout", 0.12)),
                normalize_inputs=not bool(saved.get("no_input_layernorm", False)),
                freeze_token_embeddings=bool(saved.get("freeze_token_embeddings", False)),
                pretrained_model_name=pm,
            )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(device)
        self.model.eval()
        self._is_v7 = isinstance(self.model, RouterBottleneck3BlockV7)
        self._is_v8 = isinstance(self.model, RouterBottleneck3BlockV8)

        self.threshold = float(threshold) if threshold is not None else 0.5
        print(
            f"MultitaskJSRouter: threshold={self.threshold}, "
            f"entropy_threshold={self.entropy_threshold}, "
            f"entropy_topk_k={self.entropy_topk_k}, "
            f"model_type={model_type}"
        )
        print(f"MultitaskJSRouter loaded from {model_path}")

    def route(self, outputs):
        batch_size = outputs.logits.shape[0]
        with torch.no_grad():
            next_token_logits = outputs.logits[:, -1, :]
            top_logits, _ = torch.topk(next_token_logits, k=self.logits_size, dim=-1)
            logits = top_logits.to(device=self.device, dtype=torch.float32)

            model_choices = torch.zeros(batch_size, dtype=torch.int, device=self.device)

            if self.entropy_threshold is not None:
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                high_entropy_mask = entropy > self.entropy_threshold
                model_choices[high_entropy_mask] = 1

                low_entropy_mask = ~high_entropy_mask
                if not low_entropy_mask.any():
                    self.last_route_scores = entropy.detach().cpu()
                    self.last_route_metric_name = "entropy"
                    self.state.last_model = "reference" if model_choices.any().item() else "quick"
                    return model_choices

                logits_sub = logits[low_entropy_mask]
                hidden_states_sub = outputs.hidden_states[-1][:, -1, :][low_entropy_mask].to(
                    device=self.device, dtype=torch.float32
                )
                token_sub = outputs.token[:, -1][low_entropy_mask].to(
                    device=self.device, dtype=torch.long
                )
            else:
                hidden_states_sub = outputs.hidden_states[-1][:, -1, :].to(
                    device=self.device, dtype=torch.float32
                )
                token_sub = outputs.token[:, -1].to(device=self.device, dtype=torch.long)
                logits_sub = logits

            if self._is_v7:
                ent_sub = entropy[low_entropy_mask] if self.entropy_threshold is not None else None
                if ent_sub is None:
                    probs_all = torch.softmax(logits_sub, dim=-1)
                    ent_sub = -(probs_all * torch.log(probs_all.clamp_min(1e-12))).sum(dim=-1)
                cls_logits, _ = self.model(logits_sub, hidden_states_sub, token_sub, entropy=ent_sub)
            elif self._is_v8:
                cls_logits, div_logits = self.model(logits_sub, hidden_states_sub, token_sub)
            else:
                cls_logits, _ = self.model(logits_sub, hidden_states_sub, token_sub)
            critical_prob = torch.sigmoid(cls_logits.squeeze(-1))
            if self._is_v8:
                div_prob = torch.sigmoid(div_logits.squeeze(-1))
                critical_prob = torch.maximum(critical_prob, div_prob)

            self.last_route_scores = critical_prob.detach().cpu()
            self.last_route_metric_name = "critical_prob"

            if self.entropy_threshold is not None:
                router_choices = (critical_prob >= self.threshold).to(torch.int)
                model_choices[low_entropy_mask] = router_choices
            else:
                model_choices = (critical_prob >= self.threshold).to(torch.int)

            self.state.last_model = "reference" if model_choices.any().item() else "quick"
            return model_choices



def create_switching_strategy(strategy_name: str, **kwargs) -> ModelSwitchingStrategy:
    """Factory function to create switching strategy instances"""
    strategies = {
        'immediate': ImmediateSwitching,
        'entropy': EntropySwitching,
        'entropy_neural': EntropyNeuralSwitching,
        'entropy_then_neural': EntropyNeuralSwitching,
        'entropy_neural_multitask_js': EntropyMultitaskNeuralSwitching,
        'entropy_then_neural_multitask_js': EntropyMultitaskNeuralSwitching,
        'entropy_lookahead': EntropyLookaheadSwitching,
        'entropy_variance_js': EntropyVarianceJsSwitching,
        'sliding_window_entropy': SlidingWindowEntropySwitching,
        'sliding_window_entropy_js': SlidingWindowEntropyJsSwitching,
        'entropy_drift': EntropyDriftSwitching,
        'entropy_drift_multitask_js_router': EntropyDriftMultitaskJSRouterSwitching,
        'entropy_drift_v13': EntropyDriftMultitaskJSRouterSwitching,
        'entropy_topk': EntropyTopKSwitching,
        'js': JSSwitching,
        'js_llm': JSLLMSwitching,
        'js_topk_sparse': JSTopKSparseSwitching,
        'js_topk_llm': JSTopKLLMSwitching,
        'entropy_js': EntropyJSSwitching,
        'entropy_js_llm': EntropyJSLLMSwitching,
        'entropy_js_topk_sparse': EntropyJSTopKSparseSwitching,
        'entropy_js_topk_llm': EntropyJSTopKLLMSwitching,
        'entropy_js_topk_async': EntropyJSTopKAsyncSwitching,
        'momentum': MomentumSwitching,
        'rolling': SingleRollingWindowSwitching,
        'duo_rolling': DuoRollingWindowSwitching,
        'neural': NeuralSwitching,
        'neural_rolling': NeuralRollingWindowSwitching,
        'neural_multi_input': NeuralMultiInputSwitching,
        'multitask_js_router': MultitaskJSRouterSwitching,
        'random': RandomSwitching
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available strategies: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs) 
