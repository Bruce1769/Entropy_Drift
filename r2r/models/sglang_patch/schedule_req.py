from typing import Optional, Tuple, Union, List, Dict
from sglang.srt.sampling.sampling_params import SamplingParams

class SimpleSamplingParams:
    def __init__(self, temperature: float = 1.0, top_k: int = -1, top_p: float = 1.0, max_new_tokens: int = 128):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
    
    def derive_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_new_tokens=self.max_new_tokens
        )

class WaitingReq:
    def __init__(
        self,
        rid: str,
        new_token_ids: List[int],
        sampling_params: Optional[SimpleSamplingParams] = None,
        status: str = "need",
        reference_logits=None,
        reference_logits_mode: Optional[str] = None,
        reference_topk_k: Optional[int] = None,
        reference_decision_mode: Optional[str] = None,
        reference_js_threshold: Optional[float] = None,
        reference_topk_indices=None,
        reference_topk_logits=None,
        quick_logits=None,
        quick_token_id: Optional[int] = None,
        quick_topk_indices=None,
        quick_topk_logits=None,
        final_token_source: Optional[str] = None,
        async_speculative: bool = False,
    ):
        self.rid = rid
        self.new_token_ids = new_token_ids
        self.sampling_params = sampling_params
        self.status = status
        self.reference_logits = reference_logits
        self.reference_logits_mode = reference_logits_mode
        self.reference_topk_k = reference_topk_k
        self.reference_decision_mode = reference_decision_mode
        self.reference_js_threshold = reference_js_threshold
        self.reference_topk_indices = reference_topk_indices
        self.reference_topk_logits = reference_topk_logits
        self.quick_logits = quick_logits
        self.quick_token_id = quick_token_id
        self.quick_topk_indices = quick_topk_indices
        self.quick_topk_logits = quick_topk_logits
        self.final_token_source = final_token_source
        self.async_speculative = async_speculative
