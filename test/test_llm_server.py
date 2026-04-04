from types import SimpleNamespace

import torch

from r2r.models.sglang_patch.llm_server import LLMServer


class _QueueStub:
    def __init__(self):
        self.items = []

    def put_nowait(self, item):
        self.items.append(item)


def test_llm_sparse_js_returns_quick_token_when_sparse_js_is_low():
    req = SimpleNamespace(
        rid="req-1",
        status="need",
        r2r_reference_logits_mode="topk",
        r2r_reference_topk_k=2,
        r2r_reference_decision_mode="llm_sparse_js",
        r2r_reference_js_threshold=0.2,
        r2r_quick_token_id=3,
        r2r_quick_topk_indices=torch.tensor([0, 1]),
        r2r_quick_topk_logits=torch.tensor([1.0, 1.0]),
    )
    batch = SimpleNamespace(reqs=[req], output_ids=None)
    result = SimpleNamespace(
        next_token_ids=torch.tensor([9]),
        logits_output=SimpleNamespace(
            next_token_logits=torch.tensor([[1.0, 1.0, 0.0]])
        ),
    )
    outbound_queue = _QueueStub()

    LLMServer.process_batch_results(
        rank=0,
        batch=batch,
        result=result,
        scheduler=SimpleNamespace(),
        outbound_queue=outbound_queue,
    )

    waiting_req = outbound_queue.items[0][0]
    assert waiting_req.new_token_ids == [3]
    assert waiting_req.final_token_source == "quick"
    assert waiting_req.reference_topk_indices is None
    assert waiting_req.reference_topk_logits is None


def test_llm_full_js_returns_quick_token_when_full_js_is_low():
    req = SimpleNamespace(
        rid="req-1",
        status="need",
        r2r_reference_logits_mode="full",
        r2r_reference_topk_k=None,
        r2r_reference_decision_mode="llm_full_js",
        r2r_reference_js_threshold=0.2,
        r2r_quick_logits=torch.tensor([1.0, 1.0, 0.0]),
        r2r_quick_token_id=3,
    )
    batch = SimpleNamespace(reqs=[req], output_ids=None)
    result = SimpleNamespace(
        next_token_ids=torch.tensor([9]),
        logits_output=SimpleNamespace(
            next_token_logits=torch.tensor([[1.0, 1.0, 0.0]])
        ),
    )
    outbound_queue = _QueueStub()

    LLMServer.process_batch_results(
        rank=0,
        batch=batch,
        result=result,
        scheduler=SimpleNamespace(),
        outbound_queue=outbound_queue,
    )

    waiting_req = outbound_queue.items[0][0]
    assert waiting_req.new_token_ids == [3]
    assert waiting_req.final_token_source == "quick"
    assert waiting_req.reference_logits is None
