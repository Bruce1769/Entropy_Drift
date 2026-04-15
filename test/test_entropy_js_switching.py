import torch

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.metrics import compute_js_divergence, compute_sparse_topk_js_divergence
from r2r.utils.switching import (
    EntropyDriftSwitching,
    EntropyJSSwitching,
    EntropyJSLLMSwitching,
    EntropyJSTopKAsyncSwitching,
    EntropyJSTopKSparseSwitching,
    EntropyJSTopKLLMSwitching,
)


def test_compute_js_divergence_is_zero_for_identical_logits():
    logits = torch.tensor([1.0, 1.0, 1.0])
    js = compute_js_divergence(logits, logits)
    assert abs(js) < 1e-8


def test_compute_sparse_topk_js_divergence_is_zero_for_identical_sparse_logits():
    logits = torch.tensor([2.0, 1.0])
    indices = torch.tensor([0, 1])
    js = compute_sparse_topk_js_divergence(logits, indices, logits, indices)
    assert abs(js) < 1e-8


def test_entropy_js_routes_only_high_entropy_high_js_tokens():
    switching = EntropyJSSwitching(entropy_threshold=0.45, js_threshold=0.2)

    quick_logits = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[10.0, -10.0, -10.0]],
        ]
    )
    reference_logits = torch.tensor(
        [
            [[10.0, -10.0, -10.0]],
            [[10.0, -10.0, -10.0]],
        ]
    )
    outputs = ModelOutputs(
        logits=quick_logits,
        hidden_states=[torch.zeros(2, 1, 4)],
        token=torch.tensor([[0], [0]]),
        reference_logits=reference_logits,
    )

    reference_candidates = switching.get_reference_candidates(outputs)
    model_choices = switching.route(outputs)

    assert reference_candidates.tolist() == [1, 0]
    assert model_choices.tolist() == [1, 0]


def test_entropy_js_keeps_quick_when_js_is_low():
    switching = EntropyJSSwitching(entropy_threshold=0.45, js_threshold=0.2)

    logits = torch.tensor([[[1.0, 1.0, 1.0]]])
    outputs = ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        reference_logits=logits.clone(),
    )

    assert switching.get_reference_candidates(outputs).tolist() == [1]
    assert switching.route(outputs).tolist() == [0]


def test_entropy_js_llm_reports_llm_side_full_js_request():
    switching = EntropyJSLLMSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
    )

    assert switching.get_reference_distribution_request() == {
        "mode": "full",
        "topk_k": None,
        "decision_mode": "llm_full_js",
        "js_threshold": 0.2,
    }


def test_entropy_js_llm_matches_full_route_locally():
    switching = EntropyJSLLMSwitching(entropy_threshold=0.45, js_threshold=0.2)

    logits = torch.tensor([[[1.0, 1.0, 1.0]]])
    outputs = ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        reference_logits=logits.clone(),
    )

    assert switching.get_reference_candidates(outputs).tolist() == [1]
    assert switching.route(outputs).tolist() == [0]


def test_entropy_js_topk_sparse_routes_high_entropy_high_js_tokens():
    switching = EntropyJSTopKSparseSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
        js_topk=2,
    )

    quick_logits = torch.tensor(
        [
            [[1.0, 1.0, 1.0]],
            [[10.0, -10.0, -10.0]],
        ]
    )
    outputs = ModelOutputs(
        logits=quick_logits,
        hidden_states=[torch.zeros(2, 1, 4)],
        token=torch.tensor([[0], [0]]),
        reference_topk_indices=torch.tensor([[0, 1], [0, 1]]),
        reference_topk_logits=torch.tensor([[10.0, -10.0], [10.0, -10.0]]),
    )

    assert switching.get_reference_candidates(outputs).tolist() == [1, 0]
    assert switching.route(outputs).tolist() == [1, 0]


def test_entropy_js_topk_sparse_keeps_quick_when_sparse_js_is_low():
    switching = EntropyJSTopKSparseSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
        js_topk=2,
    )

    logits = torch.tensor([[[1.0, 1.0, 1.0]]])
    outputs = ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        reference_topk_indices=torch.tensor([[0, 1]]),
        reference_topk_logits=torch.tensor([[1.0, 1.0]]),
    )

    assert switching.get_reference_candidates(outputs).tolist() == [1]
    assert switching.route(outputs).tolist() == [0]


def test_entropy_js_topk_llm_reports_llm_side_decision_request():
    switching = EntropyJSTopKLLMSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
        js_topk=64,
    )

    assert switching.get_reference_distribution_request() == {
        "mode": "topk",
        "topk_k": 64,
        "decision_mode": "llm_sparse_js",
        "js_threshold": 0.2,
    }


def test_entropy_js_topk_llm_matches_sparse_route_locally():
    switching = EntropyJSTopKLLMSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
        js_topk=2,
    )

    logits = torch.tensor([[[1.0, 1.0, 1.0]]])
    outputs = ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        reference_topk_indices=torch.tensor([[0, 1]]),
        reference_topk_logits=torch.tensor([[1.0, 1.0]]),
    )

    assert switching.get_reference_candidates(outputs).tolist() == [1]
    assert switching.route(outputs).tolist() == [0]


def test_entropy_js_topk_async_reports_async_llm_side_decision_request():
    switching = EntropyJSTopKAsyncSwitching(
        entropy_threshold=0.45,
        js_threshold=0.2,
        js_topk=64,
    )

    assert switching.async_speculative_validation is True
    assert switching.get_reference_distribution_request() == {
        "mode": "topk",
        "topk_k": 64,
        "decision_mode": "async_llm_sparse_js",
        "js_threshold": 0.2,
    }


def test_entropy_drift_resets_state_for_new_sequence():
    switching = EntropyDriftSwitching(
        alpha=0.3,
        bias=0.5,
        tau=0.2,
        warmup_steps=1,
        stochastic=False,
    )
    logits = torch.tensor(
        [
            [[10.0, -10.0, -10.0]],
            [[1.0, 1.0, 1.0]],
        ]
    )
    outputs = ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(2, 1, 4)],
        token=torch.tensor([[0], [0]]),
        sequence_ids=["sample-a", "sample-b"],
        positions=[0, 0],
    )

    assert switching.route(outputs).tolist() == [0, 0]


def test_entropy_drift_hysteresis_and_hold_reduce_flicker():
    switching = EntropyDriftSwitching(
        alpha=0.5,
        bias=0.2,
        tau=0.2,
        warmup_steps=1,
        hysteresis=0.15,
        hold_tokens=2,
        stochastic=False,
    )

    warmup = ModelOutputs(
        logits=torch.tensor([[[10.0, -10.0, -10.0]]]),
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        sequence_ids=["sample-a"],
        positions=[0],
    )
    high_drift = ModelOutputs(
        logits=torch.tensor([[[0.0, 0.0, 0.0]]]),
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        sequence_ids=["sample-a"],
        positions=[1],
    )
    low_drift = ModelOutputs(
        logits=torch.tensor([[[10.0, -10.0, -10.0]]]),
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        sequence_ids=["sample-a"],
        positions=[2],
    )

    assert switching.route(warmup).tolist() == [0]
    assert switching.route(high_drift).tolist() == [1]
    assert switching.route(low_drift).tolist() == [1]


def test_entropy_drift_confidence_filter_blocks_easy_escalations():
    switching = EntropyDriftSwitching(
        alpha=0.5,
        bias=0.2,
        tau=0.2,
        warmup_steps=1,
        max_confident_prob=0.8,
        stochastic=False,
    )

    warmup = ModelOutputs(
        logits=torch.tensor([[[10.0, -10.0, -10.0]]]),
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        sequence_ids=["sample-a"],
        positions=[0],
    )
    confident = ModelOutputs(
        logits=torch.tensor([[[3.0, 0.0, -1.0]]]),
        hidden_states=[torch.zeros(1, 1, 4)],
        token=torch.tensor([[0]]),
        sequence_ids=["sample-a"],
        positions=[1],
    )

    assert switching.route(warmup).tolist() == [0]
    assert switching.route(confident).tolist() == [0]
