import torch

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.metrics import compute_js_divergence
from r2r.utils.switching import EntropyJSSwitching


def test_compute_js_divergence_is_zero_for_identical_logits():
    logits = torch.tensor([1.0, 1.0, 1.0])
    js = compute_js_divergence(logits, logits)
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
