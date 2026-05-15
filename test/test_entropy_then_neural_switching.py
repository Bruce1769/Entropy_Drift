import torch

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.metrics import compute_entropy
from r2r.utils import switching as switching_module
from r2r.utils.switching import EntropyNeuralSwitching, create_switching_strategy


def _build_outputs(logits: torch.Tensor) -> ModelOutputs:
    batch_size = logits.shape[0]
    return ModelOutputs(
        logits=logits,
        hidden_states=[torch.zeros(batch_size, 1, 4)],
        token=torch.zeros(batch_size, 1, dtype=torch.long),
    )


def test_entropy_then_neural_alias_creates_hybrid_strategy(monkeypatch):
    class FakeNeuralSwitching:
        def __init__(self, **kwargs):
            self.threshold = float(kwargs.get("threshold", 0.5))

        def route(self, outputs):
            return torch.zeros(outputs.logits.shape[0], dtype=torch.int, device=outputs.logits.device)

    monkeypatch.setattr(switching_module, "NeuralSwitching", FakeNeuralSwitching)
    strategy = create_switching_strategy(
        "entropy_then_neural",
        model_path="dummy-router.pt",
        entropy_threshold=0.6,
        threshold=0.4,
        use_cuda_graph=False,
    )
    assert isinstance(strategy, EntropyNeuralSwitching)


def test_entropy_then_neural_high_entropy_bypasses_neural_router(monkeypatch):
    class FakeNeuralSwitching:
        def __init__(self, **kwargs):
            self.threshold = float(kwargs.get("threshold", 0.5))
            self.calls = 0

        def route(self, outputs):
            self.calls += 1
            raise AssertionError("Neural router should not run for all-high-entropy batch")

    monkeypatch.setattr(switching_module, "NeuralSwitching", FakeNeuralSwitching)
    strategy = EntropyNeuralSwitching(
        model_path="dummy-router.pt",
        entropy_threshold=0.6,
        threshold=0.4,
        use_cuda_graph=False,
    )
    outputs = _build_outputs(torch.tensor([[[1.0, 1.0, 1.0]]]))
    assert strategy.route(outputs).tolist() == [1]


def test_entropy_then_neural_low_entropy_uses_neural_router(monkeypatch):
    class FakeNeuralSwitching:
        def __init__(self, **kwargs):
            self.threshold = float(kwargs.get("threshold", 0.5))
            self.calls = 0

        def route(self, outputs):
            self.calls += 1
            return torch.ones(outputs.logits.shape[0], dtype=torch.int, device=outputs.logits.device)

    monkeypatch.setattr(switching_module, "NeuralSwitching", FakeNeuralSwitching)
    strategy = EntropyNeuralSwitching(
        model_path="dummy-router.pt",
        entropy_threshold=0.6,
        threshold=0.4,
        use_cuda_graph=False,
    )
    outputs = _build_outputs(torch.tensor([[[10.0, -10.0, -10.0]]]))
    assert strategy.route(outputs).tolist() == [1]
    assert strategy.neural.calls == 1


def test_entropy_then_neural_threshold_boundary_routes_to_reference(monkeypatch):
    class FakeNeuralSwitching:
        def __init__(self, **kwargs):
            self.threshold = float(kwargs.get("threshold", 0.5))
            self.calls = 0

        def route(self, outputs):
            self.calls += 1
            return torch.tensor([0, 1], dtype=torch.int, device=outputs.logits.device)

    monkeypatch.setattr(switching_module, "NeuralSwitching", FakeNeuralSwitching)
    boundary_logits = torch.tensor([0.0, 0.0, 0.0])
    entropy_threshold = float(compute_entropy(boundary_logits))
    strategy = EntropyNeuralSwitching(
        model_path="dummy-router.pt",
        entropy_threshold=entropy_threshold,
        threshold=0.4,
        use_cuda_graph=False,
    )
    outputs = _build_outputs(
        torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[10.0, -10.0, -10.0]],
            ]
        )
    )
    # Boundary item (entropy == threshold) must be forced to reference.
    assert strategy.route(outputs).tolist() == [1, 1]
    assert strategy.neural.calls == 1
