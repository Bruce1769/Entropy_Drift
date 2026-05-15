"""Unit tests for EntropyMultitaskNeuralSwitching (full-vocab entropy gate + multitask JS router)."""

import unittest
from unittest.mock import patch

import torch

from r2r.utils.dataclass import ModelOutputs
from r2r.utils.metrics import compute_entropy
from r2r.utils.switching import EntropyMultitaskNeuralSwitching


def _make_outputs(
    last_token_logits: torch.Tensor,
    hidden_dim: int = 8,
    min_vocab: int = 100,
) -> ModelOutputs:
    """last_token_logits: shape [batch, vocab] for the final step (vocab >= min_vocab for top-k router)."""
    if last_token_logits.dim() == 1:
        last_token_logits = last_token_logits.unsqueeze(0)
    batch, vocab = last_token_logits.shape
    if vocab < min_vocab:
        pad = torch.full(
            (batch, min_vocab - vocab),
            -1e4,
            dtype=last_token_logits.dtype,
            device=last_token_logits.device,
        )
        last_token_logits = torch.cat([last_token_logits, pad], dim=-1)
    logits = last_token_logits.unsqueeze(1)
    hidden = torch.randn(batch, 1, hidden_dim)
    token = torch.zeros(batch, 1, dtype=torch.long)
    return ModelOutputs(logits=logits, hidden_states=[hidden], token=token)


class _FakeRouterFFN(torch.nn.Module):
    """Minimal stand-in for RouterMultiTaskFFN4."""

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.forward_calls = 0

    def forward(self, logits, hidden_states, token_ids):
        self.forward_calls += 1
        b = logits.shape[0]
        # Return logits such that sigmoid gives ~0.2 (stay on quick) or ~0.9 (reference)
        val = getattr(self, "_logit_out", -2.0)
        return torch.full((b, 1), val, device=logits.device, dtype=logits.dtype), None


class TestEntropyMultitaskNeuralSwitching(unittest.TestCase):
    def _strategy(self, fake_router, entropy_threshold=1.0, threshold=0.5):
        ckpt = {
            "model_state": {},
            "args": {"ffn_dim": 64, "dropout": 0.0},
            "best_prob_threshold": 0.35,
        }

        def _fake_load(path, map_location=None, weights_only=False):
            return ckpt

        with patch("r2r.utils.switching.torch.load", side_effect=_fake_load):
            with patch(
                "r2r.models.multitask_js_router.RouterMultiTaskFFN4",
                return_value=fake_router,
            ):
                return EntropyMultitaskNeuralSwitching(
                    model_path="/fake/best.pt",
                    entropy_threshold=entropy_threshold,
                    threshold=threshold,
                    device="cpu",
                    override_init_args={"pretrained_model_name": None},
                )

    def test_all_high_entropy_never_calls_router_forward(self):
        fake = _FakeRouterFFN()
        s = self._strategy(fake, entropy_threshold=0.01)
        # Nearly uniform over many logits -> high entropy
        logits = torch.zeros(2, 50)
        out = _make_outputs(logits)
        ent = compute_entropy(out.logits[:, -1, :])
        self.assertTrue(bool((ent >= 0.01).all()))
        choices = s.route(out)
        self.assertEqual(choices.tolist(), [1, 1])
        self.assertEqual(fake.forward_calls, 0)

    def test_all_low_entropy_uses_router(self):
        fake = _FakeRouterFFN()
        fake._logit_out = -3.0  # sigmoid ~0.047 < 0.5 -> quick
        s = self._strategy(fake, entropy_threshold=5.0)
        out = _make_outputs(torch.tensor([[20.0, -20.0, -20.0], [15.0, -15.0, -15.0]]))
        choices = s.route(out)
        self.assertEqual(choices.tolist(), [0, 0])
        self.assertEqual(fake.forward_calls, 1)

    def test_mixed_batch(self):
        fake = _FakeRouterFFN()
        fake._logit_out = 3.0  # sigmoid high -> reference when neural runs
        s = self._strategy(fake, entropy_threshold=1.0)
        # row0: peaked -> low entropy; row1: flat -> high entropy
        out = _make_outputs(
            torch.stack(
                [
                    torch.tensor([10.0, -10.0, -10.0]),
                    torch.zeros(3),
                ]
            )
        )
        choices = s.route(out)
        # row1 forced to 1; row0 from neural -> high logit -> 1
        self.assertEqual(choices.tolist(), [1, 1])

    def test_threshold_none_uses_checkpoint_best_prob(self):
        fake = _FakeRouterFFN()
        ckpt = {
            "model_state": {},
            "args": {},
            "best_prob_threshold": 0.777,
        }

        def _fake_load(path, map_location=None, weights_only=False):
            return ckpt

        with patch("r2r.utils.switching.torch.load", side_effect=_fake_load):
            with patch(
                "r2r.models.multitask_js_router.RouterMultiTaskFFN4",
                return_value=fake,
            ):
                s = EntropyMultitaskNeuralSwitching(
                    model_path="/fake/best.pt",
                    entropy_threshold=10.0,
                    threshold=None,
                    device="cpu",
                    override_init_args={"pretrained_model_name": None},
                )
        self.assertAlmostEqual(s.threshold, 0.777, places=6)


if __name__ == "__main__":
    unittest.main()
