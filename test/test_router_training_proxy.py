"""Tests for r2r.utils.router_training_proxy (entropy aligned with MultitaskJSRouter)."""

import unittest

import numpy as np
import torch

from r2r.utils.router_training_proxy import (
    entropy_topk_shannon_logits,
    js_high_from_js_values,
    verify_entropy_numpy_matches_torch,
)


class TestRouterTrainingProxy(unittest.TestCase):
    def test_entropy_numpy_torch_match(self):
        rng = np.random.default_rng(0)
        logits = rng.standard_normal((32, 100)).astype(np.float32)
        self.assertTrue(verify_entropy_numpy_matches_torch(logits, rtol=1e-4, atol=1e-5))

    def test_entropy_torch_matches_softmax_formula(self):
        logits = torch.randn(4, 100, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        want = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
        got = entropy_topk_shannon_logits(logits)
        self.assertTrue(torch.allclose(got, want))

    def test_js_high_from_values(self):
        js = np.array([0.04, 0.06, 0.11], dtype=np.float32)
        y = js_high_from_js_values(js, 0.1)
        self.assertEqual(list(y.tolist()), [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
