"""CPU-only A1 fusion checks. Run: python -m unittest tests/test_candidate_history_fusion.py"""
import unittest
import torch

from minigpt4.models.candidate_history_fusion import CandidateHistoryFusion


class TestCandidateHistoryFusion(unittest.TestCase):
    def test_zero_init_and_padding(self):
        torch.manual_seed(17)
        m = CandidateHistoryFusion(dim=16, rank=4)
        u = torch.randn(3, 16)
        t = torch.randn(3, 16)
        h = torch.randn(3, 5, 16)
        valid = torch.tensor([[True, True, False, False, False],
                              [False, False, False, False, False],
                              [True, True, True, False, False]])
        delta, attn = m(u, t, h, valid)
        self.assertEqual(tuple(delta.shape), (3, 4))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(attn[1], torch.zeros_like(attn[1])))
        self.assertTrue(torch.equal(delta[1], torch.zeros_like(delta[1])))
        self.assertTrue(torch.allclose(attn[0, :2].sum(), torch.tensor(1.)))
        self.assertTrue(torch.equal(attn[0, 2:], torch.zeros_like(attn[0, 2:])))

    def test_mixed_precision_accumulation_and_backward(self):
        # Mimic FP32 GNN output with an FP16 AMP fusion residual.
        from minigpt4.models.candidate_history_fusion import accumulate_target_residuals

        delta_rows = torch.zeros(5, 4, dtype=torch.float32)
        residual = torch.tensor([[1., 2., 3., 4.],
                                 [2., 3., 4., 5.],
                                 [4., 3., 2., 1.]], dtype=torch.float16,
                                requires_grad=True)
        indices = torch.tensor([1, 1, 4], dtype=torch.long)
        result = accumulate_target_residuals(delta_rows, indices, residual)

        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.equal(
            result[1], torch.tensor([3., 5., 7., 9.], dtype=torch.float32)))
        self.assertTrue(torch.equal(result[0], torch.zeros(4)))
        self.assertTrue(torch.equal(
            result[4], torch.tensor([4., 3., 2., 1.], dtype=torch.float32)))

        result.sum().backward()
        self.assertIsNotNone(residual.grad)
        self.assertTrue(torch.equal(residual.grad, torch.ones_like(residual)))

    def test_inference_residual_multiplier(self):
        from minigpt4.models.candidate_history_fusion import scale_candidate_delta_for_eval

        delta = torch.tensor([[0.01, -0.02]], dtype=torch.float32)
        self.assertIs(scale_candidate_delta_for_eval(delta, 1.0), delta)
        self.assertTrue(torch.equal(scale_candidate_delta_for_eval(delta, 0.0),
                                    torch.zeros_like(delta)))
        self.assertTrue(torch.equal(scale_candidate_delta_for_eval(delta, 2.0),
                                    delta * 2.0))
        self.assertTrue(torch.equal(scale_candidate_delta_for_eval(delta, 5.0),
                                    delta * 5.0))
        self.assertIsNone(scale_candidate_delta_for_eval(None, 5.0))

    def test_A2_gate_identity_padding_and_grad(self):
        from minigpt4.models.candidate_history_fusion import CandidateHistoryGate
        torch.manual_seed(17)
        m = CandidateHistoryGate(dim=16, rank=4, attention="cosine")
        user = torch.randn(3, 16)
        target = torch.randn(3, 16)
        hist = torch.randn(3, 5, 16)
        valid = torch.tensor([
            [True, True, False, False, False],
            [False, False, False, False, False],
            [True, True, True, False, False]
        ])
        gate, weights = m(user, target, hist, valid)
        self.assertEqual(tuple(gate.shape), (3, 4))
        self.assertTrue(torch.equal(gate, torch.ones_like(gate)))
        self.assertTrue(torch.equal(weights[1], torch.zeros_like(weights[1])))
        self.assertAlmostEqual(float(weights[0].sum()), 1.0, places=6)
        self.assertTrue(torch.equal(weights[0, 2:], torch.zeros_like(weights[0, 2:])))

        # Zero-init outputs identity but last-layer weights receive gradient;
        # first-layer gradients can start on subsequent optimizer steps.
        (gate[0].sum()).backward()
        self.assertIsNotNone(m.gate_encoder[-1].weight.grad)
        self.assertGreater(float(m.gate_encoder[-1].weight.grad.abs().sum()), 0)

        with torch.no_grad():
            torch.nn.init.normal_(m.gate_encoder[-1].weight, std=0.1)
        hist.requires_grad_(True)
        gate, _ = m(user, target, hist, valid)
        (gate[0].sum()).backward()
        self.assertGreater(float(m.gate_encoder[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(hist.grad.abs().sum()), 0)
        self.assertTrue(torch.equal(gate[1], torch.ones_like(gate[1])))

    def test_A3_trainable_candidate_attention(self):
        from minigpt4.models.candidate_history_fusion import CandidateHistoryGate
        torch.manual_seed(23)
        m = CandidateHistoryGate(dim=16, rank=4, attention="learned")
        user = torch.randn(3, 16)
        target = torch.randn(3, 16)
        hist = torch.randn(3, 4, 16)
        valid = torch.tensor([
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, False],
        ])
        with torch.no_grad():
            torch.nn.init.normal_(m.gate_encoder[-1].weight, std=0.1)
        # Same history/user but different candidate: attention should change.
        user[1] = user[0]
        hist[1] = hist[0]
        gate, weights = m(user, target, hist, valid)
        self.assertFalse(torch.allclose(weights[0, :2], weights[1, :2]))
        self.assertTrue(torch.equal(weights[2], torch.zeros_like(weights[2])))
        self.assertTrue(torch.equal(gate[2], torch.ones_like(gate[2])))
        gate[:2].sum().backward()
        self.assertIsNotNone(m.query_proj.weight.grad)
        self.assertIsNotNone(m.key_proj.weight.grad)
        self.assertGreater(float(m.query_proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(m.key_proj.weight.grad.abs().sum()), 0)

    def test_inference_gate_ablation_identity(self):
        from minigpt4.models.candidate_history_fusion import (
            scale_candidate_gate_for_eval,
        )
        g = torch.tensor([[1.0, 1.25, 0.75], [1.0, 0.8, 1.2]])
        self.assertIsNone(scale_candidate_gate_for_eval(None, 0.0))
        self.assertIs(scale_candidate_gate_for_eval(g, 1.0), g)
        self.assertTrue(torch.equal(
            scale_candidate_gate_for_eval(g, 0.0), torch.ones_like(g)))
        self.assertTrue(torch.allclose(
            scale_candidate_gate_for_eval(g, 0.25),
            1.0 + 0.25 * (g - 1.0)))

    def test_A2_identity_rows_accumulation(self):
        from minigpt4.models.candidate_history_fusion import (
            accumulate_target_residuals,
        )
        rows = torch.ones(5, 4, dtype=torch.float32)
        gates = torch.tensor([[1.2, 0.8, 1.0, 1.1]], requires_grad=True)
        got = accumulate_target_residuals(rows, torch.tensor([3]), gates - 1)
        self.assertTrue(torch.allclose(got[3], gates[0]))
        self.assertTrue(torch.equal(got[2], torch.ones(4)))
        got.sum().backward()
        self.assertTrue(torch.equal(gates.grad, torch.ones_like(gates)))

    def test_trainable_candidate_specific(self):
        torch.manual_seed(17)
        m = CandidateHistoryFusion(dim=16, rank=4)
        torch.nn.init.normal_(m.encoder[-1].weight, std=0.1)
        u = torch.randn(2, 16, requires_grad=True)
        t = torch.randn(2, 16, requires_grad=True)
        h = torch.randn(2, 5, 16, requires_grad=True)
        valid = torch.tensor([[True, True, True, False, False],
                              [True, True, True, False, False]])
        delta, _ = m(u, t, h, valid)
        delta.sum().backward()
        self.assertGreater(float(m.encoder[0].weight.grad.norm()), 0)
        self.assertGreater(float(h.grad.norm()), 0)
        self.assertFalse(torch.allclose(delta[0], delta[1]))


if __name__ == "__main__":
    unittest.main()
