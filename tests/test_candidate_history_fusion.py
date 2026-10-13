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
