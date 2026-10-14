"""CPU tests: python -m unittest tests/test_binary_answer_score.py -v"""
import unittest

import torch
import torch.nn.functional as F
from minigpt4.models.binary_answer_score import binary_answer_score


class TestBinaryAnswerScore(unittest.TestCase):
    def test_original_mode_preserved_exactly(self):
        torch.manual_seed(6)
        logits = torch.randn(4, 7, 13, dtype=torch.float32)
        original = logits[:, -2, :][:, 3]
        got = binary_answer_score(logits, -2, yes_token_id=3,
                                  no_token_id=9, mode="yes")
        self.assertTrue(torch.equal(got, original))

    def test_margin_matches_two_answer_softmax(self):
        torch.manual_seed(9)
        logits = torch.randn(4, 7, 13)
        margin = binary_answer_score(logits, -2, 3, 9, mode="margin")
        expected = logits[:, -2, 3] - logits[:, -2, 9]
        conditional_probability = torch.softmax(
            logits[:, -2, [3, 9]], dim=-1)[:, 0]
        self.assertTrue(torch.equal(margin, expected))
        self.assertTrue(torch.allclose(torch.sigmoid(margin),
                                       conditional_probability, atol=1e-6))

    def test_margin_bce_has_yes_and_no_gradients(self):
        torch.manual_seed(4)
        logits = torch.randn(3, 4, 11, requires_grad=True)
        scores = binary_answer_score(logits, -2, 2, 8, mode="margin")
        loss = F.binary_cross_entropy_with_logits(
            scores, torch.tensor([1.0, 0.0, 1.0]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[:, -2, 2].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, -2, 8].abs().sum()), 0)
        self.assertTrue(torch.allclose(logits.grad[:, -2, 2],
                                       -logits.grad[:, -2, 8], atol=1e-7))

    def test_invalid_configuration_fails(self):
        logits = torch.randn(2, 4, 9)
        with self.assertRaises(ValueError):
            binary_answer_score(logits, -2, 1, 2, mode="unknown")
        with self.assertRaises(ValueError):
            binary_answer_score(logits, -2, 1, None, mode="margin")
        with self.assertRaises(ValueError):
            binary_answer_score(logits, -2, 1, 1, mode="margin")


if __name__ == "__main__":
    unittest.main()
