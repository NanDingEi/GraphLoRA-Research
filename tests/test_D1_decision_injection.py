"""CPU tests: python -m unittest tests.test_D1_decision_injection -v."""
import unittest
import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from minigpt4.models.d1_decision_injection import (
    build_decision_pair_features,
    decision_position,
    inject_decision_query,
)


class D1InjectionTests(unittest.TestCase):
    def test_decision_pos_matches_existing_yes_logit(self):
        # Existing S2 logits[:, -(answer_len + 1), :] predicts "Yes".
        p, a = 29, 1
        idx = decision_position(p, a, p + a)
        self.assertEqual(idx, 28)
        logits = torch.randn(2, p + a, 7)
        self.assertTrue(torch.equal(logits[:, -(a + 1), :], logits[:, idx, :]))
        with self.assertRaises(ValueError):
            decision_position(p, a, p + a + 1)

    def test_permuted_id_tokens_choose_exact_user_and_target_not_history(self):
        # 2 users, 3 items, interleaved shuffled token positions.
        # Graph global node IDs for items = local item ID + num_users(2).
        batch = torch.tensor([1, 0, 0, 1, 0, 1, 0, 1])
        node = torch.tensor([3, 2, 0, 1, 4, 4, 3, 2])
        local = torch.tensor([3, 2, 0, 1, 4, 4, 3, 2])
        gcn = torch.tensor([
            [1., 0.], [0., 1.], [1., 0.], [0., 1.],
            [1., 1.], [-1., 0.], [0., -1.],
        ])
        users = torch.tensor([0, 1])
        targets = torch.tensor([2, 1])
        feats = build_decision_pair_features(
            batch, local, node, users, targets, 2, gcn
        )
        zu0, zi0 = F.normalize(gcn[0], dim=0), F.normalize(gcn[4], dim=0)
        zu1, zi1 = F.normalize(gcn[1], dim=0), F.normalize(gcn[3], dim=0)
        self.assertTrue(torch.allclose(feats[0], torch.cat((zu0, zi0, zu0 * zi0))))
        self.assertTrue(torch.allclose(feats[1], torch.cat((zu1, zi1, zu1 * zi1))))
        self.assertFalse(feats.requires_grad)

    def test_missing_target_is_not_silently_approximated_by_history(self):
        with self.assertRaisesRegex(ValueError, "missing/ambiguous"):
            build_decision_pair_features(
                torch.tensor([0, 0]), torch.tensor([0, 2]),
                torch.tensor([0, 3]), torch.tensor([0]),
                torch.tensor([2]), 2, torch.randn(4, 8)
            )

    def test_only_prompt_decision_query_position_changes(self):
        torch.manual_seed(7)
        arr = torch.randn(2, 9, 8, requires_grad=True)
        feats = torch.randn(2, 24)
        proj = torch.nn.Linear(24, 8, bias=False)
        torch.nn.init.zeros_(proj.weight)
        # D1-B zero-initialized: equals V-only at time zero.
        z = inject_decision_query(arr, proj(feats), position=6, scale=0.1)
        self.assertTrue(torch.equal(z, arr))
        with torch.no_grad():
            proj.weight.fill_(0.1)
        z = inject_decision_query(arr, proj(feats), position=6, scale=0.1)
        self.assertTrue(torch.equal(z[:, :6, :], arr[:, :6, :]))
        self.assertTrue(torch.equal(z[:, 7:, :], arr[:, 7:, :]))
        self.assertFalse(torch.equal(z[:, 6, :], arr[:, 6, :]))
        z[:, 6, :].sum().backward()
        self.assertIsNotNone(proj.weight.grad)
        self.assertGreater(proj.weight.grad.abs().sum().item(), 0)

    def test_projector_regularization_wraps_only_forward_v2(self):
        # Regression: adding a helper immediately before forward_v2 must not
        # accidentally move the decorator onto that helper.
        model_file = (Path(__file__).resolve().parents[1] /
                      "minigpt4/models/minigpt4rec_v2.py")
        module = ast.parse(model_file.read_text(encoding="utf-8"))
        classes = [n for n in module.body
                   if isinstance(n, ast.ClassDef) and n.name == "MiniGPT4Rec_v2"]
        self.assertEqual(len(classes), 1)
        methods = {n.name: n for n in classes[0].body
                   if isinstance(n, ast.FunctionDef)}
        self.assertIn("forward_v2", methods)
        self.assertIn("_d1_query_from_feedback", methods)
        decorated = lambda m: [
            d.id for d in methods[m].decorator_list if isinstance(d, ast.Name)
        ]
        self.assertIn("projector_regularized_forward", decorated("forward_v2"))
        self.assertNotIn("projector_regularized_forward",
                         decorated("_d1_query_from_feedback"))

    def test_shape_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            inject_decision_query(
                torch.zeros(2, 5, 8), torch.zeros(3, 8), 4, .1
            )


if __name__ == "__main__":
    unittest.main()
