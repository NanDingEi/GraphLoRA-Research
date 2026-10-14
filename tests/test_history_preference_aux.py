"""CPU checks: python -m unittest tests/test_history_preference_aux.py -v"""
import unittest
import torch
from minigpt4.models.history_preference_aux import history_target_preference_loss


class HistoryPreferenceTest(unittest.TestCase):
    def setUp(self):
        # num_users=100, IDs in an intentionally shuffled token order.
        # Row0 has user 5, target item 15 and two history items 1/2.
        # Row1 has user 7, target item 17 and one history item 3.
        self.ids = torch.tensor([115, 5, 101, 102, 103, 117, 7])
        self.groups = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        self.targets = torch.tensor([15, 17])
        self.labels = torch.tensor([1, 0])
        self.z = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0],
            [0.0, 1.0], [1.0, 0.0], [1.0, 0.0],
        ], requires_grad=True)

    def test_binary_ctr_gradient_and_nonlast_target(self):
        loss, active, total = history_target_preference_loss(
            self.z, self.ids, self.groups, self.targets, self.labels,
            num_users=100, similarity_scale=4.)
        self.assertEqual((active, total), (2, 2))
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss.detach()), 0.50)
        loss.backward()
        self.assertIsNotNone(self.z.grad)
        self.assertGreater(float(self.z.grad.abs().sum()), 0)

    def test_missing_history_is_ignored_without_nan(self):
        ids = torch.tensor([3, 115, 7, 117])
        rows = torch.tensor([0, 0, 1, 1])
        z = torch.randn(4, 8, requires_grad=True)
        loss, active, total = history_target_preference_loss(
            z, ids, rows, self.targets, self.labels, 100)
        self.assertEqual((active, total), (0, 2))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(z.grad, torch.zeros_like(z)))

    def test_padding_is_not_counted(self):
        # 99 is a padding-like ID; however the GNN feedback only presents
        # genuine prompt IDs. The test excludes target duplicates as history.
        ids = torch.tensor([5, 115, 115])
        rows = torch.zeros(3, dtype=torch.long)
        emb = torch.randn(3, 8)
        loss, active, total = history_target_preference_loss(
            emb, ids, rows, torch.tensor([15]), torch.tensor([1]), 100)
        self.assertEqual((active, total), (0, 1))
        self.assertEqual(float(loss), 0.)

    def test_invalid_dimensions_and_scale(self):
        with self.assertRaises(ValueError):
            history_target_preference_loss(
                self.z, self.ids, self.groups, self.targets,
                self.labels, 100, similarity_scale=0.0)
        with self.assertRaises(ValueError):
            history_target_preference_loss(
                self.z, self.ids[:-1], self.groups, self.targets,
                self.labels, 100)


if __name__ == "__main__":
    unittest.main()
