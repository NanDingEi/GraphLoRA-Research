"""CPU tests: priority projection over accumulation-window GNN gradients."""
import unittest
import torch
from minigpt4.tasks.g2_priority_grad import project_auxiliary, accumulate_


class G2PriorityGradTest(unittest.TestCase):
    def test_conflict_does_not_decrease_primary_first_order(self):
        rec = [torch.tensor([1., 0.]), torch.tensor([0., 1.])]
        aux = [torch.tensor([-2., 3.]), torch.tensor([0., -1.])]
        projected, conflict, dot = project_auxiliary(rec, aux)
        self.assertTrue(conflict)
        self.assertAlmostEqual(dot, -3.0)
        raw_dot = sum(float(torch.dot(r, a)) for r,a in zip(rec, aux))
        after_aux = [p-r for p,r in zip(projected, rec)]
        after_dot = sum(float(torch.dot(r, a)) for r,a in zip(rec, after_aux))
        self.assertLess(raw_dot, 0.)
        self.assertAlmostEqual(after_dot, 0., places=5)

    def test_nonconflict_unchanged(self):
        rec = [torch.tensor([1., 2.])]
        aux = [torch.tensor([3., 4.])]
        projected, conflict, _ = project_auxiliary(rec, aux)
        self.assertFalse(conflict)
        self.assertTrue(torch.equal(projected[0], rec[0] + aux[0]))

    def test_accumulate_microbatches(self):
        p = torch.nn.Parameter(torch.ones(2))
        acc = accumulate_(None, [torch.tensor([0.1, 0.2])], [p])
        acc = accumulate_(acc, [torch.tensor([0.3, 0.4])], [p])
        self.assertTrue(torch.allclose(acc[0], torch.tensor([0.4, 0.6])))

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):
            project_auxiliary([torch.tensor([1.])],
                              [torch.tensor([float("nan")])])


if __name__ == "__main__":
    unittest.main()
