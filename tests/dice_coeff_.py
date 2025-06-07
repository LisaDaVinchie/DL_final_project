import unittest
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from losses import dice_coef

class TestDiceCoefficient(unittest.TestCase):
    def test_perfect_match(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        y_pred = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), 1.0, places=5)

    def test_no_overlap(self):
        y_true = torch.tensor([[1, 1, 0], [0, 1, 0]], dtype=torch.float32)
        y_pred = torch.tensor([[0, 0, 1], [1, 0, 1]], dtype=torch.float32)
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), 0.0, places=5)

    def test_partial_overlap(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        y_pred = torch.tensor([[0, 0, 1], [0, 1, 1]], dtype=torch.float32)
        expected = (2 * 3) / (5 + 4)  # 3 intersection, 5 in y_true, 4 in y_pred
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), expected, places=5)

if __name__ == '__main__':
    unittest.main()
