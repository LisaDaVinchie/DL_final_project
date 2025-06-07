import unittest
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from losses import dice_coef

class TestDiceCoefficient(unittest.TestCase):
    def test_perfect_match(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 1]])
        y_pred = torch.tensor([[1, 0, 1], [0, 1, 1]])
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), 1.0, places=5)

    def test_no_overlap(self):
        y_true = torch.tensor([[1, 1, 0], [0, 1, 0]])
        y_pred = torch.tensor([[0, 0, 1], [1, 0, 1]])
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), 0.0, places=5)

    def test_partial_overlap(self):
        y_true = torch.tensor([[1, 0, 1], [0, 1, 1]])
        y_pred = torch.tensor([[0, 0, 1], [0, 1, 0]])
        # y_true sum = 4, y_pred sum = 2, intersection = 2
        expected = (2 * 2) / (4 + 2)
        self.assertAlmostEqual(dice_coef(y_true, y_pred).item(), expected, places=5)

if __name__ == '__main__':
    unittest.main()
