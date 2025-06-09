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
        mask = torch.zeros_like(y_true, dtype=torch.bool)  # No mask, all pixels are valid
        intersection, union = dice_coef(y_true, y_pred, mask)
        dice = (2 * intersection) / (union + 1e-7)  # Adding a small value to avoid division by zero
        self.assertAlmostEqual(dice.item(), 1.0, places=5)

    def test_no_overlap(self):
        y_true = torch.tensor([[1, 1, 0], [0, 1, 0]])
        y_pred = torch.tensor([[0, 0, 1], [1, 0, 1]])
        mask = torch.zeros_like(y_true, dtype=torch.bool)  # No mask, all pixels are valid
        intersection, union = dice_coef(y_true, y_pred, mask)
        dice = (2 * intersection) / (union + 1e-7)  # Adding a small value to avoid division by zero
        self.assertAlmostEqual(dice.item(), 0.0, places=5)

    def test_partial_overlap(self):
        y_true = torch.tensor([[1, 0, 1],
                               [0, 1, 1]])
        y_pred = torch.tensor([[0, 0, 1],
                               [0, 1, 0]])
        mask = torch.zeros_like(y_true, dtype=torch.bool)
        # y_true sum = 4, y_pred sum = 2, intersection = 2
        expected_intersection = 2.0
        expected_union = (4 + 2)
        intersection, union = dice_coef(y_true, y_pred, mask)
        self.assertAlmostEqual(intersection.item(), expected_intersection, places=5)
        self.assertAlmostEqual(union.item(), expected_union, places=5)
    
    def test_partial_overlap_and_mask(self):
        y_true = torch.tensor([[1, 0, 1],
                               [0, 1, 1]])
        y_pred = torch.tensor([[0, 0, 1],
                               [0, 1, 0]])
        mask = torch.tensor([[1, 0, 1],
                             [0, 1, 0]], dtype=torch.bool)
        # The mask will ignore the second column of the first row and the first column of the second row
        # y_true sum = 4, y_pred sum = 2, intersection = 2
        expected_intersection = 0
        expected_union = (1 + 0)
        intersection, union = dice_coef(y_true, y_pred, mask)
        self.assertAlmostEqual(intersection.item(), expected_intersection, places=5)
        self.assertAlmostEqual(union.item(), expected_union, places=5)

if __name__ == '__main__':
    unittest.main()
