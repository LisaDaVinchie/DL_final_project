import torch as th
from torch.optim import Adam
import unittest
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from train import TrainModel
from model import SimpleModel
from losses import PerPixelL1

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        self.model = SimpleModel()
        self.loss_function = PerPixelL1()
        self.optimizer = Adam(self.model.parameters(), lr=0.001)
        self.train_model = TrainModel(self.model, self.loss_function, self.optimizer, 
                                      weights_path="test_weights.pth", results_path="test_results",
                                      save_every=1)

    def test_train_model_initialization(self):
        """Test if the TrainModel class initializes correctly."""
        self.assertIsNotNone(self.train_model)
        self.assertEqual(len(self.train_model.train_losses), 0)
        self.assertEqual(len(self.train_model.test_losses), 0)