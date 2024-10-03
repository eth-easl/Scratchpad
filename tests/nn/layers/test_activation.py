import unittest
import torch
from scratchpad.nn.layers.activation import ScaledActivation

import torch.nn as nn

class TestScaledActivation(unittest.TestCase):
    def setUp(self):
        self.intermediate_size = 8
        self.input_is_parallel = False
        self.params_dtype = torch.float32
        self.act_module = nn.ReLU()
        self.scaled_activation = ScaledActivation(
            self.act_module,
            self.intermediate_size,
            self.input_is_parallel,
            self.params_dtype
        )
        self.input_tensor = torch.randn(2, self.intermediate_size)

    def test_initialization(self):
        self.assertEqual(self.scaled_activation.act, self.act_module)
        self.assertEqual(self.scaled_activation.input_is_parallel, self.input_is_parallel)
        self.assertEqual(self.scaled_activation.scales.shape[0], self.intermediate_size)
        self.assertEqual(self.scaled_activation.scales.dtype, self.params_dtype)

    def test_forward(self):
        output = self.scaled_activation(self.input_tensor)
        expected_output = self.act_module(self.input_tensor) / self.scaled_activation.scales
        self.assertTrue(torch.allclose(output, expected_output, atol=1e-6))

    def test_weight_loader(self):
        loaded_weight = torch.randn(self.intermediate_size, dtype=self.params_dtype)
        self.scaled_activation.weight_loader(self.scaled_activation.scales, loaded_weight)
        self.assertTrue(torch.allclose(self.scaled_activation.scales.data, loaded_weight, atol=1e-6))

if __name__ == "__main__":
    unittest.main()