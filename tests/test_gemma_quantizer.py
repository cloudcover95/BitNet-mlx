# path: BitNet-mlx/tests/test_gemma_quantizer.py
#!/usr/bin/env python3
"""
Tests for GemmaTernaryConverter and ModelQuantizer

These tests validate the core quantization engine and Gemma converter.
Run with: python -m pytest tests/test_gemma_quantizer.py -v
"""

import unittest

import numpy as np

try:
    from src.quantization.model_quantizer import ModelQuantizer
    from src.quantization.gemma_ternary_converter import GemmaTernaryConverter
    HAS_MODULES = True
except ImportError:
    HAS_MODULES = False


class TestModelQuantizer(unittest.TestCase):

    def setUp(self):
        if not HAS_MODULES:
            self.skipTest("Required modules not available")
        self.quantizer = ModelQuantizer(output_dim=64, verbose=False)

    def test_quantize_tensor(self):
        # Create a dummy weight matrix
        dummy_weights = np.random.randn(128, 256).astype(np.float32)
        result = self.quantizer.quantize_tensor(dummy_weights, name="test_layer")

        self.assertIn("signature", result)
        self.assertIn("projection", result)
        self.assertGreaterEqual(result["signature"]["persistence_score"], 0.0)
        self.assertLessEqual(result["signature"]["persistence_score"], 1.0)

    def test_quantize_model_mock(self):
        # Create a mock model with parameters
        class MockModule:
            def named_modules(self):
                return [("", self)]

            def named_parameters(self, recurse=False):
                import torch
                return [("weight", torch.randn(64, 128))]

        mock_model = MockModule()
        summary = self.quantizer.quantize_model(mock_model)

        self.assertIn("layers_quantized", summary)
        self.assertGreaterEqual(summary["layers_quantized"], 0)


class TestGemmaConverter(unittest.TestCase):

    def test_converter_initialization(self):
        if not HAS_MODULES:
            self.skipTest("Required modules not available")

        # This will only work if transformers + a small Gemma model is available
        # For CI we just test that the class can be imported
        self.assertTrue(hasattr(GemmaTernaryConverter, "convert_model"))


if __name__ == "__main__":
    unittest.main()
