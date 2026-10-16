import unittest
import numpy as np
from typing import List

def generate_data(seed: int, num_samples: int, num_labels: int) -> List[int]:
    """Simulate a deterministic data generation process."""
    np.random.seed(seed)
    return np.random.randint(0, num_labels, size=num_samples).tolist()

class TestData(unittest.TestCase):
    def test_deterministic_generation(self):
        """Test that the same seed produces the same data."""
        data1 = generate_data(42, 10, 5)
        data2 = generate_data(42, 10, 5)
        self.assertEqual(data1, data2)
        
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        data1 = generate_data(42, 10, 5)
        data2 = generate_data(43, 10, 5)
        self.assertNotEqual(data1, data2)

    def test_data_range(self):
        """Test that generated data is within the expected range."""
        num_labels = 5
        data = generate_data(123, 100, num_labels)
        for val in data:
            self.assertTrue(0 <= val < num_labels)

if __name__ == "__main__":
    unittest.main()
