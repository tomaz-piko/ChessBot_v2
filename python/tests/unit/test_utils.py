import unittest
from utils import convert_u64_to_np

class TestGameImage(unittest.TestCase):
    def test_image_single_item(self):
        fake_image = [i for i in range(117)]
        result = convert_u64_to_np(fake_image)
        self.assertEqual(result.shape, (1, 117, 8, 8))

    def test_image_batch(self):
        fake_images = [[i for i in range(117)] for _ in range(10)]
        result = convert_u64_to_np(fake_images)
        self.assertEqual(result.shape, (10, 117, 8, 8))

if __name__ == '__main__':
    unittest.main()