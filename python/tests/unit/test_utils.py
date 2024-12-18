import unittest
from utils import convert_u64_to_np

class TestUtils(unittest.TestCase):
    def test_image_single_item(self):
        fake_image = [i for i in range(109)]
        result = convert_u64_to_np(fake_image)
        self.assertEqual(result.shape, (1, 109, 8, 8))

    def test_image_batch(self):
        fake_images = [[i for i in range(109)] for _ in range(10)]
        result = convert_u64_to_np(fake_images)
        self.assertEqual(result.shape, (10, 109, 8, 8))

if __name__ == '__main__':
    unittest.main()