import unittest
from configs import defaultConfig, selfplayConfig, trainingConfig, modelConfig, _ints, _floats, _bools

class TestConfigs(unittest.TestCase):
    def test_value_types(self):
        groupConfig = defaultConfig
        groupConfig.update(selfplayConfig)
        groupConfig.update(trainingConfig)
        groupConfig.update(modelConfig)

        for k, v in groupConfig.items():
            if k in _ints:
                self.assertIsInstance(v, int)
            elif k in _floats:
                self.assertIsInstance(v, float)
            elif k in _bools:
                self.assertIsInstance(v, bool)

        self.assertTrue(defaultConfig["model_size"] in ["sm", "md", "lg"])
        return
        


if __name__ == '__main__':
    unittest.main()