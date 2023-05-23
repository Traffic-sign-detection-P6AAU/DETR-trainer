import unittest
from evaluation.evaluate_test_data import prepare_for_coco_detection

class EvaluationTests(unittest.TestCase):
    def test_prepare_coco(self):
        def test():
            return []
        class Prediction:
            def __init__(self):
                self.items = test
        result = prepare_for_coco_detection(Prediction())
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()
