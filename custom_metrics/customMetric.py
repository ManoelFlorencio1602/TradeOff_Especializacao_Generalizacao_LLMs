from deepeval.test_case import LLMTestCase
from deepeval.metrics import BaseMetric

class customMetric(BaseMetric):
    def __init__(self):
        super().__init__()

    def measure(self, test_case: LLMTestCase):
        if test_case.actual_output == test_case.expected_output:
            score = 1.0
        else:
            score = 0.0
        self.score = score
        self.reason = "Exact match" if score == 1.0 else "Mismatch in output"
        return score