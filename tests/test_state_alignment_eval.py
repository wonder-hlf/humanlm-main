import unittest

from analysis.state_alignment_eval import summarize


class StateAlignmentEvalTest(unittest.TestCase):
    def test_summarizes_scores_and_state_errors(self):
        result = summarize(
            [
                {
                    "state_alignment": {
                        "dimension_scores": {
                            "task_understanding": {"score": 0.75, "rationale": "supported"}
                        },
                        "overall_state_alignment": 0.5,
                        "missing_state": ["cost uncertainty"],
                        "redundant_unsupported_state": ["confidence"],
                    }
                }
            ]
        )

        self.assertEqual(result["overall_state_alignment"]["mean"], 0.5)
        self.assertEqual(result["dimension_scores"]["task_understanding"]["mean"], 0.75)
        self.assertEqual(result["top_missing_states"], [("cost uncertainty", 1)])


if __name__ == "__main__":
    unittest.main()
