import unittest

from analysis.compare_state_alignment import compare


def row(key, overall, dimension):
    return {
        "sample_key": key,
        "state_alignment": {
            "overall_state_alignment": overall,
            "dimension_scores": {"goal": {"score": dimension}},
        },
    }


class CompareStateAlignmentTest(unittest.TestCase):
    def test_compares_only_shared_successful_samples(self):
        result = compare(
            {"a": row("a", 0.4, 0.5), "b": row("b", 0.8, 0.7)},
            {"a": row("a", 0.6, 0.4), "c": row("c", 0.9, 0.9)},
        )

        self.assertEqual(result["shared_successful_samples"], 1)
        self.assertEqual(result["overall"]["mean_delta"], 0.2)
        self.assertEqual(result["overall"]["trained_wins"], 1)
        self.assertEqual(result["dimensions"]["goal"]["base_wins"], 1)


if __name__ == "__main__":
    unittest.main()
