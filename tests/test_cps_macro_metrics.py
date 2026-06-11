import unittest

from scripts.evaluate_cps_macro_metrics import summarize_bundle


def event(subject, verb, obj, turn, matching=None, submit_result=None):
    row = {
        "attempt_no": 1,
        "turn_no": turn,
        "subject": subject,
        "verb": verb,
        "object": obj,
    }
    if matching:
        row["matching"] = matching
    if submit_result:
        row["submit_result"] = submit_result
    return row


class CpsMacroMetricsTest(unittest.TestCase):
    def test_summarizes_prefix_independent_human_trajectory(self):
        corpus = [
            event("A", "says", "add it", 1),
            event("B", "adds", "(1-2)", 2),
            event("T", "submits", "feedback", 3, submit_result={"abs_error": 2}),
            event("B", "adds", "(2-3)", 4),
        ]
        annotated = [
            event("A", "says", "add it", 1),
            event("B", "adds", "(1-2)", 2, matching="MISMATCH_1"),
            event("T", "submits", "feedback", 3),
            event("B", "adds", "(2-3)", 4, matching="MATCH_1"),
        ]
        row = summarize_bundle(
            {"team_no": 1, "corpus": corpus, "annotated_corpus": annotated},
            repair_window=3,
        )

        self.assertEqual(row["task_performance"]["best_cost_gap"], 2.0)
        self.assertEqual(row["language_action_alignment"]["mismatch_repair_probability"], 1.0)
        self.assertEqual(row["behavior_rhythm"]["action_counts"]["edit_add"], 2)
        self.assertEqual(row["behavior_rhythm"]["discourse_counts"]["route_instruction"], 1)
        self.assertEqual(row["behavior_rhythm"]["turn_count"], 4)
        self.assertEqual(row["behavior_rhythm"]["first_submit_relative_position"], 0.6667)
        self.assertEqual(row["behavior_rhythm"]["first_match_relative_position"], 1.0)


if __name__ == "__main__":
    unittest.main()
