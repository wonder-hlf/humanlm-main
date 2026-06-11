import unittest

from scripts.cps_humanlm_utils import prefix_environment_state
from scripts.prepare_cps_humanlm_data import attach_submit_results, make_sample


def event(subject, verb, obj, turn=1, submit_result=None):
    row = {
        "attempt_no": 1,
        "turn_no": turn,
        "subject": subject,
        "verb": verb,
        "object": obj,
    }
    if submit_result:
        row["submit_result"] = submit_result
    return row


class PrepareCpsHumanlmDataTest(unittest.TestCase):
    def test_attach_submit_results_requires_aligned_streams(self):
        with self.assertRaises(ValueError):
            attach_submit_results([event("A", "says", "hi")], [])

    def test_environment_uses_only_prefix_submit_results(self):
        prefix = [
            event("A", "adds", "(1-2)"),
            event("T", "submits", "feedback", submit_result={"abs_error": 4}),
        ]
        state = prefix_environment_state(prefix)

        self.assertEqual(state["current_edges"], [[1, 2]])
        self.assertEqual(state["performance_so_far"]["submission_count"], 1)
        self.assertEqual(state["performance_so_far"]["best_cost_gap"], 4.0)

    def test_sample_does_not_see_future_outcomes_or_actions(self):
        bundle = {
            "team_no": 9,
            "learning_features": {"RLG": 100},
            "log_features": {"task_level": {"success": True}},
        }
        events = [
            event("A", "says", "try this", turn=1),
            event("A", "adds", "(1-2)", turn=1),
            event("B", "says", "okay", turn=2),
            event("T", "submits", "done", turn=3, submit_result={"abs_error": 0}),
        ]

        sample = make_sample(bundle, events, 0, 80)

        self.assertEqual(sample["environment_state"]["performance_so_far"]["submission_count"], 0)
        self.assertEqual(len(sample["ground_truth"]["actions_before_next_human_utterance"]), 1)
        self.assertNotIn("RLG", str(sample))
        self.assertNotIn("success", str(sample))

    def test_sample_exposes_six_humanlm_state_dimensions(self):
        sample = make_sample(
            {"team_no": 9},
            [event("A", "says", "hello")],
            0,
            80,
        )

        self.assertEqual(
            set(sample["state_dimensions"]),
            {"belief", "goal", "value", "stance", "emotion", "communication"},
        )
        self.assertEqual(sample["role_state"]["declared_role"], "not_available_in_bundle")
        self.assertFalse(sample["qwen_rollout_request"]["ground_truth_visible_to_qwen"])
        self.assertEqual(sample["judge_request"]["judge_role"], "score_qwen_rollout_only")


if __name__ == "__main__":
    unittest.main()
