import unittest

from scripts.run_cps_dsv4pro_alignment import (
    judge_messages,
    qwen_rollout_messages,
    sample_key,
    validate_judge_output,
    validate_qwen_rollout,
)


class CpsDsv4ProAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "student_profile": {"source": "prefix_behavior_only"},
            "role_state": {"declared_role": "not_available_in_bundle"},
            "environment_state": {"performance_so_far": {"submission_count": 0}},
            "dialogue_and_action_history": [{"subject": "A", "verb": "says", "object": "hi"}],
            "state_dimensions": {"belief": {"cps_name": "task_understanding"}},
            "ground_truth": {"utterance": "add it", "actions_before_next_human_utterance": []},
            "computable_state_proxies": {},
        }

    def test_qwen_generates_states_and_response_without_ground_truth(self):
        text = str(qwen_rollout_messages(self.sample))

        self.assertNotIn("add it", text)
        self.assertNotIn("ground_truth", text)
        self.assertIn("latent_states", text)
        self.assertIn("utterance", text)

    def test_dsv4pro_judge_receives_ground_truth_and_qwen_rollout(self):
        rollout = {
            "latent_states": {"goal": "add an edge"},
            "utterance": "let us add it",
            "action_intent": None,
        }
        text = str(judge_messages(self.sample, rollout))

        self.assertIn("add it", text)
        self.assertIn("add an edge", text)
        self.assertIn("only a state-alignment judge", text)

    def test_qwen_rollout_requires_all_six_states_and_response(self):
        valid = {
            "latent_states": {
                "belief": "edge is missing",
                "goal": "add edge",
                "value": "low cost",
                "stance": "agree",
                "emotion": "confident",
                "communication": "instruction",
            },
            "utterance": "add that edge",
            "action_intent": {"type": "add_track", "track": ["Bern", "Zurich"]},
        }

        validate_qwen_rollout(valid)
        with self.assertRaises(ValueError):
            validate_qwen_rollout({"latent_states": {"goal": "add edge"}, "utterance": "go"})

    def test_judge_requires_scores_missing_and_unsupported_states(self):
        valid = {
            "dimension_scores": {
                key: {"score": 0.5, "rationale": "partly supported"}
                for key in (
                    "task_understanding",
                    "strategy_goal",
                    "collaboration_value",
                    "interaction_stance",
                    "error_repair_state",
                    "communication_style",
                )
            },
            "overall_state_alignment": 0.5,
            "missing_state": ["cost uncertainty"],
            "redundant_unsupported_state": [],
        }

        validate_judge_output(valid)
        with self.assertRaises(ValueError):
            validate_judge_output({"dimension_scores": {}, "overall_state_alignment": 2})

    def test_sample_key_identifies_one_human_target(self):
        sample = {
            "team_no": 2,
            "participant": "B",
            "attempt_no": 3,
            "turn_no": 4,
            "source_index": 5,
        }

        self.assertEqual(sample_key(sample), "2:B:3:4:5")


if __name__ == "__main__":
    unittest.main()
