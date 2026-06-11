import unittest

from scripts.run_cps_dsv4pro_alignment import (
    judge_messages,
    qwen_rollout_messages,
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
            "optional_action": None,
        }
        text = str(judge_messages(self.sample, rollout))

        self.assertIn("add it", text)
        self.assertIn("add an edge", text)
        self.assertIn("only a judge", text)

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
            "optional_action": {"type": "edit_add", "object": "(1-2)"},
        }

        validate_qwen_rollout(valid)
        with self.assertRaises(ValueError):
            validate_qwen_rollout({"latent_states": {"goal": "add edge"}, "utterance": "go"})


if __name__ == "__main__":
    unittest.main()
