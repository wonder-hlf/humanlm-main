import unittest

from scripts.run_cps_dsv4pro_alignment import judge_messages, latent_messages


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

    def test_latent_generation_does_not_receive_ground_truth(self):
        text = str(latent_messages(self.sample))

        self.assertNotIn("add it", text)
        self.assertNotIn("ground_truth", text)

    def test_judge_receives_ground_truth_and_generated_states(self):
        text = str(judge_messages(self.sample, {"goal": "add an edge"}))

        self.assertIn("add it", text)
        self.assertIn("add an edge", text)


if __name__ == "__main__":
    unittest.main()
