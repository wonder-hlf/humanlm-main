import unittest

from scripts.prepare_cps_sft_data import make_persona, make_sample


class PrepareCpsSftDataTest(unittest.TestCase):
    def test_persona_does_not_include_future_outcome_features(self):
        persona = make_persona(7, "A")

        self.assertNotIn("RLG", persona)
        self.assertNotIn("success", persona)
        self.assertNotIn("error", persona)
        self.assertNotIn("duration", persona)

    def test_sample_does_not_leak_bundle_outcome_features(self):
        bundle = {
            "team_no": 7,
            "learning_features": {"RLG": 0.9, "A_RLG": 1.2},
            "log_features": {
                "task_level": {"success": True, "min_error": 0.0},
            },
            "annotated_corpus": [
                {
                    "attempt_no": 1,
                    "turn_no": 1,
                    "subject": "A",
                    "verb": "says",
                    "object": "hello .",
                }
            ],
        }

        sample = make_sample(bundle, 0, 40)
        prompt_text = "\n".join(message["content"] for message in sample["prompt"])

        self.assertNotIn("RLG", prompt_text)
        self.assertNotIn("min_error", prompt_text)
        self.assertNotIn("success", prompt_text)


if __name__ == "__main__":
    unittest.main()
