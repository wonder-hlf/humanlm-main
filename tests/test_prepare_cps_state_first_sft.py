import json
import unittest

from scripts.prepare_cps_state_first_sft import (
    candidate_score,
    ground_truth_action_intent,
    make_state_first_row,
    select_candidates,
)


def candidate(index, overall, goal):
    return {
        "sample_key": "1:A:1:2:3",
        "candidate_index": index,
        "qwen_rollout": {
            "latent_states": {
                "belief": "b",
                "goal": goal,
                "value": "v",
                "stance": "s",
                "emotion": "e",
                "communication": "c",
            },
            "utterance": "generated utterance must not be used",
            "action_intent": {"type": "submit"},
        },
        "state_alignment": {
            "overall_state_alignment": overall,
            "dimension_scores": {
                "strategy_goal": {"score": overall, "rationale": "test"}
            },
        },
    }


class PrepareCpsStateFirstSftTest(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "team_no": 1,
            "participant": "A",
            "attempt_no": 1,
            "turn_no": 2,
            "source_index": 3,
            "student_profile": {},
            "role_state": {},
            "environment_state": {},
            "dialogue_and_action_history": [],
            "state_dimensions": {},
            "optional_steering_config": None,
            "ground_truth": {
                "utterance": "human truth",
                "actions_before_next_human_utterance": [
                    {"subject": "A", "verb": "adds", "object": "(1-2)"}
                ],
            },
        }

    def test_selects_highest_judged_candidate_above_threshold(self):
        rows = [candidate(0, 0.4, "low"), candidate(1, 0.8, "high")]

        selected = select_candidates(rows, minimum_score=0.5)

        self.assertEqual(selected["1:A:1:2:3"]["candidate_index"], 1)
        self.assertEqual(candidate_score(selected["1:A:1:2:3"])[0], 0.8)

    def test_target_uses_selected_states_but_human_utterance_and_action(self):
        row = make_state_first_row(self.sample, candidate(1, 0.8, "high"))
        target = json.loads(row["generation"])

        self.assertEqual(target["latent_states"]["goal"], "high")
        self.assertEqual(target["utterance"], "human truth")
        self.assertEqual(target["action_intent"], {"type": "add_track", "track": "(1-2)"})
        self.assertNotIn("generated utterance must not be used", row["generation"])

    def test_ground_truth_action_intent_can_be_null(self):
        self.assertIsNone(ground_truth_action_intent([]))


if __name__ == "__main__":
    unittest.main()
