import unittest

from scripts.merge_cps_consecutive_utterances import (
    clean_utterance_text,
    merge_consecutive_utterances,
    review_key,
)


def event(subject, verb, text, attempt=1, turn=1):
    return {
        "attempt_no": attempt,
        "turn_no": turn,
        "subject": subject,
        "verb": verb,
        "object": text,
    }


class MergeConsecutiveUtterancesTest(unittest.TestCase):
    def test_merges_directly_consecutive_incomplete_speech(self):
        merged, stats = merge_consecutive_utterances(
            [
                event("A", "says", "go to mount bern and then"),
                event("A", "says", "mount basel ."),
            ]
        )

        self.assertEqual([row["object"] for row in merged], [
            "go to mount bern and then mount basel."
        ])
        self.assertEqual(stats["merged_fragments"], 1)
        self.assertEqual(
            merged[0]["merged_utterance_parts"],
            ["go to mount bern and then", "mount basel ."],
        )

    def test_does_not_merge_complete_consecutive_speech(self):
        merged, stats = merge_consecutive_utterances(
            [
                event("A", "says", "okay ."),
                event("A", "says", "i am doing this ."),
            ]
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(stats["merged_fragments"], 0)

    def test_merges_short_continuation_after_interjection(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "go from mount bern ."),
                event("A", "says", "four ."),
                event("B", "says", "to mount basel ."),
            ]
        )

        self.assertEqual([row["object"] for row in merged], [
            "go from mount bern. to mount basel.",
            "four.",
        ])

    def test_does_not_merge_complete_response_after_interjection(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "go from mount bern ."),
                event("A", "says", "four ."),
                event("B", "says", "yes that works ."),
            ]
        )

        self.assertEqual([row["object"] for row in merged], [
            "go from mount bern.",
            "four.",
            "yes that works.",
        ])

    def test_does_not_merge_complete_sentence_starting_with_then(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "yeah i do ."),
                event("B", "says", "then you see what i did ?"),
            ]
        )

        self.assertEqual(len(merged), 2)

    def test_new_task_turn_ends_merge(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "go from mount bern .", turn=1),
                event("B", "says", "to mount basel .", turn=2),
            ]
        )

        self.assertEqual(len(merged), 2)

    def test_cleans_spaces_before_punctuation(self):
        self.assertEqual(
            clean_utterance_text("i think do we connect it , to all places ?"),
            "i think do we connect it, to all places?",
        )

    def test_marks_incomplete_fragment_for_training_exclusion(self):
        merged, stats = merge_consecutive_utterances(
            [event("A", "says", "we should go to")]
        )

        self.assertTrue(merged[0]["is_incomplete_fragment"])
        self.assertEqual(stats["incomplete_fragments"], 1)

    def test_human_review_hash_marks_only_reviewed_merge(self):
        events = [
            event("A", "says", "we should"),
            event("A", "says", "go."),
        ]
        expected_parts = ["we should", "go."]
        flag = review_key(
            7,
            {
                "attempt_no": 1,
                "turn_no": 1,
                "subject": "A",
                "merged_utterance_parts": expected_parts,
            },
        )

        merged, _ = merge_consecutive_utterances(events, team_no=7, review_flags={flag})

        self.assertEqual(merged[0]["human_review_status"], "needs_review")
        self.assertTrue(merged[0]["is_incomplete_fragment"])


if __name__ == "__main__":
    unittest.main()
