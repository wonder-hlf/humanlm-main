import unittest

from scripts.merge_cps_consecutive_utterances import merge_consecutive_utterances


def event(subject, verb, text, attempt=1, turn=1):
    return {
        "attempt_no": attempt,
        "turn_no": turn,
        "subject": subject,
        "verb": verb,
        "object": text,
    }


class MergeConsecutiveUtterancesTest(unittest.TestCase):
    def test_merges_directly_consecutive_speech(self):
        merged, stats = merge_consecutive_utterances(
            [
                event("A", "says", "go to mount bern ."),
                event("A", "says", "then mount basel ."),
            ]
        )

        self.assertEqual([row["object"] for row in merged], [
            "go to mount bern . then mount basel ."
        ])
        self.assertEqual(stats["merged_fragments"], 1)

    def test_merges_short_continuation_after_interjection(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "go from mount bern ."),
                event("A", "says", "four ."),
                event("B", "says", "to mount basel ."),
            ]
        )

        self.assertEqual([row["object"] for row in merged], [
            "go from mount bern . to mount basel .",
            "four .",
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
            "go from mount bern .",
            "four .",
            "yes that works .",
        ])

    def test_new_task_turn_ends_merge(self):
        merged, _ = merge_consecutive_utterances(
            [
                event("B", "says", "go from mount bern .", turn=1),
                event("B", "says", "to mount basel .", turn=2),
            ]
        )

        self.assertEqual(len(merged), 2)


if __name__ == "__main__":
    unittest.main()
