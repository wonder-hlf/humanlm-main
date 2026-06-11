import unittest

from scripts.judge_cps_state_candidates import load_jsonl


class JudgeCpsStateCandidatesTest(unittest.TestCase):
    def test_module_exposes_jsonl_loader(self):
        self.assertTrue(callable(load_jsonl))


if __name__ == "__main__":
    unittest.main()
