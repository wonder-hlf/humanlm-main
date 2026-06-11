import tempfile
import unittest
from pathlib import Path

from scripts.run_training_two import build_commands, load_server_config


class RunTrainingTwoTest(unittest.TestCase):
    def test_loads_server_only_dsv4pro_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dsv4pro_config.py"
            path.write_text(
                'DSV4PRO_API_KEY="key"\n'
                'DSV4PRO_BASE_URL="https://example.test"\n'
                'DSV4PRO_MODEL="judge"\n'
                'REASONING_EFFORT="high"\n'
                'THINKING_ENABLED=True\n'
            )

            config = load_server_config(path)

            self.assertEqual(config["DSV4PRO_MODEL"], "judge")
            self.assertEqual(config["DSV4PRO_REASONING_EFFORT"], "high")
            self.assertEqual(config["DSV4PRO_THINKING_ENABLED"], "true")

    def test_commands_use_base_qwen_pipeline_and_wait_for_slurm(self):
        commands = build_commands(
            workspace=Path("/workspace"),
            candidates_per_sample=3,
            minimum_score=0.5,
            limit=0,
        )

        self.assertEqual(commands[1][0:2], ["sbatch", "--wait"])
        self.assertIn("generate_cps_state_candidates_a6000.slurm", commands[1][-1])
        self.assertIn("judge_and_build_cps_state_first_sft.sh", commands[2][-1])
        self.assertEqual(commands[3][0:2], ["sbatch", "--wait"])


if __name__ == "__main__":
    unittest.main()
