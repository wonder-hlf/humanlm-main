# Training Two: State-First SFT

Training two is a small-sample, non-RL experiment. Both latent-state candidate
generation and state-first SFT start from the original Qwen3-8B base model.
Training one's response-only SFT model is not used.

## Data Flow

For each leakage-safe target turn:

1. Qwen receives only the student profile, prefix dialogue/actions, current
   environment state, current role/view state, and state definitions.
2. Qwen samples multiple candidate latent-state objects.
3. DSV4Pro receives each candidate plus the ground-truth human utterance and
   nearby human action, and scores state alignment.
4. The highest-scoring candidate above the threshold is selected.
5. The SFT target is built from:
   - selected Qwen latent states;
   - ground-truth human utterance;
   - ground-truth nearby human action intent.

Qwen-generated utterances and actions are never used as SFT targets. Final
RLG, final success, and events after the target point are never model inputs.

## Main Artifacts

- Leakage-safe source data: `data/cps_humanlm/v1/20p`
- Raw Qwen candidates: `outputs/cps_state_first_raw_candidates_base_qwen/20p`
- Candidate and judge records: `outputs/cps_state_first_candidates_base_qwen/20p`
- State-first SFT data: `data/cps_state_first_sft_base_qwen/20p`
- Training output: `humanlm_outputs/cps_qwen3_8b_base_state_first_sft_smoke`

On clusters where compute nodes cannot access the internet, run the pipeline in
two stages:

1. `generate_cps_state_candidates_a6000.slurm` generates candidates on a GPU
   compute node without calling the judge.
2. `judge_and_build_cps_state_first_sft.sh` runs on the internet-connected
   login node, calls DSV4Pro, selects candidates, and writes SFT parquet files.

Both stages are append-safe. Re-running them skips successful work.

## Important Limitation

The selected latent states are Judge-selected pseudo-labels, not observed
human mental-state labels. This experiment tests whether explicit state-first
supervision improves simulation behavior. It is not the HumanLM GRPO stage.
