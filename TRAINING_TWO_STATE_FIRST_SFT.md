# Training Two: State-First SFT

Training two is a small-sample, non-RL experiment. It starts from the merged
response-only SFT model produced by training one.

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
- Candidate and judge records: `outputs/cps_state_first_candidates/20p`
- State-first SFT data: `data/cps_state_first_sft/20p`
- Training output: `humanlm_outputs/cps_qwen3_8b_state_first_sft_smoke`

The candidate builder is append-safe. Re-running it skips successful
candidate/judge pairs and retries failed pairs with a different sampling seed.

## Important Limitation

The selected latent states are Judge-selected pseudo-labels, not observed
human mental-state labels. This experiment tests whether explicit state-first
supervision improves simulation behavior. It is not the HumanLM GRPO stage.
