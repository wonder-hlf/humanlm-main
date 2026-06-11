# CPS HumanLM Training Runbook

This repo keeps code only. Do not commit raw or derived human-participant data.

## 0. What Is Being Migrated

The response-only SFT pipeline below is a baseline. The HumanLM-style JUSThink
pipeline is built separately so that every target human utterance has:

- prefix-only student profile, dialogue/action history, environment state, and
  observed role capabilities;
- the six HumanLM state dimensions: belief, goal, value, stance, emotion, and
  communication, mapped to CPS interpretations;
- the ground-truth next utterance and the target student's actions before the
  next human utterance;
- computable action/alignment proxies and an LLM-judge request schema.

It never uses final `RLG`, final success, or final task-level statistics as
inputs. The JSON bundles do not reliably expose current cost-view/edit-view
assignment, so the pipeline records this limitation instead of inventing a
role.

Build a small HumanLM-style dataset:

```bash
python scripts/prepare_cps_humanlm_data.py \
  --input-dir /path/to/team_bundles_atc21s_full \
  --output-dir data/cps_humanlm/v1/20p \
  --max-samples-per-team 20 \
  --context-window 80
```

Run Qwen3-8B as the HumanLM policy model. Qwen generates both the six latent
states and the next utterance/optional action:

```bash
vllm serve /path/to/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --dtype auto \
  --host 127.0.0.1 \
  --port 8000
```

Then configure Qwen's OpenAI-compatible endpoint and DSV4Pro as judge only.
Keep the judge API key outside the repo and avoid typing it directly into shell
history:

```bash
export QWEN_BASE_URL='http://127.0.0.1:8000/v1'
export QWEN_MODEL='Qwen3-8B'

read -s DSV4PRO_API_KEY
export DSV4PRO_API_KEY
export DSV4PRO_BASE_URL='https://provider.example/v1'
export DSV4PRO_MODEL='provider-specific-dsv4pro-model-id'

python scripts/run_cps_dsv4pro_alignment.py \
  --input data/cps_humanlm/v1/20p/train.jsonl \
  --output data/cps_humanlm/v1/20p/train.dsv4pro_alignment.jsonl \
  --limit 5
```

Both endpoints must provide an OpenAI-compatible `/chat/completions` API. The
script first asks Qwen for `latent_states + utterance + optional_action`, then
passes that rollout and the human ground truth to DSV4Pro for scoring. DSV4Pro
is never asked to generate latent states or a response. Do not commit API keys
or alignment outputs containing human data.

Compute hard proxy and macro human-trajectory baselines:

```bash
python scripts/evaluate_cps_state_proxies.py \
  --input data/cps_humanlm/v1/20p/test.jsonl \
  --output data/cps_humanlm/v1/20p/test_proxy_summary.json

python scripts/evaluate_cps_macro_metrics.py \
  --input-dir /path/to/team_bundles_atc21s_full \
  --output data/cps_humanlm/v1/human_macro_metrics.json
```

`Qwen3-8B` is the policy model that generates latent states and response/action.
DSV4Pro is only the LLM judge. The current scripts construct rollouts and score
state alignment; wiring those scores into the HumanLM GRPO optimization loop is
the next training stage.

### Small-batch implementation route

Use this order before any full-data run:

1. Build `20 samples/team` with team-held-out train/validation/test splits.
2. Start the untrained or SFT-initialized Qwen3-8B policy through vLLM.
3. Ask Qwen to generate one structured rollout per sample:
   `six latent states + utterance + optional action`.
4. Validate the rollout schema. Invalid or incomplete Qwen output is rejected
   before it reaches the judge.
5. Ask DSV4Pro only to score Qwen's latent states against the human utterance
   and nearby action. Save per-dimension score, rationale, and overall score.
6. Inspect a very small judge batch first, such as five samples. Check for
   reward hacking, future-information assumptions, and inconsistent rationales.
7. Combine the DSV4Pro state-alignment score with computable action proxies.
8. Feed the combined reward into the HumanLM VERL/GRPO recipe to update
   Qwen3-8B. Start with the small train split and short rollouts.
9. Compare the trained policy against human held-out trajectories using task
   performance, behavior rhythm, language-action alignment, discourse
   distribution, and transition structure.
10. Expand to all samples only after the small-batch reward and rollout quality
    are stable.

Stages 1-6 and the human macro evaluation are implemented in this repository.
Stage 8, the JUSThink-specific reward function and GRPO launcher, is not yet
implemented. The existing `run_chuangzhi_cps_sft.sh` remains a response-only
SFT baseline and must not be described as HumanLM state-alignment training.

## 1. Build a Small SFT Dataset on Mac

The preparation script merges directly consecutive speech from the same
participant before sampling. Within the same logged `attempt_no` and `turn_no`,
it also joins short continuation fragments beginning with words such as `to`,
`of`, `and`, or `because`, even after a brief teammate interjection. This
handles fragments such as `to mount basel .` without combining every utterance
from a long task turn.

```bash
cd /Users/wonder-hlf/Desktop/humanlm-main
python scripts/prepare_cps_sft_data.py \
  --input-dir /Users/wonder-hlf/Desktop/CPS/score/team_bundles_atc21s_full \
  --output-dir data/cps_team_sft/sft/r_no_tag/20p \
  --max-samples-per-team 20 \
  --context-window 40
```

To inspect merged raw bundles separately:

```bash
python scripts/merge_cps_consecutive_utterances.py \
  --input-dir /Users/wonder-hlf/Desktop/CPS/score/team_bundles_atc21s_full \
  --output-dir local_data/team_bundles_atc21s_merged
```

Both `data/` and `local_data/` are ignored by Git because they contain human
participant data.

Output:

```text
data/cps_team_sft/sft/r_no_tag/20p/train.parquet
data/cps_team_sft/sft/r_no_tag/20p/val.parquet
data/cps_team_sft/sft/r_no_tag/20p/test.parquet
data/cps_team_sft/sft/r_no_tag/20p/manifest.json
```

The default split is team-held-out:

```text
train: 8 teams
val:   1 team
test:  1 team
```

The SFT prompt only uses information available up to the target utterance:
participant identity plus prior dialogue/action events. It deliberately excludes
`learning_features` such as `RLG`, task-level outcomes, attempt outcomes, and
other statistics computed after the target utterance, because those would leak
future information into training.

## 2. Push Code to GitHub

The `data/` directory is ignored by Git because it contains derived human data.

```bash
git add .gitignore scripts/prepare_cps_sft_data.py docs/cps_humanlm_training.md
git commit -m "Add CPS HumanLM SFT data preparation"
git push
```

## 3. Pull Code on Chuangzhi Server

```bash
git clone https://github.com/wonder-hlf/humanlm-main.git
cd humanlm-main
```

If the repo already exists:

```bash
cd humanlm-main
git pull
```

Based on the current Chuangzhi workspace file list, the repo already exists at:

```text
./humanlm-main
```

After pulling the latest code, go back to the workspace root and run:

```bash
cd ..
bash humanlm-main/scripts/check_chuangzhi_environment.sh
```

Send the output back if any path is unclear. It checks GPU, model directories, the training repo, Python packages, and whether the CPS raw data has been uploaded.

## 4. Copy or Generate the Small Dataset on Server

The current Chuangzhi workspace file list does not show `team_bundles_atc21s_full`, so upload this local folder to the server first:

```text
/Users/wonder-hlf/Desktop/CPS/score/team_bundles_atc21s_full
```

Recommended target path on Chuangzhi:

```text
./humanlm-main/local_data/team_bundles_atc21s_full
```

Then run the converter on Chuangzhi:

```bash
cd humanlm-main
python scripts/prepare_cps_sft_data.py \
  --input-dir local_data/team_bundles_atc21s_full \
  --output-dir data/cps_team_sft/sft/r_no_tag/20p \
  --max-samples-per-team 20 \
  --context-window 40
```

## 5. Download Required Training Assets

The current Chuangzhi workspace has the GitHub source code only. It does not contain the original HumanLM model or its Qwen3-8B base model.

The current Chuangzhi workspace file list does not show `verl-recipe-humanlm`, so clone it first:

```bash
git clone https://github.com/ehejin/verl-recipe-humanlm.git
```

Then enter the training repo:

```bash
cd verl-recipe-humanlm
git checkout 6a7dbd3f143fc0a9af599ed7a458fc503341f846
```

This pinned recipe commit does not contain the VERL core trainer package. Before training, verify whether the current Chuangzhi Python environment already provides it:

```bash
python3 -c "import verl; import verl.trainer.fsdp_sft_trainer; print(verl.__file__)"
```

If this command fails, stop and preserve the complete error output before installing a different VERL version. The HumanLM recipe may require a specific VERL revision, so do not silently use an arbitrary latest release.

Download the original base model separately:

```bash
cd /inspire/hdd/project/ai4education/qianhong-p-qianhong
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3-8B \
  --local-dir qwen_models/qwen/Qwen3-8B
```

The expected base-model path is:

```text
/inspire/hdd/project/ai4education/qianhong-p-qianhong/qwen_models/qwen/Qwen3-8B
```

## 6. Run SFT with Qwen3-8B

The local launcher calls the VERL SFT trainer directly. This avoids the pinned upstream script's cluster-specific `//llm_twin` paths and its outdated chat-template path, while training directly from the generated CPS parquet files:

```bash
cd /inspire/hdd/project/ai4education/qianhong-p-qianhong
bash humanlm-main/scripts/run_chuangzhi_cps_sft.sh
```

Default paths used by the launcher:

```text
Training repo: ./verl-recipe-humanlm
Base model:    ./qwen_models/qwen/Qwen3-8B
Dataset:       ./humanlm-main/data/cps_team_sft/sft/r_no_tag/20p
Outputs:       ./humanlm_outputs
GPUs:          0,1,2,3
```

### Submit on the shared A6000 Slurm cluster

The shared cluster only allows the `A6000:8` partition on `gpu6` and `gpu7`. Submit the smoke training job from the login node:

```bash
cd ~/workspace/hlf_workspace
mkdir -p logs
sbatch humanlm-main/scripts/train_cps_sft_a6000.slurm
```

Monitor:

```bash
squeue -u "$USER"
tail -f logs/hlf-cps-sft-smoke-<JOB_ID>.out
tail -f logs/hlf-cps-sft-smoke-<JOB_ID>.err
```

Cancel if needed:

```bash
scancel <JOB_ID>
```

For short environment debugging only:

```bash
srun -p A6000:8 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=64G \
  --gres=gpu:1 \
  --time=02:00:00 \
  --pty bash
```

Exit the interactive allocation promptly with `Ctrl-D`.

## 7. Expand After the Smoke Test

Generate a larger dataset by setting `--max-samples-per-team 0`:

```bash
python scripts/prepare_cps_sft_data.py \
  --input-dir /path/to/team_bundles_atc21s_full \
  --output-dir data/cps_team_sft/sft/r_no_tag/100p \
  --max-samples-per-team 0 \
  --context-window 40
```

Then run with the full dataset path:

```bash
DATA_DIR=/inspire/hdd/project/ai4education/qianhong-p-qianhong/humanlm-main/data/cps_team_sft/sft/r_no_tag/100p \
  bash humanlm-main/scripts/run_chuangzhi_cps_sft.sh
```
