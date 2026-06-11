# CPS HumanLM Training Runbook

This repo keeps code only. Do not commit raw or derived human-participant data.

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
