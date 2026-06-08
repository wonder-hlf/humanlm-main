# CPS HumanLM Training Runbook

This repo keeps code only. Do not commit raw or derived human-participant data.

## 1. Build a Small SFT Dataset on Mac

```bash
cd /Users/wonder-hlf/Desktop/humanlm-main
python scripts/prepare_cps_sft_data.py \
  --input-dir /Users/wonder-hlf/Desktop/CPS/score/team_bundles_atc21s_full \
  --output-dir data/cps_team_sft/sft/r_no_tag/20p \
  --max-samples-per-team 20 \
  --context-window 40
```

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

## 5. Put Data Where the Training Script Expects It

The pinned training script hardcodes its processed data root. For the first smoke test, use `enron` as a harmless placeholder dataset name:

```bash
mkdir -p //llm_twin/processed_data/enron_processed_dataset_by_post_dedup/sft/r_no_tag/20p
cp data/cps_team_sft/sft/r_no_tag/20p/*.parquet \
  //llm_twin/processed_data/enron_processed_dataset_by_post_dedup/sft/r_no_tag/20p/
```

If `//llm_twin/processed_data` does not exist on Chuangzhi, use the actual shared data path there and patch `humanlm/train_sft_humanlm.sh` in the training repo.

## 6. Run SFT with Local Qwen3-8B

The current Chuangzhi workspace file list does not show `verl-recipe-humanlm`, so clone it first:

```bash
git clone https://github.com/ehejin/verl-recipe-humanlm.git
```

Then enter the training repo:

```bash
cd verl-recipe-humanlm
git checkout 6a7dbd3f143fc0a9af599ed7a458fc503341f846
```

The file list shows this existing model path:

```text
./qwen_models/qwen/Qwen2.5-7B-Instruct
```

It does not show Qwen3-8B. If Qwen3-8B is installed elsewhere, use its real absolute path in the command below.

Run a small smoke test:

```bash
bash humanlm/train_sft_humanlm.sh \
  "0,1,2,3" \
  enron \
  no_thinking \
  20 \
  /path/to/local/Qwen3-8B \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=20 \
  trainer.test_freq=10
```

Replace:

```text
/path/to/local/Qwen3-8B
```

with the real Qwen3-8B path on Chuangzhi. If you only want to smoke-test the pipeline before locating Qwen3-8B, you can temporarily use:

```text
../qwen_models/qwen/Qwen2.5-7B-Instruct
```

## 7. Expand After the Smoke Test

Generate a larger dataset by setting `--max-samples-per-team 0`:

```bash
python scripts/prepare_cps_sft_data.py \
  --input-dir /path/to/team_bundles_atc21s_full \
  --output-dir data/cps_team_sft/sft/r_no_tag/100p \
  --max-samples-per-team 0 \
  --context-window 40
```

Then copy it to:

```text
//llm_twin/processed_data/enron_processed_dataset_by_post_dedup/sft/r_no_tag/100p/
```

and run:

```bash
bash humanlm/train_sft_humanlm.sh "0,1,2,3" enron no_thinking 100 /path/to/local/Qwen3-8B
```
