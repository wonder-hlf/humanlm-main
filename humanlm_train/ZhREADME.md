### 子模块设置

```bash
git submodule update --init --recursive
```

### 将子模块更新到最新提交

```bash
cd humanlm_train/verl-recipe-humanlm
git checkout humanlm
git pull origin humanlm
```

如需开始训练，请按照 `verl-recipe-humanlm/humanlm/README.md` 中 humanlm 分支上的说明操作。
