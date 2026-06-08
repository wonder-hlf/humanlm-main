# HumanLM：用状态对齐模拟用户，优于响应模仿

<div align="left">

[![](https://img.shields.io/badge/Website-HumanLM-purple?style=plastic&logo=Google%20Chrome)](https://humanlm.stanford.edu/)
[![](https://img.shields.io/badge/Datasets_&_Models-HuggingFace-yellow?style=plastic&logo=Hugging%20Face)](https://huggingface.co/snap-stanford/collections)
[![](https://img.shields.io/badge/Paper-arXiv-red?style=plastic&logo=arxiv)](https://humanlm.stanford.edu/HumanLM_paper.pdf)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

语言模型真的能像特定的人一样行动吗——不仅仅生成类似人类的文本，而是反映个人的价值观、观点和交流风格？**HumanLM** 通过将语言模型对齐到用户内部*状态*（立场、信念），而不是仅仅模仿表层回复，来应对这一挑战。


## 快速开始

### 1. 数据收集与处理

我们提供端到端工具，用于从六种来源收集原始数据，并将其处理为 train/val/test 划分，同时生成由 LLM 生成的用户画像。完整说明见 [`humanual_datasets/README.md`](humanual_datasets/README.md)。

### 2. 人类评估

用户研究界面允许标注者在 Reddit 帖子上比较自己的回复与模型生成的回复。

```bash
# 启动所需的 vLLM 模型服务器
vllm serve Qwen/Qwen3-8B --dtype auto --host 0.0.0.0 --port 8000 --tensor-parallel-size 3 --max-model-len 7168
vllm serve snap-stanford/humanlm-opinions --dtype auto --host 0.0.0.0 --port 63456 --tensor-parallel-size 2 --max-model-len 7168

# 启动 Gradio 标注界面
cd user_study
python gradio_app.py          # 添加 --debug 可跳过验证约束
```

### 3. 训练

HumanLM 训练所用的 VERL 配方作为 git 子模块维护在
`humanlm_train/verl-recipe-humanlm`。

如果你克隆仓库时没有包含子模块，请运行：

```bash
git submodule update --init --recursive
```
首次设置时，请运行：

```bash
git clone --recurse-submodules https://github.com/zou-group/humanlm.git
```

HumanLM 专用训练代码和设置说明位于 `humanlm_train/verl-recipe-humanlm/humanlm/README.md`。

### Bibtex

```bibtex
@article{wu2026humanlm,
  title={HUMANLM: Simulating Users with State Alignment Beats Response Imitation},
  url={https://humanlm.stanford.edu/},
  author={Wu, Shirley and Choi, Evelyn and Khatua, Arpandeep and
          Wang, Zhanghan and He-Yueya, Joy and Weerasooriya, Tharindu Cyril and
          Wei, Wei and Yang, Diyi and Leskovec, Jure and Zou, James},
  year={2026}
}
```
