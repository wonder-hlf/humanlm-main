# Humanual 数据集

本软件包提供用于收集、处理和划分 human-language modeling 数据集的工具。每个数据集都描述了 `post` 与 `comment` 之间的一对多关系。

**数据许可声明：** 此处提供的数据集来自多个第三方来源。数据许可和使用条款应以如下列出的各个来源为准。使用前请参阅各来源自己的服务条款和许可协议。

## 可用数据集

| 数据集 | 来源 | 描述 |
|---------|--------|-------------|
| **Humanual-News** | [YouTube](https://www.youtube.com/) | 新闻视频评论 |
| **Humanual-Book** | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) | 图书评论与评分 |
| **Humanual-Opinion** | [Reddit](https://www.reddit.com/) | Subreddit 帖子与评论 |
| **Humanual-Politics** | [Medium](https://medium.com/) | 政治博客文章与评论 |
| **Humanual-Chat** | [WildChat](https://wildchat.allen.ai/) | 多轮 AI 聊天对话 |
| **Humanual-Email** | [Enron Email Corpus](https://www.cs.cmu.edu/~enron/) | 公司邮件线程 |

## 原始数据集格式

原始数据集中的每一行都遵循以下 schema：

```python
{
    "prompt": [
        # 第一轮始终是帖子。metadata 包含帖子信息。
        {"role": poster_id, "content": ..., "metadata": ...},
        # 后续轮次是用户评论。metadata 包含评论信息。
        {"role": user_id, "content": ..., "metadata": ...},
        ...
    ],
    "completion": ...,  # str：目标评论
    "post_id": ...,     # str：全局唯一的帖子标识符
    "user_id": ...,     # str：全局唯一的用户标识符
    "timestamp": ...,   # int：UTC 时间戳
    "metadata": ...,    # completion 评论的元数据
}
```

---

## 第 1 步：收集原始数据

每个数据集都有自己的 scraper 模块。下面给出从各个来源收集原始数据的说明。

### Humanual-News

抓取 YouTube 视频元数据、评论，以及可选的视频转录文本。

**必需的环境变量：**
- `YOUTUBE_API_KEY`：YouTube Data API 密钥。多个密钥可用 `:` 分隔，以便轮换额度。
- `WEBSHARE_PROXY_USERNAME` 和 `WEBSHARE_PROXY_PASSWORD`：用于获取转录文本的代理凭据（见 [Webshare](https://www.webshare.io)）。

```shell
# 第 1 步：抓取视频元数据和评论
python -m humanual_datasets.humanual_news \
    --channels BBCNews CNN \
    --push_to_hub <your-hf-org>/humanual_news_raw_dataset \
    --scrape_raw \
    --config .env

# 第 2 步（可选）：单独获取视频转录文本
python -m humanual_datasets.humanual_news \
    --channels BBCNews CNN \
    --push_to_hub <your-hf-org>/humanual_news_raw_dataset \
    --transcripts \
    --config .env
```

| 参数 | 描述 |
|----------|-------------|
| `--channels` | **（必需）** YouTube 频道用户名，或一个将频道映射到播放列表名称的 JSON 文件。 |
| `--scrape_raw` | 从 YouTube API 抓取视频元数据和评论。 |
| `--transcripts` | 获取视频转录文本（与 `--scrape_raw` 分开运行）。 |
| `--top_level_only` | 只获取顶层评论（不包括回复）。 |
| `--push_to_hub` | **（必需）** 用于原始数据集的 HuggingFace Hub 仓库。 |
| `--config` | 包含 API 凭据的 `.env` 文件路径。 |
| `--max_videos_per_channel` | 限制每个频道的视频数量（用于测试）。 |
| `--verification_mode` | HuggingFace 加载验证方式：`no_checks`、`basic_checks`、`all_checks`。默认：`basic_checks`。 |
| `--no_create_raw` | 跳过创建原始数据集（适用于只获取转录文本的运行）。 |

**提示：** 将 `--scrape_raw` 和 `--transcripts` 作为独立步骤运行。由于 API 额度耗尽，抓取可能会失败，分离步骤可以让恢复更容易。

`--channels` 参数也接受 JSON 文件，用于为每个频道指定播放列表名称：
```json
{
    "BBCNews": ["US & Canada | BBC News", "Sport | BBC News", "Health | BBC News"],
    "CNN": ["World News", "Entertainment", "Science & Technology"]
}
```

### Humanual-Book

通过 HuggingFace 从 Amazon Reviews 2023 数据集中收集商品评论。

**必需的环境变量：** 无（数据在 HuggingFace 上公开可用）。

```shell
python -m humanual_datasets.humanual_book \
    --categories Books \
    --push_to_hub <your-hf-org>/humanual_book_raw_dataset
```

| 参数 | 描述 |
|----------|-------------|
| `--categories` | **（必需）** 要收集的 Amazon 商品类别列表（例如 `Books`、`All_Beauty`）。 |
| `--data_dirname` | 用于中间数据的本地目录。默认：`data`。 |
| `--category_splits` | 将每个类别拆分为 N 份以降低内存占用。默认：`1`。 |
| `--push_to_hub` | 用于上传原始数据集的 HuggingFace Hub 仓库。 |
| `--config` | 用于凭据的 `.env` 文件路径。 |
| `--max_items_per_category` | 限制每个类别的 item 数量（用于调试）。 |

### Humanual-Opinion

通过 Reddit API（PRAW）从 Reddit subreddit 收集帖子和评论。

**必需的环境变量：**
- `REDDIT_USER_AGENT`：Reddit API user agent 字符串。
- `REDDIT_CLIENT_ID` 和 `REDDIT_CLIENT_SECRET`：Reddit 应用凭据。
  - 或者：用于脚本类型认证的 `REDDIT_USERNAME` 和 `REDDIT_PASSWORD`。
  - 支持多组凭据：`REDDIT_CLIENT_ID_1`、`REDDIT_CLIENT_ID_2` 等。

设置方式见 [PRAW documentation](https://praw.readthedocs.io/en/stable/getting_started/authentication.html)。

```shell
python -m humanual_datasets.humanual_opinion \
    --subreddits AmItheAsshole \
    --push_to_hub <your-hf-org>/humanual_opinion_raw_dataset \
    --config_path .env
```

| 参数 | 描述 |
|----------|-------------|
| `--subreddits` | **（必需）** 要抓取的 Subreddit 名称（不带 `r/`）。 |
| `--sort` | 排序方式：`hot`、`new`、`top`、`controversial`、`rising`。默认：`top`。 |
| `--time_filter` | 时间过滤器：`all`、`hour`、`day`、`week`、`month`、`year`。默认：`all`。 |
| `--max_posts_per_subreddit` | 限制每个 subreddit 的帖子数量。 |
| `--max_comments_per_post` | 限制每个帖子的顶层评论数量。 |
| `--max_comments_per_user` | 限制用于帖子扩展的每位用户评论数量。 |
| `--push_to_hub` | 用于上传原始数据集的 HuggingFace Hub 仓库。 |
| `--config_path` | 包含 Reddit 凭据的 `.env` 文件路径。 |
| `--extend` | 从用户评论历史中扩展帖子。 |

### Humanual-Politics

通过 RapidAPI 从 Medium 抓取政治文章和评论。

**必需的环境变量：**
- `X-RapidAPI-Key`：RapidAPI 上 Medium API 的 API 密钥。

```shell
python -m humanual_datasets.humanual_politics \
    --tag politics \
    --hf_repo <your-hf-org>/humanual_politics_raw_dataset \
    --config .env
```

| 参数 | 描述 |
|----------|-------------|
| `--tag` | **（必需）** 要抓取的 Medium 标签 slug（例如 `politics`、`health`）。 |
| `--years_ago` | 目标年份，表示为 `current_year - years_ago`。默认：`1`。 |
| `--months` | 要抓取的月份（例如 `1-12`、`1,2,3`）。默认：`1-12`。 |
| `--out_dir` | JSONL 输出的本地目录。默认：`out_medium_rows`。 |
| `--hf_repo` | 用于上传原始数据集的 HuggingFace Hub 仓库。 |
| `--flush_every_calls` | 每进行这么多次 API 调用后处理并推送一次。默认：`2`。 |
| `--article_concurrency` | 并行处理文章的数量。默认：`5`。 |
| `--public` | 将 HuggingFace 数据集设为公开。 |

### Humanual-Chat

处理来自 WildChat 数据集的多轮 AI 对话（该数据集在 HuggingFace 上公开可用）。

**必需的环境变量：** 无（数据在 HuggingFace 上公开可用）。

```shell
python -m humanual_datasets.humanual_chat \
    --push_to_hub <your-hf-org>/humanual_chat_raw_dataset
```

| 参数 | 描述 |
|----------|-------------|
| `--push_to_hub` | 用于上传原始数据集的 HuggingFace Hub 仓库。 |
| `--config` | 用于凭据的 `.env` 文件路径。 |
| `--max_conversations` | 限制要处理的对话数量（用于测试）。 |

### Humanual-Email

将 Enron 邮件 CSV 文件处理为线程式对话。

**必需的环境变量：** 无（需要本地 Enron CSV 文件）。

**前置条件：** 你需要先使用 `utils_parser.py` 将原始 Enron maildir 解析为 CSV：
```shell
python -m humanual_datasets.utils_parser \
    --input_csv path/to/enron_emails.csv \
    --output_json path/to/enron_nested.json
```

然后创建原始数据集：
```shell
python -m humanual_datasets.humanual_email \
    --csv_path path/to/enron_emails.csv \
    --push_to_hub <your-hf-org>/humanual_email_raw_dataset
```

| 参数 | 描述 |
|----------|-------------|
| `--csv_path` | **（必需）** Enron 邮件 CSV 文件路径。 |
| `--push_to_hub` | 用于上传原始数据集的 HuggingFace Hub 仓库。 |
| `--config` | 用于凭据的 `.env` 文件路径。 |
| `--min_thread_size` | 纳入数据集所需的每个邮件线程的最少邮件数。默认：`2`。 |
| `--max_rows` | 只读取前 N 行 CSV（用于测试）。 |
| `--save_to_disk` | 通过 `Dataset.save_to_disk()` 将数据集保存到本地。 |

---

## 第 2 步：处理原始数据

使用 `process_raw.py` 过滤、生成用户画像、划分并上传处理后的数据集。该脚本适用于上述任意数据集。

### 处理流水线

```
加载原始数据集（来自 --pull_from_hub 或 --cache_dir 或 --cache_hub）
        |
根据 --min_num_comments 和 --max_num_comments 过滤
        |
应用 customized_filter_fn（如果在 <dataset_name>.py 中定义）
        |
应用 complete_dataset_fn（如果在 <dataset_name>.py 中定义）
        |
生成用户画像（使用 LLM，基于最早的 --persona_history_length 条评论）
        |
划分为 train / val / seen_test / unseen_test
        |
上传 / 保存处理后的数据集
```

### 示例命令

#### Humanual-News
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_news \
    --splits BBCNews CNN \
    --subset_mode \
    --global_frac 0.25 \
    --pull_from_hub <your-hf-org>/humanual_news_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_news_processed \
    --min_num_comments 25 \
    --max_num_comments 50 \
    --persona_history_length 20 \
    --save_dir data/humanual_news_processed \
    --cache_dir data/humanual_news_raw \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --val_frac 0.002 \
    --unseen_frac 0.01 \
    --seen_frac 0.01 \
    --memory_friendly \
    --resume_from_cache
```

#### Humanual-Book
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_book \
    --splits Books \
    --pull_from_hub <your-hf-org>/humanual_book_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_book_processed \
    --min_num_comments 100 \
    --max_num_comments 1000 \
    --persona_history_length 20 \
    --subset_mode \
    --memory_friendly \
    --cache_dir data \
    --resume_from_cache
```

#### Humanual-Opinion
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_opinion \
    --splits AmItheAsshole \
    --pull_from_hub <your-hf-org>/humanual_opinion_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_opinion_processed \
    --min_num_comments 10 \
    --max_num_comments 135 \
    --persona_history_length 20 \
    --cache_dir data/humanual_opinion_raw \
    --save_dir data/humanual_opinion_processed \
    --partition_by post \
    --compute_distribution \
    --resume_from_cache \
    --max_concurrent_users 15 \
    --max_concurrent_posts 20 \
    --max_concurrent_comments 3 \
    --post_dist_batch_size 20 \
    --compute_distribution_no_train
```

#### Humanual-Politics
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_politics \
    --splits politics health love entrepreneurship travel culture self_improvement \
    --pull_from_hub <your-hf-org>/humanual_politics_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_politics_processed \
    --min_num_comments 20 \
    --max_num_comments 100 \
    --persona_history_length 20 \
    --cache_dir data/humanual_politics_raw \
    --save_dir data/humanual_politics_processed \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --compute_distribution \
    --resume_from_cache \
    --concurrency 5
```

#### Humanual-Chat
```shell
python -m humanual_datasets.process_raw \
    --dataset_name humanual_chat \
    --splits train \
    --pull_from_hub <your-hf-org>/humanual_chat_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_chat_processed \
    --min_total_turns 5 \
    --max_total_turns 10 \
    --min_num_comments 5 \
    --max_num_comments 10 \
    --min_turns_for_train 6 \
    --val_frac 0.05 \
    --unseen_frac 0.05 \
    --seen_frac 0.05 \
    --fixed_persona "A user who is chatting with an AI assistant." \
    --save_dir data/humanual_chat_processed \
    --partition_by turn \
    --persona_history_length 0 \
    --overwrite
```

#### Humanual-Email
```shell
python -m humanual_datasets.process_raw \
    --config .env \
    --dataset_name humanual_email \
    --splits default \
    --pull_from_hub <your-hf-org>/humanual_email_raw_dataset \
    --push_to_hub <your-hf-org>/humanual_email_processed \
    --min_num_comments 10 \
    --max_num_comments 500 \
    --persona_history_length 20 \
    --cache_dir data/humanual_email_raw \
    --save_dir data/humanual_email_processed \
    --partition_by post \
    --llm_model claude-sonnet-4-5-20250929 \
    --resume_from_cache
```

### process_raw.py 参数参考

#### 核心参数

| 参数 | 描述 |
|----------|-------------|
| `--dataset_name` | **（必需）** 要处理的数据集：`humanual_news`、`humanual_book`、`humanual_opinion`、`humanual_politics`、`humanual_chat`、`humanual_email`。 |
| `--splits` | **（必需）** 原始数据集中的 split 名称（支持 Python 正则表达式，例如 `".*"` 表示全部）。 |
| `--config` | 包含用于基于 LLM 生成人物画像的 API 凭据的 `.env` 文件路径。 |

#### 过滤

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--min_num_comments` | `10` | 排除评论数少于该阈值的用户。 |
| `--max_num_comments` | `1000` | 排除评论数多于该阈值的用户。 |
| `--max_samples` | `None` | 限制总样本数（用于测试/预览）。 |

#### 用户画像生成

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--persona_history_length` | `20` | 用于生成用户画像的某用户最早评论数量。设为 `0` 可跳过用户画像生成。 |
| `--remove_used_persona_rows` | `False` | 从最终数据集中移除用于生成用户画像的行。 |
| `--fixed_persona` | `None` | 对所有用户使用固定的用户画像字符串（绕过 LLM 生成）。 |
| `--preview_personas` | `False` | 预览哪些评论将用于生成用户画像，但不发起 LLM 调用。 |
| `--max_comment_length` | `3000` | 在用户画像生成阶段移除长度超过该值的评论。 |
| `--truncate_comment_length` | `1024` | 为用户画像生成将评论截断到该词数。设为 `0` 可禁用。 |

#### LLM 配置（用于用户画像生成）

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--llm_model` | `anthropic/claude-sonnet-4-5-20250929` | 用于生成用户画像的 LLM 模型（通过 [litellm](https://docs.litellm.ai/)）。 |
| `--llm_temperature` | `0.0` | LLM 采样温度。 |
| `--llm_max_tokens` | `4096` | LLM 响应的最大 token 数。 |

#### 数据集划分

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--partition_by` | `user` | 划分策略：`user`（按用户划分）、`post`（按帖子划分）或 `turn`（按对话轮次划分）。 |
| `--unseen_frac` | `0.08` | 作为 unseen test set 留出的用户比例。 |
| `--seen_frac` | `0.08` | seen 用户评论中作为 seen test set 留出的比例。 |
| `--val_frac` | `0.02` | seen 用户评论中作为验证集留出的比例。 |

#### 分布指标

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--compute_distribution` | `False` | 在用户画像生成后计算帖子分布指标。 |
| `--compute_distribution_no_train` | `False` | 只为 val+test 划分计算分布指标。 |
| `--post_dist_batch_size` | `10` | 计算分布指标时，每次 LLM 调用包含的评论数量。 |

#### 基于轮次划分的参数（用于 `--partition_by turn`）

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--min_turns_for_train` | `0` | 训练数据所需的最少轮次数。 |
| `--min_total_turns` | `0` | prompt 对话中的最少总轮次数。 |
| `--max_total_turns` | `0` | prompt 对话中的最多总轮次数。 |

#### 下载 / 上传 / 保存

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--pull_from_hub` | `None` | 用于加载原始数据集的 HuggingFace Hub 仓库。 |
| `--push_to_hub` | `None` | 用于上传处理后数据集的 HuggingFace Hub 仓库。 |
| `--save_dir` | `None` | 保存处理后 parquet 文件的本地目录。 |
| `--overwrite` | `False` | 允许覆盖已有的 HuggingFace Hub 仓库。 |
| `--subset_mode` | `False` | 将频道/类别存储为 HuggingFace subsets，而不是 splits。 |

#### 内存优化

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--memory_friendly` | `False` | 尽量减少内存占用（可能会降低处理速度）。 |
| `--cache_dir` | `None` | 将中间数据集缓存到本地磁盘以降低内存使用。 |
| `--cache_hub` | `None` | 直接从该 HuggingFace Hub 仓库加载已过滤的数据集。 |
| `--resume_from_cache` | `False` | 如果已有缓存，则从缓存继续（OOM 后很有用）。 |
| `--compression` | `uncompressed` | 缓存 parquet 文件的压缩方式：`uncompressed`、`snappy`、`lz4`、`zstd`。 |

#### 并发

| 参数 | 默认值 | 描述 |
|----------|---------|-------------|
| `--max_concurrent_users` | `5` | 用户级用户画像生成的最大并发 LLM 调用数。 |
| `--max_concurrent_posts` | `50` | 帖子级处理的最大并发 LLM 调用数。 |
| `--max_concurrent_comments` | `3` | 评论级处理的最大并发 LLM 调用数。 |

### 提示

- **正则 splits：** 使用 `--splits ".*"` 处理所有可用的 splits。
- **不生成用户画像：** 将 `--persona_history_length 0` 设为跳过用户画像生成。
- **异常用户：** 使用 `--max_num_comments` 排除评论数量异常多的用户。
- **OOM 恢复：** 结合使用 `--cache_dir`、`--memory_friendly` 和 `--resume_from_cache` 来处理大型数据集。
  - 使用 `--cache_hub` 的方式：先用 `--cache_dir` 运行以在本地保存 parquet 文件，然后用 `datasets.Dataset.from_parquet()` 加载，并 `push_to_hub()` 到你的缓存仓库。

### 使用自定义函数扩展

每个数据集模块（例如 `humanual_opinion.py`）都可以定义可选函数，`process_raw.py` 会自动加载这些函数：

- `customized_filter_fn(row: dict) -> bool`：额外的行级过滤逻辑。
- `complete_dataset_fn(df: pl.DataFrame, split: str) -> pl.DataFrame`：过滤后对数据集进行转换。
