# 电商客服 SFT + DPO 

这是一个面向中文电商客服场景的大模型微调项目。项目目标不是只做一个“能回答问题”的聊天模型，而是把模型训练成一个更接近真实客服助手的系统：在物流、退款、退换货、发票、库存、定制、售后等高频场景下，回复更自然、更简洁、更可执行，同时尽量减少编造后台状态、虚构平台能力和过度承诺。

项目整体采用两阶段训练路线：

1. `SFT`：先让模型学会客服任务的基础表达、问答结构、多轮对话承接与保守客服口径。
2. `DPO`：再在 SFT 基础上进行偏好对齐，压制“空话、模板化安抚、无依据具体化、虚构处理进度”等高风险话术，让回答更闭环、更能推动问题解决。

整个项目尽量覆盖一条完整且可复现的工程链路：

`原始 Excel FAQ -> 清洗去重 -> 问题聚类 -> 答案冲突治理 -> FAQ 多轮化 -> SFT -> 多候选生成 -> Judge 排序 -> DPO -> 自动评测 -> 消融分析`

## 文档导航

如果你想深入看方法细节、实验设计或产物说明，可以直接跳转：

- [SFT_DATA_AND_TRAINING_REPORT.md](./SFT_DATA_AND_TRAINING_REPORT.md)：SFT 数据治理、聚类清洗、FAQ 多轮化、训练问题排查与阶段复盘。
- [DPO_PROJECT_REPORT.md](./DPO_PROJECT_REPORT.md)：DPO 偏好数据构造、reference adapter 设计、beta 消融与最终结论。
- [train/README.md](./train/README.md)：训练目录补充说明。

## 为什么选电商客服场景

这个项目选择电商客服，而不是通用指令跟随或开放域问答，主要因为这个场景同时具备三类非常典型的对齐难点：

- 事实边界敏感：物流节点、订单状态、发货时间、退款时效、库存、价格保护等都不能编。
- 回复质量不只看“答对没”：还要看是否给出下一步操作、是否闭环、是否减少用户追问。
- 数据天然有噪声：原始 FAQ 会有大量重复问法、风格不一致、策略冲突、过度承诺和模板化安抚。

因此，这个项目的核心不是“把数据喂给模型”，而是把客服语料先治理成一个更稳的训练分布，再用 SFT 学能力、用 DPO 学偏好。

## 项目亮点

| 维度 | 内容 |
| --- | --- |
| 任务场景 | 中文电商客服，覆盖售前、物流、退款、退换货、发票、库存、定制、售后 |
| 训练流程 | `SFT + DPO` 两阶段微调 |
| SFT 数据 | `9408` 条 ShareGPT 风格样本，其中训练集 `9220`、验证集 `188` |
| DPO 数据 | `2443` 条 conversational preference pairs |
| 基座模型 | `Qwen3-8B` |
| 微调方式 | `LoRA + TRL + PEFT + Transformers` |
| 自动评测 | `qwen-plus` Judge，评估 `Accuracy / Auto Resolve / Closure / CSAT` |
| 主要问题 | 压制幻觉承诺、提升可执行性、减少模板腔、提升闭环率 |
| 实验类型 | SFT LoRA target/rank 消融、DPO beta 消融、推理样例对比 |

## 项目目标

项目主要希望把模型优化成下面这种风格：

- 准确理解用户当前问题，不答非所问。
- 不轻易输出“已帮您处理”“已为您加急”“系统已同步”“今晚送达”这类无依据承诺。
- 在信息不足时优先要求补充必要信息，而不是顺着上下文编造细节。
- 如果涉及操作，给出明确路径和下一步动作，提升自动解决率。
- 保持礼貌、自然、简洁的客服语气，减少空泛安抚和无意义寒暄。

## 目录结构

```text
SFT_DPO/
├── README.md
├── SFT_DATA_AND_TRAINING_REPORT.md
├── DPO_PROJECT_REPORT.md
├── data_preprocess/
│   ├── clean_customer_service_dataset.py
│   ├── cluster_customer_service_dataset.py
│   └── 网店客服回复数据集.xlsx
├── data_sft/
│   ├── all.jsonl
│   ├── train.jsonl
│   └── val.jsonl
├── data_dpo/
│   ├── dpo_pairs.jsonl
│   ├── pair_quality_summary.json
│   ├── qwen_ranked.jsonl
│   ├── sft_candidates.jsonl
│   └── ...
└── train/
    ├── configs/
    │   ├── trl_qwen_customer_sft.yaml
    │   ├── trl_qwen_customer_dpo.yaml
    │   └── deepspeed/ds_zero2.json
    ├── train_trl_sft.py
    ├── train_trl_dpo.py
    ├── infer_trl_sft.py
    ├── evaluate_inference_judge.py
    ├── build_dpo_preference_data.py
    ├── rewrite_dpo_chosen.py
    ├── plot_loss.py
    ├── run_sft_ablation_experiments.sh
    ├── run_dpo_beta_ablation_experiments.sh
    └── outputs/
```

## 数据概览

### 1. SFT 数据

当前仓库中可直接看到的 SFT 数据位于 `data_sft/`：

- `all.jsonl`：`9408` 条
- `train.jsonl`：`9220` 条
- `val.jsonl`：`188` 条

样本采用 ShareGPT 风格：

```json
{
  "id": "sample_id",
  "conversations": [
    {"from": "user", "value": "多久可以到货?"},
    {"from": "assistant", "value": "一般情况下商品发出后通常会在3到7个工作日内送达。"}
  ],
  "metadata": {
    "canonical_question": "多久可以到货?",
    "question_type": "delivery_time"
  }
}
```

问题类型分布中，占比最高的是：

- `yesno_generic`: `2727`
- `generic`: `2704`
- `return`: `962`
- `aftersales`: `613`
- `money`: `452`
- `delivery_time`: `379`

这说明当前训练集以泛咨询、可/不可判断、退换货和售后场景为主，比较符合电商客服高频意图分布。

### 2. DPO 数据

最终 DPO 训练数据位于 `data_dpo/dpo_pairs.jsonl`，当前规模为 `2443` 条。

每条样本采用 TRL 兼容的 conversational DPO 格式：

```json
{
  "prompt": [
    {"role": "system", "content": "你是电商客服助手..."},
    {"role": "user", "content": "是否提供定制服务?"}
  ],
  "chosen": [
    {"role": "assistant", "content": "我们提供部分定制服务，具体取决于商品类型和库存情况。"}
  ],
  "rejected": [
    {"role": "assistant", "content": "我们什么都可以定制，您下单就行。"}
  ]
}
```

`data_dpo/pair_quality_summary.json` 记录了偏好数据质量摘要：

| 指标 | 数值 |
| --- | ---: |
| ranked_records | 2714 |
| raw_pairs | 2714 |
| kept_pairs | 2443 |
| dropped_pairs | 271 |
| discard_fraction | 0.10 |
| drop_threshold | 3.14 |

也就是说，最终主线实验没有直接使用全部原始 pair，而是丢弃了偏好差异最弱的 `10%`，以减少 DPO 标签噪声。

## 方法总览

```mermaid
flowchart LR
    A[原始 Excel 客服问答] --> B[清洗与去重]
    B --> C[问题聚类与答案治理]
    C --> D[FAQ 多轮化扩写]
    D --> E[SFT 数据集]
    E --> F[SFT LoRA 微调]
    F --> G[多候选生成]
    G --> H[qwen-plus Judge 排序]
    H --> I[DPO preference pairs]
    I --> J[DPO 偏好对齐]
    J --> K[推理样例导出]
    K --> L[Judge 自动评测]
    L --> M[SFT/DPO 消融分析]
```

## 完整项目流程

### 1. 数据清洗与去重

入口脚本：

- `data_preprocess/clean_customer_service_dataset.py`

这一阶段解决的是“原始客服 FAQ 能不能直接拿来训练”的问题。主要处理包括：

- HTML 反转义与标签清理
- Unicode 规范化
- 中英文标点、数字、时间表达统一
- emoji、控制字符、零宽字符、乱码替换符清理
- 过短问题、过短回答过滤
- 低价值问答过滤，例如“在吗”“好的”“稍等”
- exact dedup + fuzzy dedup

如果不先做这一步，后面的聚类会被格式噪声严重干扰，模型也容易学到模板化客服语气。

### 2. 问题聚类与答案冲突治理

入口脚本：

- `data_preprocess/cluster_customer_service_dataset.py`

核心思路不是简单做去重，而是把“同一类问题的不同问法、不同回答版本”整合成更稳定的训练样本分布。报告中记录的关键聚类配置包括：

```json
{
  "question_similarity_threshold": 0.95,
  "neighbors_k": 25,
  "answer_family_threshold": 0.82,
  "max_answers_per_cluster": 2,
  "unify_time_answers": true
}
```

这一步主要做了几件事：

- 对问题做 embedding 表示并构建近邻图
- 用 Union-Find 聚类相似问法
- 用保守规则处理短句同义问法，例如“多久能到”“多久可以到”
- 在簇内合并答案家族、剔除明显冲突答案
- 对物流时效、发货时间等敏感主题做保守统一

这一步很关键，因为原始 FAQ 往往混杂了不同店铺、不同政策、不同场景下的回答。如果不先稳住答案分布，SFT 会直接学会互相矛盾的话术。

### 3. FAQ 多轮化扩写

原始数据单轮居多，但真实客服场景是强上下文任务。为了让模型学会：

- 承接上一轮回复
- 处理用户追问
- 在信息不足时做条件判断
- 给出闭环动作

项目对约 `30%` 左右的数据做了多轮化扩写，把 FAQ 改造成更接近真实客服上下文的样本。

这一步的收益主要不在“教模型更多知识”，而在于让模型学会更真实的客服对话结构。

### 4. SFT 训练

入口脚本：

- `train/train_trl_sft.py`

SFT 主线实验基于 `Qwen3-8B + LoRA`，训练摘要显示：

| 项目 | 数值 |
| --- | ---: |
| Base model | `Qwen3-8B` |
| train samples | `9220` |
| eval samples | `188` |
| epoch | `3.0` |
| global step | `3459` |
| trainable params | `15,335,424` |
| total params | `8,206,070,784` |
| trainable percent | `0.186879%` |
| GPU | `RTX 3090` |
| peak allocated | `17955.11 MB` |
| peak reserved | `23890.0 MB` |

SFT 的系统提示词也明确约束了回复风格：

- 不要输出思考过程
- 不要编造订单、物流、金额、时间等事实
- 信息不足时可以追问，但不要重复无意义问题
- 禁止无关寒暄或模板话术
- 涉及操作时必须给出具体路径或下一步建议

### 5. SFT 消融实验

项目对 LoRA rank 和 target modules 做了系统消融。`train/outputs/sft_ablation/ablation_metrics_summary.csv` 中的结果如下：

| 实验 | LoRA 配置 | Accuracy | Auto Resolve | Closure | CSAT |
| --- | --- | ---: | ---: | ---: | ---: |
| `rank8_qv` | rank=8, `q_proj+v_proj` | 72.93 | 59.67 | 60.22 | 3.519 |
| `rank16_qv` | rank=16, `q_proj+v_proj` | 73.71 | 61.14 | 60.00 | 3.543 |
| `rank32_qv` | rank=32, `q_proj+v_proj` | 67.04 | 56.98 | 56.98 | 3.436 |
| `rank16_qvko` | rank=16, `q+k+v+o` | 75.14 | 61.33 | 60.77 | 3.608 |
| `rank16_all_linear` | rank=16, all linear | 75.69 | 63.54 | 62.43 | 3.635 |

从结果看：

- 单纯把 rank 从 `16` 提到 `32` 并没有带来收益，反而明显退化。
- 把 target 从 `qv` 扩到 `qkvo`，收益很稳定。
- `rank16_all_linear` 的 judge 指标最好，但训练参数量升到 `43,646,976`，是 `rank16_qvko` 的近 `3` 倍。
- 因此主线项目最终选择 `rank16_qvko` 作为 SFT 骨干：性能强、成本可控、也更适合作为 DPO 起点。

### 6. DPO 偏好数据构造

入口脚本：

- `train/build_dpo_preference_data.py`

DPO 阶段没有直接把原始参考答案当作 `chosen`，而是采用了“同一 prompt 下多候选生成 + Judge 排序”的方式。完整流程是：

1. 从 SFT 训练集采样 `3000` 条 prompt。
2. 移除原始 assistant 回复，保留 system + history + last user，构造 DPO prompt。
3. 用 SFT adapter 多次采样生成候选回复。
4. 去重并剔除低多样性样本。
5. 用 `qwen-plus` 从 hallucination、accuracy、closure、auto_resolve、satisfaction 等维度排序。
6. 排除严重幻觉候选。
7. 依据分数差构造 `chosen / rejected`。
8. 丢弃偏好差异最弱的 `10%` pair。

报告中给出的数据构造统计为：

| 阶段 | 数量 |
| --- | ---: |
| sampled_prompts | `3000` |
| sft_candidates | `3000` |
| qwen_ranked | `2714` |
| failed_rankings | `202` |
| skipped_low_diversity_rankings | `84` |
| raw_pairs | `2714` |
| kept_pairs | `2443` |

这套构造方式比“直接用原答案做 chosen”更适合 DPO，因为它能显式拉开“更 grounded、更闭环、更可执行”和“虽然不算错但很空、很泛、没帮助”之间的差异。

### 7. DPO 训练

入口脚本：

- `train/train_trl_dpo.py`

DPO 主线从 SFT 最优 adapter `rank16_qvko` 继续训练。关键点在于 reference model 不是裸 base model，而是复制一份 SFT adapter 作为冻结 `ref` adapter。这样：

- policy 与 reference 起点完全一致
- reward 只反映 DPO 阶段产生的偏移
- 不会把 SFT 已学到的基础客服能力也混进 reward

最终有效主线实验配置：

| 项目 | 数值 |
| --- | ---: |
| base model | `Qwen3-8B` |
| SFT adapter | `rank16_qvko` |
| DPO beta | `0.3` |
| loss type | `sigmoid` |
| train samples | `2198` |
| eval samples | `245` |
| epoch | `3.0` |
| global step | `207` |
| trainable params | `15,335,424` |
| total params | `8,221,406,208` |
| trainable percent | `0.18653%` |
| peak allocated | `20619.16 MB` |
| peak reserved | `23856.0 MB` |

## 实验结果

### 1. SFT 主模型结果

`rank16_qvko` 在自动评测上的结果为：

| 指标 | 分数 |
| --- | ---: |
| Accuracy | `75.14` |
| Auto Resolve | `61.33` |
| Closure | `60.77` |
| CSAT | `3.608` |

这是后续 DPO 的基线。

### 2. DPO beta 消融

报告明确指出，最终有效结论应以 `3 epochs` 的 beta 消融为准，而不是更早期的 `2 epochs` 版本。最终结果如下：

| 实验 | beta | epoch | reward margin | Accuracy | Auto Resolve | Closure | CSAT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `rank16_qvko_beta0p05` | 0.05 | 3.0 | 0.0126 | 73.40 | 63.83 | 63.83 | 3.601 |
| `rank16_qvko_beta0p1` | 0.10 | 3.0 | 0.0237 | 72.34 | 63.83 | 64.36 | 3.580 |
| `rank16_qvko_beta0p3` | 0.30 | 3.0 | 0.0621 | 73.40 | 66.49 | 65.96 | 3.628 |

结论很明确：

- `beta=0.3` 的 `reward margin` 最大，说明偏好边界推动最强。
- 它不是 Accuracy 最高的配置，但在 DPO 更关心的 `Auto Resolve / Closure / CSAT` 上表现最好。
- 因此项目最终把 `beta=0.3` 视为本阶段最有效配置。

### 3. SFT 与 DPO 主模型对比

| 模型 | Accuracy | Auto Resolve | Closure | CSAT |
| --- | ---: | ---: | ---: | ---: |
| SFT `rank16_qvko` | 75.14 | 61.33 | 60.77 | 3.608 |
| DPO `beta=0.3` | 73.40 | 66.49 | 65.96 | 3.628 |

从这个对比可以看到 DPO 的真实价值：

- Accuracy 略降 `1.74` 个百分点
- Auto Resolve 提升 `5.16` 个百分点
- Closure 提升 `5.19` 个百分点
- CSAT 提升 `0.020`

这非常符合本项目对 DPO 的定位：不是追求更像标准答案，而是牺牲极少量逐字准确性，换取更强的可执行性和客服闭环感。

## 结果展示与定性观察

从仓库里的 `inference_samples.json` 可以直观看到两阶段模型的典型问题与改进方向。

### 1. SFT 阶段的典型残留问题

SFT 已经能输出自然客服语气，但仍会在一些场景里：

- 顺着上下文补充并不存在的后台状态
- 把“可以进一步确认”说成“已经安排”
- 给出过度具体的物流节点、时效或处理动作

例如在偏远地区物流追问场景里，SFT 样例会出现“当前物流路由已规划为中转至满洲里再分发”“我可立即为您同步申请优先中转调度”这类无依据具体化表述。

### 2. DPO 阶段的主要改善

DPO 更偏好下面这种回答风格：

- 不乱承诺后台动作
- 保留可执行路径
- 结论更直接
- 条件判断更清楚

例如在“袋子丢了还能换吗”的场景里，DPO 样例会更稳定地收敛到：

- 先说明袋子不是关键条件
- 再明确真正判断标准是“吊牌在、商品完好、未穿着”
- 最后给出 `[我的订单 -> 申请换货]` 这样的操作路径

这类改进正对应了 DPO 后 `Auto Resolve` 和 `Closure` 的提升。

## 训练与评估脚本

核心脚本说明：

- `train/train_trl_sft.py`：SFT 主训练脚本
- `train/train_trl_dpo.py`：DPO 主训练脚本
- `train/infer_trl_sft.py`：导出推理样例
- `train/evaluate_inference_judge.py`：调用评审模型自动打分
- `train/build_dpo_preference_data.py`：构造 DPO 偏好数据
- `train/rewrite_dpo_chosen.py`：对 DPO `chosen` 做重写增强
- `train/plot_loss.py`：根据 `trainer_state.json` 重绘 loss 曲线
- `train/run_sft_ablation_experiments.sh`：SFT 消融实验脚本
- `train/run_dpo_beta_ablation_experiments.sh`：DPO beta 消融脚本

## 环境依赖

建议环境：

- Python `3.9+`
- CUDA 可用
- 单卡 `RTX 3090` 可复现主线实验

训练依赖安装：

```bash
cd SFT_DPO/train
pip install -r requirements-train.txt
```

常用核心依赖包括：

- `transformers`
- `datasets`
- `trl`
- `peft`
- `accelerate`
- `PyYAML`
- `matplotlib`

如果需要运行数据清洗和聚类脚本，通常还需要：

```bash
pip install pandas scikit-learn numpy sentence-transformers faiss-gpu
```

## 快速开始

### 0. 先检查配置路径

当前配置文件里仍保留了一些本机绝对路径和历史数据路径。直接复现前请优先检查：

- `train/configs/trl_qwen_customer_sft.yaml`
- `train/configs/trl_qwen_customer_dpo.yaml`

需要重点确认：

- `model.model_name_or_path`
- `model.sft_adapter_path`
- `dataset.train_file`
- `dataset.val_file`
- `output.output_dir`
- `output.artifact_dir`

### 1. 运行 SFT

```bash
python train/train_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

### 2. 导出推理样例

```bash
python train/infer_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

### 3. 自动评测 SFT 输出

评测脚本从环境变量读取接口配置：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url
```

然后执行：

```bash
python train/evaluate_inference_judge.py \
  --input train/outputs/your_checkpoint/artifacts/inference_samples.json \
  --model qwen-plus
```

### 4. 构造 DPO 数据

```bash
python train/build_dpo_preference_data.py \
  --config train/configs/trl_qwen_customer_sft.yaml \
  --train-file SFT_DPO/data_sft/train.jsonl \
  --adapter-path /path/to/your_sft_adapter \
  --output-dir SFT_DPO/data_dpo
```

### 5. 运行 DPO

```bash
python train/train_trl_dpo.py --config train/configs/trl_qwen_customer_dpo.yaml
```

### 6. 多卡 / DeepSpeed

如果需要启用 ZeRO-2，可在配置中设置：

```yaml
distributed:
  deepspeed: deepspeed/ds_zero2.json
```

然后使用：

```bash
torchrun --nproc_per_node=2 train/train_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

或：

```bash
torchrun --nproc_per_node=2 train/train_trl_dpo.py --config train/configs/trl_qwen_customer_dpo.yaml
```

## 训练产物

典型输出目录通常包含：

- LoRA adapter 权重
- tokenizer 相关文件
- `trainer_state.json`
- `train_results.json`
- `all_results.json`
- `artifacts/loss_curve.png`
- `artifacts/inference_samples.json`
- `artifacts/infer.jsonl`
- `artifacts/run_summary.json`

DPO 输出额外常见：

- `artifacts/dpo_metrics.csv`
- `artifacts/dpo_reward_curve.png`
- `artifacts/judge_summary.json`
- `artifacts/judge_summary.csv`

## 当前结论

截至当前仓库产物，可以把项目结论概括为三点：

1. 客服微调项目的关键不只是训练脚本，而是训练前的数据治理。清洗、聚类、冲突剔除和多轮化扩写，对最终质量影响很大。
2. SFT 已经可以把模型带到“像客服”的水平，但仍会残留顺着上下文编造后台状态的问题。
3. DPO 对“可执行性、闭环感、减少空话和减少无依据承诺”确实有效，尤其是 `beta=0.3` 的 3-epoch 配置表现最好。

## 局限与后续方向

项目当前仍有一些明确局限：

- 配置中仍残留绝对路径，开源友好度还不够。
- 自动评测依赖 `qwen-plus` Judge，仍属于模型评审，不是人工标注金标准。
- 数据分布仍偏 FAQ 和高频客服意图，长尾复杂售后场景覆盖有限。
- DPO 后 Accuracy 有小幅回落，说明偏好优化与保留逐字准确性之间仍存在权衡。

后续可以继续做：

- 统一配置为相对路径和可公开复现模板
- 增加更细颗粒度的子场景评测集
- 加入人工审阅样本池，和自动 Judge 做交叉验证
- 做更系统的 DPO 数据质量实验，例如不同丢弃比例、不同候选生成温度、不同 judge rubric

## 相关文档

- [SFT_DATA_AND_TRAINING_REPORT.md](./SFT_DATA_AND_TRAINING_REPORT.md)
- [DPO_PROJECT_REPORT.md](./DPO_PROJECT_REPORT.md)
- [train/README.md](./train/README.md)
