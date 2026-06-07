# 电商客服 SFT + DPO 

这是一个面向中文电商客服场景的大模型微调项目，目标不是只让模型“会回答”，而是让模型在物流、退款、退换货、发票、库存、定制、售后等高频客服问题上，回复得更自然、更简洁、更可执行，同时尽量减少编造后台状态、虚构平台能力和过度承诺。

项目整体采用两阶段训练方案：

1. 监督微调 `SFT`：让模型先学会客服任务的基础表达、问答结构和多轮对话承接。
2. 偏好对齐 `DPO`：在 SFT 基础上继续优化，让模型更少说空话、更少幻觉、更多闭环动作和具体路径。

## 项目亮点

- 面向真实电商客服场景，而不是通用聊天。
- 包含从原始 Excel 问答数据到训练集的完整数据治理流程。
- 重点解决客服模型常见问题：
  - 同义问法重复
  - 同类问题答案冲突
  - 模板化、低信息量回复
  - 虚构订单状态、物流状态、平台权限
- 基于 `TRL + Transformers + PEFT`，训练流程独立清晰，便于复现和二次开发。
- 提供 SFT、DPO、自动评测、loss 曲线绘制、推理样例导出和消融实验脚本。

## 项目目标

本项目主要优化以下几类能力：

- 准确理解用户问题，不答非所问。
- 尽量避免“已帮您处理”“已为您加急”“系统已同步”等无依据承诺。
- 给出明确下一步动作、操作路径或条件判断，提高自动解决率和闭环率。
- 保持客服语气礼貌、自然、简洁，减少空泛安抚和无意义寒暄。

## 目录结构

```text
SFT_DPO/
├── README.md
├── SFT_DATA_AND_TRAINING_REPORT.md
├── DPO_PROJECT_REPORT.md
├── .gitignore
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
    └── run_dpo_beta_ablation_experiments.sh
```

## 数据说明

### 一、SFT 数据

`data_sft/` 中保存监督微调数据，目前仓库内可见：

- `all.jsonl`：9408 条
- `train.jsonl`：9220 条
- `val.jsonl`：188 条

每条样本采用 ShareGPT 风格结构：

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

这些 SFT 数据并不是原始 FAQ 直接拿来训练，而是经过了：

- 文本清洗
- 低价值样本过滤
- 去重
- 相似问题聚类
- 簇内答案排序与冲突剔除
- 时间类话术统一
- 部分 FAQ 多轮化改写

### 二、DPO 数据

`data_dpo/dpo_pairs.jsonl` 是最终用于偏好对齐的数据，当前约 2443 条。

每条样本格式如下：

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

DPO 数据构造流程为：

1. 从 SFT 训练集抽取 prompt。
2. 使用 SFT adapter 对同一 prompt 多次采样，生成多个候选回复。
3. 用评审模型对候选进行打分和排序。
4. 过滤严重幻觉候选。
5. 生成 `chosen / rejected` 偏好对。

相比直接拿原始参考答案当 `chosen`，这种方式更适合偏好学习，因为它能显式拉开“更 grounded、更闭环、更可执行”和“虽然没错但太泛、没帮助”之间的差异。

## 方法流程

### 一、数据治理阶段

`data_preprocess/clean_customer_service_dataset.py`

主要负责：

- 清洗原始 Excel 客服问答
- 统一标点、数字、时间表达
- 去除乱码、低价值回复和明显噪声
- exact dedup + fuzzy dedup

`data_preprocess/cluster_customer_service_dataset.py`

主要负责：

- 对问题做向量化表示
- 构建近邻图并聚类相似问法
- 在簇内合并回答家族并排序
- 剔除互相冲突或高风险答案
- 把部分单轮 FAQ 扩写成多轮客服对话

### 二、SFT 阶段

`train/train_trl_sft.py`

主要流程：

- 加载 ShareGPT 风格 `conversations`
- 转换为 TRL 使用的 `messages`
- 基于 Qwen 系列基座模型进行 LoRA 微调
- 导出 checkpoint、loss 曲线、推理样例和训练摘要

### 三、DPO 阶段

`train/build_dpo_preference_data.py`

主要流程：

- 使用 SFT 模型生成多候选回复
- 调用评审模型排序
- 输出 TRL 兼容的 DPO 数据集

`train/train_trl_dpo.py`

主要流程：

- 从 SFT adapter 初始化 policy model
- 克隆一份冻结 adapter 作为 reference policy
- 使用 `DPOTrainer` 进行偏好对齐训练

## 环境依赖

建议使用 Python `3.9+`，并准备好 CUDA 环境。

安装训练依赖：

```bash
cd SFT_DPO/train
pip install -r requirements-train.txt
```

核心依赖包括：

- `transformers`
- `datasets`
- `trl`
- `peft`
- `accelerate`
- `PyYAML`
- `matplotlib`

如果你还需要运行数据清洗与聚类脚本，通常还要补充：

```bash
pip install pandas scikit-learn numpy sentence-transformers faiss-gpu
```

如果环境里没有 `faiss-gpu`，也可以改成 CPU 版或 `sklearn` 近邻后端。

## 快速开始

### 0. 上传 GitHub 前先处理路径

当前部分配置文件中仍保留了本机绝对路径，例如：

- `/home/txs/work/zyp/LLM/Qwen3-8B`
- `/home/txs/work/zyp/SFT_DPO/...`

如果你准备公开仓库，建议优先把这些路径改成：

- 相对路径
- 或在配置文件中留空，让使用者自行填写

重点检查：

- `train/configs/trl_qwen_customer_sft.yaml`
- `train/configs/trl_qwen_customer_dpo.yaml`

### 1. 准备基座模型

下载或放置一个可用的 Qwen 基座模型，并把配置中的 `model_name_or_path` 改成你的本地路径或 Hugging Face 模型名，例如：

```yaml
model:
  model_name_or_path: /path/to/Qwen3-8B
```

### 2. 运行 SFT

先检查 `train/configs/trl_qwen_customer_sft.yaml` 中的以下字段：

- `model.model_name_or_path`
- `dataset.train_file`
- `dataset.val_file`
- `output.output_dir`
- `output.artifact_dir`

然后启动训练：

```bash
python train/train_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

如果你只是想快速验证流程，也可以临时把数据路径改成仓库自带的数据：

```yaml
dataset:
  train_file: ./SFT_DPO/data_sft/train.jsonl
  val_file: ./SFT_DPO/data_sft/val.jsonl
```

### 3. 导出推理样例

```bash
python train/infer_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

默认会在输出目录中导出：

- `artifacts/inference_samples.json`

### 4. 自动评测推理结果

当前脚本已经改为从环境变量读取接口配置，运行前请先设置：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url
```

然后执行：

```bash
python train/evaluate_inference_judge.py \
  --input train/outputs/your_sft_checkpoint/artifacts/inference_samples.json \
  --model qwen-plus
```

默认会生成：

- `judge_results.jsonl`
- `judge_summary.json`
- `judge_summary.csv`

### 5. 构造 DPO 数据

确保你已经有一个可用的 SFT adapter，并设置好环境变量：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=your_base_url
```

然后运行：

```bash
python train/build_dpo_preference_data.py \
  --config train/configs/trl_qwen_customer_sft.yaml \
  --train-file SFT_DPO/data_sft/train.jsonl \
  --adapter-path /path/to/your_sft_adapter \
  --output-dir SFT_DPO/data_dpo
```

### 6. 运行 DPO

检查 `train/configs/trl_qwen_customer_dpo.yaml` 中的关键字段：

- `model.model_name_or_path`
- `model.sft_adapter_path`
- `dataset.train_file`
- `output.output_dir`

然后启动：

```bash
python train/train_trl_dpo.py --config train/configs/trl_qwen_customer_dpo.yaml
```

## 多卡与 DeepSpeed

项目已经为 TRL 训练流程预留了 DeepSpeed 配置入口。

如果需要启用 ZeRO-2，可在配置文件中设置：

```yaml
distributed:
  deepspeed: deepspeed/ds_zero2.json
```

然后使用：

```bash
torchrun --nproc_per_node=2 train/train_trl_sft.py --config train/configs/trl_qwen_customer_sft.yaml
```

或者：

```bash
torchrun --nproc_per_node=2 train/train_trl_dpo.py --config train/configs/trl_qwen_customer_dpo.yaml
```

## 训练产物

典型输出目录下通常会包含：

- LoRA adapter 权重
- tokenizer 相关文件
- `trainer_state.json`
- `train_results.json`
- `all_results.json`
- `artifacts/loss_curve.png`
- `artifacts/inference_samples.json`
- `artifacts/run_summary.json`

DPO 输出目录下还可能包含：

- `artifacts/dpo_metrics.jsonl`
- `artifacts/dpo_reward_curve.png`
- `artifacts/judge_summary.json`

## 评估指标

当前自动评测主要关注：

- `Accuracy`
- `Auto Resolve`
- `Closure`
- `CSAT`

其中 DPO 阶段尤其关注：

- 自动解决率是否提升
- 闭环率是否提升
- 严重幻觉是否减少

这比单纯追求“像不像训练集原句”更符合客服模型的实际可用性。

## 脚本说明

- `train/train_trl_sft.py`：SFT 主训练脚本
- `train/train_trl_dpo.py`：DPO 主训练脚本
- `train/infer_trl_sft.py`：导出推理样例
- `train/evaluate_inference_judge.py`：调用评审模型打分
- `train/build_dpo_preference_data.py`：构造 DPO 偏好对
- `train/rewrite_dpo_chosen.py`：对 DPO `chosen` 进行重写优化
- `train/plot_loss.py`：根据 `trainer_state.json` 重新绘制 loss 曲线
- `train/run_sft_ablation_experiments.sh`：SFT 消融实验脚本
- `train/run_dpo_beta_ablation_experiments.sh`：DPO beta 消融实验脚本


## 后续可继续改进的方向

- 把配置文件中的绝对路径全部改成相对路径。
- 提供一份最小可复现配置，方便别人快速跑通。
- 补一份 `run.sh` 或 `Makefile`，降低复现门槛。
- 增加更系统的离线评测集，分别覆盖物流、退款、售后、发票、库存等子场景。
- 把评测脚本、数据处理脚本和训练脚本整理成更统一的命令行风格。

## 相关文档

- [SFT_DATA_AND_TRAINING_REPORT.md](./SFT_DATA_AND_TRAINING_REPORT.md)
- [DPO_PROJECT_REPORT.md](./DPO_PROJECT_REPORT.md)
- [train/README.md](./train/README.md)

