# 电商客服 SFT 数据治理与训练复盘

本文档记录本轮电商客服 SFT 的数据处理、聚类清洗、FAQ 多轮化扩写、训练配置、问题排查、最终效果评估，以及进入 DPO 前的判断与建议。

对应本轮产物：

- 清洗与聚类脚本：`dataprocess/clean_customer_service_dataset.py`、`dataprocess/cluster_customer_service_dataset.py`
- 最终 SFT 数据：`SFT_DPO/data_clustered_5.7_1/all.jsonl`、`train.jsonl`、`val.jsonl`
- 聚类审计：`dataprocess/cluster_audit/report.json`、`cluster_sizes.jsonl`、`clusters_preview.json`
- 扩写缓存：`dataprocess/cluster_audit/faq_dialogue_augmentation_cache_qwen-plus.jsonl`
- 训练配置：`SFT_DPO/train/configs/trl_qwen_customer_sft.yaml`
- 最终模型输出样例：`SFT_DPO/train/outputs/qwen3-8b_lora_sft_5.7_1/checkpoint/artifacts/inference_samples.json`
- 训练摘要：`SFT_DPO/train/outputs/qwen3-8b_lora_sft_5.7_1/checkpoint/artifacts/run_summary.json`

## 1. 目标与核心思路

原始数据主要是电商客服 FAQ 问答，单轮居多。直接用于 SFT 会有几个明显问题：

- 问法重复、回答风格不一致，模型容易记住噪声。
- 同一类问题存在互相冲突或过度承诺的答案。
- FAQ 语言偏书面，缺少真实客服多轮场景。
- 物流、售后、退款、发票、库存等场景对事实边界要求高，模型容易编造。

本轮目标不是简单“把数据喂进去训练”，而是尽量构建一个更稳定、更贴近客服任务的 SFT 数据分布：

1. 先对基础 QA 做文本清洗和去重。
2. 再按问题语义聚类，合并相似问法。
3. 在簇内对答案做排序、冲突剔除、策略统一。
4. 对约 30% 数据做多轮化扩写，让模型学习上下文承接、追问、闭环、情绪处理。
5. 使用 Qwen3-8B 做 LoRA SFT。
6. 最后通过 inference samples 检查生成质量与剩余风险，为 DPO 做准备。

## 2. 基础数据清洗

清洗脚本为 `dataprocess/clean_customer_service_dataset.py`。

### 2.1 清洗做了什么

核心函数：

- `clean_text`：见 `dataprocess/clean_customer_service_dataset.py:199`
- `normalize_for_dedup`：见 `dataprocess/clean_customer_service_dataset.py:224`
- `is_low_value_question`：见 `dataprocess/clean_customer_service_dataset.py:232`
- `is_low_value_answer`：见 `dataprocess/clean_customer_service_dataset.py:245`

主要处理包括：

- HTML 反转义与标签清理。
- Unicode 规范化。
- 中英文标点统一。
- emoji、控制字符、零宽字符、乱码替换符清理。
- 日期、时间、带逗号数字标准化。
- 多空格、重复标点、首尾无意义符号清理。
- 过滤过短问题、过短回答。
- 过滤低价值问答，如“在吗”“好的”“请稍等”等。
- exact dedup 与 fuzzy dedup。

### 2.2 为什么先做清洗

如果不先清洗，后续聚类会被格式噪声干扰。例如：

- `多久能到？？`、`多久能到?`、`多久 能 到` 可能被当成不同问法。
- 含 emoji、HTML、零宽字符的文本会影响 embedding 与去重。
- 大量“好的”“在的”“稍等”等低价值回复会训练出模板化客服。

清洗阶段解决的是“数据能不能作为训练语料”的基础问题。

## 3. 聚类与答案稳定化

聚类脚本为 `dataprocess/cluster_customer_service_dataset.py`。

### 3.1 聚类总体流程

脚本头部定义的 pipeline 是：

1. 读取 ShareGPT 风格 `jsonl`。
2. 对问题做 embedding / TF-IDF 表征。
3. 用 FAISS 或 sklearn 近邻搜索构图。
4. 用 Union-Find 合并相似问题。
5. 每个问题簇内做答案家族聚合、质量排序、矛盾剔除。
6. 导出 `all/train/val` 与审计文件。

关键实现：

- embedding 构建：`build_embeddings`
- 近邻图：`build_neighbor_graph`，见 `dataprocess/cluster_customer_service_dataset.py:419`
- Union-Find 聚类：`cluster_questions`，见 `dataprocess/cluster_customer_service_dataset.py:474`
- 答案家族合并：`merge_answer_families`，见 `dataprocess/cluster_customer_service_dataset.py:736`
- 大簇快速路径：`build_singleton_answer_families`，见 `dataprocess/cluster_customer_service_dataset.py:780`

最终采用的核心参数来自 `dataprocess/cluster_audit/report.json`：

```json
{
  "embedding_backend_used": "sentence-transformers:cuda",
  "ann_backend_used": "faiss-gpu",
  "question_similarity_threshold": 0.95,
  "neighbors_k": 25,
  "answer_family_threshold": 0.82,
  "max_answers_per_cluster": 2,
  "unify_time_answers": true,
  "time_policy_upper_percentile": 0.85
}
```

### 3.2 问题：语义阈值默认过低导致过聚类

**现象**

使用 sentence-transformers + FAISS 时，如果使用默认阈值 `0.83`，最大簇会变得非常大，出现严重过聚类。

曾观察到：

- 最大簇超过一万条。
- 很多不同业务意图被合进同一簇。
- 输出记录过少，训练分布被压扁。

**为什么**

客服问题很多都是短句，例如：

- `支持吗`
- `可以退吗`
- `多久到`
- `有发票吗`

短文本 embedding 在语义空间里容易距离过近，特别是都带有“能否”“可以”“怎么”等客服高频词时，低阈值会让不同业务意图互相连通。Union-Find 一旦通过若干边连起来，会形成大连通分量，导致雪球式合并。

**试了什么**

- 使用 `sentence-transformers:cuda` + `faiss-gpu` 保证语义表征质量和速度。
- 将 `similarity-threshold` 提升到 `0.95`。
- 加入 `question_topics` 规则，只有主题兼容的问题才允许通过近邻边合并。

**解决了吗**

基本解决。最终聚类结果：

```text
clusters_total: 6332
multi_record_clusters: 1762
raw_records_total: 24559
kept_records_total: 9408
```

最大几个簇更合理：

```text
cluster 0: delivery_time, 587 条，canonical_question=多久可以到货?
cluster 1: invoice, 462 条，canonical_question=能否提供发票?
cluster 2: return, 436 条，canonical_question=退货流程是怎样的?
cluster 3: generic, 331 条，canonical_question=如何查询订单状态?
cluster 4: custom, 330 条，canonical_question=能否提供定制服务?
```

仍然有 4570 个单样本簇，这是客服 FAQ 的正常现象：长尾问题多，且相似度阈值为了防止误合并必须保持较严。

### 3.3 问题：短句同义问法被分到不同簇

**现象**

最初发现：

- `多久能到?`
- `多久可以到?`

被分到了不同簇。

**为什么**

在高阈值 `0.95` 下，短句中“能”和“可以”的差异会明显影响 embedding 相似度；而这两个句子本质上是同一个物流时效意图。

**试了什么**

加入保守的确定性语义归一化函数 `semantic_question_key`，见 `dataprocess/cluster_customer_service_dataset.py:376`。

它只针对非常明确的短时间类问法做别名合并：

```text
多久能到?
多久可以到?
请问多久可以到?
快递多久能到?
物流多久能到?
```

归一到：

```text
semantic:delivery_time:多久到
```

同时排除容易有不同政策含义的问题：

```text
最快多久能到?
经济型物流多久能到?
国际物流多久能到?
指定配送时间相关问题
```

**解决了吗**

解决。最终 `多久能到?` 已进入 `cluster_id=0` 的物流时效大簇：

```text
cluster_id: 0
question_type: delivery_time
canonical_question: 多久可以到货?
cluster_size: 587
unique_questions: 106
```

### 3.4 问题：同簇答案冲突、承诺不稳定

**现象**

同一类问题里，答案常常互相矛盾。例如物流时效可能同时出现：

- 1-3 天
- 3-7 天
- 24 小时发货
- 偏远地区 7 天
- 国际物流 7-30 天

如果全部保留，模型会学到不稳定政策，甚至产生过度承诺。

**为什么**

原始 FAQ 数据并不是一个统一店铺政策库，而是多个 FAQ 样式混合数据。不同来源、不同商品、不同地区、不同活动规则混在一起，回答自然会冲突。

**试了什么**

脚本里对答案做了多层处理：

1. `classify_question_type` 按问题主题分类。
2. `classify_stance` 判断正向/负向/未知。
3. `extract_time_ranges` 抽取时间范围。
4. `answer_quality_score` 给答案打分。
5. `detect_contradictions` 对冲突答案降权或剔除。
6. 对时间类问题使用保守统一策略。

时间类统一策略见 `build_time_policy_answer`，位置：`dataprocess/cluster_customer_service_dataset.py:590`。

最终写入的保守策略包括：

- 发货时间：通常 `1个工作日至3个工作日` 安排发货。
- 到货时间：发出后通常 `3个工作日至7个工作日` 送达。
- 国际/跨境：受目的地、清关、承运商、节假日影响，以页面预计时效和物流追踪为准。

**解决了吗**

大部分解决。物流时效类回答从混乱承诺统一为保守客服口径，减少了 SFT 中的政策冲突。

但后续 inference samples 仍发现模型会在订单场景里编造更具体的时间、金额、状态。这说明仅靠 SFT 数据清洗不够，后续 DPO 仍需要专门压制“无依据具体化”。

### 3.5 问题：自提、破损等场景需要保守政策回答

**现象**

某些场景如果简单聚类排序，容易保留过于绝对的答案。例如：

- “是否支持自提”
- “收到商品损坏怎么办”

不同店铺/商品/仓库政策差异很大。

**为什么**

这些不是纯语义问题，而是强依赖商品、仓库、地区、订单状态的政策问题。

**试了什么**

加入保守策略回答，见 `build_conservative_policy_answer`，位置：`dataprocess/cluster_customer_service_dataset.py:678`。

例如自提回答不直接承诺支持，而是：

- 以下单页可选配送方式为准。
- 如果页面支持自提，可选择自提点和取货时间。
- 如果页面没有自提选项，则默认快递配送。
- 具体商品是否可自提，建议下单前确认。

**解决了吗**

解决了大部分“强承诺”问题。训练数据中此类场景更保守，符合客服助手要求。

### 3.6 问题：大簇答案家族合并太慢

**现象**

使用 sentence-transformers + FAISS GPU 后，embedding 和近邻搜索很快，但后处理阶段会长时间卡住。

**为什么**

瓶颈不在 GPU，而在 Python 侧答案家族合并。`merge_answer_families` 使用 `SequenceMatcher` 比较答案文本相似度，大簇内答案多时，近似成两两比较，复杂度很高。

**试了什么**

做了两层优化：

1. 长度上界剪枝：如果两个字符串长度差异太大，理论最大相似度达不到阈值，就跳过 `SequenceMatcher`。
2. 大簇快速路径：当簇大小超过 `LARGE_CLUSTER_FAST_FAMILY_THRESHOLD=100`，或者属于时间类/保守策略类问题，直接构造 singleton answer families，跳过昂贵答案相似合并。

相关代码：

- `LARGE_CLUSTER_FAST_FAMILY_THRESHOLD`：`dataprocess/cluster_customer_service_dataset.py:125`
- 大簇快速路径判断：`dataprocess/cluster_customer_service_dataset.py:893`
- `build_singleton_answer_families`：`dataprocess/cluster_customer_service_dataset.py:780`

**解决了吗**

解决。最终可以稳定跑完 sentence-transformers + FAISS GPU 聚类，并输出审计和训练数据。

代价是：大簇内部不再精细合并相似答案，而是依赖后续排序、保守策略和 `max_answers_per_cluster` 控制输出规模。对本任务来说可接受，因为大簇主要用于稳定政策口径，不追求保留所有答案变体。

## 4. FAQ 多轮化扩写

### 4.1 为什么要扩写

原始数据大量是单轮 FAQ：

```json
[
  {"from": "user", "value": "可以开发票吗?"},
  {"from": "assistant", "value": "可以,请联系客服提供开票信息。"}
]
```

这种数据能教会模型回答单问，但不擅长：

- 多轮上下文承接。
- 用户追问后的补充说明。
- 必要时追问关键信息。
- 用户催促/不满时的情绪处理。
- 给出闭环操作路径。

因此加入约 30% 多轮改写数据。

### 4.2 扩写规则

扩写系统提示词见 `AUGMENT_SYSTEM_PROMPT`，位置：`dataprocess/cluster_customer_service_dataset.py:1050`。

目标是把 FAQ 改写为真实、多轮、可用于客服 SFT 的对话数据。

要求包括：

- 用户口语化，可包含疑惑、催促、不满。
- 客服优先给明确结论。
- 必要时追问。
- 提供操作路径或下一步建议。
- 简洁自然，不要模板废话。
- 禁止“请稍等”“帮您查询”“占位符”“索要好评”。
- 不编造 FAQ 中没有的时间、金额、政策承诺。
- 2-4 轮对话。

本地校验函数 `validate_augmented_dialogue` 会检查：

- 输出必须是 JSON object。
- 必须包含 `messages`。
- 消息数量 4-8 条，即 2-4 轮。
- user/assistant 必须交替。
- 内容不能为空。
- 禁止词不能出现。

### 4.3 抽样策略：从“每簇 30%”改成“覆盖优先 30%”

**问题**

一开始考虑每个聚类抽 30%，但最终输出数据中单样本簇很多：

```text
all.jsonl: 9408 条
clusters: 6332 个
30% 目标: 2822 条
```

如果每簇至少抽 1 条，会超过总体 30%。如果每簇按 30% 四舍五入，又会漏掉大量单样本簇。

**为什么**

客服 FAQ 是明显长尾分布：头部簇很大，尾部大量单样本簇。单一“每簇比例”无法同时满足“总体 30%”和“尽量覆盖每个簇”。

**试了什么**

实现 `coverage` 抽样策略，见 `choose_augmentation_indices`，位置：`dataprocess/cluster_customer_service_dataset.py:1293`。

逻辑：

1. 先计算总预算：`round(total_records * augment_ratio)`。
2. 打乱簇顺序。
3. 每个簇先抽 1 条，直到预算耗尽。
4. 如果预算还有剩，再按簇继续补抽。

最终：

```text
selected: 2822
succeeded: 2797
failed: 25
实际改写比例: 2797 / 9408 = 29.73%
```

**解决了吗**

解决。总体接近 30%，同时覆盖尽可能多的簇。

### 4.4 API 扩写实现方案

核心函数：

- `openai_chat_json`：`dataprocess/cluster_customer_service_dataset.py:1212`
- `post_openai_chat_json`：`dataprocess/cluster_customer_service_dataset.py:1263`
- `generate_augmented_dialogue`
- `augment_clustered_records`：`dataprocess/cluster_customer_service_dataset.py:1380`

实现要点：

1. 使用 OpenAI-compatible `/chat/completions`。
2. 优先尝试 JSON Schema 结构化输出。
3. 如果渠道不支持 JSON Schema，自动 fallback 到普通 chat 输出，再用本地 JSON 解析与校验兜底。
4. 支持并发请求：`ThreadPoolExecutor` + `as_completed`。
5. 支持缓存：成功结果写入 JSONL cache，重跑时先读缓存。
6. 支持中断恢复：已经成功的 cache 不会重复请求。
7. 支持不同模型使用不同 cache 文件，避免混用。

本轮最终使用：

```text
augment_model: qwen-plus
augment_cache: dataprocess/cluster_audit/faq_dialogue_augmentation_cache_qwen-plus.jsonl
augment_concurrency: 8
augment_ratio: 0.3
augment_sampling_strategy: coverage
```

### 4.5 问题：不同 API 站点与模型名不一致

**现象**

使用 `https://www.dmxapi.cn/v1` 时返回 token 无效；使用 `https://www.dmxapi.com/v1` 成功。

使用 `qwenplus` 时返回：

```text
model_not_found
```

**为什么**

该 token 属于国际站，应该使用：

```text
https://www.dmxapi.com/v1
```

同时平台实际模型名是：

```text
qwen-plus
```

而不是：

```text
qwenplus
```

**试了什么**

用轻量请求分别探测：

- `qwenplus`
- `qwen-plus`
- `gpt-4o`

确认 `qwen-plus` 普通 chat 可用。

**解决了吗**

解决。最终使用 `qwen-plus`。

### 4.6 问题：qwen-plus 不稳定支持 JSON Schema

**现象**

qwen-plus 普通 chat 请求成功，但带 `response_format: json_schema` 时可能断开连接。

**为什么**

部分 OpenAI-compatible 中转渠道并不完整支持 OpenAI JSON Schema 参数。

**试了什么**

实现 fallback：

1. 优先发送 JSON Schema 请求。
2. 如果失败，改为普通 chat 请求。
3. 对返回内容做更宽容的 JSON 提取：支持纯 JSON、```json 代码块、前后带少量文本的 JSON。
4. 解析后仍用 `validate_augmented_dialogue` 严格校验。

相关函数：

- `parse_json_content`
- `validate_augmented_dialogue`

**解决了吗**

解决。qwen-plus 小样本测试通过，最终完成 2797 条多轮改写。

### 4.7 问题：API 余额不足、连接断开、任务中断

**现象**

改写过程中出现过：

- `HTTP Error 403: Forbidden`
- `Remote end closed connection without response`
- `Connection reset by peer`

**为什么**

一部分是余额不足，一部分是并发请求导致上游连接不稳定。

**试了什么**

- 使用 cache 保存成功样本，保证中断后不丢。
- 从 16 并发降到 8、4、2 做稳定性测试。
- 增加 timeout 和 retries。
- 充值后继续从 cache 断点续跑。
- qwen-plus 使用独立 cache，不复用旧 gpt-4o cache。

**解决了吗**

解决。最终 qwen-plus 扩写完成：

```text
selected: 2822
succeeded: 2797
failed: 25
cache_hits: 141
```

失败的 25 条保留原始 FAQ，不影响整体数据可用性。

## 5. 最终数据集

最终数据来自 `SFT_DPO/data_clustered`。

### 5.1 数据规模

```text
all.jsonl total: 9408
train.jsonl total: 9220
val.jsonl total: 188
```

### 5.2 多轮改写比例

```text
all.jsonl:
  改写多轮: 2797
  原始 FAQ: 6611
  改写比例: 29.73%

train.jsonl:
  改写多轮: 2744
  原始 FAQ: 6476
  改写比例: 29.76%

val.jsonl:
  改写多轮: 53
  原始 FAQ: 135
  改写比例: 28.19%
```

### 5.3 对话轮数

```text
单轮 FAQ: 6611 条，conversations 长度为 2
多轮改写: 2797 条，conversations 长度为 4，即 2 轮对话
```

这说明当前 qwen-plus 扩写主要生成 2 轮对话，满足 2-4 轮要求的下界。

### 5.4 问题类型分布

`all.jsonl` 中 Top question types：

```text
yesno_generic: 2727
generic: 2704
return: 962
aftersales: 613
money: 452
delivery_time: 379
custom: 275
logistics_query: 209
warranty: 187
payment: 115
stock: 115
ship_time: 111
```

### 5.5 长度分布

按字符粗略统计：

```text
all.jsonl char length:
  p50: 95
  p90: 239
  p95: 264
  p99: 309
  max: 446
```

使用 Qwen3-8B tokenizer，加上 system prompt 和 chat template 后统计：

```text
all:
  p50: 229
  p90: 350
  p95: 369
  p99: 407
  max: 507
  >1024: 0

augmented:
  p50: 338
  p90: 382
  p95: 397
  p99: 429
  max: 507
  >1024: 0

FAQ:
  p50: 223
  p90: 240
  p95: 243
  p99: 254
  max: 285
  >1024: 0
```

因此训练配置中的 `max_seq_length: 1024` 是充足的。

## 6. SFT 训练配置

训练配置文件：`SFT_DPO/train/configs/trl_qwen_customer_sft.yaml`。

核心配置：

```yaml
model:
  model_name_or_path: /home/txs/work/zyp/LLM/Qwen3-8B
  max_seq_length: 1024
  eos_token: "<|im_end|>"
  packing: false

dataset:
  train_file: ./data_clustered/train.jsonl
  val_file: ./data_clustered/val.jsonl

lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - v_proj

training:
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  learning_rate: 1.0e-4
  num_train_epochs: 3.0
  lr_scheduler_type: cosine
  warmup_ratio: 0.1
  gradient_checkpointing: true
  assistant_only_loss: true
```

### 6.1 System Prompt 设计

训练时使用了较强约束的客服 system prompt，要求：

- 自然、简洁、礼貌。
- 优先解决当前问题。
- 不输出思考过程或模板标签。
- 不编造订单、物流、金额、时间等事实。
- 信息不足时可以追问必要信息。
- 禁止无关寒暄和模板话术。
- 涉及操作必须给路径或下一步建议。

这是对数据侧规则的补充：数据清洗减少噪声，system prompt 提供行为边界。

## 7. SFT 训练结果

最终训练摘要来自：

`SFT_DPO/train/outputs/qwen3-8b_lora_sft_5.7_1/checkpoint/artifacts/run_summary.json`

关键指标：

```text
model_name_or_path: /home/txs/work/zyp/LLM/Qwen3-8B
train_samples: 9220
eval_samples: 188
global_step: 3459
epoch: 3.0
best_metric/eval_loss: 1.1247353553771973
best_model_checkpoint: checkpoint-3459
```

训练结果：

```text
train_loss: 1.221417716970055
train_runtime: 8627.0544s
train_samples_per_second: 3.206
train_steps_per_second: 0.401
```

最后一次 eval：

```text
eval_loss: 1.1247353553771973
eval_mean_token_accuracy: 0.6661805669043926
eval_samples_per_second: 10.607
```

### 7.1 整体效果

从 `inference_samples.json` 抽查 100 条：

```text
样本数: 100
空回复: 0
超短回复: 0
过长回复: 0
平均 prediction 长度: 约 71 字
多轮 prompt: 16
单轮 prompt: 84
```

禁词/风险词只统计 prediction：

```text
请稍等: 0
帮您查询: 0
好评: 0
模板: 0
作为AI: 0
亲亲: 12
承诺: 1
立刻: 1
最晚: 1
已为您: 3
今天: 2
3天后: 1
```

这说明模型已经明显避免了常见模板废话，也没有出现空回复、格式崩坏等问题。

### 7.2 正向样例

物流节假日问题：

```text
用户：配送时间会因为节假日有所变动吗?
模型：通常情况下,商品发出后一般会在3个工作日至7个工作日内送达。 具体到货时间会受收货地区、快递线路、天气和节假日影响,请以物流追踪信息为准;如需更准确时效,可以联系客服提供收货地区后确认。
```

这个回答优点：

- 没有承诺具体送达日期。
- 说明影响因素。
- 给出下一步建议。
- 与保守物流策略一致。

海外退换货问题：

```text
用户：是否支持海外退换货服务?
模型：我们目前主要提供国内地区的退换货服务,对于海外客户,由于国际物流和海关限制较多,我们暂时不支持直接的海外退换货服务。 但您可以考虑使用第三方国际快递公司进行退货,并自行承担相关费用。
```

优点：

- 结论明确。
- 说明原因。
- 不绕弯。

运输破损问题：

```text
用户：如果商品在运输过程中损坏了怎么办?
模型：若您的商品在运输过程中受损,请立即联系我们并提供相关证据(如照片或视频)。 我们将根据具体情况协助您进行索赔,并尽快为您补发或退款。
```

优点：

- 操作明确。
- 要求凭证。
- 给出可能处理方向。

### 7.3 仍然存在的问题

#### 问题 1：具体订单/物流/日期/金额幻觉

在 100 条样例中发现 5 条明显高风险样例。

例如：

```text
用户：哦,但我家在内蒙古呼伦贝尔,是不是会慢一点?
模型：呼伦贝尔属于偏远地区,我们已为您加急匹配中转线路,实际送达时效将控制在5个工作日内。 如您3天后仍未更新物流,请直接回复本消息,我立刻为您同步催派单并升级处理。
```

问题：

- “已为您加急匹配”没有依据。
- “控制在5个工作日内”是强承诺。
- “3天后”“立刻同步催派单”属于未经工具确认的具体承诺。

又如：

```text
模型：系统已为您自动开通7天无理由退货+全额补偿(80元)权益...
```

问题：

- 编造补偿金额。
- 编造系统自动开通权益。

**原因**

一部分来自训练/评测 reference 本身存在具体订单状态、日期、金额、自动处理结果等脏样本。SFT 会模仿 reference 的表达方式，即使 system prompt 禁止编造，也无法完全抵消数据中的示范。

**是否解决**

部分解决。大多数普通问题已经保守，但订单状态类、多轮售后类仍会幻觉。这个问题建议进入 DPO 阶段重点解决。

#### 问题 2：少量“亲亲”口吻残留

100 条 prediction 中 `亲亲` 出现 12 次。

**原因**

原始电商客服数据中“亲亲”类话术较多，即便清洗和质量评分做了惩罚，仍有部分留存。

**是否解决**

部分解决。没有出现“请稍等”“帮您查询”“索要好评”等更严重模板话术，但如果目标是更专业客服风格，DPO 或二次数据过滤可以继续压低“亲亲”。

#### 问题 3：有些回答路径过具体但可能不适用

例如：

```text
请打开[我的订单]→找到该笔订单→点击[去支付]→进入支付页后,在「商品信息」下方看到「延保服务」选项...
```

如果平台真实路径不是这样，会误导用户。

**原因**

扩写数据鼓励“提供可执行路径”，但模型有时会把路径细节补得过满。

**是否解决**

部分解决。多数路径是合理泛化，但仍需 DPO 约束：不知道具体平台路径时，应说“可在订单页/售后页查看对应入口”，不要编造具体按钮名。

## 8. 是否可以进入 DPO

结论：**可以进入 DPO 准备阶段，但不建议直接无脑大规模 DPO。**

这版 SFT 已经具备：

- 稳定客服语气。
- 基本问题闭环能力。
- 多轮上下文承接能力。
- 较少模板废话。
- 较好的物流/售后/退换货常见问题回答。

但还存在：

- 订单状态幻觉。
- 具体时间/金额/补偿承诺幻觉。
- 少量淘宝式亲昵称呼。
- 部分操作路径过具体。

因此 DPO 应该聚焦“客服边界”：

```text
chosen:
  明确结论 + 不编造事实 + 信息不足时追问 + 给通用可执行路径

rejected:
  编造物流状态/订单状态/金额/日期/自动处理结果
  过度承诺
  亲亲/模板话术
  无依据具体按钮路径
```

不建议把现有 reference 全部当 chosen，因为 reference 中也有脏样本。更合理的方式是从 SFT inference 中抽取高风险 prediction，人工或规则辅助构造 rejected，再用保守客服答案作为 chosen。

建议先做一个小而精的 DPO 数据集：

```text
规模：1000-3000 条
重点场景：
  物流时效
  订单状态
  退款/退货
  售后赔付
  发票
  海外配送
  库存
  地址修改
```

## 9. 本轮经验总结

### 9.1 数据比训练更关键

单纯训练模型无法解决数据中的冲突和幻觉示范。只有先做聚类、答案统一、保守策略重写，模型输出才会稳定。

### 9.2 高阈值语义聚类 + 规则兜底比低阈值更安全

低阈值会严重过聚类；高阈值会漏掉少量短句同义。最终选择：

```text
sentence-transformers + FAISS GPU + threshold 0.95 + semantic_question_key 兜底
```

这是比较稳的折中。

### 9.3 多轮扩写要控制覆盖，而不是只看比例

客服问题长尾很多。如果只按每簇比例抽样，很多单样本簇永远不会多轮化。coverage 策略让 30% 预算尽可能覆盖更多问题类型，训练价值更高。

### 9.4 API 生成必须有缓存和校验

如果没有 cache，余额不足、连接断开、模型切换都会导致大量重复请求和成本浪费。本轮实现的 JSONL cache 让任务可以随时中断、随时恢复。

如果没有本地校验，LLM 扩写会混入格式错误、禁词、角色错位、空回复等问题。本轮通过 `validate_augmented_dialogue` 保证了最终入库质量。

### 9.5 SFT 之后仍需要 DPO

SFT 学会“怎么回答”，但不一定学会“哪些话不能说”。客服场景中，不能编造订单状态、不能承诺具体时间金额，是偏好边界问题，非常适合用 DPO 继续对齐。

## 10. 推荐下一步

1. 从当前 `inference_samples.json` 和更多自定义 prompts 中收集模型高风险输出。
2. 构建 DPO pair：
   - chosen：保守、可执行、不编造。
   - rejected：SFT 当前幻觉/过度承诺/模板话术版本。
3. 优先覆盖物流、退款、售后、订单状态、金额赔付。
4. 先训练小规模 DPO，观察是否降低幻觉与亲亲话术。
5. 用同一批固定 eval prompts 做 SFT vs DPO 对比。

最终判断：**这版 SFT 值得作为 DPO base。**

