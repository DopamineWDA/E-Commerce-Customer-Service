# DPO 阶段项目总结报告

## 1. 阶段目标与最终结论

DPO 阶段的目标不是重新学习电商客服知识本身，而是在已经完成 SFT 的基础上，把模型从“能回答”进一步推向“更像可上线客服”：更少空话、更明确下一步动作、更少虚构后台状态、更高闭环率。

本项目最终有效的 DPO 消融应以 3 epochs 实验为准：

`/home/txs/work/zyp/SFT_DPO/train/outputs/dpo_beta_ablation/dpo_pairs_single_gpu_0513_1434`

该目录下三组 beta 消融均使用同一份 DPO 数据、同一 SFT adapter、同一评估集和 judge 流程，训练轮数均为 3 epochs。早期目录：

`/home/txs/work/zyp/SFT_DPO/train/outputs/dpo_beta_ablation/dpo_pairs`

其中 beta=0.05/0.1/0.3 主要是 2 epochs 结果，虽然可以作为探索记录，但由于训练步数较少，不应作为最终有效消融结论的主要依据。

最终 3 epochs 消融结果如下：

| 实验 | beta | epoch | global step | eval reward margin | Accuracy | Auto Resolve | Closure | CSAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rank16_qvko_beta0p05 | 0.05 | 3.0 | 207 | 0.0126 | 73.40 | 63.83 | 63.83 | 3.601 |
| rank16_qvko_beta0p1 | 0.10 | 3.0 | 207 | 0.0237 | 72.34 | 63.83 | 64.36 | 3.580 |
| rank16_qvko_beta0p3 | 0.30 | 3.0 | 207 | 0.0621 | 73.40 | 66.49 | 65.96 | 3.628 |

综合判断：beta=0.3 是本阶段最有效配置。它没有带来最高 Accuracy，但显著提升了 DPO 最关心的自动解决率、闭环率和总体满意度，并且训练内部偏好信号最强，eval reward margin 最高。

与 SFT 主力模型 rank16_qvko 的 judge 结果相比：

| 模型 | Accuracy | Auto Resolve | Closure | CSAT |
|---|---:|---:|---:|---:|
| SFT rank16_qvko | 75.14 | 61.33 | 60.77 | 3.608 |
| DPO beta=0.3, 3 epochs | 73.40 | 66.49 | 65.96 | 3.628 |

DPO 后 Accuracy 略降 1.74 个百分点，但 Auto Resolve 提升 5.16 个百分点，Closure 提升 5.19 个百分点，CSAT 小幅提升 0.020。这个结果符合 DPO 阶段的定位：牺牲极少量逐字准确性，换取更强的可执行性和客服闭环感。

## 2. 数据构造：从 SFT 样本到 DPO preference pair

### 2.1 为什么不能直接用原始参考答案做 chosen

最初的问题是：原始客服数据虽然可以用于 SFT，但不天然适合 DPO。原因包括：

1. 很多参考答案只是“可接受答案”，不一定是同一上下文下明显优于另一个回复的偏好样本。
2. 原始答案里存在客服常见但高风险的话术，例如“系统已处理”“已为您加急”“预计今晚发出”等，这类表达对 SFT 来说可能只是模仿语料，对 DPO 来说却会被强化成偏好。
3. DPO 需要的是 chosen/rejected 的相对差异，而不是单条答案的绝对好坏。如果 chosen 与 rejected 都很泛，训练信号会很弱；如果 rejected 全是严重错误，模型只学会避开极端坏样本，对“泛泛但安全”这种真实短板帮助有限。

因此最终采用了“同一 prompt 下多候选生成 + judge 排序”的方式构造偏好对。

### 2.2 数据生成流程

核心脚本是：

`SFT_DPO/train/build_dpo_preference_data.py`

流程如下：

1. 从 SFT 训练集随机采样 3000 条样本。
2. 保留 system prompt、历史对话和最后一轮 user 问题，移除原最后 assistant 回复，得到 DPO prompt。
3. 使用 SFT adapter 生成多个候选回复。默认采样温度为 0.5、0.7、0.9，top_p 为 0.9，max_new_tokens 为 160。
4. 对候选去重，如果唯一候选少于 2 个，则跳过该样本。
5. 使用 qwen-plus 作为偏好标注 judge，对候选按照 hallucination、accuracy、closure、auto_resolve、satisfaction、overall_score 打分排序。
6. 排除严重幻觉候选作为 chosen 的可能性。
7. 构造 TRL conversational DPO 格式：

```json
{
  "prompt": [...],
  "chosen": [{"role": "assistant", "content": "..."}],
  "rejected": [{"role": "assistant", "content": "..."}]
}
```

### 2.3 最终 DPO 数据规模

最终主用数据位于：

`SFT_DPO/data_dpo/dpo_pairs.jsonl`

数据构造统计：

| 阶段 | 数量 |
|---|---:|
| sampled_prompts | 3000 |
| sft_candidates | 3000 |
| qwen_ranked | 2714 |
| failed_rankings | 202 |
| skipped_low_diversity_rankings | 84 |
| raw_pairs | 2714 |
| dropped_pairs | 271 |
| kept_pairs | 2443 |

失败与跳过主要来自两类问题：

1. 低多样性：84 条样本在多温度采样后仍不足 2 个唯一候选。这说明部分 FAQ 型问题已经被 SFT 学得过于稳定，候选之间几乎没有偏好差异。
2. 全候选严重幻觉：202 条 ranking 失败，错误信息为“all candidates have severe hallucination=1; no valid chosen candidate”。这类样本多发生在多轮客服上下文中，用户追问订单、补发、审核、物流状态等细节，而 SFT 候选容易顺着原参考答案编造后台事实。

最终 2443 条 DPO pair 的质量摘要：

| 指标 | 数值 |
|---|---:|
| chosen 平均长度 | 74.3 字 |
| chosen 中位长度 | 72 字 |
| rejected 平均长度 | 72.3 字 |
| rejected 中位长度 | 70 字 |
| score_delta 均值 | 73.91 |
| score_delta 中位数 | 51.72 |
| score_delta 最小值 | 3.14 |
| score_delta 最大值 | 202.0 |

问题类型分布 Top10：

| question_type | 数量 |
|---|---:|
| yesno_generic | 730 |
| generic | 714 |
| return | 257 |
| aftersales | 161 |
| money | 119 |
| custom | 96 |
| delivery_time | 62 |
| logistics_query | 51 |
| warranty | 49 |
| payment | 36 |

数据集中 yes/no 泛问题和通用咨询占比较高，这解释了后续 DPO 改善主要集中在“下一步路径”“闭环表达”“条件判断”，而不是深层业务知识。

### 2.4 偏好强度过滤

构造完成后没有直接使用所有 raw pairs，而是按 best_score - worst_score 的偏好强度丢弃最低 10%。最终阈值为 3.14：

| 指标 | 数值 |
|---|---:|
| ranked_records | 2714 |
| raw_pairs | 2714 |
| discard_fraction | 0.10 |
| drop_threshold | 3.14 |
| kept_pairs | 2443 |
| dropped_pairs | 271 |

这样做的原因是：DPO 对标签噪声比较敏感，如果 chosen/rejected 差异很小，模型会被迫学习不稳定的微小风格差异。早期数据缓存 `data_cache/data_dpo_0` 曾使用 30% 丢弃比例，只保留 2041 条，但后续发现过度丢弃会减少覆盖面，因此主线改为更温和的 10%。

## 3. 训练方法与关键技术细节

### 3.1 基础模型与 adapter

DPO 从 SFT 最优 adapter 继续训练：

| 配置 | 值 |
|---|---|
| Base model | `/home/txs/work/zyp/LLM/Qwen3-8B` |
| SFT adapter | `/home/txs/work/zyp/SFT_DPO/train/outputs/sft_ablation/rank16_qvko/checkpoint` |
| LoRA rank | 16 |
| LoRA target modules | q_proj, v_proj, k_proj, o_proj |
| DPO trainable parameters | 15,335,424 |
| Total parameters | 8,221,406,208 |
| Trainable ratio | 0.18653% |

DPO 训练时只更新 LoRA adapter，base model 冻结。

### 3.2 DPO reference model 的处理

一个关键问题是 reference policy 应该是谁。

对于“先 SFT，再 DPO”的流程，DPO 的 reference policy 应该是 DPO 开始前的 SFT 模型，而不是裸 base model。否则 reward 计算会变成“当前 adapter 相对 base model 的变化”，这会把 SFT 阶段学到的客服能力也混进 DPO reward，导致数值异常且优化方向不稳定。

项目中在 `train_trl_dpo.py` 里实现了 `clone_peft_adapter`：

1. 加载 SFT adapter 作为可训练的 `default` adapter。
2. 复制一份同权重 adapter，命名为 `ref`。
3. 将 `ref` adapter 冻结。
4. 在 TRL DPOConfig 中传入 `model_adapter_name=default` 和 `ref_adapter_name=ref`。

这样 policy 与 reference 的初始状态完全一致，DPO reward 只反映 DPO 训练带来的偏移。

早期没有正确使用 SFT ref 的实验表现异常：例如 `rank16_qvko_beta0.1` 1 epoch 的 train_loss 达到 2.363，eval reward margin 为负，chosen/rejected reward 都变成十几量级，说明 reference 不合适或训练目标被放大。修正 reference adapter 后，loss 回到约 0.69，reward margin 从接近 0 开始逐步拉开，训练稳定很多。

### 3.3 训练超参数

最终 beta 消融使用的共同配置：

| 参数 | 值 |
|---|---|
| loss_type | sigmoid |
| learning_rate | 5e-7 |
| num_train_epochs | 3.0 |
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 16 |
| effective batch size | 32 |
| lr_scheduler_type | cosine |
| warmup_ratio | 0.1 |
| weight_decay | 0 |
| max_grad_norm | 1.0 |
| max_seq_length | 1024 |
| validation_split | 0.1 |
| train samples | 2198 |
| eval samples | 245 |
| metric_for_best_model | rewards/margins |
| greater_is_better | true |

显存情况：单张 RTX 3090，峰值 allocated 约 20.6 GB，reserved 约 23.7 GB。训练 3 epochs 约 4992 秒。

### 3.4 为什么选择 rewards/margins 作为 best metric

DPO 训练中 eval_loss 可以反映优化难度，但不一定直接代表偏好区分能力。项目最终更关注模型是否把 chosen 拉到 rejected 前面，因此使用 `rewards/margins` 作为 best metric：

`reward_margin = reward(chosen) - reward(rejected)`

3 epochs 消融中，beta 越大，eval reward margin 越明显：

| beta | eval reward margin | eval reward accuracy |
|---:|---:|---:|
| 0.05 | 0.0126 | 0.6423 |
| 0.10 | 0.0237 | 0.6260 |
| 0.30 | 0.0621 | 0.6382 |

beta=0.3 的 margin 最大，说明它对偏好边界的推动最强；同时 judge 结果也显示它在闭环和自动解决方面最好，因此训练指标与外部 judge 指标方向一致。

## 4. 评估方法

### 4.1 推理评估集

消融实验统一在：

`/home/txs/work/zyp/SFT_DPO/data_sft/val.jsonl`

上进行推理，最终 judge 样本数为 188。推理脚本为：

`SFT_DPO/train/infer_trl_sft.py`

每个 DPO checkpoint 推理后保存到：

`checkpoint/artifacts/infer.jsonl`

随后由 judge 脚本评估。

### 4.2 Judge 指标

评估脚本：

`SFT_DPO/train/evaluate_inference_judge.py`

judge 使用 qwen-plus，对模型回复按四个维度评分：

1. Accuracy：是否准确回答用户最后问题，且无明显虚构或错误政策。
2. Auto Resolve：是否给出明确下一步动作、操作路径或可执行方案。
3. Closure：是否形成闭环，让用户知道结论或后续判断。
4. Satisfaction：1 到 5 分客服满意度。

该 judge 不要求模型与参考答案逐字一致，更关注真实客服体验。这一点很重要，因为 DPO 阶段的目标本来就是从“拟合参考答案”转向“可执行、闭环、少幻觉”的服务质量。

## 5. 消融实验与结论

### 5.1 早期 2 epochs beta 消融

早期 `dpo_pairs` 目录中的 2 epochs 消融结果如下：

| beta | epoch | global step | eval reward margin | Accuracy | Auto Resolve | Closure | CSAT |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 2.0 | 70 | 0.0031 | 73.94 | 62.77 | 63.83 | 3.596 |
| 0.10 | 2.0 | 70 | 0.0061 | 75.53 | 61.70 | 62.77 | 3.622 |
| 0.30 | 2.0 | 70 | 0.0309 | 75.00 | 62.77 | 64.89 | 3.622 |

这组实验的问题是训练轮数较少，global step 只有 70，DPO reward margin 刚刚开始拉开。它可以说明 beta=0.3 已经有更强偏好信号，但不足以支撑最终结论。

同目录下还有一个 `rank16_qvko_beta0.3_3epochs`，其训练结果与最终 3 epochs beta=0.3 的训练指标一致，说明 beta=0.3 3 epochs 的可复现性较好；但完整三组 beta=0.05/0.1/0.3 的公平对比应以 `dpo_pairs_single_gpu_0513_1434` 为准。

### 5.2 最终 3 epochs beta 消融

最终有效消融：

| beta | train_loss | eval_loss | eval reward margin | Judge 侧主要表现 |
|---:|---:|---:|---:|---|
| 0.05 | 0.6887 | 0.6871 | 0.0126 | 改善有限，Auto Resolve/Closure 略高于 SFT，但力度较弱 |
| 0.10 | 0.6844 | 0.6821 | 0.0237 | 训练偏好更强，但外部 judge Accuracy 和 CSAT 不占优 |
| 0.30 | 0.6699 | 0.6677 | 0.0621 | Auto Resolve、Closure、CSAT 最好，综合最优 |

beta=0.3 的最后训练日志显示：

| 指标 | 值 |
|---|---:|
| train rewards/chosen | 0.3067 |
| train rewards/rejected | 0.2295 |
| train rewards/margins | 0.0772 |
| train rewards/accuracies | 0.7000 |
| eval rewards/chosen | 0.3081 |
| eval rewards/rejected | 0.2460 |
| eval rewards/margins | 0.0621 |
| eval rewards/accuracies | 0.6382 |

这说明模型确实学到了“chosen 相对 rejected 更优”的偏好方向，并且没有出现 reward margin 发散或 eval loss 异常。

### 5.3 为什么 beta=0.3 更合适

DPO 中 beta 控制偏好优化相对于 reference policy 的强度。beta 太小，模型变化保守，DPO 信号难以转化为可观察的回复风格变化；beta 太大可能导致模型过度追逐 chosen 风格，损害准确性或稳定性。

本项目中 beta=0.05 和 0.1 的 reward margin 分别只有 0.0126 和 0.0237，外部 judge 改善也有限。beta=0.3 将 margin 提升到 0.0621，同时 Auto Resolve 从 SFT 的 61.33 提升到 66.49，Closure 从 60.77 提升到 65.96，说明它足以把偏好数据中的“明确路径、闭环表达、条件化建议”转化为模型输出习惯。

需要注意：beta=0.3 并没有让 Accuracy 最高。原因可能是 DPO 数据主要优化回复形态和服务流程，而不是补充事实知识；同时 judge 对 accuracy 的判定会惩罚部分条件化表达不够贴合参考答案的情况。因此最终选择 beta=0.3，是因为项目目标更偏客服可解决性，而不是单纯追求 Accuracy。

## 6. 过程中的尝试、失败与复盘

### 6.1 问题一：DPO 数据容易强化幻觉式客服

现象：

在多轮客服场景中，用户经常追问“是否已发货”“能否补发”“审核多久”“是否会自动处理”等问题。SFT 生成候选时，为了显得像客服，容易生成“系统已自动审核”“已为您补发”“预计明早物流更新”“已帮您加急”等无依据后台动作。

为什么：

原始客服语料中存在大量确定式服务承诺。SFT 学会了这种话术风格，但模型并没有真实订单系统状态。DPO 如果把这类回答选为 chosen，就会进一步强化“编造后台状态换取闭环感”的坏习惯。

试了什么：

1. 在 ranking prompt 中明确 hallucination 字段，把严重虚构订单号、订单状态、物流节点、金额、时间、库存、后台已操作等定义为 hallucination=1。
2. 规定严重 hallucination=1 的候选绝对不能作为 chosen。
3. 如果所有候选都是严重 hallucination，则整条样本失败，不进入 DPO 数据。
4. rejected 不总是选择最严重幻觉，而是优先选择“基本正确但不够具体、安全但无帮助、不闭环”的候选，避免训练只学会区分极端错误。

是否解决：

部分解决。最终有 202 条样本因“all candidates have severe hallucination=1”被剔除，避免了污染 DPO 数据。但这也暴露了 SFT 在订单状态型、多轮追问型样本上的候选生成质量仍然不足。

### 6.2 问题二：候选之间差异过小，偏好信号弱

现象：

一些 FAQ 或标准政策问题，多温度采样后候选几乎完全相同，甚至只有 1 个唯一候选。

为什么：

这些问题在 SFT 训练集中频繁出现，模型输出已经高度模式化；同时客服场景本身有很多标准答案，采样温度变化不足以制造有效偏好差异。

试了什么：

1. 使用 0.5、0.7、0.9 三个 temperature 生成候选。
2. 对候选文本去重。
3. 对唯一候选数少于 2 的样本直接跳过。
4. 对 best_score - worst_score 最低的 10% pair 进行过滤。

是否解决：

基本解决。84 条低多样性样本被跳过，271 条弱偏好 pair 被过滤。保留下来的 2443 条数据 score_delta 中位数为 51.72，偏好强度足够训练。

### 6.3 问题三：Reference policy 选择错误会导致 DPO 训练异常

现象：

早期 DPO 实验中出现 loss 约 2.36、reward 数值十几、eval reward margin 为负的异常结果，训练信号不可信。

为什么：

对 PEFT 模型做 DPO 时，如果 TRL 默认通过禁用 adapter 作为 reference，就会把 reference 变成 base model，而不是 SFT model。这样当前 policy 与 reference 初始差距过大，DPO reward 不再只表示偏好学习，而混入了 SFT adapter 相对 base model 的整体差异。

试了什么：

在训练脚本中复制一份 SFT adapter 作为 frozen reference adapter：

`default` adapter 用于训练，`ref` adapter 用作 reference。

是否解决：

解决。修正后 loss 回到 0.69 附近，reward margin 从接近 0 稳定上升，训练曲线正常。

### 6.4 问题四：chosen 改写看似提升训练指标，但没有提升外部 judge

现象：

尝试使用 qwen-plus 改写 chosen，让 chosen 更符合 judge 高分标准。改写后训练内部指标非常强，例如 `rank16_qvko_beta0.1_5.12_rewritten` 的 eval reward margin 达到 0.7035，eval reward accuracy 达到 0.9919。

但外部 judge 结果并没有变好：

| 实验 | epoch | Accuracy | Auto Resolve | Closure | CSAT |
|---|---:|---:|---:|---:|---:|
| beta0.1 原始 DPO 2 epochs | 2 | 75.00 | 63.30 | 64.89 | 3.633 |
| beta0.1 rewritten 2 epochs | 2 | 75.00 | 62.23 | 63.83 | 3.580 |
| beta0.1 rewritten 1 epoch | 1 | 73.94 | 59.57 | 59.57 | 3.548 |

为什么：

改写 chosen 让 chosen/rejected 之间出现了很强的形式差异，DPO 内部很容易区分二者，所以 reward margin 暴涨。但这种差异不一定等价于真实泛化能力。改写文本可能引入统一风格、长度偏差或过度“高分模板化”，模型学到的是“改写 chosen 的分布特征”，而不是更可靠的客服决策。

试了什么：

1. 阶段 2 改写：输入 2443 条，成功 2370 条，失败 73 条，平均长度比 1.0888。
2. 阶段 3 更严格改写：输入 2443 条，成功 2017 条，失败 426 条，平均长度比 1.0782。
3. 加入高风险词和高风险时间/状态模式检测，阻止改写新增订单状态、物流节点、自动处理、上门取件、赔付、免费等无依据承诺。

是否解决：

没有作为主线方案采用。改写流程对减少显性风险有帮助，但训练结果显示它没有带来外部 judge 增益，反而可能造成偏好过强、泛化不足。因此最终主线回到原始 SFT 候选 pair，而不是 rewritten chosen。

### 6.5 问题五：高质量过滤后数据更干净，但覆盖不足

现象：

尝试从 2443 条主数据中筛出更高质量 pair，只保留 chosen 满足 accuracy=1、closure=1、auto_resolve=1、satisfaction>=4、无风险内容，并要求 rejected 非幻觉且具备特定弱点。

过滤结果：

| 阶段 | 数量 |
|---|---:|
| 原始 DPO pair | 2443 |
| 高质量过滤后 | 791 |
| 进一步保留 rejected weak | 669 |

高质量过滤 drop reasons：

| 原因 | 数量 |
|---|---:|
| chosen_risky_content | 806 |
| rejected_hallucination_not_0 | 560 |
| chosen_auto_resolve_not_1 | 549 |
| rejected_accuracy_not_1 | 372 |
| chosen_satisfaction_lt_4 | 294 |
| chosen_closure_not_1 | 290 |
| chosen_accuracy_not_1 | 83 |

训练结果：

| 实验 | train samples | global step | Accuracy | Auto Resolve | Closure | CSAT |
|---|---:|---:|---:|---:|---:|---:|
| beta0.1 highquality 2 epochs | 711 | 24 | 72.87 | 63.30 | 63.30 | 3.569 |

为什么：

高质量过滤让数据更干净，但样本量下降明显，训练只有 711 个 train samples、24 个 global steps。覆盖不足导致模型学不到足够广泛的客服场景，外部 judge 没有提升。

是否解决：

结论是：高质量过滤可以作为后续数据清洗方向，但当前 669/791 条规模太小，不适合作为主训练数据。更可行的方向是扩大候选生成规模，或者只做轻过滤，不做过严筛选。

## 7. 最终有效方案

最终采用的 DPO 方案可以概括为：

1. 使用 SFT rank16_qvko adapter 作为起点。
2. 对 SFT 训练样本采样 3000 条，使用 SFT adapter 多温度生成候选。
3. 用 qwen-plus 做偏好排序，不把原参考答案直接作为 chosen。
4. 严格排除严重幻觉作为 chosen 的样本。
5. 丢弃偏好差异最弱的 10% pair，保留 2443 条。
6. 使用 TRL DPOTrainer，sigmoid loss，SFT adapter clone 为 reference adapter。
7. LoRA rank16 qvko，learning rate 5e-7，effective batch size 32，训练 3 epochs。
8. beta 消融后选择 beta=0.3。

最终推荐 checkpoint：

`/home/txs/work/zyp/SFT_DPO/train/outputs/dpo_beta_ablation/dpo_pairs_single_gpu_0513_1434/rank16_qvko_beta0p3/checkpoint/checkpoint-207`

对应 artifacts：

`/home/txs/work/zyp/SFT_DPO/train/outputs/dpo_beta_ablation/dpo_pairs_single_gpu_0513_1434/rank16_qvko_beta0p3/checkpoint/artifacts`

## 8. 对实验结果的解释

DPO 后最明显的变化不是“知道更多”，而是“回答更像能把事办完”。

SFT 模型已经有较高 Accuracy，但常见问题是：

1. 回答偏泛，例如“建议联系客服处理”“请关注订单状态”。
2. 给了结论但没有操作路径。
3. 有礼貌但没闭环。
4. 为了闭环偶尔编造后台动作。

DPO 数据的 chosen/rejected 构造正好围绕这些问题：chosen 偏向 grounded、条件化、有操作路径、有下一步、有真实客服感；rejected 多是安全但无帮助、不够具体、不闭环的回答。因此 DPO beta=0.3 主要提升 Auto Resolve 和 Closure，而 Accuracy 变化不大甚至略降。

从业务目标看，这是可接受的。客服机器人真正影响体验的往往不是单句是否与参考答案一致，而是用户看完后是否知道下一步怎么做。DPO 在这个方向上是有效的。

## 9. 当前局限

1. Judge 样本数只有 188，能比较趋势，但统计置信度有限。
2. DPO 数据来自 SFT 自采样，候选上限受 SFT 能力限制；如果 SFT 对某类问题全都幻觉，该样本只能丢弃，无法产生优质 pair。
3. 数据类型分布偏 yesno_generic 和 generic，对复杂售后、物流异常、订单状态类问题覆盖仍不足。
4. qwen-plus 同时参与 ranking 和 final judge，存在评价同源风险。
5. 高质量过滤数据规模不足，说明当前数据生成策略在“严格安全 + 明确闭环 + rejected 有教学价值”三者之间还不够平衡。
6. 改写 chosen 会放大训练内部 margin，但不一定提升外部泛化，后续不能只看 DPO reward 指标。


## 10. 后续建议

1. 扩大 DPO 数据生成规模，从 3000 prompt 提升到 8000 到 15000 prompt，并对售后、物流、退款、订单状态追问进行定向过采样。
2. 候选生成不只依赖同一个 SFT adapter，可以加入不同温度、不同 checkpoint、规则模板候选或弱模型候选，提升 rejected 多样性。
3. 对“全候选幻觉”的 202 类样本单独建安全拒答/条件化回答数据，而不是简单丢弃。
4. 高质量过滤不要一次性收紧到 669 条，可以采用分层采样：高质量强 pair + 普通覆盖 pair 混合训练。
5. 引入人工抽检或不同 judge 模型交叉评估，降低 qwen-plus ranking 与 qwen-plus judge 的同源偏差。
6. 对 beta=0.3 继续做更细消融，例如 beta=0.2、0.4，以及 2/3/4 epochs 对比，确认是否存在继续训练后的 Accuracy 下滑。
7. 建议新增维度化错误分析，按 question_type 分别统计 Accuracy、Auto Resolve、Closure，定位 DPO 对哪些问题类型最有效、哪些类型有副作用。

## 11. 一句话复盘

这次 DPO 最关键的收益来自三点：用 SFT 多采样构造真实相对偏好、把 SFT adapter 正确复制为 reference policy、以 beta=0.3 训练足够 3 epochs。失败尝试也很有价值：chosen 改写和过严高质量过滤都说明，DPO 不能只追求训练内部 margin，最终仍要看外部 judge 下的真实客服闭环能力。


## 12. 为什么 DPO 收益有限

第一，偏好数据来自同一个 SFT 模型的多采样候选。这样做成本低、风格一致，但 chosen/rejected 的能力上限受 SFT 模型限制。如果候选池里没有真正高质量答案，DPO 只能在几个相似回复里选较好者，难以学到新的业务能力。

第二，客服任务的错误往往不是“偏好强弱”，而是事实约束问题。DPO 可以让模型更偏向 chosen，但无法可靠知道哪些订单状态、物流节点、时间承诺是不可编造的。除非数据中大量、系统性地覆盖这些负例，否则模型仍会在 held-out 样本中复发。

第三，chosen/rejected 的差异有时和最终 judge 指标不完全一致。训练时优化 pairwise preference，评估时看 accuracy、closure、auto_resolve、CSAT。reward margin 增大说明模型学会区分训练 pair，但并不保证所有评估维度同步提升。

第四，数据规模偏小。2443 条 pair 对 8B 模型的 LoRA DPO 来说能产生方向性影响，但不足以稳定改写所有高风险客服模式。high_quality 过滤后只剩 791/669 条，更不足以覆盖完整业务分布。

第五，部分业务标签/立场存在冲突。例如同是“是否提供定制服务”，不同样本的 source metadata stance 可能出现 positive/negative 差异。DPO 如果没有商品级、店铺级、平台级上下文，就只能学习通用表达，无法精准决定业务结论。

第六，推理评估使用 greedy decoding，而训练中样本生成/训练内部 sample 使用了不同生成设置。DPO 改变 logits 分布后，greedy 解码可能放大某些模板或安全表达，导致自然度不一定提升。