# 检索评测报告（model-qwen38-nothink）

> 本文件由 `python -m eval.make_report` 从 `summary.json` / `per_query.jsonl` 直接生成，不是手抄。改了代码就重跑、重新生成，别手工改数字。

## 1. 可复现信息

| 项 | 值 |
| --- | --- |
| 运行时间 | 2026-11-12 15:49:35 |
| commit | `9671dc4`（评测跑于该提交；后续文档/脚本提交不改变指标口径） |
| 题目数 | 50（{'fact': 12, 'table': 10, 'procedure': 10, 'multi_turn': 8, 'out_of_scope': 6, 'negative': 4}） |
| 语料 | 22 切片，fingerprint `a40f15b80b1f` |
| 数据集 | fingerprint `304f87a2a19b` |
| 编码器 / 精排 | `bge-m3` / `bge` |
| LLM | `qwen3.8-flash`（思考模式：未记录） |
| 指标 | Hit / Recall / Precision / MRR / nDCG @5 |

> 该 run 跑于元信息补丁之前，`llm_thinking` 未落库（run_id 后缀 `nothink` 表示关闭思考）；口径与三次 run 的完整对照见 `eval/model-compare.md`。

## 2. 检索消融

每次只加一个变量：稠密 → +稀疏 → +HyDE&RRF → +精排 → +断崖截断。

| 配置 | Hit@5 | Recall@5 | Precision@5 | MRR@5 | nDCG@5 | 平均文档数 | 平均上下文字数 | P95 检索耗时 | HyDE 额外调用 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `A1_dense` | 0.875 | 0.875 | 0.175 | 0.758 | 0.788 | 10.0 | 3803 | 0.2 ms | — |
| `A2_hybrid` | 0.900 | 0.900 | 0.180 | 0.756 | 0.792 | 10.0 | 3755 | 0.2 ms | — |
| `A3_hyde_rrf` | 0.875 | 0.875 | 0.175 | 0.728 | 0.764 | 10.0 | 3830 | 578.8 ms | 5236 ms/题（冷） |
| `A4_rerank` | 0.950 | 0.950 | 0.190 | 0.885 | 0.902 | 10.0 | 3830 | 76.6 ms | 缓存命中 |
| `A5_cutoff` | 0.950 | 0.950 | 0.190 | 0.885 | 0.902 | 8.2 | 3163 | 0.7 ms | 缓存命中 |

## 3. 分题型

最佳配置：`A4_rerank`

| 题型 | 题数 | Hit@5 | MRR@5 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `fact` | 12 | 0.917 | 0.854 | 0.869 |
| `table` | 10 | 1.000 | 1.000 | 1.000 |
| `procedure` | 10 | 1.000 | 0.883 | 0.913 |
| `multi_turn` | 8 | 0.875 | 0.792 | 0.812 |
| `out_of_scope` | 6 | — | — | — |
| `negative` | 4 | — | — | — |

> `out_of_scope` / `negative` 没有 gold 切片，检索指标无定义；它们的正确性只能由答案层的弃答率衡量（`eval/service_eval.py`）。

## 4. HyDE 调用延迟（冷启动）

| 调用数 | p50 | p95 | 平均 | 最大 |
| --- | --- | --- | --- | --- |
| 50 | 5206 ms | 6737 ms | 5236 ms | 10184 ms |

## 5. 未命中的题目（最佳配置之外的全量核对）

| 题号 | 题型 | 问题 | gold | 实际 top-1 |
| --- | --- | --- | --- | --- |
| F01 | `fact` | RS-12 的显示屏是几位数？ | c007 | ## 交流电压测量 |
| M08 | `multi_turn` | 尺寸是多少？ | c007 | ## 控制与端口 |

## 6. 对照 run

| run | LLM | A2 Hit@5 | A3 Hit@5 | A4 Hit@5 | A4 MRR@5 | HyDE 冷启动平均 |
| --- | --- | --- | --- | --- | --- | --- |
| `model-qwen38-nothink` | `qwen3.8-flash` | 0.900 | 0.875 | 0.950 | 0.885 | 5236 ms |
| `model-qwen-flash` | `qwen-flash` | 0.900 | 0.850 | 0.950 | 0.885 | 3014 ms |
| `model-qwen38-flash` | `qwen3.8-flash` | 0.900 | 0.850 | 0.950 | 0.885 | 21964 ms |

> 检索指标与 LLM 选型无关：若表中 A2/A3 出现差异，先怀疑单题翻转（50 题下 1 题 = 2.5 个点），而不是模型优劣。详见 `eval/model-compare.md`。

## 7. 复现

```bash
python -m eval.link_gold                       # 校验标注能落到当前语料
python -m eval.run_eval --run-id <run_id>      # 重跑消融
python -m eval.make_report --runs eval/results/<run_id> --out eval/REPORT-baseline.md
```

## 8. 解读边界

- 本地是精确检索，线上 Milvus 为 HNSW 近似检索，**绝对值有差异，只做相对比较**；
- `WeightedRanker` 用 min-max 归一化近似复刻，与 Milvus 的分数不等价；
- 50 题规模下，单题翻转即 ±2.5 个百分点，**单题差异不解读**；
- 多轮题使用 `rewrite`（理想改写）检索，衡量的是检索上限，改写质量另见 `service_eval.py`；
- 语料从干净 Markdown 重切，不是 Milvus 里那份编码损坏的旧数据。
