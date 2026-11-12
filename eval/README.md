# 评测层（eval/）

> 目标：让检索链路的每个组件都有可对比的数字——好多少、贵多少、崩在哪。
> 这里产出的每一个指标都要能追溯到具体的 run、语料指纹与提交。

---

## 1. 为什么需要这一层

项目原先只有"跑通"这一条证据：混合检索、HyDE、RRF、断崖截断都写在 README 里，
却没有任何对比数据说明这些组件是否真的有用；改了切分或权重，也无法判断是变好还是变坏。

这一层补上的就是：**可复现的指标 + 可归因的消融 + 有记录的失败案例**。

---

## 2. 快速开始

在仓库根目录（`shoper_brain/`）执行：

```bash
# 0) 依赖：pytest 用于单测（其余依赖项目 venv 里已有）
knowledge/.venv/Scripts/python.exe -m pip install pytest

# 1) 单测：指标与检索组件
knowledge/.venv/Scripts/python.exe -m pytest eval/tests -q

# 2) 校验标注：把「章节 + 证据句」解析成 chunk_id
knowledge/.venv/Scripts/python.exe -m eval.link_gold

# 3) 全量消融（需要 GPU + BGE 权重 + DASHSCOPE_API_KEY）
knowledge/.venv/Scripts/python.exe -m eval.run_eval --run-id baseline-v1

# 4) 没有模型时只验证链路（指标无意义，仅供 CI 冒烟）
knowledge/.venv/Scripts/python.exe -m eval.run_eval --embedder hash --hyde none \
    --configs A1_dense,A2_hybrid --limit 5
```

输出在 `eval/results/<run_id>/`：

| 文件 | 用途 |
| --- | --- |
| `summary.md` | 可直接粘进 README 的表格 |
| `summary.json` | 机器可读的完整指标（含分题型） |
| `per_query.jsonl` | 每题 top-k、gold、耗时、上下文字数，用于 badcase 复盘 |
| `config.json` | 实验参数、语料/数据集指纹、git sha，保证可复现 |

`eval/results/` 不进版本库（体积 + 含本机路径）。要留证据就固化：

```bash
python -m eval.make_report --runs eval/results/<run_id> \
    --compare eval/results/<另一组 run> --out eval/REPORT-baseline.md
```

仓库里的 [`REPORT-baseline.md`](REPORT-baseline.md) 就是这么生成的——
**报告里的数字不是手抄的，改代码后重跑、重新生成即可**。

---

## 3. 数据集协议（50 题）

| 题型 | 数量 | 测什么 | gold 是否为空 |
| --- | --- | --- | --- |
| `fact` | 12 | 单跳事实（参数、指标、符号含义） | 否 |
| `table` | 10 | 表格/规格数值（最容易被切分和线性化搞坏） | 否 |
| `procedure` | 10 | 多步骤操作（步骤顺序、端口、档位） | 否 |
| `multi_turn` | 8 | 指代消解（"它/那/呢"依赖上一轮） | 否 |
| `out_of_scope` | 6 | 手册外的领域问题（蓝牙、电容、价格、竞品、校准、IP 等级） | 是（应弃答） |
| `negative` | 4 | 领域内但手册没写（保修、螺丝数量、背光颜色、固件） | 是（应弃答） |

### 3.1 标注规则（重要）

- **不手抄 chunk_id**。切片参数一变 chunk_id 全废，所以每道题标注的是
  `gold_sections`（章节名）+ `evidence`（原文证据句，必须在切片正文里能原样找到）。
- `link_gold.py` 负责把锚点翻译成 `chunk_id`，**解析不到就报错退出**——
  等于用机器反向校验"这道题是否真的能从语料里找到答案"，防止拍脑袋出题。
- 证据优先：证据句能唯一定位时，不把整个章节都算 gold，否则「规格」被拆成
  2 片后 Recall 永远封顶 50%。
- 弃答题（`out_of_scope` / `negative`）不参与检索指标均值，单独统计**弃答正确率**。

### 3.2 为什么只有 50 题、单本手册

手册语料只有一本（22 个切片），能写出的"不重复、可解析"的问题上限就在 50 左右。
**这是当前评测的真实边界，不是设计缺陷**。要扩大规模，正确做法是加第二、第三本
手册（也就顺带补上"跨手册对比"这一题型），而不是把一道题换个问法凑数。

---

## 4. 指标口径

| 指标 | 定义 | 为什么需要 |
| --- | --- | --- |
| `Hit@k` | 前 k 条里有没有 gold | 最直观的"能不能找到" |
| `Recall@k` | 前 k 条覆盖了多少 gold | 多 gold 场景（表格被拆片） |
| `Precision@k` | 前 k 条里 gold 的占比 | 衡量上下文被噪声稀释的程度 |
| `MRR@k` | 第一个 gold 的位置倒数 | 关心"排得靠不靠前"（影响生成质量） |
| `nDCG@k` | 二值相关性的排序质量 | 比 MRR 更平滑，适合做消融对比 |
| `avg_docs` / `avg_context_chars` | 平均喂给 LLM 的文档数与字数 | **成本代理指标**：断崖截断到底省了多少 |
| `P95 latency` | 检索阶段耗时 | 只做相对比较；真实延迟看 `service_eval.py` |

---

## 5. 消融配置（每次只加一个变量）

| 配置 | 稠密 | 稀疏 | HyDE | RRF | 精排 | 断崖截断 |
| --- | --- | --- | --- | --- | --- | --- |
| `A1_dense` | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `A2_hybrid` | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `A3_hyde_rrf` | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| `A4_rerank` | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ |
| `A5_cutoff` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

对应到代码：`eval/configs.py`。要加"dense/sparse 权重网格"，直接在
`ABLATIONS` 里加 `A2b_hybrid_w07`（`dense_weight=0.7`）即可，不需要改任何流程。

---

## 6. 端到端评测（答案层）

```bash
# 先起服务
python -m knowledge.api.query_router

# 打真实接口，导出待人工打分的 CSV
knowledge/.venv/Scripts/python.exe -m eval.service_eval --endpoint http://127.0.0.1:8001

# 人工把 human_score 填 0/1 后汇总
knowledge/.venv/Scripts/python.exe -m eval.service_eval --score eval/results/answers_xxx.csv
```

**答案正确性不交给 LLM 自动判分**。自评偏差无法复核，做法是人工打 0/1
（50 题规模下全量抽检成本可控），`keyword_hit` / `abstain_ok` 只作为辅助信号；
同义表述会造成漏判，见 `answer_ab.py` 里 `verdict()` 的注释。

多轮题会先把 `history` 逐条发给同一 `session_id` 制造真实历史，再发最终问题，
而不是把历史拼进 query 里假装多轮。

---

## 7. 已知限制（便于复现与判断适用边界）

1. **本地是精确检索**，线上 Milvus 是 HNSW 近似检索，绝对数值会有小幅差异；
2. **WeightedRanker 用 min-max 归一化近似复刻**，绝对分数与 Milvus 不等价，只用于相对比较；
   副作用见 `eval/tests/test_retrieval.py::test_weighted_rank_minmax_tie_artifact`；
3. **多轮题本地用 `rewrite`（理想改写）检索**，等价于"检索上限"；改写质量只在
   `service_eval.py` 里测，两者不要混着说；
4. **联网检索（web MCP）不在本地消融里**，需要走服务模式；
5. **单本手册**，无法测跨设备串答案（`item_name` 过滤的必要性）；
6. 语料默认从**干净的 Markdown 重新切分**，不是从 Milvus 导出——原因见 `eval/corpus.py` 顶部注释
   （当前集合里的 `content` 是乱码、`item_name` 存着整段 JSON）。

---

## 8. CI 冒烟（可选）

```yaml
- run: python -m pytest eval/tests -q
- run: python -m eval.link_gold           # 标注一旦失效立刻失败
- run: python -m eval.run_eval --embedder hash --hyde none --limit 3 --configs A1_dense
```

前两条不依赖 GPU 和网络，适合放进 GitHub Actions。

> 注意：`--embedder hash` 只用字符级词袋，链路可跑通但指标无意义，仅用于冒烟，
> 任何结论都必须来自 `bge-m3` 的 run。