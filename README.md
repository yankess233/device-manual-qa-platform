# 家电说明书智能问答平台

[![CI](https://github.com/yankess233/device-manual-qa-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/yankess233/device-manual-qa-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-3776AB)
![Milvus](https://img.shields.io/badge/vector--db-Milvus%202.5-00A1EA)

把几十页的家电说明书变成**可按型号检索、能出操作步骤、带原文配图、答不出来会认怂**的问答服务。

语料是两台真实设备的官方说明书（美的变频空调 40 页、米家云台摄像机 16 页），
链路覆盖 **MinerU 解析 → 标题感知切分 → BGE-M3 混合检索 → RRF → 精排 → 断崖截断 → 约束生成 → SSE 流式**，
并配了一套**可复现的评测**：两套语料、80 道标注题、5 组消融、以及"调参到底有没有用"的对照实验。

> 这个项目的重点不是"接上大模型"，而是**每一步都能说清楚它值多少**：
> 精排带来多少 MRR、断崖截断省多少上下文、HyDE 在什么场景下是负收益、调 `max_k` 为什么不可能提 Hit@5。

---

## 目录

- [项目简介](#项目简介)
- [核心能力](#核心能力)
- [系统架构](#系统架构)
- [导入流水线](#导入流水线)
- [查询流水线](#查询流水线)
- [技术栈](#技术栈)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [语料与评测](#语料与评测)
- [接口文档](#接口文档)
- [数据模型](#数据模型)
- [关键配置项](#关键配置项)
- [工程化能力](#工程化能力)
- [已知限制](#已知限制)
- [路线图](#路线图)
- [更新记录](#更新记录)

---

## 项目简介

家电说明书的典型使用场景是"我家里这台机器出了个状况，说明书第几页怎么说的"。
直接问大模型会编，直接全文塞进去又贵又不可溯源。这个项目要解决三件事：

1. **先认机器，再答题**。说明书是"一型号一本册"，不确认型号就在多本手册里瞎捞。
   系统用 LLM 抽取产品名（含多轮指代消解）→ 向量匹配主体库 → 高置信直接确认、中置信反问用户、
   型号 token 精确命中则短路确认。
2. **检索要准，且知道为什么准**。稠密（语义）+ 稀疏（型号/数值）+ HyDE（假设文档）三路召回，
   RRF 融合后用 cross-encoder 精排，再用断崖截断砍掉陪跑的尾部。
3. **答案受约束、可追溯**。Prompt 要求只依据参考片段作答，没有就说不知道；
   需要图时从原文片段里取图；答案带 `[n]` 角标，流式返回。

工程上没有停在"能跑"：API Key 鉴权、上传校验（修过真实路径穿越漏洞）、
Redis 任务态（支持多 worker）、JSON 结构化日志、Locust 压测、Docker 编排、GitHub Actions CI。

## 核心能力

| 能力 | 实现要点 |
| --- | --- |
| PDF 解析 | MinerU 4.x 出 Markdown + 结构化内容 + 图片；扫描版走 OCR |
| 图片直链 | 图片转存 MinIO，MD 中的相对路径改写为可访问 URL；文件名保留 `page_N_image_M` |
| 表格可检索 | HTML/Markdown 表格线性化为 `列名: 值`，覆盖合并单元格、无表头 KV 表 |
| 标题感知切分 | 按 `#`/`##` 切；超长二次切（1000 字）、过短合并（500 字）、代码围栏保护 |
| 主体识别 | LLM 抽「品牌+型号+品类」；JSON 污染自动剥离；型号 token 精确匹配短路 |
| 混合检索 | BGE-M3 稠密 + 稀疏，`WeightedRanker(0.5, 0.5)` |
| 融合 | RRF（k=60），多路结果按 `chunk_id` 累加，天然去重 |
| 精排 | `bge-reranker-large` 逐对打分重排 |
| 断崖截断 | 相邻分差 >0.15 处截断，至少 3 条、最多 10 条 |
| 多轮对话 | MongoDB 存历史，指代消解后回填产品名 |
| 流式输出 | SSE 三类事件：`progress` / `delta` / `final` |
| 任务可视化 | 导入 8 个节点的进度、耗时实时回传前端时间线 |

## 系统架构

```
┌────────────────────── 导入链路（离线，每份文档一次） ──────────────────────┐
│                                                                            │
│  PDF ──▶ 上传校验 ──▶ MinerU 解析 ──▶ 图片转存 MinIO + VLM 图注            │
│                                    │                                       │
│                                    ▼                                       │
│            标题感知切分（表格线性化） ──▶ 主体名称识别（LLM）               │
│                                    │                                       │
│                                    ▼                                       │
│                 BGE-M3 稠密+稀疏向量化 ──▶ 写入 Milvus                     │
│                                            ├─ 切片集合 kb_chunks_home_v1   │
│                                            └─ 主体集合 kb_item_names_home_v1│
└────────────────────────────────────────────────────────────────────────────┘
                                     │
┌────────────────────── 查询链路（在线，每次请求） ──────────────────────────┐
│                                                                            │
│  问题 ──▶ 主体抽取（LLM，带历史） ──▶ 主体确认（向量匹配 + 型号短路）       │
│              │                                                             │
│              ▼                                                             │
│        三路召回：稠密 / 稀疏 / HyDE ──▶ RRF 融合 ──▶ bge-reranker 精排     │
│              │                                                             │
│              ▼                                                             │
│        断崖截断 ──▶ 上下文组装（≤12000 字） ──▶ 约束生成 ──▶ SSE 流式      │
└────────────────────────────────────────────────────────────────────────────┘
        │                        │                        │
   Milvus（向量+标量）      MinIO（原图/原文件）      MongoDB（会话历史）
```

## 导入流水线

| # | 节点 | 做什么 | 失败模式与处理 |
| --- | --- | --- | --- |
| 1 | 上传文件 | 文件名净化、后缀白名单、50MB 流式限额；本地 + MinIO 双写 | 超限删半成品；非法后缀 400 |
| 2 | 检查文件 | 校验路径与文件存在 | 缺失抛 `ValidationError` |
| 3 | PDF→Markdown | 调 `mineru-kit` 子进程，产出 `markdown.md` + `structured_content.json` + `images/` | 解析器偶发显存报错，重试即过 |
| 4 | 图片处理 | 图片上传 MinIO、改写链接、VLM 生成图注 | 单图失败跳过，不阻断整条链路 |
| 5 | 文档切分 | 按标题切、超长二次切、过短合并、表格线性化 | 超长章节无二级标题会导致同标题多片（见已知限制） |
| 6 | 主体名称识别 | LLM 读前 3 片抽产品名，写主体集合并回填每个切片 | 返回 JSON 串时用 `normalize_item_name` 剥离 |
| 7 | 向量生成 | BGE-M3 同时产出稠密与稀疏向量 | 模型不可用则该任务失败 |
| 8 | 导入 Milvus | 建集合、建索引、insert | 集合已存在则复用 |

实测耗时（RTX 3060）：

| 文档 | 页数 | 切片 | 总耗时 | 其中 PDF→MD | 其中图注 |
| --- | --- | --- | --- | --- | --- |
| 美的变频空调使用安装说明书 | 40 | 50 | **476 s** | 337 s | 124 s |
| 米家智能摄像机云台版说明书 | 16 | 22 | **81 s** | 67 s | 14 s |

## 查询流水线

### 1. 主体确认

- **抽取**：LLM 结合最近 5 轮历史，抽 `item_names` 并改写问题（指代消解）；
- **匹配**：用抽到的名字去主体集合做混合检索，得到候选与分数；
- **决策**：
  - 分数 ≥0.7 且唯一 → 直接确认；
  - 0.6~0.7 或多条高分 → **反问用户选哪个**；
  - 型号 token 精确命中（如 `KFR-51LW/N8KS1-1P`）→ 短路确认；
  - 完全无候选且库里只有一个产品 → 单产品兜底确认。

### 2. 三路召回

| 路 | 手段 | 解决的问题 |
| --- | --- | --- |
| 稠密 | BGE-M3 句向量 | 同义改写（"清洗滤网" ↔ "滤尘网清洁"） |
| 稀疏 | BGE-M3 稀疏权重 | 型号、数值、专有名词的精确匹配 |
| HyDE | LLM 先写"假想答案"再检索 | 问题过短、与文档用词差距大 |

稠密与稀疏用 `WeightedRanker(0.5, 0.5)` 融合，HyDE 那一路与主路再做 RRF（k=60）。

### 3. 精排与截断

- `bge-reranker-large` 对 `(query, chunk)` 逐对打分重排，**单步增益最大**；
- 断崖截断：从前往后找第一处相邻分差 >0.15 的位置截断，至少 3 条、最多 10 条。
  它的目标不是提命中，而是**在信息量已经衰减的尾部省上下文**。

### 4. 生成与流式

Prompt 给模型三样东西：带 `[n]` 角标的参考片段、历史对话、确认到的产品名；
要求只依据参考片段作答、无据则弃答、需要配图时在末尾输出 `【图片】` 区块。
SSE 分三类事件：`progress`（节点进度）、`delta`（逐字增量）、`final`（完整答案 + 图片）。

## 技术栈

| 层 | 选型 |
| --- | --- |
| 编排 | LangGraph（导入 8 节点 / 查询 7 个实际节点 + 2 个路由与汇合的虚拟节点） |
| 解析 | MinerU 4.x（版面 + OCR + 公式表格） |
| 向量 | BGE-M3（稠密 1024 维 + 稀疏词权重） |
| 精排 | bge-reranker-large |
| 向量库 | Milvus 2.5（HNSW/AUTOINDEX + SPARSE_INVERTED_INDEX） |
| 对象存储 | MinIO |
| 会话存储 | MongoDB 7 |
| 任务态 | 内存 / Redis 7（可切换） |
| 服务 | FastAPI + Uvicorn（导入 :8000 / 查询 :8001） |
| 大模型 | DashScope 兼容接口（默认 `qwen3.8-flash`，关思考） |
| 可观测 | JSON 结构化日志 + `request_id` + 节点耗时 |
| 压测 | Locust |
| 交付 | Docker Compose + GitHub Actions |

## 目录结构

```text
shoper_brain/
├── knowledge/
│   ├── api/                       # 两个 FastAPI 入口 + security.py（鉴权/CORS）
│   ├── core/                      # Service 单例注入、路径常量
│   ├── front/                     # 导入台 / 问答台（原生 HTML，自动带 API Key）
│   ├── processor/
│   │   ├── import_process/        # 导入图：8 节点 + 状态 + 配置 + 异常体系
│   │   └── query_process/         # 查询图：主体确认 / 三路召回 / RRF / 精排 / 生成
│   ├── prompt/                    # 提示词模板（领域示例集中在 import_prompt / query_prompt）
│   ├── schema/                    # Pydantic 请求响应模型
│   ├── service/                   # 上传双写、图调度、历史读写
│   ├── utils/
│   │   ├── client/                # AI / 存储客户端单例（双检锁 + 延迟初始化）
│   │   ├── embedding_util.py      # BGE-M3 混合向量
│   │   ├── milvus_util.py         # 混合检索请求与执行
│   │   ├── markdown_util.py       # 表格线性化
│   │   ├── ranking.py             # 断崖截断 + 主体置信度决策（纯函数）
│   │   ├── text_normalize.py      # 产品名归一化 + 型号 token 匹配
│   │   ├── upload_guard.py        # 文件名净化 / 后缀白名单 / 流式限额
│   │   ├── task_store.py          # 任务态：内存 / Redis 双后端
│   │   ├── sse_util.py            # SSE 队列 / Redis Pub/Sub
│   │   └── log_context.py         # request_id / task_id / JSON 日志 / 耗时中间件
│   └── requirements.txt
├── eval/                          # 评测层（与线上同一条代码路径）
│   ├── dataset.jsonl              # 旧域 50 题（万用表，保留做回归）
│   ├── dataset-home.jsonl         # 新域 30 题（家电）
│   ├── corpus/home.jsonl          # 新域语料快照（72 切片）
│   ├── corpus/rs12.jsonl          # 旧域语料快照（22 切片）
│   ├── retrieval.py metrics.py    # 检索组件与指标
│   ├── run_eval.py                # 消融入口，输出 summary.md/json
│   ├── link_gold.py               # 证据句 → chunk_id，解析失败即报错
│   ├── answer_ab.py               # 答案层模型 A/B（固定上下文换模型）
│   ├── badcase.md                 # 已归因的失败案例（BC-01 ~ BC-11）
│   └── REPORT-baseline.md         # 固化的评测报告
├── tests/                         # 114 个单测（纯逻辑，不依赖 torch）
│   ├── unit/                      # 切分 / 表格 / RRF / 截断 / 决策 / 上传守卫 / 日志
│   └── conftest.py
├── tools/
│   ├── make_home_dataset.py       # 生成家电域评测集
│   ├── experiment_cutoff_hyde.py  # max_k 与 HyDE 权重扫描实验
│   ├── check_ci_isolated.py       # 屏蔽重依赖，模拟 CI 环境
│   └── pdf2md_zip.py              # MinerU 独立解析工具（校验图片断链）
├── deploy/                        # Docker 编排（基础设施 + 应用 profile）
├── loadtest/                      # Locust 脚本与压测记录
├── requirements-ci.txt            # CI 专用轻量依赖（不含 torch）
├── pyproject.toml                 # pytest / coverage / ruff 配置
└── README.md
```

## 快速开始

### 1. 环境要求

- Python **3.12**
- 一台能跑 CUDA 的机器（BGE-M3 与精排需要；CPU 可跑但慢）
- Milvus 2.5、MinIO、MongoDB（本地 Docker 或远程均可）
- DashScope API Key（LLM 与 VLM）

### 2. 拉起依赖服务

```bash
# 方式一：用仓库自带的 compose（含 etcd/minio/milvus/mongo/redis）
cd deploy
cp .env.example .env      # 填 DASHSCOPE_API_KEY、MODEL_DIR、数据卷目录
docker compose up -d
docker compose ps          # 五个容器都应为 healthy

# 方式二：已有 Milvus/MinIO/Mongo，直接在 .env 里填地址
```

离线镜像导入、GPU 写法、故障排查见 [`deploy/README.md`](deploy/README.md)。

### 3. 安装依赖

```bash
cd knowledge
python -m venv .venv
.venv/Scripts/activate            # Windows；Linux 用 source .venv/bin/activate
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

最少要改这几项：

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 大模型 Key |
| `MILVUS_URL` / `MINIO_ENDPOINT` / `MONGO_URL` | 三个存储地址 |
| `BGE_M3_PATH` / `BGE_RERANKER_LARGE` | 本地权重目录 |
| `BGE_DEVICE` / `BGE_FP16` | GPU 用 `cuda:0` / `1`，CPU 用 `cpu` / `false` |
| `API_KEYS` | 留空 = 不鉴权（仅本地）；填了就必须带 `X-API-Key` 或 `?key=` |
| `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` | 换语料时**务必换名**，避免混库 |

### 5. 启动服务

```bash
# 导入服务（:8000）
python -m knowledge.api.import_router

# 问答服务（:8001）
python -m knowledge.api.query_router
```

### 6. 使用

| 页面 | 地址 |
| --- | --- |
| 知识导入台 | `http://127.0.0.1:8000/import`（配了 API Key 时加 `?key=你的Key`） |
| 问答客服台 | `http://127.0.0.1:8001/chat?key=你的Key` |

上传一份说明书 → 看 8 个节点的时间线 → 到问答页提问，答案带 `[n]` 角标与配图。

### 7. 跑评测

```bash
# 单测（不需要 GPU）
python -m pytest tests eval/tests -q

# 校验标注：证据句 → chunk_id，解析失败会点名哪道题
python -m eval.link_gold --dataset eval/dataset-home.jsonl \
  --corpus eval/corpus/home.jsonl --corpus-source milvus

# 5 组检索消融（需要 GPU + 权重 + Key）
python -m eval.run_eval --dataset eval/dataset-home.jsonl \
  --corpus eval/corpus/home.jsonl --run-id home-v2 \
  --configs A1_dense,A2_hybrid,A3_hyde_rrf,A4_rerank,A5_cutoff

# 没有权重时只验证链路（数字无意义）
python -m eval.run_eval --embedder hash --hyde none --limit 5 --configs A1_dense,A2_hybrid
```

结果写到 `eval/results/<run_id>/`，`summary.md` 可直接贴文档。

### 8. 可观测性与压测

```bash
# 结构化日志：每条带 request_id / task_id，节点打 node_end + duration_ms
docker compose logs -f query-app | jq 'select(.task_id=="a1b2c3d4")'

# 压测
pip install locust
locust -f loadtest/locustfile.py --headless -u 5 -r 1 -t 2m --csv=lt-5 --host http://127.0.0.1:8001
```

实测（RTX 3060 / qwen3.8-flash 关思考 / 单进程 uvicorn / 旧域 22 切片语料）：

| 并发 | 请求数/时长 | P50 | P95 | 错误率 | 吞吐 |
| --- | --- | --- | --- | --- | --- |
| 5 | 10 / 60s | 11.0 s | 12.0 s | 0% | 0.17 QPS |
| 8 | 12 / 45s | 11.0 s | 11.0 s | 0% | 0.27 QPS |
| 10 | 20 / 60s | 12.0 s | 12.0 s | 0% | 0.33 QPS |

5→10 并发 P95 不变、吞吐翻倍，说明请求是并行的；**瓶颈在单卡串行推理**
（BGE-M3 与精排共用一把进程内锁）。要提吞吐只能模型服务化 / 批推理 / 语义缓存。

## 语料与评测

### 两套语料

| 语料 | 文档 | 页数 | 切片 | 题目 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 家电（主） | 美的变频空调 + 米家摄像机 | 40 + 16 | **72** | 30 | 主评测 |
| 万用表（旧） | RS-12 数字万用表 | 6 | 22 | 50 | 换域回归对照 |

### 评测集构成（新域 30 题）

| 题型 | 数量 | 例子 |
| --- | --- | --- |
| `fact` | 8 | 「51 机的建议制冷面积是多少？」 |
| `table` | 7 | 「51 机的 HSPF 值是多少？」 |
| `procedure` | 7 | 「怎么清洁空调滤尘网？」 |
| `multi_turn` | 5 | 「那 72 机的制冷面积呢？」（带历史） |
| `out_of_scope` | 3 | 「这台空调的整机保修期是几年？」（手册没有，应弃答） |

标注协议：每题写「章节 + 逐字证据句」，不绑 `chunk_id`。
`python -m eval.link_gold` 会把证据句映射到当前语料，**解析不到直接报错**——
等于用机器反查"这道题到底能不能从手册里找到答案"。当前 **30/30** 解析通过。

### 指标定义

| 指标 | 含义 |
| --- | --- |
| Hit@k | 前 k 条里有没有命中 gold |
| Recall@k | gold 被召回的比例（多 gold 时更敏感） |
| Precision@k | 前 k 条里有效片段占比 |
| MRR@k | 第一个正确片段的倒数名次 |
| nDCG@k | 排序质量（考虑名次与多 gold） |
| 平均文档数 / 平均上下文字数 | 成本代理指标 |
| P95 检索耗时 | 首 token 之外的检索段延迟 |

### 消融结果（新域，30 题 / 72 切片）

| 配置 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | 文档数 | 上下文字数 |
| --- | --- | --- | --- | --- | --- | --- |
| A1 纯稠密 | 0.852 | 0.815 | 0.711 | 0.723 | 10.0 | 7702 |
| A2 +稀疏（混合） | 0.852 | 0.815 | 0.738 | 0.738 | 10.0 | 7556 |
| A3 +HyDE & RRF | 0.852 | 0.815 | 0.753 | 0.747 | 10.0 | 7573 |
| A4 +精排 | **0.889** | **0.852** | **0.840** | **0.816** | 10.0 | 7573 |
| A5 +断崖截断 | **0.889** | **0.852** | **0.840** | **0.816** | **7.4** | **5692** |

### 消融结果（旧域，50 题 / 22 切片）

| 配置 | Hit@5 | MRR@5 | nDCG@5 | 文档数 | 上下文字数 |
| --- | --- | --- | --- | --- | --- |
| A1 纯稠密 | 0.875 | 0.758 | 0.788 | 10.0 | 3803 |
| A2 +稀疏 | 0.900 | 0.756 | 0.792 | 10.0 | 3755 |
| A3 +HyDE & RRF | 0.875 | 0.728 | 0.764 | 10.0 | 3830 |
| A4 +精排 | **0.950** | **0.885** | **0.902** | 10.0 | 3830 |
| A5 +断崖截断 | **0.950** | **0.885** | **0.902** | **8.2** | **3163** |

### 四条结论

1. **精排是唯一"确定赚"的组件**：新域 MRR 0.753→0.840、旧域 0.756→0.885，是单步增益最大的一环。
2. **断崖截断只省钱不影响质量**：新域上下文 7573→5692 字（-25%）、旧域 3830→3163 字（-17%），
   Hit/Recall/MRR/nDCG 一格没变。
3. **HyDE 在旧域是负收益**（0.900→0.875），在新域零收益——短问题 + 强型号场景下，
   假想文档会把不存在的数值灌进检索空间。生产默认关闭，只留开关。
4. **换域后绝对指标下降可归因**：0.950→0.889，根因是切分层（40 页手册里一个章节占 46% 的切片）
   与文档跨章重复（同一证据句出现两次导致假失败），不是换域动作本身引入。

### 调参实验：max_k 与 HyDE 权重都提不动 Hit@5

`tools/experiment_cutoff_hyde.py` 扫描两个旋钮（27 道计分题）：

| 旋钮 | 取值 | Hit@5 | 平均文档数 | 平均上下文字数 |
| --- | --- | --- | --- | --- |
| 断崖 `max_k` | 5 / 10 / 15 / 20 / 30 | 0.889（**全同**） | 4.4 → 7.1 → 7.7 | 3820 → 5740 → 6048 |
| HyDE 融合权重 | 0 / 0.25 / 0.5 / 1.0 / 1.5 / 2.0 | 0.889（**全同**） | 6.9 → 7.1 | 5427 → 5740 |
| 召回深度 | 10 / 20 / 30 | 0.889（全同） | 7.1 → 7.3 | 5740 → 6241 |

原因：截断只砍尾部、不改前五顺序；HyDE 换的是候选顺序，而精排会用 cross-encoder 彻底重排。
三个未命中题的 gold **都在前十之外**，所以加大 `max_k` 也捞不到；加深召回只把 Recall@5
从 0.833 抬到 0.852。**要提高 Hit@5 得动切分与标注，不是这几个旋钮。**

### 方差提醒

30 题基数下**一题 = 3.7 个百分点**，50 题下是 2 个百分点。所以：

- 单题级差异不解读（同一配置不同 run 会翻转 1 题）；
- 只讲多次运行稳定复现的差异：精排 +、截断降本、HyDE 负收益，这三条在多次 run 里都成立。

## 接口文档

启动后访问 `/docs` 看 Swagger。配了 `API_KEYS` 时，除页面外的接口都需要
`X-API-Key: <key>` 头（或 `?key=<key>`，供 SSE 的 `EventSource` 使用）。

### 导入服务 `:8000`

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| GET | `/import` | 导入页面 | 否 |
| POST | `/upload` | 上传说明书并异步入库 | 是 |
| GET | `/status/{task_id}` | 任务进度（状态 / 已完成 / 进行中 / 各节点耗时） | 是 |

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -H "X-API-Key: dev-key-1" \
  -F "file=@./美的空调.pdf"
# => {"task_id":"9c5ae9ff","message":"文件上传成功,请等待导入流程完成"}

curl -H "X-API-Key: dev-key-1" http://127.0.0.1:8000/status/9c5ae9ff
```

### 查询服务 `:8001`

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| GET | `/chat` | 问答页面 | 否 |
| POST | `/query` | 提交问题（流式 / 非流式） | 是 |
| GET | `/stream/{task_id}` | SSE 事件流 | 是 |
| GET | `/history/{session_id}` | 拉取会话历史（默认 50 条） | 是 |
| DELETE | `/history/{session_id}` | 清空会话 | 是 |

```bash
# 非流式
curl -X POST http://127.0.0.1:8001/query \
  -H "X-API-Key: dev-key-1" -H "Content-Type: application/json" \
  -d '{"query":"这款空调的适用面积是多少？","session_id":"demo","is_stream":false}'

# 流式：先拿 task_id，再订阅 SSE
curl -N "http://127.0.0.1:8001/stream/<task_id>?key=dev-key-1"
```

SSE 事件：

```text
event: progress
data: {"status":"processing","done_list":["确认问题产品"],"running_list":["切片搜索"]}

event: delta
data: {"delta":"将功能转盘置于"}

event: final
data: {"answer":"……完整答案……","image_urls":["http://.../page_19_image_8.jpg"]}
```

## 数据模型

### Milvus：切片集合（`CHUNKS_COLLECTION`）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_id` | INT64（主键，auto_id） | 切片唯一 ID |
| `dense_vector` | FLOAT_VECTOR(1024) | BGE-M3 稠密向量，`AUTOINDEX` / `COSINE` |
| `sparse_vector` | SPARSE_FLOAT_VECTOR | BGE-M3 稀疏权重，`SPARSE_INVERTED_INDEX` / `IP` |
| `content` | VARCHAR(65535) | 切片正文（含图片 URL） |
| `title` | VARCHAR(65535) | 章节标题 |
| `parent_title` | VARCHAR(65535) | 上级标题 |
| `file_title` | VARCHAR(65535) | 来源文档 |
| `item_name` | VARCHAR(65535) | 产品名，检索时作为过滤条件 |

### Milvus：主体集合（`ITEM_NAME_COLLECTION`）

`pk / file_title / item_name / dense_vector / sparse_vector`，
用于"用户说的型号 → 知识库里哪份手册"。

### MongoDB：`chat_message`

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `session_id` | string | 会话 ID |
| `role` | string | `user` / `assistant` |
| `text` | string | 消息正文 |
| `rewritten_query` | string | 指代消解后的问题 |
| `item_names` | array | 关联产品名 |
| `ts` | float | 时间戳 |

## 关键配置项

导入侧（`knowledge/processor/import_process/config.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `max_content_length` | 1000 | 切片上限，超出二次切分 |
| `min_content_length` | 500 | 切片下限，不足与相邻合并 |
| `img_content_length` | 200 | 图片上下文抓取字数 |
| `item_name_chunk_k` | 3 | 主体识别使用前几片 |
| `embedding_batch_size` | 8 | 批量嵌入大小 |
| `requests_per_minute` | 20 | VLM 图注限速 |

查询侧（`knowledge/processor/query_process/config.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_CONTEXT_CHARS` | 12000 | 上下文预算 |
| `RRF_K` / `RRF_MAX_RESULTS` | 60 / 10 | RRF 平滑常数与保留数 |
| `RERANK_MAX_TOP_K` / `MIN_TOP_K` / `GAP_ABS` | 10 / 3 / 0.15 | 断崖截断三参数 |
| `EMBEDDING_SEARCH_LIMIT` / `HYDE_SEARCH_LIMIT` | 10 / 5 | 两路召回深度 |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` / `dense` / `sparse` |
| `EMBEDDING_DENSE_WEIGHT` / `SPARSE_WEIGHT` | 0.5 / 0.5 | 混合检索权重 |
| `ITEM_NAME_HIGH_CONFIDENCE` / `MID_CONFIDENCE` | 0.7 / 0.6 | 主体确认阈值 |
| `ITEM_NAME_GAP_ABS` / `MAX_OPTIONS` | 0.15 / 5 | 候选过滤与反问上限 |

安全与上传（`knowledge/api/security.py` + `knowledge/utils/upload_guard.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `API_KEYS` | 空 | 逗号分隔；空 = 不鉴权（仅本地） |
| `ALLOWED_ORIGINS` | 本机 8000/8001 | CORS 白名单 |
| `MAX_UPLOAD_MB` | 50 | 上传上限，流式校验 |
| `ALLOWED_UPLOAD_SUFFIXES` | `.pdf,.md,.markdown` | 后缀白名单 |

模型侧（`knowledge/utils/client/ai_clients.py`）：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_DEFAULT_MODEL` | `qwen3.8-flash` | 主体抽取 / HyDE / 答案生成 |
| `LLM_ENABLE_THINKING` | `false` | qwen3 系列默认开思考（单次 20s+），必须关 |
| `ITEM_MODEL` / `VL_MODEL` | `qwen3.8-flash` / `qwen3-vl-plus` | 主体识别 / 图注 |

任务态：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TASK_BACKEND` | `memory` | `memory` / `redis` |
| `REDIS_URL` | `redis://127.0.0.1:6379/0` | Redis 地址 |
| `TASK_TTL_SECONDS` | 86400 | 任务状态保留时长 |

## 工程化能力

| 能力 | 实现 | 关键点 |
| --- | --- | --- |
| 鉴权 | `API_KEYS` 依赖 | 头或查询参数；`hmac.compare_digest` 防时序侧信道 |
| 上传安全 | `upload_guard` | 修掉 `os.path.join(dir, filename)` 的路径穿越；流式限额不信 `Content-Length` |
| 任务态 | `TaskStore` 协议 | 内存 / Redis 双实现，24h TTL，连不上自动降级 |
| 流式 | SSE + Redis Pub/Sub | 推流与消费可跨进程，支持多 worker |
| 日志 | `log_context` | JSON、`request_id`/`task_id`、`node_end + duration_ms`、纯 ASGI 中间件 |
| 部署 | `deploy/docker-compose.yml` | 五容器 + 两个应用服务，全 healthcheck，依赖 `service_healthy` |
| CI | `.github/workflows/ci.yml` | ruff + pytest + 标注校验，22~40s |
| 压测 | `loadtest/locustfile.py` | 问答与状态轮询两类用户；404 探活属预期 |

测试：**114 个用例、覆盖率 46%**。核心纯逻辑模块 88~94%（切分、表格线性化、上传守卫、产品名归一化）。
`tools/check_ci_isolated.py` 用 import 钩子把 torch/pymilvus 变"不存在"，本地即可复现 CI 依赖环境。

## 已知限制

1. **超长章节缺二级标题**：40 页手册里 `# 安装说明` 一个标题占 46% 的切片，标题维度失去区分度，
   稀疏检索被同质词带偏。这是新域 Hit@5 低于旧域的主因。
2. **文档跨章重复**：同一事实在"快速指引"与"详细说明"各写一次，gold 只标一片会产生假失败。
3. **无页码溯源**：解析产物 `structured_content.json` 里有 `page_idx`，但尚未写入 Milvus 标量字段，
   答案暂时给不出"第 X 页"。
4. **单租户**：没有按部门/密级过滤，多租户需扩展为 `tenant_id` 标量过滤 + JWT 角色。
5. **无语义缓存**：相同/相似问题重复走完整链路，成本与延迟都有压缩空间。
6. **单卡串行推理**：5 并发 P95 约 12s、吞吐 0.33 QPS，瓶颈是 GPU 上的模型串行化。
7. **应用容器未在低配 VM 落地**：镜像 5~6G，实测机磁盘仅剩 3.5G，当前是"容器跑基础设施、应用跑宿主机"。

## 路线图

| 优先级 | 事项 | 预期收益 |
| --- | --- | --- |
| P1 | 章节内隐式二级切分（按编号条目/段落切） | 唯一标题/切片从 21/50 提升，Hit@5 回升 |
| P1 | 跨章重复检测 + gold 多标 | 修掉假失败，预计 +1 题 |
| P2 | 页码溯源（`page_idx` → 标量字段 → 答案引用 + PDF 跳页） | 可溯源性质变 |
| P2 | 语义缓存（相似问题命中） | 延迟与 token 成本下降 |
| P3 | 多租户与权限过滤 | 面向 B 端交付 |
| P3 | 模型服务化（embedding/rerank 独立服务） | 吞吐提升，解开单卡锁 |

## 更新记录

| 日期 | 变更 |
| --- | --- |
| 2026-11-14 | 换域到家电说明书（美的空调 + 米家摄像机）；30 题新评测集；全仓注释去除旧域残留 |
| 2026-11-14 | 新增 max_k / HyDE 权重对照实验，确认两者对 Hit@5 无影响 |
| 2026-11-13 | P1 工程化：鉴权与上传校验、Redis 任务态、结构化日志、Locust 压测、GitHub Actions |
| 2026-11-12 | 评测层落地：50 题评测集、5 组消融、badcase 归因、报告生成 |

---

## 相关文档

- [`eval/README.md`](eval/README.md)：评测协议、指标口径、复现方式
- [`eval/REPORT-baseline.md`](eval/REPORT-baseline.md)：旧域完整评测报告（机器生成）
- [`eval/badcase.md`](eval/badcase.md)：BC-01 ~ BC-11 失败案例与归因
- [`deploy/README.md`](deploy/README.md)：离线镜像导入、模型挂载、GPU 写法、故障排查
- [`loadtest/README.md`](loadtest/README.md)：压测方法与实测数据