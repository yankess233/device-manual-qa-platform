# 压测记录（Locust）

环境：Windows 宿主机 · RTX 3060 · `qwen3.8-flash`（关闭思考）· 知识库 1 份手册（22 切片）
· 基础设施在 VM 192.168.10.150 · 单进程 `uvicorn`（无 worker）

```bash
export LOADTEST_API_KEY=dev-key-1
python -m locust -f loadtest/locustfile.py --headless -u 8 -r 2 -t 45s \
  --csv=loadtest/results/lt-8 --host http://127.0.0.1:8001 --only-summary
```

## 实测结果（/query，非流式）

| 并发 | 请求数 | P50 | P95 | 错误率 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 5 | 10 / 60s | 11.0s | 12.0s | 0% | 与 1 并发时的单题延迟接近 |
| 8 | 12 / 45s | 11.0s | 11.0s | 0% | 吞吐 0.27 QPS |
| 10 | 20 / 60s | 12.0s | 12.0s | 0% | 吞吐 0.33 QPS |

`/status` 探活（不存在的 task_id）P95 16~18ms，随并发线性增长，说明任务态查询本身不是瓶颈。

## 结论

1. **延迟是"单题成本"决定的，不是排队决定的**：5→10 并发 P95 只从 12.0s 涨到 12.0s，
   吞吐却从 0.17 涨到 0.33 QPS，说明请求基本是并行的，没有出现雪崩式退化。
2. **真正的瓶颈是单卡 GPU 上的模型推理串行化**：BGE-M3 编码与 bge-reranker 走同一把
   进程内锁，同一时刻只有一个请求在用卡；再叠一次 LLM 生成（约 3~5s），单题 11s 就定了。
3. **要提吞吐只有三条路**：多 worker + 模型服务化（把 embedding/rerank 拆成独立服务）、
   批量推理、或降低单题成本（关 HyDE、减少 top_k、语义缓存）。
4. 简历上应写**实测口径**：`P95 12s @ 5 并发，吞吐 0.17 QPS，瓶颈为单卡串行推理`，
   不要写"支持高并发"这类没有数字支撑的话。

## 注意

- 压测前先手动问一次做**模型预热**，否则第一批请求的冷启动会污染 P95。
- `ImportUser.poll_status` 用不存在的 task_id 探活，**404 是预期行为**，
  只有 5xx 才算失败（`catch_response` 里已按此判定）。
- 结果 CSV 在 `loadtest/results/`，已被 `.gitignore` 忽略，需要留档就贴进本文。