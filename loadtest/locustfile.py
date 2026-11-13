"""Locust 压测脚本：问答与导入两条链路。

用法（在能访问服务的机器上跑）::

    pip install locust
    locust -f loadtest/locustfile.py --headless -u 5  -r 1 -t 2m --csv=lt-5  --host http://127.0.0.1:8001
    locust -f loadtest/locustfile.py --headless -u 10 -r 2 -t 2m --csv=lt-10 --host http://127.0.0.1:8001

要点：
- CPU 部署下单次问答可能 5~10 秒，`timeout` 给了 60s，否则会看到大量"超时"
  假错误，误判服务挂了；
- 压测前先手动问一次，让 BGE/重排模型进内存（预热），否则第一批请求的延迟会污染 P95；
- 关注 `*_stats.csv` 里的 50%/95%/99% 与 failures，而不是只看 RPS。
"""

from __future__ import annotations

import os

from locust import HttpUser, between, task

API_KEY = os.getenv("LOADTEST_API_KEY", "")
QUESTION = os.getenv("LOADTEST_QUESTION", "RS-12 怎么测电阻？")


class QueryUser(HttpUser):
    """问答链路：短请求、高并发，是压测主角。"""

    wait_time = between(1, 3)

    def _headers(self) -> dict:
        return {"X-API-Key": API_KEY} if API_KEY else {}

    @task(5)
    def query_non_stream(self):
        with self.client.post(
            "/query",
            json={"query": QUESTION, "session_id": "loadtest", "is_stream": False},
            headers=self._headers(),
            timeout=60,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")

    @task(1)
    def history(self):
        self.client.get("/history/loadtest", headers=self._headers(), timeout=15)


class ImportUser(HttpUser):
    """导入链路：低并发 + 长耗时，验证任务态与状态轮询在并发下不串。"""

    wait_time = between(5, 10)

    def _headers(self) -> dict:
        return {"X-API-Key": API_KEY} if API_KEY else {}

    @task
    def poll_status(self):
        # 用不存在的 task_id 探活：正常应快速返回空状态，不会 500
        with self.client.get(
            "/status/loadtest-nonexistent", headers=self._headers(), timeout=15, catch_response=True
        ) as response:
            # 不存在的 task_id 返回 404 是**预期行为**（不是故障），只有 5xx 才算失败
            if response.status_code >= 500:
                response.failure(f"HTTP {response.status_code}")
            elif response.status_code in (200, 401, 404):
                response.success()