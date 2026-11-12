# 虚拟机 Docker 部署

面向的场景：**镜像来自离线 tar，全部服务跑在一台 Linux 虚拟机的 Docker 里**。
当前环境里的 192.168.10.150 就是这台机器。

```
                     ┌─────────────────────── kb-net ───────────────────────┐
 浏览器 / curl ──────▶│ import-app :8000   query-app :8001                   │
                     │        │                 │                           │
                     │        ├── milvus :19530 ── etcd :2379               │
                     │        ├── minio  :9000  ── (Milvus 存储 + 图片直链)  │
                     │        └── mongodb:27017 (会话历史)                   │
                     └──────────────────────────────────────────────────────┘
```

---

## 1. 导入镜像（离线 tar）

在你现成的 `docker_images` 目录里，逐个导入（文件名以实际为准）：

```bash
cd /path/to/docker_images
docker load -i quay.io.coreos.etcd.tar
docker load -i minio.tar
docker load -i milvusedb.tar
docker load -i mongo7.0.tar
docker load -i zilliz.attu.tar
docker images | grep -E "etcd|minio|milvus|mongo|attu"
```

版本必须与 `docker-compose.yml` 里的 tag 对得上：

| 服务 | 镜像 |
| --- | --- |
| etcd | `quay.io/coreos/etcd:v3.5.25` |
| minio | `minio/minio:RELEASE.2024-12-18T13-15-44Z` |
| milvus | `milvusdb/milvus:v2.5.5` |
| mongo | `mongo:7.0` |
| attu（可选） | `zilliz/attu:v2.5.10` |

> 如果 `docker load` 出来的 tag 与上面不同（比如只有 `mongo:latest`），
> 用 `docker tag <实际tag> <期望tag>` 对齐再起服务，否则 compose 会尝试联网拉取。

---

## 2. 准备目录与模型

```bash
sudo mkdir -p /opt/shoper-brain/{volumes,models}
sudo chown -R $USER:$USER /opt/shoper-brain

# 模型目录结构（必须叫这两个名字，容器内挂到 /models）
/opt/shoper-brain/models/
├── bge-m3/                # BGE-M3 权重（dense + sparse）
└── bge-reranker-large/    # 精排权重
```

模型直接从你现在 Windows 盘的 `F:\ai_models\modelscope_cache\models\BAAI\` 拷过来即可：

```bash
scp -r F:/ai_models/modelscope_cache/models/BAAI/bge-m3            user@192.168.10.150:/opt/shoper-brain/models/
scp -r F:/ai_models/modelscope_cache/models/BAAI/bge-reranker-large user@192.168.10.150:/opt/shoper-brain/models/
```

> 虚拟机没有 GPU 时：`.env` 里设 `BGE_DEVICE=cpu`、`BGE_FP16=false`。
> CPU 上精排 10 条约 1~3 秒，单次问答可接受，但不要指望并发。

---

## 3. 配置 `.env`

```bash
cd deploy
cp .env.example .env
vi .env      # 至少改这两处：
             #   DASHSCOPE_API_KEY=<你的 key>
             #   DOCKER_VOLUME_DIRECTORY / MODEL_DIR 改成上面创建的路径
```

要点：

- `MODEL_DIR` 指向宿主机模型目录，容器只读挂载到 `/models`；
- `MILVUS_URL` / `MINIO_ENDPOINT` / `MONGO_URL` **不要改**，compose 会强制覆盖成服务名；
- `.env` 已在 `.gitignore` 里（`deploy/.env`），不要提交。

---

## 4. 起服务

```bash
# 只起基础设施（和现在 192.168.10.150 上跑的等价）
docker compose up -d

# 看健康状态：三个 infra 都要 healthy
docker compose ps

# 首次起 Milvus 会跑约 60~90 秒初始化，healthcheck 有 90s start_period
docker compose logs -f milvus
```

确认 infra 就绪后再起应用：

```bash
# 构建并启动两个应用服务（首次构建 10~20 分钟，主要是 torch + mineru）
docker compose --profile app up -d --build

# 只想跑问答、不需要 PDF 解析时，用小镜像：
#   .env 里设 WITH_MINERU=0，再 --profile app up -d --build
```

需要 Milvus 图形界面时：

```bash
docker compose --profile tools up -d      # Attu: http://<vm-ip>:7000
```

---

## 5. 没有外网的机器怎么构建应用镜像

虚拟机装不了依赖时，在**有网的机器**上构建，再搬过去：

```bash
# 有网机器（注意平台要和虚拟机一致，通常是 amd64）
docker build -f deploy/Dockerfile \
  --build-arg WITH_MINERU=1 \
  -t shoper-brain/app:0.1 .

docker save shoper-brain/app:0.1 | gzip > shoper-app-0.1.tar.gz

# 虚拟机
docker load -i shoper-app-0.1.tar.gz
docker compose --profile app up -d        # 镜像已存在，不会再 build
```

---

## 6. 验证

```bash
# 宿主机有 python3 时
python3 deploy/verify.py --host 127.0.0.1

# 没有的话用 curl
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs   # 200
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8001/docs   # 200
```

浏览器打开：

| 页面 | 地址 |
| --- | --- |
| 知识导入台 | `http://<vm-ip>:8000/import` |
| 问答客服台 | `http://<vm-ip>:8001/chat` |

跑一次真实链路：上传一本 PDF → 看 7 个节点进度 → 到问答页提问 → 确认答案带图。

---

## 7. 有 GPU 的机器

在 `docker-compose.yml` 的 `x-app-base` 里放开设备预留（需要宿主机装好 NVIDIA Container Toolkit）：

```yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

同时 `.env` 改成 `BGE_DEVICE=cuda:0`、`BGE_FP16=true`。
镜像里的 torch 是 CPU 版，GPU 场景需要换基础镜像（`pytorch/pytorch:2.5-cuda12.4-cudnn9-runtime`）并去掉 CPU 轮子的安装步骤。

---

## 8. 常见问题

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `milvus` 一直 unhealthy | etcd/minio 没就绪，或宿主机内存不足（建议 ≥ 8G） | `docker compose logs milvus etcd minio`；先确保两个依赖 healthy |
| 应用容器起不来，日志报 `ConnectionError` | 依赖还没 healthy 就被拉起 | compose 已配 `condition: service_healthy`；若手改过 compose 记得保留 |
| `BGE_M3_PATH 不存在` | 模型目录名不对 / 没挂进去 | `docker compose exec query-app ls /models` 确认能看到两个子目录 |
| 上传 PDF 报 MinerU 相关错误 | `WITH_MINERU=0` 构建的镜像，或模型没下 | 用 `WITH_MINERU=1` 重建；MinerU 首次解析会下载模型，需要外网或预置缓存 |
| 端口被占用 | 宿主机已有同端口服务 | 改 `.env` 里的 `IMPORT_PORT` / `QUERY_PORT` 等 |
| 图片链接 404 | `MINIO_ENDPOINT` 被写成 `127.0.0.1`，浏览器在别的机器上访问不到 | 图片 URL 由 MinIO 生成，跨机器访问要保证 `9000` 端口对外可达 |
| `docker compose` 仍去联网拉镜像 | 本地 tar 的 tag 与 compose 不一致 | `docker images` 对齐 tag，或 `docker tag` 重命名 |
| `deploy/.env` 里中文注释导致解析异常 | Windows 换行/编码 | 用 `file .env` 确认是 UTF-8；必要时 `sed -i 's/\r$//' .env` |

---

## 9. 与你原 compose 的差异

| 项 | 原来 | 现在 | 说明 |
| --- | --- | --- | --- |
| 服务名 | `standalone` | `milvus` | 应用里用 `milvus:19530`，比 `standalone` 可读 |
| 网络名 | `milvus` | `kb-net` | 换网络后需整体 `down` 再 `up`；bind mount 的数据不受影响 |
| MinIO 凭据变量 | `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | 映射到 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | 新版本 MinIO 只认 ROOT_*，应用侧仍读 ACCESS/SECRET，两边用同一份 `.env` 对齐 |
| 应用服务 | 无（跑在 Windows 宿主机） | `import-app` / `query-app`（profile `app`） | 与 infra 同网络，不用再写 `192.168.10.150` |
| healthcheck | 部分有 | 全部有 + `depends_on: condition` | 避免"应用比 Milvus 先起来"的启动竞态 |

> 切换步骤：`docker compose -f <旧文件> down` → 用本目录的 compose `up -d`。
> 数据都在 `${DOCKER_VOLUME_DIRECTORY}/volumes` 的 bind mount 里，不会丢。