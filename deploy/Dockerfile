# ============================================================
# 应用镜像：两个服务共用（导入 :8000 / 问答 :8001）
#
# 设计取舍：
#   - 基础镜像用 slim，torch 先装 CPU 版：默认 GPU 版会把镜像撑到 6GB+，
#     而虚拟机里通常没有 GPU；需要 GPU 时用 --build-arg 或改用基础镜像自带 CUDA；
#   - MinerU 用 build arg 控制：只跑问答服务时 WITH_MINERU=0，镜像小一半以上；
#   - 不把 .env 打进镜像，配置在运行时通过 env_file 注入。
# ============================================================
FROM python:3.12-slim

ARG WITH_MINERU=1
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    HF_HOME=/home/app/.cache/huggingface \
    MODELSCOPE_CACHE=/home/app/.cache/modelscope \
    XDG_CACHE_HOME=/home/app/.cache

# opencv / torch / mineru 运行时需要的系统库；curl 供 compose healthcheck 之外的排查使用
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libmagic1 \
        poppler-utils \
        fonts-noto-cjk \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖，利用层缓存：只改业务代码时不会重装 torch
COPY knowledge/requirements.txt /app/knowledge/requirements.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && if [ "$WITH_MINERU" = "1" ]; then \
         python -m pip install -r /app/knowledge/requirements.txt; \
       else \
         grep -vi '^mineru' /app/knowledge/requirements.txt > /tmp/req-no-mineru.txt \
         && python -m pip install -r /tmp/req-no-mineru.txt; \
       fi

# 再拷业务代码
COPY knowledge /app/knowledge

# 非 root 运行；temp_data 用命名卷挂载，这里先建好目录并交权
# 命名卷首次挂载会继承镜像内目录的属主，所以这两个目录必须先建好并 chown 给 app，
# 否则容器里以非 root 身份跑会写不进去（temp_data 与模型缓存）
RUN useradd --create-home --shell /bin/bash app \
    && mkdir -p /app/knowledge/temp_data /home/app/.cache \
    && chown -R app:app /app /home/app
USER app

EXPOSE 8000 8001

CMD ["python", "-m", "knowledge.api.query_router"]