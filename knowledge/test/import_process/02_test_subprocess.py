import os
import subprocess
from pathlib import Path

# 基础用法：run() - 同步执行，等待完成
os.environ['MINERU_MODEL_SOURCE'] = 'modelscope'#设置MinerU的模型来源 ModelScope（阿里魔搭）
os.environ['MODELSCOPE_OFFLINE'] = '1'#1 表示开启离线模式（MODELSCOPE_CACHE 指定的目录）
os.environ['HF_HUB_OFFLINE'] = '1'#启用 HuggingFace Hub 离线模式
os.environ['TRANSFORMERS_OFFLINE'] = '1'#启用 Transformers 库离线模式
os.environ['MINERU_MODEL_BASE_DIR'] = r'F:\mineru_model'#MinerU 模型根目录（已从 C 盘迁到 F 盘，子进程会继承）

print("[OK] 环境变量已配置")   # 原为 emoji ✅，GBK 控制台会 UnicodeEncodeError
print(f"   MINERU_MODEL_SOURCE = {os.environ.get('MINERU_MODEL_SOURCE')}")
print(f"   MODELSCOPE_OFFLINE = {os.environ.get('MODELSCOPE_OFFLINE')}")
print(f"   HF_HOME = {os.environ.get('HF_HOME')}")
print()

# MinerU 4.x 没有 2.x 的 -p / -b / --backend / --source（--source 只在 models download 上有）
# 等价写法：mineru-kit parse <pdf> -o <目录> -f zip --tier standard --ocr-mode auto
# 关键：必须 -f zip，才会把图片外置成 images/（默认 -f markdown 只写一个 md，图片会丢）
OUT_DIR = r"E:\project\agent_project\shoper_brain\knowledge\processor\import_process\temp_dir"
PDF_PATH = r"F:\BaiduNetdiskDownload\temp_dir\万用表RS-12的使用.pdf"
MINERU_KIT = r"E:\project\agent_project\shoper_brain\knowledge\.venv\Scripts\mineru-kit.exe"

result = subprocess.run(
    [
        MINERU_KIT,          # 必须用 venv 里的可执行文件，mineru/mineru-kit 都不在 PATH 上
        "parse",
        PDF_PATH,
        "-o",
        OUT_DIR,
        "-f", "zip",         # 出 markdown.md + images/ + json，图片不丢
        "--tier", "standard",# basic=纯小模型(快) / standard=小模型+VLM(更准)
        "--ocr-mode", "auto",
    ],
    capture_output=True,  # 捕获子进程的标准输出和标准错误，结果可通过 .stdout / .stderr 获取
    encoding="utf-8",  # 以 UTF-8 编码解码输出，返回字符串而非字节
    #errors="strict"（默认）→ 直接报错 UnicodeDecodeError
    errors="replace",  # 遇到无法解码的字符用  替换       errors="ignore"直接丢弃无效字节
    check=True,  # 子进程返回非零退出码时**自动抛出** `CalledProcessError` 异常
)

print(result.stdout)

# ---------------- 解压并校验（4.x 的 zip 里是 markdown.md / middle_json.json / images/） ----------------
import zipfile
from pathlib import Path

zip_path = Path(OUT_DIR) / f"{Path(PDF_PATH).stem}.zip"
dest_dir = Path(OUT_DIR) / Path(PDF_PATH).stem
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(dest_dir)

md_text = (dest_dir / "markdown.md").read_text(encoding="utf-8")
refs = md_text.count("![")
img_count = len(list((dest_dir / "images").glob("*")))
print("解压到:", dest_dir)
print(f"md 字符数: {len(md_text)} | 图片引用: {refs} | images 文件数: {img_count}")