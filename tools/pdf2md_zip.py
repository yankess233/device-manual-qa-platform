"""方案 A: MinerU 4.x PDF -> Markdown 全量产物（外置图片，图片不丢）

用法:
    python tools/pdf2md_zip.py <pdf路径或目录> [输出目录] [选项]

选项:
    --tier {basic,standard}   解析档位，默认 standard（basic 更快，纯小模型）
    --pages 1-5,r2           只解析指定页（默认全部）
    --ocr-mode {auto,txt,ocr} 默认 auto
    --keep-zip               保留中间 zip（默认保留，--no-keep-zip 删除）

做的事:
    1) 调 mineru-kit parse -f zip --tier ... --ocr-mode ...（默认全页，避免"只解析前 10 页"的坑）
    2) 解压到 <输出目录>/<pdf名>/
    3) markdown.md -> <pdf名>.md; middle_json.json -> <pdf名>_middle.json 等（对齐 2.x 命名习惯）
    4) 校验：md 里每个 images/xxx 引用都真实存在，打印统计；有断链则退出码 1
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINERU_KIT = ROOT / "knowledge" / ".venv" / "Scripts" / "mineru-kit.exe"

RENAME = {
    "markdown.md": "{stem}.md",
    "middle_json.json": "{stem}_middle.json",
    "structured_content.json": "{stem}_structured_content.json",
    "model_output.json": "{stem}_model_output.json",
}


def parse_to_zip(pdf: Path, out_dir: Path, tier: str, ocr_mode: str, pages: str | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(MINERU_KIT), "parse", str(pdf), "-o", str(out_dir), "-f", "zip",
           "--tier", tier, "--ocr-mode", ocr_mode]
    if pages:
        cmd += ["--pages", pages]
    print("[1/3] 解析:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    zip_path = out_dir / f"{pdf.stem}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"没有生成压缩包: {zip_path}")
    return zip_path


def unzip_and_rename(zip_path: Path, pdf: Path, out_dir: Path) -> Path:
    dest = out_dir / pdf.stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    print("[2/3] 解压 ->", dest)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    for old, new in RENAME.items():
        src = dest / old
        if src.exists():
            src.rename(dest / new.format(stem=pdf.stem))
    return dest


def verify(dest: Path, pdf: Path) -> bool:
    md_path = dest / f"{pdf.stem}.md"
    text = md_path.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    broken = [r for r in refs if not r.startswith("data:") and not (dest / r).exists()]
    images = sorted(p.name for p in (dest / "images").glob("*")) if (dest / "images").is_dir() else []
    print("[3/3] 校验")
    print("  md:", md_path.name, "| 字符数:", len(text))
    print("  图片引用:", len(refs), "| images/ 文件数:", len(images))
    print("  断链引用:", broken if broken else "无")
    print("  表格(table 标签):", text.count("<table"), "| 内嵌 base64 图片:", "base64" in text)
    ok = bool(refs) and not broken
    print("  结论:", "OK 图片完整" if ok else "警告: 图片引用异常")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description="MinerU 4.x PDF->MD（方案A：md + 外置 images/）")
    ap.add_argument("pdf", help="PDF 文件路径，或包含多个 PDF 的目录")
    ap.add_argument("out_dir", nargs="?", help="输出目录，默认取 PDF 所在目录")
    ap.add_argument("--tier", default="standard", choices=["basic", "standard"])
    ap.add_argument("--pages", default=None, help="如 1-5,8 或 all")
    ap.add_argument("--ocr-mode", default="auto", choices=["auto", "txt", "ocr"])
    ap.add_argument("--keep-zip", dest="keep_zip", action="store_true", default=True)
    ap.add_argument("--no-keep-zip", dest="keep_zip", action="store_false")
    a = ap.parse_args()

    src = Path(a.pdf)
    if not src.exists():
        print("找不到:", src)
        return 2
    out_dir = Path(a.out_dir) if a.out_dir else (src.parent if src.is_file() else src)
    targets = [src] if src.is_file() else sorted(p for p in src.glob("*") if p.suffix.lower() == ".pdf")
    if not targets:
        print("目录下没有 PDF:", src)
        return 2

    ok_all = True
    for pdf in targets:
        print(f"=== {pdf.name} ===")
        zip_path = parse_to_zip(pdf, out_dir, a.tier, a.ocr_mode, a.pages)
        dest = unzip_and_rename(zip_path, pdf, out_dir)
        ok_all &= verify(dest, pdf)
        if not a.keep_zip:
            zip_path.unlink(missing_ok=True)
        print("产物目录:", dest)
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
