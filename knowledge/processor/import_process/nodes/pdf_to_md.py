import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.exceptions import (
    FileProcessingError,
    PdfConversionError,
    ValidationError,
)
from knowledge.processor.import_process.state import ImportGraphState


class PdfToMdNode(BaseNode):
    name = "pdf_to_md_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        import_file_path_obj,file_dir_path_obj = self._validate_state_inputs_path(state)
        processed_code = self._execute_mineru(import_file_path_obj, file_dir_path_obj)

        if processed_code == 0:
            self.logger.info(f"PDF转MD成功!{import_file_path_obj} - 输出：{file_dir_path_obj}")
        else:
            self.logger.error("PDF转MD失败")
            raise PdfConversionError("PDF转MD失败",self.name)

        md_path = self._get_md_paths(import_file_path_obj, file_dir_path_obj)

        state["md_path"] = md_path

        return state

    def _validate_state_inputs_path(self, state: ImportGraphState) -> Tuple[Path,Path] :

        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        if not import_file_path:
            raise ValidationError("输入的文件为空",self.name)

        import_file_path_obj = Path(import_file_path)

        if not import_file_path_obj:
            raise FileProcessingError("输入的文件路径不存在",self.name)

        if not file_dir:
            file_dir = import_file_path_obj.parent
        file_dir_obj = Path(file_dir)

        return import_file_path_obj,file_dir_obj

    @lru_cache(maxsize=1)  # noqa: B019 - 单例节点，缓存不会跨实例泄漏


    def _execute_mineru(self, import_file_path_obj, file_dir_path_obj):
        proc = subprocess.Popen(
            args=["mineru-kit",          # 必须用 venv 里的可执行文件，mineru/mineru-kit 都不在 PATH 上
                "parse",
                import_file_path_obj,
                "-o",
                file_dir_path_obj,
                "-f", "zip",         # 出 markdown.md + images/ + json，图片不丢
                "--tier", "standard",# basic=纯小模型(快) / standard=小模型+VLM(更准)
                "--ocr-mode", "auto", ],
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.STDOUT,  # 合并错误到标准输出
            text=True,
            encoding="utf-8",
            errors="replace",  # 遇到乱码时替换
            bufsize=1  # 行缓冲，实时输出
        )

        for line in proc.stdout:
            print(line.rstrip())

        import zipfile
        from pathlib import Path

        zip_path = Path(file_dir_path_obj) / f"{Path(import_file_path_obj).stem}.zip"
        dest_dir = Path(file_dir_path_obj) / Path(import_file_path_obj).stem
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest_dir)

        md_text = (dest_dir / "markdown.md").read_text(encoding="utf-8")
        refs = md_text.count("![")
        img_count = len(list((dest_dir / "images").glob("*")))
        print("解压到:", dest_dir)
        print(f"md 字符数: {len(md_text)} | 图片引用: {refs} | images 文件数: {img_count}")

        processed_code = proc.wait()


        return processed_code

    def _get_md_paths(self, import_file_path_obj, file_dir_path_obj):
        import_file_name = import_file_path_obj.stem
        md_path = file_dir_path_obj / import_file_name / "markdown.md"
        return str(md_path)


if __name__ == "__main__":
    setup_logging()
    import_file_path = r"F:\BaiduNetdiskDownload\temp_dir\美的空调.pdf"
    file_dir = r"E:\project\agent_project\shoper_brain\knowledge\processor\import_process\temp_dir"
    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    PdfToMdNode = PdfToMdNode()

    processed_state = PdfToMdNode(state)  # __call__ -> process()

    # entry.process(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))





