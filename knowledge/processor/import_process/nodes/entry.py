import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class EntryNode(BaseNode):
    """入口节点
        根据上传文件后缀，来决定走哪个分支节点：[.pdf,.md]
    """

    name = "entry_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        self.log_step("step1","获取文件路径")
        import_file_path = state.get("import_file_path","")
        file_dir = state.get("file_dir","")

        self.log_step("step2","检测文件路径是否为空")
        if not import_file_path:
            raise ValidationError(f"文件不存在",self.name)

        path = Path(import_file_path)

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path
        elif suffix == ".md":
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}暂支持")
            raise ValidationError(f"文件类型{suffix}暂支持")

        file_title = path.stem
        state['file_title'] = file_title

        return state

if __name__ == "__main__":
    setup_logging()
    import_file_path = r"E:\project\agent_project\shoper_brain\knowledge\processor\import_process\temp_dir\万用表RS-12的使用.pdf"
    file_dir = r"E:\project\agent_project\shoper_brain\knowledge\processor\import_process\temp_dir"
    state = {
        "is_pdf_read_enabled": False,
        "is_md_read_enabled": False,
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }

    entry = EntryNode()

    processed_state = entry(state)  # __call__ -> process()

    # entry.process(state)

    print(json.dumps(processed_state, indent=4, ensure_ascii=False))