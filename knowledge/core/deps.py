from functools import cache

from knowledge.service.import_file_service import ImportFileService
from knowledge.service.query_service import QueryService


# 专业创建对象，并进行缓存管理。可以实现对象共享。
@cache  # 缓存满了，不清理。  OOM 问题
# @lru_cache # 缓存满了，可以根据lru(最近最少使用)策略进行缓存清理
def get_import_file_service() -> ImportFileService:
    return ImportFileService()


@cache
def get_query_service() -> QueryService:
    return QueryService()
