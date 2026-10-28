import ast
import time
from typing import List, Optional, Union

# import aiohttp
# import requests
# from bs4 import BeautifulSoup
# from langchain_community.docstore.base import Docstore
from langchain_community.docstore.document import Document
from langchain_community.tools.tavily_search import TavilySearchResults


# Langchain自带包搜索
class TavilySearch:
    all_times = []

    def __init__(self, benchmark=False):
        self.benchmark = benchmark

    def search(self, entity: str, is_retry: bool = False) -> Union[str, Document]:
        """Try to search for information using TavilySearchResults.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        try:
            daily = TavilySearchResults(max_results=5)
            ret = daily.invoke(input=entity)

            content_list = []
            for obj in ret:
                content_list.append(obj["content"])

            result = "\n".join(content_list)
            if self.benchmark and not is_retry:
                # we only benchmark the outermost call
                self.all_times.append(round(time.time() - s, 2))
            return result
        except Exception as e:
            return "search error:{}".format(e)

    async def asearch(
        self, entity: str, is_retry: bool = False
        ) -> Union[str, Document]:
        """Try to search for information using TavilySearchResults asynchronously.

        Args:
            entity: entity string.

        Returns: a Document object or error message.
        """
        s = time.time()
        try:
            daily = TavilySearchResults(max_results=5)
            ret = await self._async_invoke(daily, entity)

            content_list = []
            for obj in ret:
                content_list.append(obj["content"])

            result = "\n".join(content_list)
            if self.benchmark and not is_retry:
                # we only benchmark the outermost call
                self.all_times.append(round(time.time() - s, 2))
            return result
        except Exception as e:
            return "search error:{}".format(e)

    async def _async_invoke(self, search_obj, entity):
        """
        处理异步调用并将结果转换为可等待对象的辅助方法。

        这里假设search_obj.invoke已被修改为返回可等待对象（如异步生成器或协程等），
        通过async for循环正确地迭代获取结果，并将每个结果项收集到一个列表中返回，以便后续正确处理。
        """
        result_list = []
        # 假设 invoke 返回一个普通的列表，不是异步生成器
        ret = search_obj.invoke(input=entity)
        for item in ret:
            result_list.append(item)
        return result_list