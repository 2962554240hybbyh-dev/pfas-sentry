"""
检索器模块
"""


class Retriever:
    """基于关键词的检索器"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def retrieve(self, query, top_k=5):
        """
        检索相关文档

        Args:
            query: 查询字符串
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        results = self.kb.search(query, top_k=top_k)
        return results

    def retrieve_by_compound(self, compound_name):
        """按化合物检索"""
        results = []
        for triple in self.kb.triples:
            if compound_name.lower() in triple.get('head', '').lower():
                results.append({
                    'type': 'triple',
                    'content': f"{triple['head']} {triple['relation']} {triple['tail']}",
                    'data': triple,
                })
        return results

    def retrieve_by_endpoint(self, endpoint):
        """按毒性终点检索"""
        results = []
        for triple in self.kb.triples:
            if endpoint.lower() in triple.get('tail', '').lower():
                results.append({
                    'type': 'triple',
                    'content': f"{triple['head']} {triple['relation']} {triple['tail']}",
                    'data': triple,
                })
        return results

    def retrieve_regulations(self, query=''):
        """检索法规标准"""
        results = []
        for reg in self.kb.regulations:
            if not query or query.lower() in reg.get('name', '').lower():
                results.append({
                    'type': 'regulation',
                    'content': f"{reg['name']}: {reg.get('content', '')}",
                    'data': reg,
                })
        return results
