"""
知识库加载模块
"""
import json
import os


class KnowledgeBase:
    """PFAS知识库"""

    def __init__(self, data_path=None):
        self.triples = []
        self.documents = []
        self.regulations = []

        if data_path and os.path.exists(data_path):
            self.load(data_path)
        else:
            self._init_default()

    def _init_default(self):
        """初始化默认知识库"""
        self.triples = self._get_default_triples()
        self.documents = self._get_default_documents()
        self.regulations = self._get_default_regulations()

    def load(self, path):
        """加载知识图谱数据"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.triples = data.get('triples', [])
            self.documents = data.get('documents', [])
            self.regulations = data.get('regulations', [])
        except Exception as e:
            print(f"加载知识库失败: {e}")
            self._init_default()

    def save(self, path):
        """保存知识图谱数据"""
        data = {
            'triples': self.triples,
            'documents': self.documents,
            'regulations': self.regulations,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def search(self, query, top_k=5):
        """搜索相关知识"""
        results = []
        query_lower = query.lower()

        # 搜索三元组
        for triple in self.triples:
            text = f"{triple.get('head', '')} {triple.get('relation', '')} {triple.get('tail', '')}"
            if self._match(query_lower, text.lower()):
                results.append({
                    'type': 'triple',
                    'content': text,
                    'data': triple,
                })

        # 搜索文档
        for doc in self.documents:
            if self._match(query_lower, doc.get('content', '').lower()):
                results.append({
                    'type': 'document',
                    'content': doc.get('content', '')[:200],
                    'data': doc,
                })

        # 搜索法规
        for reg in self.regulations:
            if self._match(query_lower, reg.get('name', '').lower()):
                results.append({
                    'type': 'regulation',
                    'content': f"{reg['name']}: {reg.get('content', '')}",
                    'data': reg,
                })

        return results[:top_k]

    def _match(self, query, text):
        """简单的关键词匹配"""
        keywords = query.split()
        return any(kw in text for kw in keywords if len(kw) > 1)

    def _get_default_triples(self):
        """默认三元组"""
        return [
            {'head': 'PFOA', 'relation': 'has_endpoint', 'tail': 'NR-AR'},
            {'head': 'PFOA', 'relation': 'has_endpoint', 'tail': 'NR-AhR'},
            {'head': 'PFOA', 'relation': 'has_endpoint', 'tail': 'SR-MMP'},
            {'head': 'PFOA', 'relation': 'via_mechanism', 'tail': 'PPARα激活'},
            {'head': 'PFOA', 'relation': 'via_mechanism', 'tail': '氧化应激'},
            {'head': 'PFOA', 'relation': 'causes', 'tail': '肝毒性'},
            {'head': 'PFOA', 'relation': 'causes', 'tail': '免疫抑制'},
            {'head': 'PFOS', 'relation': 'has_endpoint', 'tail': 'NR-AR'},
            {'head': 'PFOS', 'relation': 'has_endpoint', 'tail': 'NR-AhR'},
            {'head': 'PFOS', 'relation': 'has_endpoint', 'tail': 'SR-MMP'},
            {'head': 'PFOS', 'relation': 'via_mechanism', 'tail': '甲状腺激素干扰'},
            {'head': 'PFOS', 'relation': 'causes', 'tail': '免疫抑制'},
            {'head': 'GenX', 'relation': 'has_endpoint', 'tail': 'NR-AhR'},
            {'head': 'GenX', 'relation': 'via_mechanism', 'tail': '肝脏毒性'},
        ]

    def _get_default_documents(self):
        """默认文档"""
        return [
            {
                'title': 'PFAS毒理学综述',
                'content': '全氟和多氟烷基物质(PFAS)是一类人工合成的有机化合物，因其独特的化学稳定性被广泛应用于工业和消费品。',
                'source': 'Sunderland et al., 2019',
            },
            {
                'title': 'PFAS环境归趋',
                'content': 'PFAS因其碳-氟键的极高稳定性，在环境中具有极强的持久性，被称为永久化学品。',
                'source': 'Buck et al., 2011',
            },
        ]

    def _get_default_regulations(self):
        """默认法规"""
        return [
            {
                'name': 'GB 5749-2022',
                'content': '生活饮用水卫生标准：PFOS+PFOA ≤ 40 ng/L',
                'country': '中国',
            },
            {
                'name': 'EPA NPDWR',
                'content': 'PFOA ≤ 4 ng/L, PFOS ≤ 4 ng/L',
                'country': '美国',
            },
            {
                'name': 'WHO Guidelines',
                'content': 'PFOA ≤ 100 ng/L, PFOS ≤ 40 ng/L',
                'country': '国际',
            },
        ]
