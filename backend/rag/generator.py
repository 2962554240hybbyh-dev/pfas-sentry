"""
生成器模块 - 基于模板的回答生成
"""


class Generator:
    """基于模板的回答生成器"""

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.templates = self._load_templates()

    def generate(self, question, retrieved_docs=None):
        """
        生成回答

        Args:
            question: 用户问题
            retrieved_docs: 检索到的文档

        Returns:
            包含回答文本和来源的字典
        """
        question_lower = question.lower()

        # 匹配模板
        for keywords, template_func in self.templates.items():
            if any(kw in question_lower for kw in keywords):
                answer = template_func(retrieved_docs)
                return answer

        # 默认回答
        return self._default_answer(question, retrieved_docs)

    def _load_templates(self):
        """加载回答模板"""
        return {
            ('pfoa', '毒'): self._answer_pfoa_toxicity,
            ('pfos', '毒'): self._answer_pfos_toxicity,
            ('机制',): self._answer_mechanism,
            ('生物富集', '积累'): self._answer_bioaccumulation,
            ('健康', '危害'): self._answer_health_effects,
            ('去除', '处理'): self._answer_removal,
            ('标准', '限值'): self._answer_standards,
            ('genx', '安全'): self._answer_genx_safety,
            ('短链', '长链'): self._answer_chain_length,
        }

    def _answer_pfoa_toxicity(self, docs):
        """PFOA毒性解释"""
        return {
            'text': """**PFOA（全氟辛酸，CAS: 335-67-1）有毒的原因：**

**1. PPARα受体激活** 🔴
PFOA进入人体后，会激活肝脏中的PPARα受体，导致肝脏过度工作，引起肝细胞增生和肝肿大。
> 📖 来源：Sunderland et al., J Expo Sci Environ Epidemiol, 2019 [PMID: 30464233]

**2. 氧化应激** 🔴
PFOA在细胞内产生大量活性氧自由基（ROS），攻击DNA和蛋白质，导致细胞损伤。
> 📖 来源：Fenton et al., Environ Health Perspect, 2021 [PMID: 34009096]

**3. 免疫抑制** 🔴
PFOA抑制B淋巴细胞的分化和抗体产生，降低疫苗接种后的特异性抗体水平。
> 📖 来源：Grandjean et al., JAMA, 2012 [PMID: 22274686]

**4. 甲状腺激素干扰** 🟠
PFOA可竞争性结合甲状腺激素转运蛋白（TTR），干扰T4的正常运输和代谢。
> 📖 来源：Post et al., Environ Health Perspect, 2017

**5. 内分泌干扰** 🟠
PFOA可干扰多种激素的正常功能，导致生殖发育异常。
> 📖 来源：ATSDR, 2021
""",
            'sources': [
                'Sunderland et al., 2019 [PMID: 30464233]',
                'Fenton et al., 2021 [PMID: 34009096]',
                'Grandjean et al., 2012 [PMID: 22274686]',
            ],
            'confidence': 0.95,
        }

    def _answer_pfos_toxicity(self, docs):
        """PFOS毒性解释"""
        return {
            'text': """**PFOS（全氟辛烷磺酸，CAS: 1763-23-1）的毒性：**

PFOS与PFOA的毒性机制相似，但PFOS的生物半衰期更长（约5.4年），生物富集性更强。

**主要毒性效应：**
1. 免疫抑制：降低疫苗抗体反应
2. 甲状腺干扰：影响甲状腺激素代谢
3. 肝毒性：导致肝脏脂肪变性
4. 发育毒性：影响胎儿发育

> 📖 来源：EFSA Scientific Opinion, 2020
> 📋 标准：GB 5749-2022 PFOS+PFOA ≤ 40 ng/L
""",
            'sources': ['EFSA, 2020', 'GB 5749-2022'],
            'confidence': 0.90,
        }

    def _answer_mechanism(self, docs):
        """毒性机制解释"""
        return {
            'text': """**PFAS的主要毒性机制：**

1. **PPARα激活**：激活肝脏受体，导致脂质代谢紊乱
2. **氧化应激**：产生自由基，损伤DNA和蛋白质
3. **甲状腺激素干扰**：竞争性结合转运蛋白
4. **免疫抑制**：降低B细胞抗体产生能力
5. **线粒体功能障碍**：干扰细胞能量代谢

> 📖 来源：Sunderland et al., 2019; Fenton et al., 2021
""",
            'sources': ['Sunderland et al., 2019', 'Fenton et al., 2021'],
            'confidence': 0.90,
        }

    def _answer_bioaccumulation(self, docs):
        """生物富集解释"""
        return {
            'text': """**PFAS为什么会生物富集：**

1. **化学稳定性极高** - 碳-氟键是最强的化学键，极难被酶分解
2. **与蛋白质高度结合** - 与血浆蛋白紧密结合，难以排出
3. **食物链放大** - 每经过一个营养级，浓度放大2-10倍

> 📖 来源：Buck et al., 2011; Conder et al., 2008
""",
            'sources': ['Buck et al., 2011', 'Conder et al., 2008'],
            'confidence': 0.90,
        }

    def _answer_health_effects(self, docs):
        """健康效应解释"""
        return {
            'text': """**PFAS对人体的主要危害：**

| 健康效应 | 证据等级 | 关键研究 |
|---------|---------|---------|
| 肝毒性 | 强 | 血清PFAS与肝酶升高正相关 |
| 甲状腺疾病 | 强 | 干扰甲状腺激素代谢 |
| 免疫抑制 | 强 | 降低疫苗抗体反应 |
| 发育毒性 | 强 | 与低出生体重相关 |
| 肾癌 | 中 | IARC列为2B类致癌物 |

> 📖 来源：ATSDR, 2021; EPA PFAS Roadmap, 2021
""",
            'sources': ['ATSDR, 2021', 'EPA, 2021'],
            'confidence': 0.90,
        }

    def _answer_removal(self, docs):
        """去除方法解释"""
        return {
            'text': """**饮用水中PFAS的去除方法：**

1. **活性炭吸附** ⭐推荐 - 去除率>90%
2. **离子交换树脂** ⭐推荐 - 去除率可达99%
3. **反渗透膜** - 去除率>95%
4. **高级氧化** - 可彻底分解，但成本高

> 📋 标准：GB 5749-2022 PFOS+PFOA ≤ 40 ng/L
""",
            'sources': ['GB 5749-2022'],
            'confidence': 0.85,
        }

    def _answer_standards(self, docs):
        """标准解释"""
        return {
            'text': """**各国PFAS管控标准：**

| 标准 | 限值 | 国家 |
|------|------|------|
| GB 5749-2022 | PFOS+PFOA ≤ 40 ng/L | 中国 |
| EPA NPDWR | PFOA ≤ 4 ng/L | 美国 |
| WHO Guidelines | PFOA ≤ 100 ng/L | 国际 |
| EU REACH | 全面限制中 | 欧盟 |
""",
            'sources': ['GB 5749-2022', 'EPA', 'WHO'],
            'confidence': 0.95,
        }

    def _answer_genx_safety(self, docs):
        """GenX安全性"""
        return {
            'text': """**GenX vs PFOA 安全性对比：**

| 项目 | PFOA | GenX |
|------|------|------|
| 碳链长度 | 8个碳 | 3个碳（含醚键） |
| 生物半衰期 | 3.8年 | 约30天 |
| 生物富集性 | 高 | 中等 |
| IARC分类 | 2B类 | 未分类 |

**结论：GenX相对更安全**，但仍需关注。

> 📖 来源：Gomis et al., 2018
""",
            'sources': ['Gomis et al., 2018'],
            'confidence': 0.85,
        }

    def _answer_chain_length(self, docs):
        """碳链长度影响"""
        return {
            'text': """**碳链长度对PFAS毒性的影响：**

| 特性 | 长链(≥7碳) | 短链(<7碳) |
|------|-----------|-----------|
| 生物半衰期 | 数年 | 数天-数周 |
| 生物富集性 | 高 | 低-中 |
| 毒性 | 高 | 中 |
| 环境持久性 | 极高 | 高 |

短链PFAS相对更安全，但并非完全安全。
""",
            'sources': ['Zheng et al., 2022'],
            'confidence': 0.85,
        }

    def _default_answer(self, question, docs):
        """默认回答"""
        # 从检索结果构建回答
        sources = []
        if docs:
            for doc in docs[:3]:
                if 'data' in doc:
                    data = doc['data']
                    if 'source' in data:
                        sources.append(data['source'])

        return {
            'text': f"""关于"{question}"的回答：

PFAS（全氟和多氟烷基物质）是一类含有碳-氟键的人工合成有机化合物，被称为"永久化学品"。

主要来源：不粘锅涂层、防水织物、消防泡沫、食品包装。

> 📖 来源：EPA PFAS Strategic Roadmap, 2021
> 📋 标准：中国《新污染物治理行动方案》, 2022
""",
            'sources': sources or ['EPA, 2021', '国务院, 2022'],
            'confidence': 0.70,
        }
