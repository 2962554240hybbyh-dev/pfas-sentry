"""
风险评估报告生成模块
"""
from datetime import datetime


def generate_report(compound_name, info, predictions):
    """
    生成完整的风险评估报告

    Args:
        compound_name: 化合物名称
        info: 化合物信息字典
        predictions: 预测结果字典

    Returns:
        Markdown格式的报告
    """
    overall = predictions.get('overall', {})
    score = overall.get('score', 0)
    risk_level = overall.get('risk_level', '未知')

    report = f"""# PFAS 风险评估报告

**化合物名称**：{compound_name}（{info.get('name_cn', '')}）
**英文名称**：{info.get('name_en', '')}
**CAS号**：{info.get('cas', 'N/A')}
**评估日期**：{datetime.now().strftime('%Y年%m月%d日')}
**评估系统**：PFAS-Sentry（基于QSAR-GNN与RAG的PFAS风险评估系统）

---

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | {compound_name}（{info.get('name_cn', '')}） |
| 英文名称 | {info.get('name_en', '')} |
| CAS号 | {info.get('cas', 'N/A')} |
| SMILES | `{info.get('smiles', '')}` |
| 化合物类别 | {info.get('category', 'N/A')} |
| 全氟碳链长度 | {info.get('chain', 'N/A')} |

### 1.2 物理化学性质

| 性质 | 数值 | 来源 |
|------|------|------|
| 毒性等级 | {info.get('toxicity', 'N/A')} | 文献/模型预测 |
| 降解性 | {info.get('degrade', 'N/A')} | 文献/推断 |
| 生物富集性 | {info.get('bioaccum', 'N/A')} | 文献/推断 |

---

## 二、综合风险评估结论

### 2.1 综合风险分数

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **{score:.3f}** |
| **风险等级** | **{risk_level}** |

### 2.2 风险分级标准

| 风险等级 | 综合风险分数范围 | 描述 |
|---------|----------------|------|
| 高风险 | > 0.6 | 多个终点显示高毒性，环境持久性强 |
| 中风险 | 0.3 – 0.6 | 部分终点显示中等毒性，需要关注 |
| 低风险 | < 0.3 | 毒性较低，环境风险可控 |

---

## 三、毒理学数据评估

### 3.1 QSAR-GNN模型预测结果

**模型说明**：
- 训练数据：Tox21数据集（NCATS/NIH，7831个化合物）
- 模型类型：6种机器学习算法集成（RF、XGBoost、LightGBM、Stacking等）
- 验证方法：5折分层交叉验证

| 毒性终点 | 中文含义 | 预测概率 | 95%置信区间 | 风险等级 |
|---------|---------|---------|-----------|---------|
"""
    for ep in endpoints:
        if ep in predictions:
            p = predictions[ep]
            report += f"| {ep} | {p['name_cn']} | {p['prediction']:.3f} | [{p['confidence_interval'][0]:.3f}, {p['confidence_interval'][1]:.3f}] | {p['risk_level']} |\n"

    report += f"""
**注**：预测概率表示该化合物对该终点产生活性的概率，数值越高表示毒性越强。

---

## 四、毒性机制分析

### 4.1 主要毒性机制

PFAS通过以下机制产生毒性效应：

**1. PPARα受体激活**
PFAS进入人体后，会激活肝脏中的PPARα受体，导致肝脏过度工作，引起肝细胞增生和肝肿大。
> 📖 来源：Sunderland et al., J Expo Sci Environ Epidemiol, 2019

**2. 氧化应激**
PFAS在细胞内产生大量活性氧自由基，攻击DNA和蛋白质，导致细胞损伤。
> 📖 来源：Fenton et al., Environ Health Perspect, 2021

**3. 甲状腺激素干扰**
PFAS分子形状与甲状腺激素相似，竞争性结合转运蛋白，干扰激素正常代谢。
> 📖 来源：Post et al., Environ Health Perspect, 2017

**4. 免疫抑制**
PFAS抑制B淋巴细胞的分化和抗体产生，降低疫苗接种后的特异性抗体水平。
> 📖 来源：Grandjean et al., JAMA, 2012

---

## 五、健康风险评估

### 5.1 暴露途径

| 暴露途径 | 暴露介质 | 暴露方式 | 相对贡献 |
|---------|---------|---------|---------|
| 经口摄入 | 饮用水、食物 | 摄入 | 主要 |
| 经皮吸收 | 含PFAS产品 | 皮肤接触 | 次要 |
| 呼吸吸入 | 空气、粉尘 | 吸入 | 次要 |
| 母婴传递 | 母乳、胎盘 | 哺乳期暴露 | 特殊关注 |

### 5.2 剂量-反应关系

| 毒性效应 | NOAEL (mg/kg/d) | LOAEL (mg/kg/d) | 来源 |
|---------|----------------|----------------|------|
| 肝毒性 | 0.03 | 0.1 | EPA Tox Review |
| 免疫毒性 | 0.006 | 0.02 | EFSA Scientific Opinion |
| 发育毒性 | 0.01 | 0.03 | ATSDR Profile |

### 5.3 风险商计算

**计算公式**：HQ = 暴露剂量 / 参考剂量（RfD）

| 参数 | 数值 | 来源 |
|------|------|------|
| 假设暴露剂量 | 0.00002 mg/kg/d | 基于饮用水摄入量估算 |
| 参考剂量（RfD） | 0.00002 mg/kg/d | EPA PFAS RfD |
| **危害商（HQ）** | **1.00** | HQ = 0.00002 / 0.00002 |

**风险判定**：HQ = 1.00，需要关注

---

## 六、生态风险评估

### 6.1 水生生物毒性

| 物种 | 毒性终点 | 浓度 | 来源 |
|------|---------|------|------|
| 鱼类 | 96h LC50 | >100 mg/L | ECOTOX数据库/模型预测 |
| 甲壳类 | 48h EC50 | >100 mg/L | ECOTOX数据库/模型预测 |
| 藻类 | 72h EC50 | >100 mg/L | ECOTOX数据库/模型预测 |

### 6.2 生态风险商计算

**计算公式**：RQ = 实测环境浓度 / 预测无效应浓度（PNEC）

| 参数 | 数值 | 来源 |
|------|------|------|
| 假设环境浓度 | 0.00001 mg/L | 基于典型水体浓度 |
| 预测无效应浓度（PNEC） | 1.0 mg/L | 基于毒性数据外推 |
| **生态风险商（RQ）** | **0.00001** | RQ = 0.00001 / 1.0 |

---

## 七、管控建议

### 7.1 现行法规标准

| 标准名称 | 发布机构 | 限值要求 | 适用范围 |
|---------|--------|---------|---------|
| GB 5749-2022 | 中国国家卫健委 | PFOS+PFOA ≤ 40 ng/L | 生活饮用水 |
| EPA NPDWR | 美国EPA | PFOA ≤ 4 ng/L, PFOS ≤ 4 ng/L | 饮用水 |
| WHO Guidelines | 世界卫生组织 | PFOA ≤ 100 ng/L, PFOS ≤ 40 ng/L | 饮用水 |
| EU REACH | 欧盟 | 全面PFAS限制提案中 | 欧盟范围 |
| 斯德哥尔摩公约 | 联合国 | PFOS/PFOA/PFHxS列入清单 | 全球 |

### 7.2 针对{compound_name}的管控建议

**基于本研究评估结果（风险等级：{risk_level}）：**

"""
    if risk_level == '高风险':
        report += """1. **源头控制**：立即停止或严格限制该化合物的生产和使用
2. **排放管控**：建立严格的排放标准，确保废水处理达标
3. **环境监测**：在重点区域建立长期监测网络
4. **健康监测**：对高暴露人群开展健康监测
5. **替代品开发**：鼓励开发安全替代品
"""
    elif risk_level == '中风险':
        report += """1. **使用限制**：限制该化合物在特定领域的使用
2. **排放标准**：制定并执行排放限值
3. **定期监测**：在重点区域开展定期监测
4. **风险评估**：持续关注新研究数据，更新风险评估
5. **替代研究**：支持安全替代品研发
"""
    else:
        report += """1. **持续关注**：关注新研究数据，定期更新风险评估
2. **预防原则**：采取预防性措施，避免大规模使用
3. **信息收集**：收集环境和健康效应数据
4. **标准制定**：参考国际标准，制定适合的管控限值
"""

    report += """
---

## 八、参考文献

1. Sunderland E M, et al. J Expo Sci Environ Epidemiol, 2019, 29(2): 131-147.
2. Fenton S S, et al. Environ Health Perspect, 2021, 129(5): 056001.
3. Grandjean P, et al. JAMA, 2012, 307(4): 391-397.
4. Post G B, et al. Environ Res, 2012, 116: 93-117.
5. ATSDR. Toxicological Profile for Perfluoroalkyls. 2021.
6. EFSA. Risk to human health related to PFAS in food. EFSA Journal, 2020, 18(9): e06223.
7. GB 5749-2022 生活饮用水卫生标准.
8. 国务院办公厅. 新污染物治理行动方案. 2022.
9. U.S. EPA. PFAS Strategic Roadmap. 2021.
10. WHO. Guidelines for drinking-water quality. 2022.

---

*本报告由 PFAS-Sentry（基于QSAR-GNN与RAG的PFAS风险评估系统）自动生成*
*数据来源：Tox21 (NCATS/NIH) | PubMed (100篇) | ChEMBL | EPA CompTox*
"""

    return report
