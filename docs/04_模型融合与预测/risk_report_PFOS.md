# PFAS 风险评估报告

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | 全氟辛烷磺酸（PFOS） |
| CAS号 | 1763-23-1 |
| SMILES | `OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F` |
| 化合物类别 | PFSA |
| 分子式 | C8HF17O3S |
| 分子量 | 500.13 g/mol |
| LogP | 4.84 |
| 评估日期 | 2026-05-31 17:28 |
| 评估系统 | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 二、毒理学预测数据

### 集成模型预测结果（6算法加权平均）

| 毒性终点 | 中文含义 | 预测概率 | 标准差 | 风险等级 |
|---------|---------|---------|--------|---------|
| NR-AR | 雄激素受体拮抗 | 0.646 | 0.270 | 高 |
| NR-AR-LBD | 配体结合域活性 | 0.399 | 0.272 | 中 |
| NR-AhR | 芳香烃受体激活 | 0.651 | 0.260 | 高 |
| SR-HSE | 热休克元件响应 | 0.423 | 0.261 | 中 |
| SR-MMP | 线粒体膜电位异常 | 0.645 | 0.263 | 高 |
| SR-p53 | p53通路激活 | 0.569 | 0.303 | 中 |

### 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **0.555** |
| **风险等级** | **中风险** |

---

## 三、已知毒性数据（知识图谱）

- PFOS --[contains]--> sulfonic acid group
- PFOS --[contains]--> perfluoroalkyl chain
- PFOS --[has_endpoint]--> NR-AR
- PFOS --[has_endpoint]--> NR-AhR
- PFOS --[has_endpoint]--> SR-HSE
- PFOS --[has_endpoint]--> SR-MMP
- PFOS --[has_endpoint]--> SR-p53
- PFOS --[has_endpoint]--> BCF
- PFOS --[via_mechanism]--> PPARα激活
- PFOS --[via_mechanism]--> 甲状腺激素干扰
- PFOS --[via_mechanism]--> 免疫抑制
- PFOS --[via_mechanism]--> 氧化应激
- PFOS --[exists_in]--> 饮用水
- PFOS --[exists_in]--> 地表水
- PFOS --[exists_in]--> 地下水
- PFOS --[exists_in]--> 土壤
- PFOS --[exists_in]--> 海洋
- PFOS --[exists_in]--> 沉积物
- PFOS --[exists_in]--> 生物体
- PFOS --[affects]--> 鱼类
- PFOS --[affects]--> 甲壳类
- PFOS --[affects]--> 鸟类
- PFOS --[affects]--> 哺乳动物
- PFOS --[affects]--> 人类
- PFOS --[causes]--> 肝毒性
- PFOS --[causes]--> 甲状腺功能异常
- PFOS --[causes]--> 免疫功能抑制
- PFOS --[causes]--> 发育毒性
- PFOS --[causes]--> 生殖毒性
- PFOS --[causes]--> 内分泌干扰
- PFOA --[applies_to]--> GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- PFOS --[applies_to]--> 中国新污染物治理行动方案
- PFOS --[applies_to]--> 斯德哥尔摩公约 (POPs)
- PFOS --[applies_to]--> GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- PFOS --[applies_to]--> EPA PFAS 管控标准 (70 ng/L)
- PFOS --[applies_to]--> 欧盟 REACH 限制
- 人类 --[protected_by]--> GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- 鱼类 --[protected_by]--> GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- 甲壳类 --[protected_by]--> GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)

---

## 四、毒性机制分析

1. **PPARα激活**：激活过氧化物酶体增殖物激活受体α，导致脂质代谢紊乱
2. **氧化应激**：诱导活性氧产生，导致DNA氧化损伤
3. **线粒体功能障碍**：干扰线粒体电子传递链
4. **内分泌干扰**：竞争性结合甲状腺激素转运蛋白
5. **免疫抑制**：抑制免疫细胞功能

---

## 五、管控建议

| 标准 | 限值 |
|------|------|
| GB 5749-2022 饮用水标准 | PFOS+PFOA ≤ 40 ng/L |
| 美国EPA (2023) | PFOA ≤ 4 ng/L, PFOS ≤ 4 ng/L |
| 欧盟REACH | 全面PFAS限制提案进行中 |
| WHO (2022) | PFOA ≤ 100 ng/L, PFOS ≤ 40 ng/L |
| 斯德哥尔摩公约 | PFOS/PFOA/PFHxS列入POPs清单 |

---

*本报告由 QSAR-GNN + RAG PFAS风险评估系统自动生成*
