# PFAS 风险评估报告

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | 全氟癸酸（PFDA） |
| CAS号 | 335-76-2 |
| SMILES | `OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F` |
| 化合物类别 | PFCA |
| 分子式 | C10HF19O2 |
| 分子量 | 514.08 g/mol |
| LogP | 5.72 |
| 评估日期 | 2026-05-31 17:28 |
| 评估系统 | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 二、毒理学预测数据

### 集成模型预测结果（6算法加权平均）

| 毒性终点 | 中文含义 | 预测概率 | 标准差 | 风险等级 |
|---------|---------|---------|--------|---------|
| NR-AR | 雄激素受体拮抗 | 0.690 | 0.261 | 高 |
| NR-AR-LBD | 配体结合域活性 | 0.347 | 0.304 | 中 |
| NR-AhR | 芳香烃受体激活 | 0.644 | 0.263 | 高 |
| SR-HSE | 热休克元件响应 | 0.452 | 0.248 | 中 |
| SR-MMP | 线粒体膜电位异常 | 0.647 | 0.262 | 高 |
| SR-p53 | p53通路激活 | 0.572 | 0.307 | 中 |

### 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **0.559** |
| **风险等级** | **中风险** |

---

## 三、已知毒性数据（知识图谱）

- PFDA --[contains]--> carboxylic acid group
- PFDA --[contains]--> perfluoroalkyl chain
- PFDA --[has_endpoint]--> NR-AhR
- PFDA --[has_endpoint]--> SR-p53
- PFDA --[has_endpoint]--> BCF
- PFDA --[via_mechanism]--> PPARα激活
- PFDA --[via_mechanism]--> 脂质代谢干扰
- PFDA --[exists_in]--> 海洋
- PFDA --[exists_in]--> 生物体
- PFDA --[causes]--> 肝毒性
- PFDA --[applies_to]--> 中国新污染物治理行动方案
- PFDA --[applies_to]--> 斯德哥尔摩公约 (POPs)

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
