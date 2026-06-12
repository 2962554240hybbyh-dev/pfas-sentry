# PFAS 风险评估报告

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | 全氟辛酸（PFOA） |
| CAS号 | 335-67-1 |
| SMILES | `OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F` |
| 化合物类别 | PFCA |
| 分子式 | C8HF15O2 |
| 分子量 | 414.06 g/mol |
| LogP | 4.45 |
| TPSA | 37.30 Å² |
| 氢键供体 | 1 |
| 氢键受体 | 1 |
| 评估日期 | 2026-05-31 16:39 |
| 评估系统 | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 二、毒理学预测数据

### 2.1 集成模型预测结果（6算法加权平均）

| 毒性终点 | 中文含义 | 预测概率 | 标准差 | 风险等级 |
|---------|---------|---------|--------|---------|
| NR-AR | 雄激素受体拮抗 | 0.333 | 0.307 | 中 |
| NR-AR-LBD | 配体结合域活性 | 0.389 | 0.202 | 中 |
| NR-AhR | 芳香烃受体激活 | 0.286 | 0.155 | 低 |
| SR-HSE | 热休克元件响应 | 0.461 | 0.293 | 中 |
| SR-MMP | 线粒体膜电位异常 | 0.472 | 0.298 | 中 |
| SR-p53 | p53通路激活 | 0.277 | 0.202 | 低 |

### 2.2 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **0.370** |
| **风险等级** | **中风险** |
| 评估终点数 | 6 |

---

## 三、已知毒性数据（知识图谱）

### 分子结构包含
- PFOA → carboxylic acid group
- PFOA → perfluoroalkyl chain

### 具有毒性终点
- PFOA → NR-AR
- PFOA → NR-AhR
- PFOA → SR-MMP
- PFOA → SR-p53
- PFOA → BCF

### 毒性机制
- PFOA → PPARα激活
- PFOA → 氧化应激
- PFOA → 线粒体功能障碍

### 环境存在
- PFOA → 饮用水
- PFOA → 地表水
- PFOA → 地下水
- PFOA → 土壤
- PFOA → 海洋
- PFOA → 生物体

### 影响物种
- PFOA → 鱼类
- PFOA → 哺乳动物
- PFOA → 人类

### 导致健康效应
- PFOA → 肝毒性
- PFOA → 肾毒性
- PFOA → 甲状腺功能异常
- PFOA → 免疫功能抑制
- PFOA → 发育毒性
- PFOA → 生殖毒性
- PFOA → 致癌性

### 适用管控措施
- PFOA → 中国新污染物治理行动方案
- PFOA → 斯德哥尔摩公约 (POPs)
- PFOA → GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- PFOA → EPA PFAS 管控标准 (70 ng/L)
- PFOA → 欧盟 REACH 限制
- PFOS → GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)

### protected_by
- 人类 → GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- 鱼类 → GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)
- 甲壳类 → GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)

---

## 四、毒性机制分析

基于知识图谱和文献数据，该化合物可能通过以下机制产生毒性：

1. **PPARα激活**：激活过氧化物酶体增殖物激活受体α，导致脂质代谢紊乱和肝肿大
2. **氧化应激**：诱导活性氧(ROS)产生，导致DNA氧化损伤和脂质过氧化
3. **线粒体功能障碍**：干扰线粒体电子传递链，影响细胞能量代谢
4. **内分泌干扰**：竞争性结合甲状腺激素转运蛋白，干扰激素正常代谢
5. **免疫抑制**：抑制免疫细胞功能，降低疫苗接种后的抗体反应

---

## 五、健康效应

根据毒理学研究和流行病学证据：

| 健康效应 | 证据等级 | 说明 |
|---------|---------|------|
| 肝毒性 | 强 | 血清PFAS水平与肝酶升高正相关 |
| 肾毒性 | 中 | 与慢性肾病风险增加有关 |
| 甲状腺功能异常 | 强 | 干扰甲状腺激素代谢 |
| 免疫功能抑制 | 强 | 降低疫苗抗体反应 |
| 发育毒性 | 强 | 与低出生体重、早产相关 |
| 生殖毒性 | 中 | 影响生育力 |
| 致癌性 | 中 | IARC列为2B类可能致癌物 |

---

## 六、环境归趋

| 环境介质 | 存在情况 | 持久性 |
|---------|---------|--------|
| 饮用水 | 广泛检出 | 极高 |
| 地表水 | 广泛检出 | 极高 |
| 地下水 | 检出 | 极高 |
| 土壤 | 检出 | 高 |
| 大气 | 前体物质传输 | 中 |
| 海洋 | 全球分布 | 极高 |
| 生物体 | 生物富集 | 极高 |

---

## 七、管控建议

### 7.1 国内法规标准

| 法规/标准 | 要求 |
|----------|------|
| 《新污染物治理行动方案》(2022) | 将PFAS列为重点管控新污染物 |
| GB 5749-2022 生活饮用水标准 | PFOS+PFOA ≤ 40 ng/L |
| GB 3838-2002 地表水标准 | 参照执行 |

### 7.2 国际法规标准

| 法规/标准 | 要求 |
|----------|------|
| 美国EPA (2023) | PFOA ≤ 4 ng/L, PFOS ≤ 4 ng/L |
| 欧盟REACH | 正在推进全面PFAS限制提案 |
| WHO (2022) | PFOA ≤ 100 ng/L, PFOS ≤ 40 ng/L |
| 斯德哥尔摩公约 | PFOS/PFOA/PFHxS列入POPs清单 |

### 7.3 建议措施

1. **源头控制**：减少不必要的PFAS使用，开发安全替代品
2. **过程管控**：加强生产过程中的PFAS排放控制
3. **末端治理**：采用高效水处理技术去除PFAS
4. **监测预警**：建立PFAS环境监测网络
5. **风险管理**：建立PFAS污染场地风险管控体系

---

## 八、参考文献

1. Sunderland E M, et al. A review of the pathways of human exposure to poly- and perfluoroalkyl substances (PFASs) and present understanding of health effects. *J Expo Sci Environ Epidemiol*, 2019.
2. Fenton S E, et al. Per- and Polyfluoroalkyl Substance Exposure and Fetal Growth: A Systematic Review. *Environ Health Perspect*, 2021.
3. 国务院办公厅. 新污染物治理行动方案. 2022.
4. GB 5749-2022 生活饮用水卫生标准. 2022.
5. US EPA. PFAS Strategic Roadmap: EPA's Commitments to Action 2021-2024. 2021.
6. ECHA. Restriction Proposal on PFAS. 2023.

---

*本报告由 QSAR-GNN + RAG PFAS风险评估一体化系统自动生成*
*模型: LR + SVM + RF + XGBoost + LightGBM + GBDT (6算法集成)*
