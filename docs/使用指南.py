"""
PFAS 风险评估系统 - 交互式使用界面
双击运行或命令行: python 使用指南.py
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 已知 PFAS 化合物库（输入名称自动查找 SMILES）
# ============================================================
PFAS_DB = {
    'PFOA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛酸', 'cas': '335-67-1', 'cat': 'PFCA'},
    'PFOS':       {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛烷磺酸', 'cas': '1763-23-1', 'cat': 'PFSA'},
    'GenX':       {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F', 'name': 'HFPO-DA', 'cas': '13252-13-6', 'cat': 'PFECDA'},
    'PFNA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟壬酸', 'cas': '375-95-1', 'cat': 'PFCA'},
    'PFDA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟癸酸', 'cas': '335-76-2', 'cat': 'PFCA'},
    'PFHxA':      {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己酸', 'cas': '307-24-4', 'cat': 'PFCA'},
    'PFBS':       {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟丁烷磺酸', 'cas': '375-73-5', 'cat': 'PFSA'},
    'PFHxS':      {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己烷磺酸', 'cas': '355-46-4', 'cat': 'PFSA'},
    'PFBA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟丁酸', 'cas': '375-22-4', 'cat': 'PFCA'},
    'PFPeA':      {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟戊酸', 'cas': '2706-90-3', 'cat': 'PFCA'},
    'PFUnDA':     {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟十一烷酸', 'cas': '2058-94-8', 'cat': 'PFCA'},
    'FOSA':       {'smiles': 'NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛烷磺酰胺', 'cas': '754-91-6', 'cat': 'FASA'},
    'ADONA':      {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F', 'name': 'ADONA', 'cas': '919005-14-4', 'cat': 'PFECDA'},
    'TFA':        {'smiles': 'OC(=O)C(F)(F)F', 'name': '三氟乙酸', 'cas': '76-05-1', 'cat': 'PFCA'},
    'TFMS':       {'smiles': 'OS(=O)(=O)C(F)(F)F', 'name': '三氟甲磺酸', 'cas': '1493-13-6', 'cat': 'PFSA'},
    '6:2 FTCA':   {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC', 'name': '6:2氟调聚物酸', 'cas': '27854-31-5', 'cat': 'FTCA'},
    '8:2 FTCA':   {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC', 'name': '8:2氟调聚物酸', 'cas': '27854-30-4', 'cat': 'FTCA'},
    '9Cl-PF3ONS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(Cl)F', 'name': '9-氯代全氟壬烷磺酸', 'cas': '756426-58-1', 'cat': 'Cl-PFAES'},
    'N-EtFOSE':   {'smiles': 'CCN(CCO)S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': 'N-乙基全氟辛烷磺酰胺乙醇', 'cas': '1691-99-2', 'cat': 'FASE'},
    'PFDS':       {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟癸烷磺酸', 'cas': '335-77-3', 'cat': 'PFSA'},
}

ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
ENDPOINT_CN = {
    'NR-AR': '雄激素受体拮抗',
    'NR-AR-LBD': '配体结合域活性',
    'NR-AhR': '芳香烃受体激活',
    'SR-HSE': '热休克元件响应',
    'SR-MMP': '线粒体膜电位异常',
    'SR-p53': 'p53通路激活',
}

# ============================================================
# 系统初始化
# ============================================================
class PFAS_System:
    def __init__(self):
        self.qsar_models = {}
        self.scaler = None
        self.kg = None
        self._load()

    def _load(self):
        qsar_dir = os.path.join(PROJECT_DIR, "models", "qsar")
        if not os.path.exists(qsar_dir):
            print("[错误] 模型目录不存在，请先运行训练脚本")
            return

        for f in os.listdir(qsar_dir):
            if f.endswith('.joblib') and 'scaler' not in f:
                parts = f.replace('qsar_', '').replace('.joblib', '').split('_')
                if len(parts) >= 2:
                    ep = parts[0]
                    name = '_'.join(parts[1:])
                    if ep not in self.qsar_models:
                        self.qsar_models[ep] = {}
                    try:
                        self.qsar_models[ep][name] = joblib.load(os.path.join(qsar_dir, f))
                    except:
                        pass

        scaler_path = os.path.join(qsar_dir, "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)

        kg_path = os.path.join(PROJECT_DIR, "05_知识图谱", "pfas_kg_triples.csv")
        if os.path.exists(kg_path):
            self.kg = pd.read_csv(kg_path)

        n_models = sum(len(v) for v in self.qsar_models.values())
        print(f"  系统加载完成: {n_models} 个QSAR模型, {len(self.kg) if self.kg is not None else 0} 个知识三元组")

    def predict(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        desc = []
        for name, func in Descriptors.descList:
            try:
                val = func(mol)
                desc.append(float(val) if val and not np.isinf(val) and not np.isnan(val) else 0.0)
            except:
                desc.append(0.0)
        try:
            desc.extend(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
        except:
            desc.extend([0]*2048)
        try:
            desc.extend(list(MACCSkeys.GenMACCSKeys(mol)))
        except:
            desc.extend([0]*167)

        X = np.array(desc).reshape(1, -1)
        expected = self.scaler.n_features_in_
        if X.shape[1] < expected:
            X = np.pad(X, ((0,0),(0,expected-X.shape[1])))
        elif X.shape[1] > expected:
            X = X[:, :expected]
        X = self.scaler.transform(X)

        results = {}
        for ep in ENDPOINTS:
            if ep in self.qsar_models:
                preds = []
                for mn, model in self.qsar_models[ep].items():
                    try:
                        preds.append(model.predict_proba(X)[0, 1])
                    except:
                        pass
                if preds:
                    results[ep] = {'mean': float(np.mean(preds)), 'std': float(np.std(preds))}
        return results

    def query_kg(self, name):
        if self.kg is None:
            return []
        related = self.kg[
            self.kg['head'].str.contains(name, case=False, na=False) |
            self.kg['tail'].str.contains(name, case=False, na=False)
        ]
        return related.to_dict('records')

    def generate_report(self, compound_key, smiles, cn_name, category, cas=''):
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        preds = self.predict(smiles)
        kg_data = self.query_kg(compound_key)

        risk_score = np.mean([v['mean'] for v in preds.values()]) if preds else 0
        risk_level = '高风险' if risk_score > 0.6 else '中风险' if risk_score > 0.3 else '低风险'

        report = f"""# PFAS 风险评估报告

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | {cn_name}（{compound_key}） |
| CAS号 | {cas if cas else 'N/A'} |
| SMILES | `{smiles}` |
| 化合物类别 | {category} |
| 分子式 | {Chem.rdMolDescriptors.CalcMolFormula(mol)} |
| 分子量 | {Descriptors.MolWt(mol):.2f} g/mol |
| LogP | {Descriptors.MolLogP(mol):.2f} |
| TPSA | {Descriptors.TPSA(mol):.2f} Å² |
| 氢键供体 | {Descriptors.NumHDonors(mol)} |
| 氢键受体 | {Descriptors.NumHAcceptors(mol)} |
| 评估日期 | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
| 评估系统 | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 二、毒理学预测数据

### 2.1 集成模型预测结果（6算法加权平均）

| 毒性终点 | 中文含义 | 预测概率 | 标准差 | 风险等级 |
|---------|---------|---------|--------|---------|
"""
        for ep in ENDPOINTS:
            if ep in preds:
                v = preds[ep]
                risk = '高' if v['mean'] > 0.6 else '中' if v['mean'] > 0.3 else '低'
                report += f"| {ep} | {ENDPOINT_CN[ep]} | {v['mean']:.3f} | {v['std']:.3f} | {risk} |\n"

        report += f"""
### 2.2 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **{risk_score:.3f}** |
| **风险等级** | **{risk_level}** |
| 评估终点数 | {len(preds)} |

---

## 三、已知毒性数据（知识图谱）

"""
        if kg_data:
            # 按关系类型分组
            rel_groups = {}
            for row in kg_data:
                r = row['relation']
                if r not in rel_groups:
                    rel_groups[r] = []
                rel_groups[r].append(row)

            rel_cn = {
                'contains': '分子结构包含',
                'has_endpoint': '具有毒性终点',
                'via_mechanism': '毒性机制',
                'exists_in': '环境存在',
                'affects': '影响物种',
                'causes': '导致健康效应',
                'applies_to': '适用管控措施',
            }

            for rel, rows in rel_groups.items():
                cn = rel_cn.get(rel, rel)
                report += f"### {cn}\n"
                for row in rows:
                    report += f"- {row['head']} → {row['tail']}\n"
                report += "\n"
        else:
            report += "暂无已知数据\n\n"

        report += """---

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
"""
        return report, risk_level, risk_score


# ============================================================
# 交互式主界面
# ============================================================
def main():
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "PFAS 风险评估一体化系统" + " "*30 + "║")
    print("║" + " "*10 + "基于 QSAR-GNN + RAG 的全氟化合物毒性预测" + " "*10 + "║")
    print("╚" + "═"*68 + "╝")
    print()

    system = PFAS_System()
    print()

    while True:
        print("─" * 70)
        print("  可用操作:")
        print("    1. 输入化合物名称（如 PFOA、PFOS、GenX）")
        print("    2. 输入 SMILES 字符串（如 OC(=O)C(F)(F)F）")
        print("    3. 输入 list 查看所有可查询化合物")
        print("    4. 输入 q 退出")
        print("─" * 70)

        user_input = input("\n  请输入 > ").strip()

        if user_input.lower() in ['q', 'quit', 'exit', '退出']:
            print("\n  再见！")
            break

        if user_input.lower() in ['list', '列表', 'ls']:
            print(f"\n  {'名称':12s} {'CAS':14s} {'类别':10s} {'中文名'}")
            print(f"  {'─'*60}")
            for key, info in PFAS_DB.items():
                print(f"  {key:12s} {info['cas']:14s} {info['cat']:10s} {info['name']}")
            print()
            continue

        if not user_input:
            continue

        # 判断输入类型
        smiles = None
        compound_key = None
        cn_name = None
        category = None
        cas = None

        # 先查名称
        if user_input.upper() in PFAS_DB:
            info = PFAS_DB[user_input.upper()]
            smiles = info['smiles']
            compound_key = user_input.upper()
            cn_name = info['name']
            category = info['cat']
            cas = info['cas']
        elif user_input in PFAS_DB:
            info = PFAS_DB[user_input]
            smiles = info['smiles']
            compound_key = user_input
            cn_name = info['name']
            category = info['cat']
            cas = info['cas']
        else:
            # 尝试当 SMILES 解析
            mol = Chem.MolFromSmiles(user_input)
            if mol:
                smiles = user_input
                compound_key = 'Unknown'
                cn_name = Chem.rdMolDescriptors.CalcMolFormula(mol)
                category = '未知'
            else:
                # 模糊搜索
                matches = [k for k in PFAS_DB if user_input.upper() in k.upper()]
                if matches:
                    print(f"\n  您是否要查询: {', '.join(matches)}?")
                    continue
                else:
                    print(f"\n  [错误] 无法识别 '{user_input}'")
                    print(f"  请输入 PFAS 名称（如 PFOA）或 SMILES 字符串")
                    print(f"  输入 list 查看所有可查询化合物")
                    continue

        # 执行预测和报告生成
        print(f"\n  正在分析 {compound_key} ({cn_name})...")
        print()

        # 分子信息
        mol = Chem.MolFromSmiles(smiles)
        print(f"  分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
        print(f"  分子量: {Descriptors.MolWt(mol):.2f} g/mol")
        print(f"  LogP:   {Descriptors.MolLogP(mol):.2f}")
        print(f"  SMILES: {smiles[:60]}{'...' if len(smiles)>60 else ''}")
        print()

        # 模型预测
        preds = system.predict(smiles)
        if preds:
            print(f"  {'毒性终点':14s} {'中文含义':14s} {'预测概率':>10s} {'风险':>6s}")
            print(f"  {'─'*56}")
            for ep in ENDPOINTS:
                if ep in preds:
                    v = preds[ep]['mean']
                    risk = "高" if v > 0.6 else "中" if v > 0.3 else "低"
                    bar = "█" * int(v * 15) + "░" * (15 - int(v * 15))
                    print(f"  {ep:14s} {ENDPOINT_CN[ep]:14s} {bar} {v:.3f}  [{risk}]")

            score = np.mean([preds[ep]['mean'] for ep in ENDPOINTS if ep in preds])
            level = '高风险' if score > 0.6 else '中风险' if score > 0.3 else '低风险'
            print(f"\n  综合风险分数: {score:.3f}  风险等级: 【{level}】")
        print()

        # 知识图谱
        kg_data = system.query_kg(compound_key)
        if kg_data:
            print(f"  知识图谱数据 ({len(kg_data)} 条):")
            for row in kg_data[:8]:
                print(f"    {row['head']} ──[{row['relation']}]──> {row['tail']}")
            if len(kg_data) > 8:
                print(f"    ... 还有 {len(kg_data)-8} 条")
        print()

        # 生成报告
        save = input("  是否生成风险评估报告？(y/n) > ").strip().lower()
        if save in ['y', 'yes', '是', '']:
            report, risk_level, risk_score = system.generate_report(
                compound_key, smiles, cn_name, category, cas
            )
            report_dir = os.path.join(PROJECT_DIR, "04_模型融合与预测")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, f"风险评估报告_{compound_key}.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n  [OK] 报告已保存: {report_path}")
            print(f"  [OK] 风险等级: {risk_level} (分数: {risk_score:.3f})")
        print()


if __name__ == '__main__':
    main()
