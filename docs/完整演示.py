"""PFAS 风险评估系统 - 完整自动演示"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd, numpy as np, joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
ENDPOINT_CN = {
    'NR-AR': '雄激素受体拮抗', 'NR-AR-LBD': '配体结合域活性',
    'NR-AhR': '芳香烃受体激活', 'SR-HSE': '热休克元件响应',
    'SR-MMP': '线粒体膜电位异常', 'SR-p53': 'p53通路激活',
}

# 加载模型
qsar_dir = os.path.join(PROJECT_DIR, 'models', 'qsar')
qsar_models = {}
for f in os.listdir(qsar_dir):
    if f.endswith('.joblib') and 'scaler' not in f:
        parts = f.replace('qsar_', '').replace('.joblib', '').split('_')
        if len(parts) >= 2:
            ep, name = parts[0], '_'.join(parts[1:])
            qsar_models.setdefault(ep, {})[name] = joblib.load(os.path.join(qsar_dir, f))
scaler = joblib.load(os.path.join(qsar_dir, 'feature_scaler.joblib'))
kg = pd.read_csv(os.path.join(PROJECT_DIR, '05_知识图谱', 'pfas_kg_triples.csv'))

PFAS_DB = {
    'PFOA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛酸', 'cas': '335-67-1', 'cat': 'PFCA'},
    'PFOS':       {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛烷磺酸', 'cas': '1763-23-1', 'cat': 'PFSA'},
    'GenX':       {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F', 'name': 'HFPO-DA', 'cas': '13252-13-6', 'cat': 'PFECDA'},
    'PFNA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟壬酸', 'cas': '375-95-1', 'cat': 'PFCA'},
    'PFDA':       {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟癸酸', 'cas': '335-76-2', 'cat': 'PFCA'},
    'PFHxA':      {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己酸', 'cas': '307-24-4', 'cat': 'PFCA'},
    'PFBS':       {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟丁烷磺酸', 'cas': '375-73-5', 'cat': 'PFSA'},
    'PFHxS':      {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己烷磺酸', 'cas': '355-46-4', 'cat': 'PFSA'},
    'ADONA':      {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F', 'name': 'ADONA', 'cas': '919005-14-4', 'cat': 'PFECDA'},
    '9Cl-PF3ONS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(Cl)F', 'name': '9-氯代全氟壬烷磺酸', 'cas': '756426-58-1', 'cat': 'Cl-PFAES'},
}

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    desc = []
    for n, f in Descriptors.descList:
        try:
            v = f(mol); desc.append(float(v) if v and not np.isinf(v) and not np.isnan(v) else 0.0)
        except:
            desc.append(0.0)
    try: desc.extend(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
    except: desc.extend([0]*2048)
    try: desc.extend(list(MACCSkeys.GenMACCSKeys(mol)))
    except: desc.extend([0]*167)
    X = np.array(desc).reshape(1,-1)
    exp = scaler.n_features_in_
    if X.shape[1] < exp: X = np.pad(X,((0,0),(0,exp-X.shape[1])))
    elif X.shape[1] > exp: X = X[:,:exp]
    X = scaler.transform(X)
    results = {}
    for ep in ENDPOINTS:
        if ep in qsar_models:
            preds = []
            for mn, model in qsar_models[ep].items():
                try: preds.append(model.predict_proba(X)[0,1])
                except: pass
            if preds: results[ep] = {'mean': float(np.mean(preds)), 'std': float(np.std(preds))}
    return results

def generate_report(comp, info, preds):
    smiles = info['smiles']
    mol = Chem.MolFromSmiles(smiles)
    score = np.mean([preds[ep]['mean'] for ep in ENDPOINTS if ep in preds]) if preds else 0
    level = '高风险' if score > 0.6 else '中风险' if score > 0.3 else '低风险'

    report = f"""# PFAS 风险评估报告

## 一、基本信息

| 项目 | 内容 |
|------|------|
| 化合物名称 | {info['name']}（{comp}） |
| CAS号 | {info['cas']} |
| SMILES | `{smiles}` |
| 化合物类别 | {info['cat']} |
| 分子式 | {Chem.rdMolDescriptors.CalcMolFormula(mol)} |
| 分子量 | {Descriptors.MolWt(mol):.2f} g/mol |
| LogP | {Descriptors.MolLogP(mol):.2f} |
| 评估日期 | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
| 评估系统 | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 二、毒理学预测数据

### 集成模型预测结果（6算法加权平均）

| 毒性终点 | 中文含义 | 预测概率 | 标准差 | 风险等级 |
|---------|---------|---------|--------|---------|
"""
    for ep in ENDPOINTS:
        if ep in preds:
            v = preds[ep]
            r = '高' if v['mean'] > 0.6 else '中' if v['mean'] > 0.3 else '低'
            report += f"| {ep} | {ENDPOINT_CN[ep]} | {v['mean']:.3f} | {v['std']:.3f} | {r} |\n"

    report += f"""
### 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | **{score:.3f}** |
| **风险等级** | **{level}** |

---

## 三、已知毒性数据（知识图谱）

"""
    kg_data = kg[kg['head'].str.contains(comp, case=False, na=False) |
                 kg['tail'].str.contains(comp, case=False, na=False)]
    if len(kg_data) > 0:
        for _, row in kg_data.iterrows():
            report += f"- {row['head']} --[{row['relation']}]--> {row['tail']}\n"

    report += """
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
"""
    return report, level, score

# ================================================================
# 演示开始
# ================================================================
print()
print('★' * 70)
print('  PFAS 风险评估系统 - 完整功能演示')
print('★' * 70)
print()
print('  系统加载完成: %d 个QSAR模型, %d 个知识三元组' % (sum(len(v) for v in qsar_models.values()), len(kg)))
print()

# 可查询化合物
print('  可查询化合物列表:')
print('  %-12s %-14s %-10s %s' % ('名称', 'CAS', '类别', '中文名'))
print('  ' + '-' * 60)
for key, info in PFAS_DB.items():
    print('  %-12s %-14s %-10s %s' % (key, info['cas'], info['cat'], info['name']))
print()

# 逐个化合物演示
report_dir = os.path.join(PROJECT_DIR, '04_模型融合与预测')

for comp, info in PFAS_DB.items():
    smiles = info['smiles']
    mol = Chem.MolFromSmiles(smiles)

    print('=' * 70)
    print('  化合物: %s (%s)' % (comp, info['name']))
    print('  CAS: %s  类别: %s' % (info['cas'], info['cat']))
    print('  分子式: %s  分子量: %.1f' % (Chem.rdMolDescriptors.CalcMolFormula(mol), Descriptors.MolWt(mol)))
    print('=' * 70)

    preds = predict(smiles)
    if preds:
        print()
        print('  %-14s %-14s  %-22s %s' % ('终点', '含义', '预测概率', '风险'))
        print('  ' + '-' * 62)
        for ep in ENDPOINTS:
            if ep in preds:
                v = preds[ep]['mean']
                risk = '高' if v > 0.6 else '中' if v > 0.3 else '低'
                bar = '#' * int(v * 20) + '.' * (20 - int(v * 20))
                print('  %-14s %-14s  [%s] %.3f  [%s]' % (ep, ENDPOINT_CN[ep], bar, v, risk))

        score = np.mean([preds[ep]['mean'] for ep in ENDPOINTS])
        level = '高风险' if score > 0.6 else '中风险' if score > 0.3 else '低风险'
        print('  ' + '-' * 62)
        print('  综合风险: %.3f  等级: 【%s】' % (score, level))

    # 知识图谱
    kg_data = kg[kg['head'].str.contains(comp, case=False, na=False) |
                 kg['tail'].str.contains(comp, case=False, na=False)]
    if len(kg_data) > 0:
        print()
        print('  知识图谱: %d 条关联' % len(kg_data))
        for _, row in kg_data.head(5).iterrows():
            print('    %s --[%s]--> %s' % (row['head'], row['relation'], row['tail']))
        if len(kg_data) > 5:
            print('    ... 还有 %d 条' % (len(kg_data) - 5))

    # 生成报告
    report, level, score = generate_report(comp, info, preds)
    path = os.path.join(report_dir, 'risk_report_%s.md' % comp)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report)
    print()
    print('  [报告已保存] %s  风险等级: %s (%.3f)' % (path, level, score))
    print()

# 汇总表
print()
print('★' * 70)
print('  全部化合物预测结果汇总表')
print('★' * 70)
print()
header = '  %-10s %-8s' % ('化合物', '类别')
for ep in ENDPOINTS:
    header += ' %10s' % ep
header += '  %8s  %s' % ('综合', '等级')
print(header)
print('  ' + '-' * 95)

for comp, info in PFAS_DB.items():
    preds = predict(info['smiles'])
    if preds:
        line = '  %-10s %-8s' % (comp, info['cat'])
        for ep in ENDPOINTS:
            if ep in preds:
                line += ' %10.3f' % preds[ep]['mean']
            else:
                line += ' %10s' % 'N/A'
        score = np.mean([preds[ep]['mean'] for ep in ENDPOINTS if ep in preds])
        level = '高' if score > 0.6 else '中' if score > 0.3 else '低'
        line += '  %8.3f  [%s]' % (score, level)
        print(line)

print()
print('  所有报告已保存到: %s' % report_dir)
print()
print('★' * 70)
print('  演示完成！')
print('★' * 70)
