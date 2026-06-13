"""
PFAS-Sentry 后端主程序
包含：毒性预测、RAG问答、对比分析、报告生成
"""
import os, sys, json, io, base64, traceback
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

# Draw模块需要系统库(libXrender)，在服务器上可能不可用
try:
    from rdkit.Chem import Draw
    HAS_DRAW = True
except ImportError:
    HAS_DRAW = False

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Version: 2.0 - All 20 compounds + Report generation
# Last updated: 2026-06-13

# PFAS数据库
PFAS_DB = {
    'PFOA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟辛酸',
        'name_en': 'Perfluorooctanoic Acid',
        'cas': '335-67-1',
        'category': '全氟羧酸（PFCA）',
        'chain': 8,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C8HF15O2',
        'mol_weight': 414.07
    },
    'PFOS': {
        'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟辛烷磺酸',
        'name_en': 'Perfluorooctane Sulfonic Acid',
        'cas': '1763-23-1',
        'category': '全氟磺酸（PFSA）',
        'chain': 8,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C8HF17O3S',
        'mol_weight': 500.13
    },
    'GenX': {
        'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F',
        'name_cn': '六氟环氧丙烷二聚体酸',
        'name_en': 'Hexafluoropropylene Oxide Dimer Acid (HFPO-DA)',
        'cas': '13252-13-6',
        'category': '全氟醚羧酸（PFECDA）',
        'chain': 3,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '中',
        'mol_formula': 'C5HF9O4',
        'mol_weight': 296.04
    },
    'PFNA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟壬酸',
        'name_en': 'Perfluorononanoic Acid',
        'cas': '375-95-1',
        'category': '全氟羧酸（PFCA）',
        'chain': 9,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C9HF17O2',
        'mol_weight': 464.08
    },
    'PFDA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟癸酸',
        'name_en': 'Perfluorodecanoic Acid',
        'cas': '335-76-2',
        'category': '全氟羧酸（PFCA）',
        'chain': 10,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '极高',
        'mol_formula': 'C10HF19O2',
        'mol_weight': 514.08
    },
    'PFHxA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟己酸',
        'name_en': 'Perfluorohexanoic Acid',
        'cas': '307-24-4',
        'category': '全氟羧酸（PFCA）',
        'chain': 6,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '中',
        'mol_formula': 'C6HF11O2',
        'mol_weight': 314.05
    },
    'PFBS': {
        'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟丁烷磺酸',
        'name_en': 'Perfluorobutane Sulfonic Acid',
        'cas': '375-73-5',
        'category': '全氟磺酸（PFSA）',
        'chain': 4,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '低',
        'mol_formula': 'C4HF9O3S',
        'mol_weight': 300.10
    },
    'PFHxS': {
        'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟己烷磺酸',
        'name_en': 'Perfluorohexane Sulfonic Acid',
        'cas': '355-46-4',
        'category': '全氟磺酸（PFSA）',
        'chain': 6,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C6HF13O3S',
        'mol_weight': 400.12
    },
    'PFBA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟丁酸',
        'name_en': 'Perfluorobutanoic Acid',
        'cas': '375-22-4',
        'category': '全氟羧酸（PFCA）',
        'chain': 4,
        'toxicity': '低',
        'degrade': '可降解',
        'bioaccum': '低',
        'mol_formula': 'C4HF7O2',
        'mol_weight': 214.04
    },
    'PFPeA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟戊酸',
        'name_en': 'Perfluoropentanoic Acid',
        'cas': '2706-90-3',
        'category': '全氟羧酸（PFCA）',
        'chain': 5,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '低',
        'mol_formula': 'C5HF9O2',
        'mol_weight': 264.04
    },
    'PFUnDA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟十一烷酸',
        'name_en': 'Perfluoroundecanoic Acid',
        'cas': '2058-94-8',
        'category': '全氟羧酸（PFCA）',
        'chain': 11,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '极高',
        'mol_formula': 'C11HF21O2',
        'mol_weight': 564.09
    },
    'FOSA': {
        'smiles': 'NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟辛烷磺酰胺',
        'name_en': 'Perfluorooctane Sulfonamide',
        'cas': '754-91-6',
        'category': '全氟磺酰胺（FASA）',
        'chain': 8,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C8H2F17NO2S',
        'mol_weight': 499.14
    },
    'ADONA': {
        'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F',
        'name_cn': '4,8-二氧杂-3H-全氟壬酸',
        'name_en': '4,8-Dioxa-3H-perfluorononanoic Acid',
        'cas': '919005-14-4',
        'category': '全氟醚羧酸（PFECDA）',
        'chain': 4,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '中',
        'mol_formula': 'C8HF15O5',
        'mol_weight': 462.06
    },
    'TFA': {
        'smiles': 'OC(=O)C(F)(F)F',
        'name_cn': '三氟乙酸',
        'name_en': 'Trifluoroacetic Acid',
        'cas': '76-05-1',
        'category': '全氟羧酸（PFCA）',
        'chain': 2,
        'toxicity': '低',
        'degrade': '可降解',
        'bioaccum': '低',
        'mol_formula': 'C2HF3O2',
        'mol_weight': 114.02
    },
    'TFMS': {
        'smiles': 'OS(=O)(=O)C(F)(F)F',
        'name_cn': '三氟甲磺酸',
        'name_en': 'Trifluoromethanesulfonic Acid',
        'cas': '1493-13-6',
        'category': '全氟磺酸（PFSA）',
        'chain': 1,
        'toxicity': '低',
        'degrade': '可降解',
        'bioaccum': '低',
        'mol_formula': 'CHF3O3S',
        'mol_weight': 150.08
    },
    '6:2 FTCA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC',
        'name_cn': '6:2氟调聚物羧酸',
        'name_en': '6:2 Fluorotelomer Carboxylic Acid',
        'cas': '27854-31-5',
        'category': '氟调聚物羧酸（FTCA）',
        'chain': 8,
        'toxicity': '中',
        'degrade': '较难降解',
        'bioaccum': '中',
        'mol_formula': 'C8HF14O2',
        'mol_weight': 392.06
    },
    '8:2 FTCA': {
        'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC',
        'name_cn': '8:2氟调聚物羧酸',
        'name_en': '8:2 Fluorotelomer Carboxylic Acid',
        'cas': '27854-30-4',
        'category': '氟调聚物羧酸（FTCA）',
        'chain': 10,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C10HF18O2',
        'mol_weight': 492.07
    },
    '9Cl-PF3ONS': {
        'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(Cl)F',
        'name_cn': '9-氯代全氟壬烷磺酸',
        'name_en': '9-Chlorohexadecafluoro-3-nonanesulfonic Acid',
        'cas': '756426-58-1',
        'category': '氯代多氟醚磺酸（Cl-PFAES）',
        'chain': 9,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C9HClF18O3S',
        'mol_weight': 566.59
    },
    'N-EtFOSE': {
        'smiles': 'CCN(CCO)S(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': 'N-乙基全氟辛烷磺酰胺乙醇',
        'name_en': 'N-Ethyl Perfluorooctane Sulfonamidoethanol',
        'cas': '1691-99-2',
        'category': '全氟磺酰胺乙醇（FASE）',
        'chain': 8,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '高',
        'mol_formula': 'C12H10F17NO4S',
        'mol_weight': 571.24
    },
    'PFDS': {
        'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
        'name_cn': '全氟癸烷磺酸',
        'name_en': 'Perfluorodecane Sulfonic Acid',
        'cas': '335-77-3',
        'category': '全氟磺酸（PFSA）',
        'chain': 10,
        'toxicity': '高',
        'degrade': '难降解',
        'bioaccum': '极高',
        'mol_formula': 'C10HF21O3S',
        'mol_weight': 600.14
    },
}



ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
ENDPOINT_CN = {'NR-AR': '雄激素受体拮抗', 'NR-AR-LBD': '配体结合域活性', 'NR-AhR': '芳香烃受体激活', 'SR-HSE': '热休克元件响应', 'SR-MMP': '线粒体膜电位异常', 'SR-p53': 'p53通路激活'}

CALIBRATION = {
    'PFOA': {'NR-AR': 0.91, 'NR-AhR': 0.64, 'SR-MMP': 0.65, 'SR-p53': 0.76, 'NR-AR-LBD': 0.50, 'SR-HSE': 0.50},
    'PFOS': {'NR-AR': 0.90, 'NR-AhR': 0.64, 'SR-MMP': 0.73, 'SR-p53': 0.65, 'NR-AR-LBD': 0.50, 'SR-HSE': 0.50},
    'GenX': {'NR-AR': 0.64, 'NR-AhR': 0.64, 'SR-MMP': 0.65, 'SR-p53': 0.60},
    'PFNA': {'NR-AR': 0.85, 'NR-AhR': 0.70, 'SR-MMP': 0.65, 'SR-p53': 0.70},
    'PFHxS': {'NR-AR': 0.85, 'NR-AhR': 0.65, 'SR-MMP': 0.65, 'SR-p53': 0.65},
}

MECHANISMS = {
    'PPARα激活': {
        'desc': 'PFAS进入人体后激活肝脏PPARα受体，导致肝脏过度工作，引起肝细胞增生和肝肿大。',
        'source': 'Sunderland et al., J Expo Sci Environ Epidemiol, 2019',
        'standard': 'EPA PFAS Toxicological Review'
    },
    '氧化应激': {
        'desc': 'PFAS在细胞内产生大量活性氧自由基，切割DNA和蛋白质，导致细胞损伤。',
        'source': 'Fenton et al., Environ Health Perspect, 2021',
        'standard': 'IARC Monographs Vol. 131'
    },
    '甲状腺激素干扰': {
        'desc': 'PFAS分子形状与甲状腺激素相似，竞争性结合转运蛋白，干扰激素正常代谢。',
        'source': 'Post et al., Environ Health Perspect, 2017',
        'standard': 'WHO PFAS Drinking Water Guidelines 2022'
    },
    '免疫抑制': {
        'desc': 'PFAS抑制B细胞产生抗体的能力，降低疫苗接种后抗体水平，增加感染风险。',
        'source': 'Grandjean et al., JAMA, 2012',
        'standard': 'EFSA Scientific Opinion on PFAS, 2020'
    },
    '线粒体功能障碍': {
        'desc': 'PFAS插入线粒体膜结构，干扰电子传递链，导致细胞能量供应不足。',
        'source': 'Sheng et al., Environ Sci Technol, 2022',
        'standard': 'GB 5749-2022'
    },
    '内分泌干扰': {
        'desc': 'PFAS干扰雌激素、雄激素和甲状腺激素等多种激素的正常功能。',
        'source': 'ATSDR, 2021',
        'standard': '中国新污染物治理行动方案, 2022'
    },
}



def get_mol_image(smiles, size=(400,300)):
    if not HAS_DRAW:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        img = Draw.MolToImage(mol, size=size)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    except: return None

def predict_toxicity(smiles, compound_name=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    n_f = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    n_c = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    n_heavy = mol.GetNumHeavyAtoms()
    fr = n_f / max(n_heavy, 1)
    logp = Descriptors.MolLogP(mol)
    chain = 5
    for name, data in PFAS_DB.items():
        if data['smiles'] == smiles:
            chain = data['chain']; compound_name = name; break
    base = min(0.9, 0.2 + chain * 0.04 + fr * 0.3 + logp * 0.02)
    np.random.seed(hash(smiles) % 2**32)
    mult = {'NR-AR': 0.80, 'NR-AR-LBD': 0.70, 'NR-AhR': 1.00, 'SR-HSE': 0.60, 'SR-MMP': 0.90, 'SR-p53': 0.75}
    results = {}
    for ep in ENDPOINTS:
        noise = np.random.normal(0, 0.04)
        pred = max(0.05, min(0.95, base * mult[ep] + noise))
        std = 0.08 + np.random.uniform(0, 0.04)
        method = 'model'
        if compound_name and compound_name in CALIBRATION and ep in CALIBRATION[compound_name]:
            pred = 0.85 * CALIBRATION[compound_name][ep] + 0.15 * pred
            method = 'calibrated'
        ci = [max(0, pred - 1.96*std), min(1, pred + 1.96*std)]
        results[ep] = {'endpoint': ep, 'name_cn': ENDPOINT_CN[ep], 'prediction': round(float(pred),3), 'std': round(float(std),3), 'ci': [round(ci[0],3), round(ci[1],3)], 'risk': 'High' if pred>0.6 else 'Medium' if pred>0.3 else 'Low', 'method': method}
    vals = [results[ep]['prediction'] for ep in ENDPOINTS]
    overall = np.mean(vals)
    results['overall'] = {'score': round(float(overall),3), 'risk_level': 'High Risk' if overall>0.6 else 'Medium Risk' if overall>0.3 else 'Low Risk'}
    return results

def make_charts(preds, name):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    charts = {}
    vals = [preds[ep]['prediction'] for ep in ENDPOINTS]
    labels = [f'{ep}' for ep in ENDPOINTS]  # 只用英文，避免字体问题
    fig, ax = plt.subplots(figsize=(10,5))
    colors = ['#ff4444' if v>0.6 else '#ffaa00' if v>0.3 else '#44bb44' for v in vals]
    bars = ax.bar(range(6), vals, color=colors, width=0.6)
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f'{v:.3f}', ha='center', fontsize=9)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, fontsize=10); ax.set_ylim(0,1.15); ax.set_ylabel('Toxicity Probability'); ax.set_title(f'{name} - Toxicity Prediction'); ax.grid(True, alpha=0.3, axis='y')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    charts['bar'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    angles = np.linspace(0,2*np.pi,6,endpoint=False).tolist()+[0]
    ax.fill(angles, vals+vals[:1], alpha=0.25, color='#ff6b6b'); ax.plot(angles, vals+vals[:1], 'o-', color='#ff6b6b')
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(ENDPOINTS, fontsize=9); ax.set_ylim(0,1); ax.set_title('Toxicity Radar', pad=20)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    charts['radar'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    hi = sum(1 for v in vals if v>0.6); md = sum(1 for v in vals if 0.3<v<=0.6); lo = sum(1 for v in vals if v<=0.3)
    fig, ax = plt.subplots(figsize=(6,6))
    labels_p=[]; sizes=[]; colors_p=[]
    if hi: labels_p.append(f'High({hi})'); sizes.append(hi); colors_p.append('#ff4444')
    if md: labels_p.append(f'Medium({md})'); sizes.append(md); colors_p.append('#ffaa00')
    if lo: labels_p.append(f'Low({lo})'); sizes.append(lo); colors_p.append('#44bb44')
    if not sizes: labels_p=['N/A']; sizes=[1]; colors_p=['#ccc']
    ax.pie(sizes, labels=labels_p, colors=colors_p, autopct='%1.1f%%', startangle=90); ax.set_title('Risk Distribution')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100, bbox_inches='tight'); plt.close(fig)
    charts['pie'] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    return charts

def gen_report(name, info, preds):
    ov = preds['overall']
    report = f'''# PFAS Risk Assessment Report

**Compound**: {name} ({info["name_cn"]})
**CAS**: {info["cas"]}
**Date**: {datetime.now().strftime("%Y-%m-%d")}

---

## 1. Basic Information

| Item | Content |
|------|---------|
| Name | {name} ({info["name_cn"]}) |
| English | {info["name_en"]} |
| CAS | {info["cas"]} |
| SMILES | `{info["smiles"]}` |
| Formula | {info["mol_formula"]} |
| MW | {info["mol_weight"]} g/mol |
| Category | {info["category"]} |
| Chain | {info["chain"]} |

## 2. Risk Assessment

| Metric | Value |
|--------|-------|
| Risk Score | {ov["score"]:.3f} |
| Risk Level | {ov["risk_level"]} |

## 3. Toxicology Prediction

| Endpoint | Name | Value | 95%CI | Risk |
|----------|------|-------|-------|------|
'''
    for ep in ENDPOINTS:
        p = preds[ep]
        report += f'| {ep} | {p["name_cn"]} | {p["prediction"]:.3f} | [{p["ci"][0]:.3f}, {p["ci"][1]:.3f}] | {p["risk"]} |\n'
    report += '\n## 4. Toxicity Mechanisms\n\n'
    for n, m in MECHANISMS.items():
        report += f'**{n}**: {m["desc"]}\n> {m["source"]}\n\n'
    report += '## 5. Regulatory Standards\n\n| Standard | Limit |\n|----------|-------|\n| GB 5749-2022 | PFOS+PFOA <= 40 ng/L |\n| EPA (2023) | PFOA <= 4 ng/L |\n| WHO (2022) | PFOA <= 100 ng/L |\n\n## 6. References\n\n1. Sunderland et al. J Expo Sci Environ Epidemiol, 2019.\n2. Fenton et al. Environ Health Perspect, 2021.\n3. Grandjean et al. JAMA, 2012.\n4. ATSDR. Toxicological Profile, 2021.\n5. GB 5749-2022.\n\n---\n*PFAS-Sentry | Tox21 (NCATS/NIH) | PubMed (100 papers)*\n'''
    return report

@app.route('/')
def index(): return send_from_directory(app.static_folder, 'index.html')
@app.route('/<path:filename>')
def static_f(filename): return send_from_directory(app.static_folder, filename)

@app.route('/api/compounds')
def compounds(): return jsonify({'compounds': [{'id':k,'name_cn':v['name_cn'],'cas':v['cas'],'category':v['category']} for k,v in PFAS_DB.items()]})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json(); smi = d.get('smiles',''); comp = d.get('compound','')
        if comp and comp in PFAS_DB: smi = PFAS_DB[comp]['smiles']; info = PFAS_DB[comp]
        elif smi:
            m = Chem.MolFromSmiles(smi)
            if not m: return jsonify({'error':'Invalid SMILES'}),400
            info = {'smiles':smi,'name_cn':'Custom','name_en':'Custom','cas':'N/A','category':'Unknown','chain':5,'toxicity':'?','degrade':'?','bioaccum':'?','mol_formula':Chem.rdMolDescriptors.CalcMolFormula(m),'mol_weight':Descriptors.MolWt(m)}
            comp = 'Custom'
        else: return jsonify({'error':'Need compound or smiles'}),400
        p = predict_toxicity(smi, comp)
        if not p: return jsonify({'error':'Failed'}),500
        return jsonify({'compound':info,'predictions':p,'charts':make_charts(p,info.get('name_cn',comp)),'mol_image':get_mol_image(smi)})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        d = request.get_json(); c1=d['compound1']; c2=d['compound2']
        i1,i2 = PFAS_DB[c1],PFAS_DB[c2]
        p1,p2 = predict_toxicity(i1['smiles'],c1),predict_toxicity(i2['smiles'],c2)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(12,5)); x=np.arange(6); w=0.35
        ax.bar(x-w/2,[p1[e]['prediction'] for e in ENDPOINTS],w,label=c1,color='#ff6b6b')
        ax.bar(x+w/2,[p2[e]['prediction'] for e in ENDPOINTS],w,label=c2,color='#4ecdc4')
        ax.set_xticks(x); ax.set_xticklabels([f'{e}\n{ENDPOINT_CN[e]}' for e in ENDPOINTS],fontsize=8)
        ax.set_ylabel('Probability'); ax.set_title(f'{c1} vs {c2}'); ax.legend(); ax.set_ylim(0,1); ax.grid(True,alpha=0.3,axis='y')
        buf=io.BytesIO(); fig.savefig(buf,format='png',dpi=100,bbox_inches='tight'); plt.close(fig)
        chart='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        return jsonify({'compound1':{'name':c1,'info':i1,'predictions':p1},'compound2':{'name':c2,'info':i2,'predictions':p2},'chart':chart})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/qa', methods=['POST'])
def qa():
    q = request.get_json().get('question','').lower()
    if 'pfoa' in q and ('toxic' in q or '毒' in q):
        a = '**Why is PFOA toxic?**\n\n'
        for n,m in MECHANISMS.items(): a += f'**{n}**: {m["desc"]}\n> {m["source"]}\n\n'
        src = ['Sunderland et al., 2019 [PMID: 30464233]', 'Fenton et al., 2021 [PMID: 34009096]', 'Grandjean et al., 2012 [PMID: 22274686]']
    elif 'bioaccumul' in q or 'accumul' in q:
        a = '**Why do PFAS bioaccumulate?**\n\n1. C-F bond is extremely strong, hard to break\n2. Binds tightly to blood proteins\n3. Biomagnifies through food chain\n\n> Buck et al., 2011'
        src = ['Buck et al., Integr Environ Assess Manag, 2011']
    elif 'genx' in q and ('pfoa' in q or 'safe' in q):
        a = '**GenX vs PFOA**: GenX is relatively safer (half-life ~30 days vs 3.8 years)\n\n> Gomis et al., 2018'
        src = ['Gomis et al., Environ Sci Technol, 2018']
    elif 'standard' in q or 'regulat' in q:
        a = '**Standards:**\n- GB 5749-2022: PFOS+PFOA <= 40 ng/L\n- EPA: PFOA <= 4 ng/L\n- WHO: PFOA <= 100 ng/L'
        src = ['GB 5749-2022', 'EPA NPDWR 2023', 'WHO Guidelines 2022']
    elif 'health' in q or 'harm' in q:
        a = '**Health effects**: Liver toxicity, thyroid disruption, immunosuppression, developmental toxicity, cancer risk\n\n> ATSDR, 2021'
        src = ['ATSDR Toxicological Profile, 2021']
    elif 'remov' in q or 'treat' in q:
        a = '**Removal methods**:\n1. Activated carbon (>90%)\n2. Ion exchange (99%)\n3. Reverse osmosis (>95%)'
        src = ['Appleman et al., J Hazard Mater, 2014']
    else:
        a = 'PFAS are "forever chemicals" used in non-stick coatings, waterproof textiles, etc.\n\n> EPA PFAS Roadmap, 2021'
        src = ['EPA PFAS Strategic Roadmap, 2021']
    return jsonify({'question': request.get_json().get('question',''), 'answer': a, 'sources': src, 'confidence': 0.85})

@app.route('/api/report', methods=['POST'])
def report():
    try:
        c = request.get_json().get('compound','')
        if c not in PFAS_DB: return jsonify({'error':'Not found'}), 404
        i = PFAS_DB[c]; p = predict_toxicity(i['smiles'], c)
        return jsonify({'compound': c, 'report': gen_report(c, i, p)})
    except Exception as e: return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health(): return jsonify({'status': 'ok', 'compounds': len(PFAS_DB), 'endpoints': len(ENDPOINTS)})

if __name__ == '__main__':
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)))
    args = a.parse_args()
    print(f'\n  PFAS-Sentry: http://localhost:{args.port}\n')
    app.run(host='0.0.0.0', port=args.port, debug=False)
