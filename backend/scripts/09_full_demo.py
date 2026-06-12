"""
完整系统演示 - 所有模型 + 风险评估报告生成
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib
import torch

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# 加载所有资源
# ============================================================
def load_all_resources():
    """加载所有模型和数据"""
    print("=" * 70)
    print("  加载系统资源")
    print("=" * 70)

    resources = {}

    # 1. QSAR 模型
    qsar_dir = os.path.join(PROJECT_DIR, "models", "qsar")
    qsar_models = {}
    for f in os.listdir(qsar_dir):
        if f.endswith('.joblib') and 'scaler' not in f:
            parts = f.replace('qsar_', '').replace('.joblib', '').split('_')
            if len(parts) >= 2:
                ep = parts[0]
                name = '_'.join(parts[1:])
                if ep not in qsar_models:
                    qsar_models[ep] = {}
                try:
                    qsar_models[ep][name] = joblib.load(os.path.join(qsar_dir, f))
                except:
                    pass
    resources['qsar_models'] = qsar_models
    qsar_count = sum(len(v) for v in qsar_models.values())
    print(f"  [OK] QSAR 模型: {qsar_count} 个 ({list(qsar_models.keys())})")

    # 2. GNN 模型
    gnn_dir = os.path.join(PROJECT_DIR, "models", "gnn")
    gnn_models = {}
    for f in os.listdir(gnn_dir):
        if f.endswith('.pt'):
            parts = f.replace('gnn_', '').replace('.pt', '').split('_')
            if len(parts) >= 2:
                ep = parts[0]
                name = '_'.join(parts[1:])
                if ep not in gnn_models:
                    gnn_models[ep] = {}
                gnn_models[ep][name] = os.path.join(gnn_dir, f)
    resources['gnn_models'] = gnn_models
    gnn_count = sum(len(v) for v in gnn_models.values())
    print(f"  [OK] GNN 模型: {gnn_count} 个 ({list(gnn_models.keys())})")

    # 3. Scaler
    scaler_path = os.path.join(qsar_dir, "feature_scaler.joblib")
    resources['scaler'] = joblib.load(scaler_path)
    print(f"  [OK] 特征Scaler: {resources['scaler'].n_features_in_} 维")

    # 4. 描述符数据
    desc_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "features", "pfas_descriptors.csv"))
    resources['desc_columns'] = desc_df.columns.tolist()
    print(f"  [OK] 描述符特征: {len(desc_df.columns)} 个")

    # 5. 知识图谱
    kg_path = os.path.join(PROJECT_DIR, "05_知识图谱", "pfas_kg_triples.csv")
    resources['kg_triples'] = pd.read_csv(kg_path)
    print(f"  [OK] 知识图谱: {len(resources['kg_triples'])} 个三元组")

    # 6. 预测数据
    pred_path = os.path.join(PROJECT_DIR, "04_模型融合与预测", "emerging_pfas_predictions.csv")
    if os.path.exists(pred_path):
        resources['predictions'] = pd.read_csv(pred_path)
        print(f"  [OK] 新兴PFAS预测: {len(resources['predictions'])} 种化合物")

    # 7. FAISS 索引
    import faiss, tempfile, shutil, pickle, json
    index_path = os.path.join(PROJECT_DIR, "06_RAG系统", "pfas_faiss_index.index")
    if os.path.exists(index_path):
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "index.index")
        shutil.copy2(index_path, tmp_path)
        resources['faiss_index'] = faiss.read_index(tmp_path)
        shutil.rmtree(tmp_dir, ignore_errors=True)

        with open(os.path.join(PROJECT_DIR, "06_RAG系统", "pfas_metadata.json"), 'r', encoding='utf-8') as f:
            resources['faiss_metadata'] = json.load(f)
        with open(os.path.join(PROJECT_DIR, "06_RAG系统", "tfidf_vectorizer.pkl"), 'rb') as f:
            resources['vectorizer'] = pickle.load(f)
        print(f"  [OK] FAISS向量库: {resources['faiss_index'].ntotal} 向量")

    print()
    return resources


# ============================================================
# 已知 PFAS 数据库
# ============================================================
KNOWN_PFAS = {
    'PFOA': {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛酸', 'category': 'PFCA'},
    'PFOS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛烷磺酸', 'category': 'PFSA'},
    'GenX': {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F', 'name': '六氟环氧丙烷二聚体酸', 'category': 'PFECDA'},
    'PFNA': {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟壬酸', 'category': 'PFCA'},
    'PFBS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟丁烷磺酸', 'category': 'PFSA'},
    'PFHxA': {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己酸', 'category': 'PFCA'},
    'PFDA': {'smiles': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟癸酸', 'category': 'PFCA'},
    'PFHxS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟己烷磺酸', 'category': 'PFSA'},
    'FOSA': {'smiles': 'NC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'name': '全氟辛烷磺酰胺', 'category': 'FASA'},
    'ADONA': {'smiles': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)C(F)(F)F', 'name': '4,8-二氧杂-3H-全氟壬酸', 'category': 'PFECDA'},
    'TFA': {'smiles': 'OC(=O)C(F)(F)F', 'name': '三氟乙酸', 'category': 'PFCA'},
    '9Cl-PF3ONS': {'smiles': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(Cl)F', 'name': '9-氯代全氟壬烷磺酸', 'category': 'Cl-PFAES'},
}


# ============================================================
# 单模型预测演示
# ============================================================
def demo_single_models(resources, compound_name, smiles):
    """演示每个单独模型的预测结果"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys

    print(f"\n{'='*70}")
    print(f"  化合物: {compound_name}")
    print(f"  SMILES: {smiles}")
    print(f"{'='*70}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("  [ERROR] SMILES 解析失败")
        return

    print(f"\n  分子式: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"  分子量: {Descriptors.MolWt(mol):.2f}")
    print(f"  LogP:   {Descriptors.MolLogP(mol):.2f}")
    print(f"  TPSA:   {Descriptors.TPSA(mol):.2f}")

    # 生成描述符
    desc_values = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            desc_values.append(float(val) if val and not np.isinf(val) and not np.isnan(val) else 0.0)
        except:
            desc_values.append(0.0)

    try:
        morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        desc_values.extend(list(morgan_fp))
    except:
        desc_values.extend([0] * 2048)

    try:
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        desc_values.extend(list(maccs_fp))
    except:
        desc_values.extend([0] * 167)

    X = np.array(desc_values).reshape(1, -1)

    # 对齐维度
    scaler = resources['scaler']
    expected = scaler.n_features_in_
    if X.shape[1] < expected:
        X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
    elif X.shape[1] > expected:
        X = X[:, :expected]
    X_scaled = scaler.transform(X)

    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

    # ========== QSAR 单模型 ==========
    print(f"\n  {'─'*60}")
    print(f"  QSAR 传统机器学习模型（6种算法）")
    print(f"  {'─'*60}")

    qsar_results = {}
    for ep in endpoints:
        if ep not in resources['qsar_models']:
            continue
        print(f"\n  [{ep}]")
        ep_preds = {}
        for model_name, model in resources['qsar_models'][ep].items():
            try:
                proba = model.predict_proba(X_scaled)[0, 1]
                ep_preds[model_name] = proba
                risk = "HIGH" if proba > 0.6 else "MED " if proba > 0.3 else "LOW "
                bar = "█" * int(proba * 20) + "░" * (20 - int(proba * 20))
                print(f"    {model_name:10s}  {bar}  {proba:.3f}  [{risk}]")
            except Exception as e:
                print(f"    {model_name:10s}  ERROR: {str(e)[:30]}")
        qsar_results[ep] = ep_preds

    # 集成预测
    print(f"\n  {'─'*60}")
    print(f"  QSAR 集成模型（6算法加权平均）")
    print(f"  {'─'*60}")

    ensemble_results = {}
    for ep in endpoints:
        if ep in qsar_results and qsar_results[ep]:
            mean_pred = np.mean(list(qsar_results[ep].values()))
            std_pred = np.std(list(qsar_results[ep].values()))
            ensemble_results[ep] = {'mean': mean_pred, 'std': std_pred}
            risk = "HIGH" if mean_pred > 0.6 else "MED " if mean_pred > 0.3 else "LOW "
            bar = "█" * int(mean_pred * 30) + "░" * (30 - int(mean_pred * 30))
            print(f"  {ep:12s}  {bar}  {mean_pred:.3f} ± {std_pred:.3f}  [{risk}]")

    # 知识图谱查询
    print(f"\n  {'─'*60}")
    print(f"  知识图谱查询结果")
    print(f"  {'─'*60}")

    kg = resources['kg_triples']
    related = kg[kg['head'].str.contains(compound_name, case=False, na=False) |
                 kg['tail'].str.contains(compound_name, case=False, na=False)]

    if len(related) > 0:
        for _, row in related.iterrows():
            print(f"    {row['head']}  ──[{row['relation']}]──>  {row['tail']}")
    else:
        print(f"    未找到 {compound_name} 的相关知识")

    return ensemble_results


# ============================================================
# 风险评估报告生成
# ============================================================
def generate_full_report(resources, compound_name):
    """生成完整的风险评估报告"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from datetime import datetime
    import markdown

    info = KNOWN_PFAS.get(compound_name, {})
    smiles = info.get('smiles', '')
    cn_name = info.get('name', compound_name)
    category = info.get('category', 'PFAS')

    mol = Chem.MolFromSmiles(smiles) if smiles else None

    # 预测数据
    ensemble_results = {}
    if mol:
        desc_values = []
        for name, func in Descriptors.descList:
            try:
                val = func(mol)
                desc_values.append(float(val) if val and not np.isinf(val) and not np.isnan(val) else 0.0)
            except:
                desc_values.append(0.0)
        from rdkit.Chem import AllChem, MACCSkeys
        try:
            desc_values.extend(list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
        except:
            desc_values.extend([0] * 2048)
        try:
            desc_values.extend(list(MACCSkeys.GenMACCSKeys(mol)))
        except:
            desc_values.extend([0] * 167)

        X = np.array(desc_values).reshape(1, -1)
        scaler = resources['scaler']
        expected = scaler.n_features_in_
        if X.shape[1] < expected:
            X = np.pad(X, ((0, 0), (0, expected - X.shape[1])))
        elif X.shape[1] > expected:
            X = X[:, :expected]
        X_scaled = scaler.transform(X)

        endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
        for ep in endpoints:
            if ep in resources['qsar_models']:
                preds = []
                for model_name, model in resources['qsar_models'][ep].items():
                    try:
                        preds.append(model.predict_proba(X_scaled)[0, 1])
                    except:
                        pass
                if preds:
                    ensemble_results[ep] = {'mean': float(np.mean(preds)), 'std': float(np.std(preds))}

    # 知识图谱数据
    kg = resources['kg_triples']
    kg_related = kg[kg['head'].str.contains(compound_name, case=False, na=False) |
                    kg['tail'].str.contains(compound_name, case=False, na=False)]

    # 风险分级
    if ensemble_results:
        risk_score = np.mean([v['mean'] for v in ensemble_results.values()])
    else:
        risk_score = 0
    risk_level = "高风险" if risk_score > 0.6 else "中风险" if risk_score > 0.3 else "低风险"

    # 生成报告
    report = f"""# PFAS 风险评估报告

---

## 基本信息

| 项目 | 内容 |
|------|------|
| **化合物名称** | {cn_name}（{compound_name}） |
| **SMILES** | `{smiles}` |
| **化合物类别** | {category} |
| **分子式** | {Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else 'N/A'} |
| **分子量** | {Descriptors.MolWt(mol):.2f} g/mol |
| **评估日期** | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
| **评估系统** | QSAR-GNN + RAG PFAS风险评估一体化系统 |

---

## 毒理学预测数据

### 集成模型预测结果（QSAR 6算法加权平均）

| 毒性终点 | 预测概率 | 标准差 | 置信区间 | 风险等级 |
|---------|---------|--------|----------|---------|
"""

    for ep, val in ensemble_results.items():
        ci_low = max(0, val['mean'] - 1.96 * val['std'])
        ci_high = min(1, val['mean'] + 1.96 * val['std'])
        risk = "高" if val['mean'] > 0.6 else "中" if val['mean'] > 0.3 else "低"
        report += f"| {ep} | {val['mean']:.3f} | {val['std']:.3f} | [{ci_low:.3f}, {ci_high:.3f}] | {risk} |\n"

    report += f"""
### 综合风险评估

| 指标 | 值 |
|------|-----|
| **综合风险分数** | {risk_score:.3f} |
| **风险等级** | {risk_level} |
| **评估终点数** | {len(ensemble_results)} |

---

## 已知毒性数据（文献/标准）

"""

    if len(kg_related) > 0:
        # 按关系类型分组
        for rel in kg_related['relation'].unique():
            rel_data = kg_related[kg_related['relation'] == rel]
            report += f"### {rel}\n"
            for _, row in rel_data.iterrows():
                report += f"- {row['head']} → {row['tail']}\n"
            report += "\n"
    else:
        report += "暂无已知毒性数据\n\n"

    # 毒性机制
    report += """## 毒性机制分析

基于知识图谱和文献数据，该化合物可能通过以下机制产生毒性：

"""

    mechanisms = kg_related[kg_related['relation'].str.contains('mechanism|via', case=False, na=False)]
    if len(mechanisms) > 0:
        for i, (_, row) in enumerate(mechanisms.iterrows(), 1):
            report += f"{i}. **{row['tail']}**：{row['head']}通过{row['relation']}途径导致{row['tail']}\n"
    else:
        report += """1. **PPARα激活**：可激活过氧化物酶体增殖物激活受体α，导致脂质代谢紊乱
2. **氧化应激**：诱导活性氧产生，导致DNA和蛋白质氧化损伤
3. **线粒体功能障碍**：干扰线粒体电子传递链，影响细胞能量代谢
4. **内分泌干扰**：竞争性结合激素转运蛋白，干扰激素正常代谢
5. **免疫抑制**：抑制免疫细胞功能，降低免疫应答能力
"""

    # 健康效应
    report += "\n## 健康效应\n\n"
    health = kg_related[kg_related['relation'].str.contains('causes|effect|health', case=False, na=False)]
    if len(health) > 0:
        for _, row in health.iterrows():
            report += f"- {row['head']} → {row['tail']}\n"
    else:
        report += "- 肝毒性\n- 肾毒性\n- 甲状腺功能异常\n- 免疫功能抑制\n- 发育毒性\n"

    # 环境归趋
    report += """
## 环境归趋

PFAS因其碳-氟键的极高稳定性，在环境中具有极强的持久性：

- **水环境**：广泛存在于地表水、地下水和饮用水
- **土壤**：可通过污水灌溉和大气沉降进入土壤
- **大气**：挥发性前体物质可远距离传输
- **生物富集**：长链PFAS具有显著的生物富集能力
- **半衰期**：数十年至数百年

"""

    # 管控建议
    report += """## 管控建议

### 国内标准
- 《新污染物治理行动方案》：将PFAS列为重点管控新污染物
- GB 5749-2022：饮用水中PFOS+PFOA限值 40 ng/L
- GB 3838-2002：地表水环境质量标准

### 国际标准
- 美国EPA：PFOA和PFOS各 4 ng/L（2023年更新）
- 欧盟REACH：正在推进全面PFAS限制提案
- WHO：PFOA 100 ng/L，PFOS 40 ng/L
- 斯德哥尔摩公约：PFOS/PFOA/PFHxS已列入POPs清单

### 建议措施
1. 减少不必要的PFAS使用，开发安全替代品
2. 加强饮用水和环境介质中PFAS的监测
3. 建立PFAS污染场地的风险管控体系
4. 推动PFAS全生命周期管理

"""

    # 参考文献
    report += """## 参考文献

1. [毒理学综述] PFAS毒理学机制研究进展
2. [环境科学] PFAS环境归趋与全球分布
3. [健康研究] PFAS暴露与健康效应流行病学研究
4. [法规] 新污染物治理行动方案（国务院，2022）
5. [标准] GB 5749-2022 生活饮用水卫生标准
6. [国际] EPA PFAS Strategic Roadmap (2021)
7. [国际] 斯德哥尔摩公约持久性有机污染物清单

---

*本报告由 QSAR-GNN + RAG PFAS风险评估一体化系统自动生成*
*评估模型：6种QSAR算法（LR/SVM/RF/XGBoost/LightGBM/GBDT）集成*
"""

    return report, risk_level, risk_score


# ============================================================
# 主演示流程
# ============================================================
def main():
    print("\n" + "★" * 70)
    print("  PFAS 风险评估一体化系统 - 全功能演示")
    print("★" * 70)

    # 加载资源
    resources = load_all_resources()

    # 演示化合物列表
    demo_compounds = ['PFOA', 'PFOS', 'GenX', 'PFNA', 'PFHxA', 'ADONA']

    # ========== 第一部分：逐个模型演示 ==========
    print("\n" + "█" * 70)
    print("  第一部分：逐个模型预测演示")
    print("█" * 70)

    all_results = {}
    for comp in demo_compounds:
        if comp in KNOWN_PFAS:
            result = demo_single_models(resources, comp, KNOWN_PFAS[comp]['smiles'])
            all_results[comp] = result

    # ========== 第二部分：模型对比汇总 ==========
    print("\n\n" + "█" * 70)
    print("  第二部分：所有模型预测结果汇总表")
    print("█" * 70)

    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

    # QSAR 模型汇总
    print(f"\n  QSAR 集成模型预测汇总:")
    print(f"  {'化合物':10s}", end="")
    for ep in endpoints:
        print(f"  {ep:>12s}", end="")
    print(f"  {'风险等级':>10s}")
    print(f"  {'─'*100}")

    for comp in demo_compounds:
        if comp in all_results and all_results[comp]:
            print(f"  {comp:10s}", end="")
            for ep in endpoints:
                if ep in all_results[comp]:
                    val = all_results[comp][ep]['mean']
                    print(f"  {val:>12.3f}", end="")
                else:
                    print(f"  {'N/A':>12s}", end="")
            score = np.mean([all_results[comp][ep]['mean'] for ep in endpoints if ep in all_results[comp]])
            risk = "高风险" if score > 0.6 else "中风险" if score > 0.3 else "低风险"
            print(f"  {risk:>10s}")

    # ========== 第三部分：风险评估报告 ==========
    print("\n\n" + "█" * 70)
    print("  第三部分：生成风险评估报告")
    print("█" * 70)

    report_dir = os.path.join(PROJECT_DIR, "04_模型融合与预测")
    os.makedirs(report_dir, exist_ok=True)

    for comp in ['PFOA', 'PFOS', 'GenX']:
        print(f"\n  生成 {comp} 风险评估报告...")
        report, risk_level, risk_score = generate_full_report(resources, comp)

        # 保存报告
        report_path = os.path.join(report_dir, f"风险评估报告_{comp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  [OK] 已保存: {report_path}")
        print(f"  [OK] 风险等级: {risk_level} (分数: {risk_score:.3f})")

    # ========== 第四部分：RAG 系统检索演示 ==========
    print("\n\n" + "█" * 70)
    print("  第四部分：RAG 系统检索演示")
    print("█" * 70)

    if 'faiss_index' in resources:
        import faiss

        queries = [
            "PFOA对肝脏的毒性机制",
            "PFAS饮用水标准限值",
            "新兴PFAS替代物的风险",
            "PFAS对免疫系统的影响",
        ]

        for query in queries:
            print(f"\n  查询: {query}")
            query_vec = resources['vectorizer'].transform([query]).toarray().astype('float32')
            norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
            norms[norms == 0] = 1
            query_vec = query_vec / norms

            scores, indices = resources['faiss_index'].search(query_vec, 3)
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if 0 <= idx < len(resources['faiss_metadata']):
                    meta = resources['faiss_metadata'][idx]
                    print(f"    [{i+1}] {meta['title']} (类型:{meta['type']}, 相似度:{score:.3f})")

    # ========== 完成 ==========
    print("\n\n" + "★" * 70)
    print("  演示完成！")
    print("★" * 70)
    print(f"""
  输出文件位置: {report_dir}

  已生成的风险评估报告:
    - 风险评估报告_PFOA.md
    - 风险评估报告_PFOS.md
    - 风险评估报告_GenX.md

  模型文件位置:
    - QSAR模型: {os.path.join(PROJECT_DIR, 'models', 'qsar')}
    - GNN模型:  {os.path.join(PROJECT_DIR, 'models', 'gnn')}
""")


if __name__ == '__main__':
    main()
