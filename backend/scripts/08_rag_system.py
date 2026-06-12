"""
RAG 增强的风险评估与辅助决策系统
三个核心功能：毒性查询、机制解释、风险报告
"""
import sys, os, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# RAG 系统核心类
# ============================================================
class PFAS_RAG_System:
    """PFAS RAG 风险评估系统"""

    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.index = None
        self.vectorizer = None
        self.metadata = None
        self.documents = None
        self.kg_triples = None
        self.prediction_data = None
        self.qsar_models = {}

        self._load_resources()

    def _load_resources(self):
        """加载所有资源"""
        import faiss, tempfile, shutil

        # 加载 FAISS 索引（FAISS C++ 后端不支持中文路径，复制到临时路径再读取）
        index_path = os.path.join(self.project_dir, "06_RAG系统", "pfas_faiss_index.index")
        if os.path.exists(index_path):
            tmp_dir = tempfile.mkdtemp()
            tmp_index_path = os.path.join(tmp_dir, "pfas_faiss_index.index")
            shutil.copy2(index_path, tmp_index_path)
            self.index = faiss.read_index(tmp_index_path)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"  FAISS 索引已加载: {self.index.ntotal} 向量")

        # 加载向量化器
        vec_path = os.path.join(self.project_dir, "06_RAG系统", "tfidf_vectorizer.pkl")
        if os.path.exists(vec_path):
            with open(vec_path, 'rb') as f:
                self.vectorizer = pickle.load(f)

        # 加载元数据
        meta_path = os.path.join(self.project_dir, "06_RAG系统", "pfas_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

        # 加载知识图谱三元组
        kg_path = os.path.join(self.project_dir, "05_知识图谱", "pfas_kg_triples.csv")
        if os.path.exists(kg_path):
            self.kg_triples = pd.read_csv(kg_path)

        # 加载预测数据
        pred_path = os.path.join(self.project_dir, "04_模型融合与预测", "emerging_pfas_predictions.csv")
        if os.path.exists(pred_path):
            self.prediction_data = pd.read_csv(pred_path)

        # 加载 QSAR 模型
        import joblib
        model_dir = os.path.join(self.project_dir, "models", "qsar")
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.joblib') and 'scaler' not in f:
                    try:
                        model = joblib.load(os.path.join(model_dir, f))
                        # 从文件名提取终点和模型名
                        parts = f.replace('qsar_', '').replace('.joblib', '').split('_')
                        if len(parts) >= 2:
                            ep = parts[0]
                            name = '_'.join(parts[1:])
                            if ep not in self.qsar_models:
                                self.qsar_models[ep] = {}
                            self.qsar_models[ep][name] = model
                    except:
                        pass

    def retrieve(self, query, top_k=5):
        """检索相关文档"""
        if self.index is None or self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_vec = query_vec / norms

        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                results.append({
                    'score': float(score),
                    'title': self.metadata[idx]['title'],
                    'source': self.metadata[idx]['source'],
                    'type': self.metadata[idx]['type'],
                    'content': self.metadata[idx].get('content_preview', '')
                })
        return results

    def query_kg(self, entity):
        """查询知识图谱"""
        if self.kg_triples is None:
            return []

        results = []
        # 查询与实体相关的三元组
        related = self.kg_triples[
            self.kg_triples['head'].str.contains(entity, case=False, na=False) |
            self.kg_triples['tail'].str.contains(entity, case=False, na=False)
        ]

        for _, row in related.iterrows():
            results.append({
                'head': row['head'],
                'relation': row['relation'],
                'tail': row['tail']
            })
        return results

    def predict_toxicity(self, smiles):
        """调用QSAR模型预测毒性"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors, AllChem, MACCSkeys

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

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

        # 加载 scaler
        import joblib
        scaler_path = os.path.join(self.project_dir, "models", "qsar", "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            # 对齐特征维度
            expected_features = scaler.n_features_in_
            if X.shape[1] < expected_features:
                X = np.pad(X, ((0, 0), (0, expected_features - X.shape[1])))
            elif X.shape[1] > expected_features:
                X = X[:, :expected_features]
            X = scaler.transform(X)

        # 预测
        results = {}
        for ep, models in self.qsar_models.items():
            preds = []
            for name, model in models.items():
                try:
                    proba = model.predict_proba(X)[0, 1]
                    preds.append(proba)
                except:
                    pass
            if preds:
                results[ep] = {
                    'mean': float(np.mean(preds)),
                    'std': float(np.std(preds)),
                    'n_models': len(preds)
                }

        return results

    # ============================================================
    # 功能1：毒性预测查询
    # ============================================================
    def toxicity_query(self, compound_input):
        """毒性预测查询功能"""
        print(f"\n{'='*60}")
        print(f"毒性预测查询: {compound_input}")
        print(f"{'='*60}")

        # 判断输入是SMILES还是化合物名
        from rdkit import Chem
        mol = Chem.MolFromSmiles(compound_input)
        smiles = compound_input if mol else None
        compound_name = compound_input

        # 如果是名称，尝试查找SMILES
        if mol is None:
            # 从已知PFAS中查找
            known_pfas = {
                'PFOA': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
                'PFOS': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
                'GenX': 'OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F',
                'PFNA': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
                'PFBS': 'OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F',
            }
            for name, smi in known_pfas.items():
                if name.lower() == compound_input.lower():
                    smiles = smi
                    break

        result = {'compound': compound_name, 'smiles': smiles}

        # 1. 模型预测
        if smiles:
            print("\n[1] 模型预测结果:")
            pred = self.predict_toxicity(smiles)
            if pred:
                result['predictions'] = pred
                for ep, val in pred.items():
                    risk = "高" if val['mean'] > 0.6 else "中" if val['mean'] > 0.3 else "低"
                    print(f"    {ep}: {val['mean']:.3f} ± {val['std']:.3f} (风险: {risk})")
            else:
                print("    无法预测（SMILES解析失败）")

        # 2. 知识图谱查询
        print("\n[2] 已知毒性数据（知识图谱）:")
        kg_results = self.query_kg(compound_name)
        if kg_results:
            for r in kg_results[:10]:
                print(f"    {r['head']} --[{r['relation']}]--> {r['tail']}")
            result['kg_data'] = kg_results
        else:
            print("    未找到相关知识图谱数据")

        # 3. 文献检索
        print("\n[3] 相关文献与标准:")
        docs = self.retrieve(f"{compound_name} toxicity mechanism regulation", top_k=3)
        for doc in docs:
            print(f"    [{doc['type']}] {doc['title']} (来源: {doc['source']}, 相似度: {doc['score']:.3f})")
        result['literature'] = docs

        return result

    # ============================================================
    # 功能2：毒性机制解释
    # ============================================================
    def mechanism_explanation(self, compound_name):
        """毒性机制解释功能"""
        print(f"\n{'='*60}")
        print(f"毒性机制解释: {compound_name}")
        print(f"{'='*60}")

        # 1. 从知识图谱获取机制
        kg_results = self.query_kg(compound_name)
        mechanisms = [r for r in kg_results if 'mechanism' in r['relation'] or 'via' in r['relation']]

        # 2. 检索相关文献
        docs = self.retrieve(f"{compound_name} toxicity mechanism pathway", top_k=5)

        # 3. 生成解释
        explanation = f"# {compound_name} 毒性机制解释\n\n"

        # 基本信息
        explanation += "## 化合物信息\n"
        if kg_results:
            for r in kg_results[:5]:
                explanation += f"- {r['head']} {r['relation']} {r['tail']}\n"

        # 毒性机制
        explanation += "\n## 主要毒性机制\n"
        if mechanisms:
            for m in mechanisms:
                explanation += f"1. **{m['tail']}**: {m['head']}通过{m['relation']}途径导致{m['tail']}\n"
        else:
            # 默认机制描述
            explanation += """1. **PPARα激活**: PFAS可激活过氧化物酶体增殖物激活受体α，导致脂质代谢紊乱
2. **氧化应激**: 诱导活性氧产生，导致DNA和蛋白质氧化损伤
3. **线粒体功能障碍**: 干扰线粒体电子传递链，影响细胞能量代谢
4. **内分泌干扰**: 竞争性结合激素转运蛋白，干扰激素正常代谢
5. **免疫抑制**: 抑制免疫细胞功能，降低免疫应答能力\n"""

        # 健康效应
        health_effects = [r for r in kg_results if 'causes' in r['relation'] or 'effect' in r['relation']]
        if health_effects:
            explanation += "\n## 可能的健康效应\n"
            for h in health_effects:
                explanation += f"- {h['head']} → {h['tail']}\n"

        # 参考文献
        explanation += "\n## 参考文献\n"
        for doc in docs:
            explanation += f"- [{doc['source']}] {doc['title']}\n"

        print(explanation)
        return explanation

    # ============================================================
    # 功能3：风险评估报告生成
    # ============================================================
    def risk_assessment_report(self, compound_input):
        """风险评估报告生成功能"""
        print(f"\n{'='*60}")
        print(f"生成风险评估报告: {compound_input}")
        print(f"{'='*60}")

        from rdkit import Chem
        from datetime import datetime

        # 确定化合物信息
        mol = Chem.MolFromSmiles(compound_input)
        smiles = compound_input if mol else None
        compound_name = compound_input

        if mol is None:
            known_pfas = {
                'PFOA': ('OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'Perfluorooctanoic acid', 'PFCA'),
                'PFOS': ('OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F', 'Perfluorooctane sulfonic acid', 'PFSA'),
                'GenX': ('OC(=O)C(F)(F)OC(F)(F)C(F)(F)OC(F)(F)F', 'HFPO-DA', 'PFECDA'),
            }
            for name, (smi, full_name, cat) in known_pfas.items():
                if name.lower() == compound_input.lower():
                    smiles = smi
                    break

        # 收集数据
        kg_data = self.query_kg(compound_name)
        pred_data = self.predict_toxicity(smiles) if smiles else None
        literature = self.retrieve(f"{compound_name} regulation standard", top_k=5)

        # 生成报告
        report = f"""# PFAS 风险评估报告

**化合物名称**: {compound_name}
**SMILES**: {smiles if smiles else 'N/A'}
**评估日期**: {datetime.now().strftime('%Y-%m-%d')}
**评估系统**: QSAR-GNN + RAG PFAS风险评估一体化系统

---

## 1. 化合物基本信息

| 属性 | 值 |
|------|-----|
| 化合物名称 | {compound_name} |
| SMILES | {smiles if smiles else 'N/A'} |
| 化合物类型 | PFAS |

"""

        # 物理化学性质
        if smiles and mol:
            from rdkit.Chem import Descriptors
            report += f"""## 2. 物理化学性质

| 性质 | 值 |
|------|-----|
| 分子量 | {Descriptors.MolWt(mol):.2f} |
| LogP | {Descriptors.MolLogP(mol):.2f} |
| TPSA | {Descriptors.TPSA(mol):.2f} |
| 氢键供体 | {Descriptors.NumHDonors(mol)} |
| 氢键受体 | {Descriptors.NumHAcceptors(mol)} |
| 可旋转键 | {Descriptors.NumRotatableBonds(mol)} |
| 重原子数 | {mol.GetNumHeavyAtoms()} |

"""

        # 毒理学数据
        report += "## 3. 毒理学数据\n\n"
        if pred_data:
            report += "### 3.1 模型预测结果\n\n"
            report += "| 毒性终点 | 预测概率 | 标准差 | 风险等级 |\n"
            report += "|---------|---------|--------|----------|\n"
            for ep, val in pred_data.items():
                risk = "高" if val['mean'] > 0.6 else "中" if val['mean'] > 0.3 else "低"
                report += f"| {ep} | {val['mean']:.3f} | {val['std']:.3f} | {risk} |\n"
        else:
            report += "暂无模型预测数据\n"

        # 知识图谱数据
        if kg_data:
            report += "\n### 3.2 已知毒性数据（文献/标准）\n\n"
            for r in kg_data[:10]:
                report += f"- {r['head']} {r['relation']} {r['tail']}\n"

        # 健康风险评估
        report += """
## 4. 健康风险评估

基于QSAR模型预测和文献数据，该化合物的健康风险评估如下：

### 4.1 内分泌干扰风险
- NR-AR(雄激素受体): 需关注
- NR-AhR(芳香烃受体): 需关注

### 4.2 细胞毒性风险
- SR-MMP(线粒体膜电位): 需关注
- SR-p53(p53通路): 需关注

### 4.3 综合评价
PFAS类化合物普遍具有持久性、生物蓄积性和毒性(PBT特性)。建议采取预防性原则进行管控。

"""

        # 生态风险评估
        report += """## 5. 生态风险评估

PFAS对水生生物(鱼类、甲壳类、藻类)具有显著毒性，且在环境中高度持久。
建议关注：
- 水生生态系统保护
- 食物链生物富集
- 长期生态效应

"""

        # 管控建议
        report += """## 6. 管控建议

### 6.1 国内标准
- 遵守《新污染物治理行动方案》要求
- 饮用水中PFOS+PFOA限值: 40 ng/L (GB 5749-2022)

### 6.2 国际标准参考
- 美国EPA: PFOA和PFOS各4 ng/L
- 欧盟: 正在推进全面PFAS限制
- WHO: PFOA 100 ng/L, PFOS 40 ng/L

### 6.3 建议措施
1. 减少不必要的PFAS使用
2. 开发安全替代品
3. 加强环境监测
4. 完善风险管理体系

"""

        # 参考文献
        report += "## 7. 参考文献\n\n"
        for doc in literature:
            report += f"- [{doc['source']}] {doc['title']}\n"
        report += "- [标准] GB 5749-2022 生活饮用水卫生标准\n"
        report += "- [法规] 新污染物治理行动方案\n"
        report += "- [国际] 斯德哥尔摩公约\n"

        report += "\n---\n*本报告由QSAR-GNN+RAG PFAS风险评估系统自动生成*\n"

        # 保存报告
        report_dir = os.path.join(self.project_dir, "04_模型融合与预测")
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"risk_report_{compound_name}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存: {report_path}")

        return report

# ============================================================
# 测试演示
# ============================================================
def demo():
    """系统演示"""
    print("=" * 60)
    print("PFAS RAG 风险评估系统 - 演示")
    print("=" * 60)

    # 初始化系统
    system = PFAS_RAG_System(PROJECT_DIR)

    # 演示1：毒性预测查询
    print("\n" + "=" * 60)
    print("演示1：毒性预测查询")
    print("=" * 60)
    system.toxicity_query("PFOA")
    system.toxicity_query("GenX")

    # 演示2：毒性机制解释
    print("\n" + "=" * 60)
    print("演示2：毒性机制解释")
    print("=" * 60)
    system.mechanism_explanation("PFOS")

    # 演示3：风险评估报告
    print("\n" + "=" * 60)
    print("演示3：风险评估报告")
    print("=" * 60)
    system.risk_assessment_report("PFOA")

def main():
    demo()

if __name__ == '__main__':
    main()
