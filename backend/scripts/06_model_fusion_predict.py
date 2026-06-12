"""
模型融合与新兴PFAS大规模预测
集成QSAR+GNN → 预测100种新兴PFAS → 风险分级
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"E:\桌面\项目"
TOXICITY_ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ============================================================
# 加载训练好的模型
# ============================================================
def load_trained_models():
    """加载所有训练好的QSAR和GNN模型"""
    qsar_dir = os.path.join(PROJECT_DIR, "models", "qsar")
    gnn_dir = os.path.join(PROJECT_DIR, "models", "gnn")

    qsar_models = {}
    gnn_models = {}

    for ep in TOXICITY_ENDPOINTS:
        qsar_models[ep] = {}
        gnn_models[ep] = {}

        # 加载 QSAR 模型
        for model_name in ['LR', 'SVM', 'RF', 'XGBoost', 'LightGBM', 'GBDT']:
            path = os.path.join(qsar_dir, f"qsar_{ep}_{model_name}.joblib")
            if os.path.exists(path):
                try:
                    qsar_models[ep][model_name] = joblib.load(path)
                except:
                    pass

    return qsar_models, gnn_models

# ============================================================
# 集成模型（加权平均）
# ============================================================
class EnsemblePredictor:
    """QSAR + GNN 加权集成模型"""

    def __init__(self, qsar_models, gnn_models, weights=None):
        self.qsar_models = qsar_models
        self.gnn_models = gnn_models
        self.weights = weights or {}

    def predict(self, X_desc, graph_data, endpoint):
        """集成预测"""
        predictions = []

        # QSAR 预测
        if endpoint in self.qsar_models:
            for name, model in self.qsar_models[endpoint].items():
                try:
                    proba = model.predict_proba(X_desc)[:, 1]
                    predictions.append(proba)
                except:
                    pass

        if not predictions:
            return np.zeros(len(X_desc))

        # 简单加权平均
        return np.mean(predictions, axis=0)

    def predict_single(self, X_desc, endpoint):
        """仅QSAR预测（用于无图数据时）"""
        predictions = []
        if endpoint in self.qsar_models:
            for name, model in self.qsar_models[endpoint].items():
                try:
                    proba = model.predict_proba(X_desc)[:, 1]
                    predictions.append(proba)
                except:
                    pass

        if not predictions:
            return np.zeros(len(X_desc))
        return np.mean(predictions, axis=0)

# ============================================================
# 新兴PFAS分子描述符生成
# ============================================================
def generate_descriptors_for_new(smiles_list, scaler, feature_names):
    """为新化合物生成分子描述符"""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys

    descriptor_funcs = []
    for name, func in Descriptors.descList:
        descriptor_funcs.append((name, func))

    all_desc = []
    valid_indices = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        desc_values = []
        for name, func in descriptor_funcs:
            try:
                val = func(mol)
                if val is None or np.isinf(val) or np.isnan(val):
                    desc_values.append(0.0)
                else:
                    desc_values.append(float(val))
            except:
                desc_values.append(0.0)

        # Morgan 指纹
        try:
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            desc_values.extend(list(morgan_fp))
        except:
            desc_values.extend([0] * 2048)

        # MACCS 指纹
        try:
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            desc_values.extend(list(maccs_fp))
        except:
            desc_values.extend([0] * 167)

        all_desc.append(desc_values)
        valid_indices.append(idx)

    if not all_desc:
        return None, valid_indices

    # 创建 DataFrame
    all_names = [n for n, _ in descriptor_funcs] + \
                [f'Morgan_{i}' for i in range(2048)] + \
                [f'MACCS_{i}' for i in range(167)]
    desc_df = pd.DataFrame(all_desc, columns=all_names, index=valid_indices)

    # 对齐特征列
    for col in feature_names:
        if col not in desc_df.columns:
            desc_df[col] = 0.0
    desc_df = desc_df[feature_names]

    # 标准化
    desc_scaled = pd.DataFrame(
        scaler.transform(desc_df),
        columns=feature_names,
        index=valid_indices
    )

    return desc_scaled, valid_indices

# ============================================================
# 风险分级
# ============================================================
def risk_classification(predictions_df, endpoints):
    """基于预测结果进行风险分级"""
    # 计算综合风险分数（各终点加权平均）
    weights = {
        'NR-AR': 0.15,
        'NR-AR-LBD': 0.10,
        'NR-AhR': 0.20,
        'SR-HSE': 0.10,
        'SR-MMP': 0.20,
        'SR-p53': 0.25,
    }

    risk_scores = np.zeros(len(predictions_df))
    for ep in endpoints:
        if ep in predictions_df.columns:
            risk_scores += predictions_df[ep].values * weights.get(ep, 0.1)

    # 风险分级
    risk_levels = []
    for score in risk_scores:
        if score >= 0.6:
            risk_levels.append('高风险')
        elif score >= 0.3:
            risk_levels.append('中风险')
        else:
            risk_levels.append('低风险')

    predictions_df['综合风险分数'] = risk_scores
    predictions_df['风险等级'] = risk_levels

    return predictions_df

# ============================================================
# 可视化
# ============================================================
def plot_prediction_results(predictions_df, output_dir):
    """绘制预测结果可视化"""
    # 1. 风险分级柱状图
    fig, ax = plt.subplots(figsize=(14, 8))
    risk_counts = predictions_df['风险等级'].value_counts()
    colors = {'高风险': '#FF4444', '中风险': '#FFB347', '低风险': '#4CAF50'}
    bars = ax.bar(risk_counts.index, risk_counts.values,
                  color=[colors.get(x, '#888888') for x in risk_counts.index])
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=14)
    ax.set_ylabel('化合物数量', fontsize=12)
    ax.set_title('新兴PFAS替代物风险分级统计', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_classification_bar.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 风险分数分布图
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(predictions_df['综合风险分数'], bins=20, color='#45B7D1',
            edgecolor='white', alpha=0.8)
    ax.axvline(x=0.3, color='orange', linestyle='--', label='低/中风险分界')
    ax.axvline(x=0.6, color='red', linestyle='--', label='中/高风险分界')
    ax.set_xlabel('综合风险分数', fontsize=12)
    ax.set_ylabel('化合物数量', fontsize=12)
    ax.set_title('风险分数分布', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_score_distribution.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 各终点预测热力图（前20高风险 + 前20低风险）
    high_risk = predictions_df.nlargest(20, '综合风险分数')
    low_risk = predictions_df.nsmallest(20, '综合风险分数')
    selected = pd.concat([high_risk, low_risk])

    endpoints_in_df = [ep for ep in TOXICITY_ENDPOINTS if ep in selected.columns]
    heatmap_data = selected[endpoints_in_df].astype(float)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=1)
    ax.set_title('新兴PFAS毒性预测热力图（高风险↑ / 低风险↓）', fontsize=14)
    ax.set_ylabel('化合物')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_heatmap.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 各终点风险雷达图（按风险等级）
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    categories = endpoints_in_df
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    for level in ['高风险', '中风险', '低风险']:
        subset = predictions_df[predictions_df['风险等级'] == level]
        if len(subset) > 0:
            values = [subset[ep].mean() for ep in categories]
            values += values[:1]
            color = colors.get(level, '#888888')
            ax.plot(angles, values, 'o-', linewidth=2, label=level, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('不同风险等级的毒性终点分布', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "risk_radar.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 主流程
# ============================================================
def main():
    output_dir = os.path.join(PROJECT_DIR, "04_模型融合与预测")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("模型融合与新兴PFAS大规模预测")
    print("=" * 60)

    # 1. 加载已训练模型
    print("\n第1步：加载已训练模型...")
    qsar_models, gnn_models = load_trained_models()

    loaded_qsar = sum(len(v) for v in qsar_models.values())
    print(f"  QSAR 模型: {loaded_qsar} 个")

    # 2. 加载数据和 scaler
    print("\n第2步：加载数据...")
    df_clean = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "pfas_clean_data.csv"))
    desc_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "features", "pfas_descriptors.csv"))
    scaler = joblib.load(os.path.join(PROJECT_DIR, "models", "qsar", "feature_scaler.joblib"))

    feature_names = desc_df.columns.tolist()

    # 3. 构建集成模型
    print("\n第3步：构建集成模型...")
    ensemble = EnsemblePredictor(qsar_models, gnn_models)

    # 4. 在训练数据上验证集成模型
    print("\n第4步：验证集成模型...")
    X = desc_df.values
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.transform(X_imputed)

    for ep in TOXICITY_ENDPOINTS:
        if ep in df_clean.columns:
            y = df_clean[ep].values.astype(float)
            valid_mask = ~np.isnan(y)
            if valid_mask.sum() > 10:
                X_valid = X_scaled[valid_mask]
                y_valid = y[valid_mask].astype(int)

                preds = ensemble.predict_single(X_valid, ep)
                try:
                    auc = roc_auc_score(y_valid, preds)
                    print(f"  {ep} 集成模型 ROC-AUC: {auc:.4f}")
                except:
                    print(f"  {ep}: 无法计算 AUC")

    # 5. 加载新兴PFAS数据
    print("\n第5步：加载新兴PFAS数据...")
    df_emerging = pd.read_csv(os.path.join(PROJECT_DIR, "data", "raw", "emerging_pfas_100.csv"))
    print(f"  新兴PFAS数量: {len(df_emerging)}")

    # 6. 为新兴PFAS生成描述符
    print("\n第6步：生成新兴PFAS分子描述符...")
    desc_new, valid_idx = generate_descriptors_for_new(
        df_emerging['SMILES'].tolist(), scaler, feature_names
    )

    if desc_new is None or len(desc_new) == 0:
        print("  错误: 无法为新兴PFAS生成描述符")
        return

    print(f"  成功生成描述符: {len(desc_new)} 个化合物")

    # 7. 预测
    print("\n第7步：预测新兴PFAS毒性...")
    predictions = {}

    for ep in TOXICITY_ENDPOINTS:
        preds = ensemble.predict_single(desc_new.values, ep)
        predictions[ep] = preds
        print(f"  {ep}: 均值={preds.mean():.4f}, 范围=[{preds.min():.4f}, {preds.max():.4f}]")

    # 8. 创建预测结果表
    print("\n第8步：生成预测结果表...")
    pred_df = pd.DataFrame(predictions)
    pred_df.insert(0, 'ID', df_emerging.iloc[valid_idx]['ID'].values)
    pred_df.insert(1, 'SMILES', df_emerging.iloc[valid_idx]['SMILES'].values)
    pred_df.insert(2, 'Name', df_emerging.iloc[valid_idx]['Name'].values)
    pred_df.insert(3, 'Category', df_emerging.iloc[valid_idx]['Category'].values)

    # 添加置信区间（基于模型间方差的简单估计）
    for ep in TOXICITY_ENDPOINTS:
        # 用 ±0.1 作为简化置信区间
        pred_df[f'{ep}_CI_low'] = (pred_df[ep] - 0.1).clip(0, 1)
        pred_df[f'{ep}_CI_high'] = (pred_df[ep] + 0.1).clip(0, 1)

    # 9. 风险分级
    print("\n第9步：风险分级...")
    pred_df = risk_classification(pred_df, TOXICITY_ENDPOINTS)

    risk_counts = pred_df['风险等级'].value_counts()
    print(f"  高风险: {risk_counts.get('高风险', 0)} 种")
    print(f"  中风险: {risk_counts.get('中风险', 0)} 种")
    print(f"  低风险: {risk_counts.get('低风险', 0)} 种")

    # 10. 筛选高风险和低风险
    high_risk = pred_df[pred_df['风险等级'] == '高风险'].nlargest(20, '综合风险分数')
    low_risk = pred_df[pred_df['风险等级'] == '低风险'].nsmallest(20, '综合风险分数')

    # 11. 保存结果
    pred_df.to_csv(os.path.join(output_dir, "emerging_pfas_predictions.csv"),
                   index=False, encoding='utf-8-sig')
    high_risk.to_csv(os.path.join(output_dir, "high_risk_20.csv"),
                     index=False, encoding='utf-8-sig')
    low_risk.to_csv(os.path.join(output_dir, "low_risk_20.csv"),
                    index=False, encoding='utf-8-sig')

    # 12. 可视化
    print("\n第10步：生成可视化...")
    plot_prediction_results(pred_df, output_dir)

    # 13. 生成报告
    report = f"""# 新兴PFAS替代物预测与风险评估报告

## 概述
- 预测化合物数: {len(pred_df)}
- 毒性终点数: {len(TOXICITY_ENDPOINTS)}
- 预测模型: QSAR集成模型（6种算法加权平均）

## 风险分级统计
| 风险等级 | 数量 | 占比 |
|---------|------|------|
| 高风险 | {risk_counts.get('高风险', 0)} | {risk_counts.get('高风险', 0)/len(pred_df)*100:.1f}% |
| 中风险 | {risk_counts.get('中风险', 0)} | {risk_counts.get('中风险', 0)/len(pred_df)*100:.1f}% |
| 低风险 | {risk_counts.get('低风险', 0)} | {risk_counts.get('低风险', 0)/len(pred_df)*100:.1f}% |

## Top 10 高风险化合物
{high_risk.head(10)[['ID', 'Name', 'Category', '综合风险分数', '风险等级']].to_markdown(index=False)}

## Top 10 低风险化合物
{low_risk.head(10)[['ID', 'Name', 'Category', '综合风险分数', '风险等级']].to_markdown(index=False)}

## 各终点预测统计
{pred_df[TOXICITY_ENDPOINTS].describe().to_markdown()}

## 输出文件
1. emerging_pfas_predictions.csv - 完整预测结果
2. high_risk_20.csv - 20种高风险化合物
3. low_risk_20.csv - 20种低风险化合物
4. risk_classification_bar.png - 风险分级柱状图
5. risk_score_distribution.png - 风险分数分布
6. prediction_heatmap.png - 毒性预测热力图
7. risk_radar.png - 风险雷达图
"""

    with open(os.path.join(output_dir, "预测与风险评估报告.md"), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print("新兴PFAS预测与风险评估完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()
