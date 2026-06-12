"""
模型优化：更细粒度超参数搜索 + Stacking集成 + 更严格验证
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

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r'E:\桌面\项目'
ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ============================================================
# 高级特征工程
# ============================================================
def advanced_feature_engineering(df):
    """高级特征工程：添加更多有意义的特征"""
    print("\n高级特征工程...")

    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem, MACCSkeys, rdMolDescriptors

    desc_data = []
    valid_indices = []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue

        features = {}

        # 基本描述符
        features['MolWt'] = Descriptors.MolWt(mol)
        features['LogP'] = Descriptors.MolLogP(mol)
        features['TPSA'] = Descriptors.TPSA(mol)
        features['HBD'] = Descriptors.NumHDonors(mol)
        features['HBA'] = Descriptors.NumHAcceptors(mol)
        features['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        features['HeavyAtoms'] = mol.GetNumHeavyAtoms()
        features['RingCount'] = Descriptors.RingCount(mol)
        features['AromaticRings'] = Descriptors.NumAromaticRings(mol)

        # 氟相关特征（PFAS特异性）
        n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
        n_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        n_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
        n_s = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)

        features['FluorineCount'] = n_f
        features['CarbonCount'] = n_c
        features['OxygenCount'] = n_o
        features['SulfurCount'] = n_s
        features['F_C_Ratio'] = n_f / max(n_c, 1)
        features['F_HeavyRatio'] = n_f / max(mol.GetNumHeavyAtoms(), 1)
        features['CF2_Count'] = row['SMILES'].count('C(F)(F)')
        features['CF3_Count'] = row['SMILES'].count('C(F)(F)F')

        # 电子特征
        features['MR'] = Descriptors.MolMR(mol)
        features['LabuteASA'] = Descriptors.LabuteASA(mol)
        features['BalabanJ'] = Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) else 0

        # 复杂度特征
        features['BertzCT'] = Descriptors.BertzCT(mol)
        features['Chi0'] = Descriptors.Chi0(mol)
        features['Chi1'] = Descriptors.Chi1(mol)
        features['HallKierAlpha'] = Descriptors.HallKierAlpha(mol)

        # Morgan指纹（512位）
        try:
            morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512))
            for i, bit in enumerate(morgan):
                features[f'Morgan_{i}'] = bit
        except:
            for i in range(512):
                features[f'Morgan_{i}'] = 0

        # MACCS指纹（167位）
        try:
            maccs = list(MACCSkeys.GenMACCSKeys(mol))
            for i, bit in enumerate(maccs):
                features[f'MACCS_{i}'] = bit
        except:
            for i in range(167):
                features[f'MACCS_{i}'] = 0

        desc_data.append(features)
        valid_indices.append(idx)

        if (idx + 1) % 1000 == 0:
            print(f"  已处理 {idx+1}/{len(df)}")

    desc_df = pd.DataFrame(desc_data, index=valid_indices)

    # 移除常数列
    desc_df = desc_df.loc[:, desc_df.std() > 0.001]

    # 移除高相关列
    corr = desc_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    desc_df = desc_df.drop(columns=to_drop)

    print(f"  特征维度: {desc_df.shape[1]}")
    return df.loc[valid_indices].reset_index(drop=True), desc_df


# ============================================================
# 超参数优化（更细粒度）
# ============================================================
def get_optimized_models():
    """获取优化后的模型和超参数空间"""
    models = {
        'LR': {
            'model': LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'),
            'params': {
                'C': [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42, class_weight='balanced'),
            'params': {
                'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        },
        'RF': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        },
        'XGBoost': {
            'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'scale_pos_weight': [1, 2, 3, 5]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced'),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 9, -1],
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'num_leaves': [15, 31, 50, 100],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1.0],
                'reg_lambda': [0, 0.1, 0.5, 1.0]
            }
        },
        'GBDT': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.001, 0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    return models


# ============================================================
# Stacking 集成
# ============================================================
def build_stacking_model():
    """构建Stacking集成模型"""
    base_estimators = [
        ('lr', LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced', C=1)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('xgb', xgb.XGBClassifier(n_estimators=200, random_state=42, eval_metric='logloss', use_label_encoder=False)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1, class_weight='balanced')),
    ]

    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=10000, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )
    return stacking


# ============================================================
# 主优化流程
# ============================================================
def main():
    print("\n" + "★"*70)
    print("  模型优化：超参数搜索 + Stacking集成 + 严格验证")
    print("★"*70)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'raw', 'tox21_real_data.csv'))
    if 'smiles' in df.columns:
        df = df.rename(columns={'smiles': 'SMILES'})
    df = df.dropna(subset=['SMILES'])
    print(f"  化合物数: {len(df)}")

    # 高级特征工程
    df, desc_df = advanced_feature_engineering(df)

    # 数据预处理
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(desc_df.values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存预处理器
    model_dir = os.path.join(PROJECT_DIR, 'models', 'qsar')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.joblib'))
    joblib.dump(imputer, os.path.join(model_dir, 'feature_imputer.joblib'))

    output_dir = os.path.join(PROJECT_DIR, '02_QSAR模型')
    os.makedirs(output_dir, exist_ok=True)

    # 获取优化模型
    models = get_optimized_models()

    all_results = {}

    for ep in ENDPOINTS:
        if ep not in df.columns:
            continue

        y = df[ep].values.astype(float)
        valid_mask = ~np.isnan(y)
        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask].astype(int)

        if len(np.unique(y_valid)) < 2:
            continue

        print(f"\n{'='*70}")
        print(f"  [{ep}] 样本数={len(y_valid)}, 活性={y_valid.sum()} ({y_valid.mean():.1%})")
        print(f"{'='*70}")

        # 划分数据
        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
        )

        # 特征选择
        selector = SelectKBest(f_classif, k=150)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)

        joblib.dump(selector, os.path.join(model_dir, f'selector_{ep}.joblib'))

        results = {}
        best_models = {}

        # 训练单模型
        for name, model_info in models.items():
            print(f"\n  训练 {name}...")
            try:
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    model_info['model'], model_info['params'],
                    cv=cv, scoring='roc_auc', n_jobs=-1, refit=True,
                    verbose=0
                )
                grid.fit(X_train_sel, y_train)

                y_pred = grid.predict(X_test_sel)
                y_proba = grid.predict_proba(X_test_sel)[:, 1]

                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, zero_division=0),
                    'Recall': recall_score(y_test, y_pred, zero_division=0),
                    'F1': f1_score(y_test, y_pred, zero_division=0),
                    'ROC-AUC': roc_auc_score(y_test, y_proba),
                    'PR-AUC': average_precision_score(y_test, y_proba),
                    'CV_Score': grid.best_score_,
                    'Best_Params': str(grid.best_params_),
                }

                results[name] = metrics
                best_models[name] = grid.best_estimator_

                joblib.dump(grid.best_estimator_, os.path.join(model_dir, f'qsar_{ep}_{name}.joblib'))

                print(f"    AUC={metrics['ROC-AUC']:.3f} F1={metrics['F1']:.3f} CV={grid.best_score_:.3f}")

            except Exception as e:
                print(f"    失败: {e}")

        # 训练Stacking集成
        print(f"\n  训练 Stacking集成...")
        try:
            stacking = build_stacking_model()
            stacking.fit(X_train_sel, y_train)

            y_pred = stacking.predict(X_test_sel)
            y_proba = stacking.predict_proba(X_test_sel)[:, 1]

            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'PR-AUC': average_precision_score(y_test, y_proba),
            }

            results['Stacking'] = metrics
            best_models['Stacking'] = stacking

            joblib.dump(stacking, os.path.join(model_dir, f'qsar_{ep}_Stacking.joblib'))

            print(f"    AUC={metrics['ROC-AUC']:.3f} F1={metrics['F1']:.3f}")

        except Exception as e:
            print(f"    Stacking失败: {e}")

        all_results[ep] = results

        # 绘制ROC曲线
        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        for i, (name, model) in enumerate(best_models.items()):
            try:
                y_proba = model.predict_proba(X_test_sel)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc = roc_auc_score(y_test, y_proba)
                ax.plot(fpr, tpr, color=colors[i % len(colors)],
                        label=f'{name} (AUC={auc:.3f})', linewidth=2)
            except:
                pass

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curves - {ep} (Optimized)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'roc_{ep}_optimized.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 综合热力图
    if all_results:
        all_models = set()
        for ep in all_results:
            all_models.update(all_results[ep].keys())
        all_models = sorted(all_models)

        heatmap_data = pd.DataFrame(index=list(all_results.keys()), columns=all_models)
        for ep in all_results:
            for model in all_results[ep]:
                heatmap_data.loc[ep, model] = all_results[ep][model]['ROC-AUC']
        heatmap_data = heatmap_data.astype(float)

        fig, ax = plt.subplots(figsize=(14, 7))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0)
        ax.set_title('ROC-AUC: Models × Endpoints (Optimized)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'qsar_performance_heatmap_optimized.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 生成报告
    report = "# QSAR 模型优化报告\n\n"
    report += "## 优化策略\n"
    report += "1. **高级特征工程**: 添加PFAS特异性特征（氟原子数、F/C比、CF2/CF3计数）\n"
    report += "2. **更细粒度超参数搜索**: 扩大搜索范围，更多参数组合\n"
    report += "3. **Stacking集成**: LR+RF+XGBoost+LightGBM作为基模型，LR作为元模型\n"
    report += "4. **更严格验证**: 5折分层交叉验证\n\n"

    report += "## 数据集\n"
    report += f"- 化合物数: {len(df)}\n"
    report += f"- 特征维度: {desc_df.shape[1]}\n"
    report += f"- 毒性终点: {len(ENDPOINTS)}\n\n"

    report += "## 模型性能\n\n"
    for ep in all_results:
        report += f"### {ep}\n\n"
        report += "| 模型 | ROC-AUC | F1 | Precision | Recall | CV-Score |\n"
        report += "|------|---------|-----|-----------|--------|----------|\n"
        for model, metrics in all_results[ep].items():
            report += f"| {model} | {metrics['ROC-AUC']:.3f} | {metrics['F1']:.3f} | {metrics['Precision']:.3f} | {metrics['Recall']:.3f} | {metrics.get('CV_Score', 'N/A')} |\n"

        best = max(all_results[ep], key=lambda k: all_results[ep][k]['ROC-AUC'])
        report += f"\n**最佳模型**: {best} (AUC={all_results[ep][best]['ROC-AUC']:.3f})\n\n"

    with open(os.path.join(output_dir, 'QSAR优化报告.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'★'*70}")
    print(f"  优化完成！")
    print(f"{'★'*70}")

if __name__ == '__main__':
    main()
