"""
快速优化：Stacking集成 + 关键特征工程
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif

import xgboost as xgb
import lightgbm as lgb

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys

PROJECT_DIR = r'E:\桌面\项目'
ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

def main():
    print("快速模型优化...")

    # 加载数据
    df = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'raw', 'tox21_real_data.csv'))
    if 'smiles' in df.columns:
        df = df.rename(columns={'smiles': 'SMILES'})
    df = df.dropna(subset=['SMILES'])

    # 快速特征工程
    desc_data = []
    valid_indices = []

    for idx, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol is None:
            continue

        features = {}
        features['MolWt'] = Descriptors.MolWt(mol)
        features['LogP'] = Descriptors.MolLogP(mol)
        features['TPSA'] = Descriptors.TPSA(mol)
        features['HBD'] = Descriptors.NumHDonors(mol)
        features['HBA'] = Descriptors.NumHAcceptors(mol)
        features['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        features['HeavyAtoms'] = mol.GetNumHeavyAtoms()

        # PFAS特异性特征
        n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
        n_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        features['FluorineCount'] = n_f
        features['F_C_Ratio'] = n_f / max(n_c, 1)
        features['F_HeavyRatio'] = n_f / max(mol.GetNumHeavyAtoms(), 1)
        features['CF2_Count'] = row['SMILES'].count('C(F)(F)')

        # Morgan指纹（256位）
        try:
            morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))
            for i, bit in enumerate(morgan):
                features[f'Morgan_{i}'] = bit
        except:
            for i in range(256):
                features[f'Morgan_{i}'] = 0

        desc_data.append(features)
        valid_indices.append(idx)

        if (idx + 1) % 2000 == 0:
            print(f"  已处理 {idx+1}/{len(df)}")

    desc_df = pd.DataFrame(desc_data, index=valid_indices)
    desc_df = desc_df.loc[:, desc_df.std() > 0.001]

    df_valid = df.loc[valid_indices].reset_index(drop=True)

    # 预处理
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(desc_df.values)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 保存
    model_dir = os.path.join(PROJECT_DIR, 'models', 'qsar')
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, 'feature_scaler.joblib'))
    joblib.dump(imputer, os.path.join(model_dir, 'feature_imputer.joblib'))

    output_dir = os.path.join(PROJECT_DIR, '02_QSAR模型')
    os.makedirs(output_dir, exist_ok=True)

    all_results = {}

    for ep in ENDPOINTS:
        if ep not in df_valid.columns:
            continue

        y = df_valid[ep].values.astype(float)
        valid_mask = ~np.isnan(y)
        X_valid = X_scaled[valid_mask]
        y_valid = y[valid_mask].astype(int)

        if len(np.unique(y_valid)) < 2:
            continue

        print(f"\n  [{ep}] n={len(y_valid)}, active={y_valid.sum()} ({y_valid.mean():.1%})")

        X_train, X_test, y_train, y_test = train_test_split(
            X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
        )

        selector = SelectKBest(f_classif, k=100)
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_test_sel = selector.transform(X_test)
        joblib.dump(selector, os.path.join(model_dir, f'selector_{ep}.joblib'))

        results = {}

        # 1. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2,
            random_state=42, eval_metric='logloss', use_label_encoder=False
        )
        xgb_model.fit(X_train_sel, y_train)
        y_proba = xgb_model.predict_proba(X_test_sel)[:, 1]
        results['XGBoost'] = {'ROC-AUC': roc_auc_score(y_test, y_proba), 'F1': f1_score(y_test, (y_proba>0.5).astype(int))}
        joblib.dump(xgb_model, os.path.join(model_dir, f'qsar_{ep}_XGBoost.joblib'))

        # 2. LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=7, learning_rate=0.05,
            num_leaves=50, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1
        )
        lgb_model.fit(X_train_sel, y_train)
        y_proba = lgb_model.predict_proba(X_test_sel)[:, 1]
        results['LightGBM'] = {'ROC-AUC': roc_auc_score(y_test, y_proba), 'F1': f1_score(y_test, (y_proba>0.5).astype(int))}
        joblib.dump(lgb_model, os.path.join(model_dir, f'qsar_{ep}_LightGBM.joblib'))

        # 3. Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_split=5,
            class_weight='balanced', random_state=42
        )
        rf_model.fit(X_train_sel, y_train)
        y_proba = rf_model.predict_proba(X_test_sel)[:, 1]
        results['RF'] = {'ROC-AUC': roc_auc_score(y_test, y_proba), 'F1': f1_score(y_test, (y_proba>0.5).astype(int))}
        joblib.dump(rf_model, os.path.join(model_dir, f'qsar_{ep}_RF.joblib'))

        # 4. Stacking集成
        print(f"    训练Stacking...")
        base_estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss', use_label_encoder=False)),
            ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)),
        ]
        stacking = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=10000, random_state=42),
            cv=5, stack_method='predict_proba', n_jobs=-1
        )
        stacking.fit(X_train_sel, y_train)
        y_proba = stacking.predict_proba(X_test_sel)[:, 1]
        results['Stacking'] = {'ROC-AUC': roc_auc_score(y_test, y_proba), 'F1': f1_score(y_test, (y_proba>0.5).astype(int))}
        joblib.dump(stacking, os.path.join(model_dir, f'qsar_{ep}_Stacking.joblib'))

        all_results[ep] = results

        for name, m in results.items():
            print(f"    {name}: AUC={m['ROC-AUC']:.3f}, F1={m['F1']:.3f}")

    # 总结
    print("\n" + "="*70)
    print("  优化结果总结")
    print("="*70)
    for ep in all_results:
        best = max(all_results[ep], key=lambda k: all_results[ep][k]['ROC-AUC'])
        print(f"  {ep}: 最佳={best} (AUC={all_results[ep][best]['ROC-AUC']:.3f})")

    # 保存报告
    report = "# QSAR 优化报告\n\n"
    report += f"化合物数: {len(df_valid)}, 特征数: {desc_df.shape[1]}\n\n"
    for ep in all_results:
        report += f"## {ep}\n\n"
        report += "| 模型 | ROC-AUC | F1 |\n|------|---------|-----|\n"
        for name, m in all_results[ep].items():
            report += f"| {name} | {m['ROC-AUC']:.3f} | {m['F1']:.3f} |\n"
        best = max(all_results[ep], key=lambda k: all_results[ep][k]['ROC-AUC'])
        report += f"\n最佳: {best}\n\n"

    with open(os.path.join(output_dir, 'QSAR优化报告.md'), 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n优化完成！")

if __name__ == '__main__':
    main()
