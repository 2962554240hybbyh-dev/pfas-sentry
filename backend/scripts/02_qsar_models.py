"""
QSAR 传统机器学习模型构建
6种算法 × 8个毒性终点
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
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, average_precision_score,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import KNNImputer
import joblib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# 毒性终点定义
# ============================================================
TOXICITY_ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
REGRESSION_ENDPOINTS = ['BCF']
ALL_ENDPOINTS = TOXICITY_ENDPOINTS + REGRESSION_ENDPOINTS

# ============================================================
# 数据加载与预处理
# ============================================================
def load_and_prepare_data():
    """加载清洗后的数据"""
    print("加载数据...")
    df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "cleaned", "pfas_clean_data.csv"))
    desc_df = pd.read_csv(os.path.join(PROJECT_DIR, "data", "features", "pfas_descriptors.csv"))

    print(f"  主数据形状: {df.shape}")
    print(f"  描述符矩阵形状: {desc_df.shape}")

    # 对齐索号
    common_idx = df.index.intersection(desc_df.index)
    df = df.loc[common_idx].reset_index(drop=True)
    desc_df = desc_df.loc[common_idx].reset_index(drop=True)

    # 填充缺失值
    imputer = KNNImputer(n_neighbors=5)
    desc_filled = pd.DataFrame(
        imputer.fit_transform(desc_df),
        columns=desc_df.columns
    )

    # 标准化
    scaler = StandardScaler()
    desc_scaled = pd.DataFrame(
        scaler.fit_transform(desc_filled),
        columns=desc_filled.columns
    )

    return df, desc_scaled, scaler

# ============================================================
# 模型定义与超参数
# ============================================================
def get_models():
    """定义6种模型及其超参数搜索空间"""
    models = {
        'LR': {
            'model': LogisticRegression(max_iter=5000, random_state=42),
            'params': {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto']
            }
        },
        'RF': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        'XGBoost': {
            'model': None,  # 动态导入
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': None,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        },
        'GBDT': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
        }
    }

    # 动态导入 XGBoost 和 LightGBM
    try:
        import xgboost as xgb
        models['XGBoost']['model'] = xgb.XGBClassifier(
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
    except ImportError:
        print("  警告: XGBoost 未安装，跳过")
        del models['XGBoost']

    try:
        import lightgbm as lgb
        models['LightGBM']['model'] = lgb.LGBMClassifier(random_state=42, verbose=-1)
    except ImportError:
        print("  警告: LightGBM 未安装，跳过")
        del models['LightGBM']

    return models

# ============================================================
# 单终点模型训练
# ============================================================
def train_endpoint_models(X, y, endpoint, models, output_dir):
    """训练一个终点的所有模型"""
    print(f"\n{'='*50}")
    print(f"训练终点: {endpoint}")
    print(f"{'='*50}")

    # 检查类别分布
    unique, counts = np.unique(y[~np.isnan(y)], return_counts=True)
    if len(unique) < 2:
        print(f"  跳过: 只有一个类别")
        return None

    print(f"  类别分布: {dict(zip(unique.astype(int), counts))}")

    # 处理缺失值
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask].astype(int)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
    )

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train  # 0.125 * 0.8 = 0.1
    )

    print(f"  训练集: {len(y_train_sub)}, 验证集: {len(y_val)}, 测试集: {len(y_test)}")

    results = {}
    best_models = {}

    for name, model_info in models.items():
        print(f"\n  训练 {name}...")
        try:
            # 网格搜索
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                model_info['model'],
                model_info['params'],
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                refit=True
            )

            grid.fit(X_train_sub, y_train_sub)

            # 预测
            y_pred = grid.predict(X_test)
            y_proba = grid.predict_proba(X_test)[:, 1]

            # 计算指标
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_proba),
                'PR-AUC': average_precision_score(y_test, y_proba),
                'Best_Params': str(grid.best_params_),
                'CV_Score': grid.best_score_
            }

            results[name] = metrics
            best_models[name] = grid.best_estimator_

            print(f"    ROC-AUC: {metrics['ROC-AUC']:.4f}, F1: {metrics['F1']:.4f}")

            # 保存模型
            model_path = os.path.join(output_dir, f"qsar_{endpoint}_{name}.joblib")
            joblib.dump(grid.best_estimator_, model_path)

        except Exception as e:
            print(f"    训练失败: {e}")
            continue

    if not results:
        return None

    # 绘制 ROC 曲线
    plot_roc_curves(best_models, X_test, y_test, endpoint, output_dir)

    # 保存结果表
    results_df = pd.DataFrame(results).T
    results_path = os.path.join(output_dir, f"qsar_{endpoint}_results.csv")
    results_df.to_csv(results_path, encoding='utf-8-sig')

    return results_df

# ============================================================
# ROC 曲线绘制
# ============================================================
def plot_roc_curves(models, X_test, y_test, endpoint, output_dir):
    """绘制所有模型的 ROC 曲线"""
    plt.figure(figsize=(10, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, (name, model) in enumerate(models.items()):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc = roc_auc_score(y_test, y_proba)
            plt.plot(fpr, tpr, color=colors[i % len(colors)],
                     label=f'{name} (AUC={auc:.3f})', linewidth=2)
        except:
            continue

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - {endpoint}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    roc_path = os.path.join(output_dir, f"roc_{endpoint}.png")
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# SHAP 特征重要性分析
# ============================================================
def shap_analysis(best_model, X_train, X_test, feature_names, endpoint, output_dir):
    """SHAP 全局和局部解释"""
    import shap

    print(f"\n  SHAP 分析 ({endpoint})...")

    try:
        # 创建 SHAP 解释器
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        # 如果返回的是列表（二分类），取正类
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # 1. 全局特征重要性（前20）
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          max_display=20, show=False)
        plt.title(f'SHAP Feature Importance - {endpoint}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_global_{endpoint}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 2. 局部解释（前3个样本）
        for i in range(min(3, len(X_test))):
            plt.figure(figsize=(12, 4))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[i],
                    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list) else explainer.expected_value[1],
                    data=X_test.iloc[i] if hasattr(X_test, 'iloc') else X_test[i],
                    feature_names=feature_names
                ),
                max_display=15,
                show=False
            )
            plt.title(f'Sample {i+1} Explanation - {endpoint}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"shap_local_{endpoint}_sample{i+1}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

        # 3. 依赖图（最重要的特征）
        plt.figure(figsize=(10, 6))
        top_feature_idx = np.argmax(np.abs(shap_values).mean(axis=0))
        shap.dependence_plot(top_feature_idx, shap_values, X_test,
                             feature_names=feature_names, show=False)
        plt.title(f'SHAP Dependence - {feature_names[top_feature_idx]}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_dependence_{endpoint}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 返回最重要的特征
        importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(importance)[::-1][:20]
        return [(feature_names[i], importance[i]) for i in top_indices]

    except Exception as e:
        print(f"    SHAP 分析失败: {e}")
        return []

# ============================================================
# 综合性能对比图
# ============================================================
def plot_comprehensive_comparison(all_results, output_dir):
    """绘制综合性能对比图"""
    # 1. 所有终点 × 所有模型的 ROC-AUC 热力图
    endpoints = list(all_results.keys())
    all_models = set()
    for ep in endpoints:
        all_models.update(all_results[ep].index)
    all_models = sorted(all_models)

    heatmap_data = pd.DataFrame(index=endpoints, columns=all_models)
    for ep in endpoints:
        for model in all_results[ep].index:
            heatmap_data.loc[ep, model] = all_results[ep].loc[model, 'ROC-AUC']

    heatmap_data = heatmap_data.astype(float)

    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                linewidths=0.5, vmin=0.5, vmax=1.0)
    plt.title('ROC-AUC: Models × Endpoints', fontsize=14)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Endpoint', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qsar_performance_heatmap.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 每个终点的最佳模型对比柱状图
    best_per_endpoint = {}
    for ep in endpoints:
        if len(all_results[ep]) > 0:
            best_model = all_results[ep]['ROC-AUC'].idxmax()
            best_auc = all_results[ep].loc[best_model, 'ROC-AUC']
            best_per_endpoint[ep] = (best_model, best_auc)

    fig, ax = plt.subplots(figsize=(12, 6))
    eps = list(best_per_endpoint.keys())
    aucs = [best_per_endpoint[ep][1] for ep in eps]
    models = [best_per_endpoint[ep][0] for ep in eps]

    bars = ax.bar(eps, aucs, color=sns.color_palette("husl", len(eps)))
    for bar, model in zip(bars, models):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                model, ha='center', va='bottom', fontsize=9, rotation=45)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Best Model per Endpoint (ROC-AUC)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qsar_best_per_endpoint.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 主流程
# ============================================================
def main():
    output_dir = os.path.join(PROJECT_DIR, "02_QSAR模型")
    model_dir = os.path.join(PROJECT_DIR, "models", "qsar")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # 加载数据
    df, desc_scaled, scaler = load_and_prepare_data()

    # 保存 scaler
    joblib.dump(scaler, os.path.join(model_dir, "feature_scaler.joblib"))

    # 获取模型
    models = get_models()
    print(f"\n可用模型: {list(models.keys())}")

    # 训练所有终点
    all_results = {}
    all_top_features = {}

    for endpoint in TOXICITY_ENDPOINTS:
        if endpoint in df.columns:
            y = df[endpoint].values.astype(float)
            X = desc_scaled.copy()

            results = train_endpoint_models(X, y, endpoint, models, model_dir)
            if results is not None:
                all_results[endpoint] = results

                # SHAP 分析（对最佳模型）
                best_name = results['ROC-AUC'].idxmax()
                best_model = joblib.load(os.path.join(model_dir, f"qsar_{endpoint}_{best_name}.joblib"))

                valid_mask = ~np.isnan(y)
                X_valid = X[valid_mask]
                y_valid = y[valid_mask].astype(int)
                _, X_test, _, y_test = train_test_split(
                    X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
                )

                top_features = shap_analysis(
                    best_model, X_valid, X_test,
                    desc_scaled.columns.tolist(), endpoint, output_dir
                )
                all_top_features[endpoint] = top_features

    # 综合对比
    if all_results:
        plot_comprehensive_comparison(all_results, output_dir)

    # 生成总结报告
    summary_lines = ["# QSAR 模型训练结果总结\n"]
    for endpoint, results in all_results.items():
        summary_lines.append(f"\n## {endpoint}\n")
        summary_lines.append(results.to_markdown())
        best = results['ROC-AUC'].idxmax()
        summary_lines.append(f"\n**最佳模型**: {best} (ROC-AUC={results.loc[best, 'ROC-AUC']:.4f})\n")

    if all_top_features:
        summary_lines.append("\n## 关键特征（SHAP 分析）\n")
        for endpoint, features in all_top_features.items():
            summary_lines.append(f"\n### {endpoint}\n")
            for feat, imp in features[:10]:
                summary_lines.append(f"- {feat}: {imp:.4f}")

    summary_path = os.path.join(output_dir, "QSAR模型训练报告.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))

    print(f"\n{'='*60}")
    print("QSAR 模型训练完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")
    print(f"模型目录: {model_dir}")

if __name__ == '__main__':
    main()
