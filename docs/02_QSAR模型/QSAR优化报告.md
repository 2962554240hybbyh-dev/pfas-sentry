# QSAR 优化报告

化合物数: 7823, 特征数: 267

## NR-AR

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.728 | 0.539 |
| LightGBM | 0.747 | 0.463 |
| RF | 0.754 | 0.539 |
| Stacking | 0.741 | 0.552 |

最佳: RF

## NR-AR-LBD

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.866 | 0.587 |
| LightGBM | 0.852 | 0.721 |
| RF | 0.863 | 0.597 |
| Stacking | 0.874 | 0.571 |

最佳: Stacking

## NR-AhR

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.880 | 0.534 |
| LightGBM | 0.863 | 0.556 |
| RF | 0.872 | 0.512 |
| Stacking | 0.870 | 0.482 |

最佳: XGBoost

## SR-HSE

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.805 | 0.286 |
| LightGBM | 0.758 | 0.379 |
| RF | 0.787 | 0.303 |
| Stacking | 0.785 | 0.311 |

最佳: XGBoost

## SR-MMP

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.916 | 0.616 |
| LightGBM | 0.910 | 0.653 |
| RF | 0.900 | 0.569 |
| Stacking | 0.909 | 0.566 |

最佳: XGBoost

## SR-p53

| 模型 | ROC-AUC | F1 |
|------|---------|-----|
| XGBoost | 0.842 | 0.291 |
| LightGBM | 0.819 | 0.319 |
| RF | 0.856 | 0.218 |
| Stacking | 0.841 | 0.233 |

最佳: RF

