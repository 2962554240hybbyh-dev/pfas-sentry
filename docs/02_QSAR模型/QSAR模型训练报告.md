# QSAR 模型训练结果总结


## NR-AR

|          |   Accuracy |   Precision |   Recall |   F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|-----:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.555556 |    0        |      0   |  0   |  0.571429 | 0.333333 | {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}                               |     0.7275 |
| SVM      |   0.555556 |    0        |      0   |  0   |  0.714286 | 0.45     | {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}                                 |     0.66   |
| RF       |   0.555556 |    0        |      0   |  0   |  0.357143 | 0.277778 | {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}                  |     0.575  |
| XGBoost  |   0.555556 |    0        |      0   |  0   |  0.642857 | 0.366667 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |     0.6025 |
| LightGBM |   0.777778 |    0        |      0   |  0   |  0.5      | 0.222222 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |     0.5    |
| GBDT     |   0.666667 |    0.333333 |      0.5 |  0.4 |  0.571429 | 0.333333 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}                   |     0.5325 |

**最佳模型**: SVM (ROC-AUC=0.7143)


## NR-AR-LBD

|          |   Accuracy |   Precision |   Recall |   F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|-----:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.666667 |           0 | 0        |  0   |  0.5      | 0.333333 | {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}                            |      0.5   |
| SVM      |   0.666667 |           0 | 0        |  0   |  0.277778 | 0.31746  | {'C': 0.1, 'gamma': 'scale', 'kernel': 'rbf'}                                  |      0.44  |
| RF       |   0.777778 |           1 | 0.333333 |  0.5 |  0.722222 | 0.698413 | {'max_depth': 5, 'min_samples_split': 2, 'n_estimators': 200}                  |      0.39  |
| XGBoost  |   0.777778 |           1 | 0.333333 |  0.5 |  0.722222 | 0.666667 | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}  |      0.675 |
| LightGBM |   0.666667 |           0 | 0        |  0   |  0.5      | 0.333333 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |      0.5   |
| GBDT     |   0.777778 |           1 | 0.333333 |  0.5 |  0.361111 | 0.527778 | {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100}                    |      0.375 |

**最佳模型**: RF (ROC-AUC=0.7222)


## NR-AhR

|          |   Accuracy |   Precision |   Recall |       F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|---------:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.666667 |    0        | 0        | 0        |  0.5      | 0.333333 | {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}                            |   0.5      |
| SVM      |   0.555556 |    0.333333 | 0.333333 | 0.333333 |  0.5      | 0.387302 | {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}                                 |   0.558333 |
| RF       |   0.555556 |    0        | 0        | 0        |  0.166667 | 0.275794 | {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}                  |   0.45     |
| XGBoost  |   0.222222 |    0        | 0        | 0        |  0.277778 | 0.302778 | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8}  |   0.375    |
| LightGBM |   0.666667 |    0        | 0        | 0        |  0.5      | 0.333333 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |   0.5      |
| GBDT     |   0.222222 |    0        | 0        | 0        |  0.222222 | 0.286905 | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}                    |   0.4125   |

**最佳模型**: LR (ROC-AUC=0.5000)


## SR-HSE

|          |   Accuracy |   Precision |   Recall |       F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|---------:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.555556 |    0.333333 | 0.333333 | 0.333333 |  0.666667 | 0.642857 | {'C': 1, 'penalty': 'l2', 'solver': 'liblinear'}                               |   0.625    |
| SVM      |   0.555556 |    0        | 0        | 0        |  0.5      | 0.402778 | {'C': 10, 'gamma': 'scale', 'kernel': 'linear'}                                |   0.625    |
| RF       |   0.666667 |    0.5      | 0.333333 | 0.4      |  0.638889 | 0.609524 | {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 100}                 |   0.508333 |
| XGBoost  |   0.666667 |    0.5      | 0.333333 | 0.4      |  0.333333 | 0.527778 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'subsample': 1.0} |   0.425    |
| LightGBM |   0.666667 |    0        | 0        | 0        |  0.5      | 0.333333 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |   0.5      |
| GBDT     |   0.555556 |    0.333333 | 0.333333 | 0.333333 |  0.5      | 0.555556 | {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}                   |   0.466667 |

**最佳模型**: LR (ROC-AUC=0.6667)


## SR-MMP

|          |   Accuracy |   Precision |   Recall |   F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|-----:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.444444 |        0    |     0    | 0    |      0.2  | 0.365476 | {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}                               |   0.522222 |
| SVM      |   0.333333 |        0    |     0    | 0    |      0.2  | 0.364087 | {'C': 0.1, 'gamma': 'scale', 'kernel': 'linear'}                               |   0.544444 |
| RF       |   0.444444 |        0    |     0    | 0    |      0.2  | 0.364087 | {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}                  |   0.427778 |
| XGBoost  |   0.333333 |        0.25 |     0.25 | 0.25 |      0.25 | 0.388194 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200, 'subsample': 1.0} |   0.525    |
| LightGBM |   0.555556 |        0    |     0    | 0    |      0.5  | 0.444444 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |   0.5      |
| GBDT     |   0.333333 |        0.25 |     0.25 | 0.25 |      0.2  | 0.367361 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 200}                   |   0.497222 |

**最佳模型**: LightGBM (ROC-AUC=0.5000)


## SR-p53

|          |   Accuracy |   Precision |   Recall |       F1 |   ROC-AUC |   PR-AUC | Best_Params                                                                    |   CV_Score |
|:---------|-----------:|------------:|---------:|---------:|----------:|---------:|:-------------------------------------------------------------------------------|-----------:|
| LR       |   0.777778 |         0   |        0 | 0        |  0.5      | 0.222222 | {'C': 0.01, 'penalty': 'l1', 'solver': 'liblinear'}                            |      0.5   |
| SVM      |   0.777778 |         0.5 |        1 | 0.666667 |  0.142857 | 0.196429 | {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}                                 |      0.405 |
| RF       |   0.777778 |         0   |        0 | 0        |  0.285714 | 0.22619  | {'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 100}                  |      0.1   |
| XGBoost  |   0.777778 |         0   |        0 | 0        |  0.285714 | 0.225    | {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200, 'subsample': 0.8}  |      0.22  |
| LightGBM |   0.777778 |         0   |        0 | 0        |  0.5      | 0.222222 | {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 100, 'num_leaves': 31} |      0.5   |
| GBDT     |   0.777778 |         0   |        0 | 0        |  0.571429 | 0.325    | {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100}                   |      0.235 |

**最佳模型**: GBDT (ROC-AUC=0.5714)


## 关键特征（SHAP 分析）


### NR-AR


### NR-AR-LBD


### NR-AhR


### SR-HSE


### SR-MMP

- MACCS_164: 0.0000
- MACCS_163: 0.0000
- MACCS_162: 0.0000
- MACCS_159: 0.0000
- MACCS_158: 0.0000
- MACCS_157: 0.0000
- MACCS_156: 0.0000
- MACCS_153: 0.0000
- MACCS_152: 0.0000
- MACCS_151: 0.0000

### SR-p53
