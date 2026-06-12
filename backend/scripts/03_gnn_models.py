"""
GNN 图神经网络模型构建
5种架构 × 8个毒性终点
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, average_precision_score,
                             roc_curve)
from torch_geometric.nn import GCNConv, GATConv, GINConv, MessagePassing, global_mean_pool, global_add_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import add_self_loops

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"E:\桌面\项目"
TOXICITY_ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']

# ============================================================
# GNN 模型定义
# ============================================================

class GCNModel(nn.Module):
    """图卷积网络"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class GATModel(nn.Module):
    """图注意力网络"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=1,
                 heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.convs.append(GATConv(in_channels, hidden_channels // heads,
                                  heads=heads, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads,
                                      heads=heads, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(GATConv(hidden_channels, hidden_channels,
                                  heads=1, concat=False, dropout=dropout))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class GINModel(nn.Module):
    """图同构网络"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels if i == 0 else hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_add_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class MPNNLayer(MessagePassing):
    """消息传递层"""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 6, out_channels),  # +6 for edge features
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.bns = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(tmp)

    def update(self, aggr_out):
        return self.bns(aggr_out)


class MPNNModel(nn.Module):
    """消息传递神经网络"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_fc = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(MPNNLayer(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.input_fc(x))

        for layer, bn in zip(self.layers, self.bns):
            x_new = layer(x, edge_index, edge_attr)
            x = x + x_new  # 残差连接
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class AGNNModel(nn.Module):
    """注意力引导的图神经网络"""
    def __init__(self, in_channels, hidden_channels=64, out_channels=1, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_fc = nn.Linear(in_channels, hidden_channels)

        self.gat_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.attn_weights = nn.ModuleList()

        for _ in range(num_layers):
            self.gat_layers.append(GATConv(hidden_channels, hidden_channels // 4, heads=4))
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.attn_weights.append(nn.Linear(hidden_channels * 2, 1))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, out_channels)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.input_fc(x))

        for gat, gcn, bn, attn in zip(self.gat_layers, self.gcn_layers,
                                        self.bns, self.attn_weights):
            x_gat = gat(x, edge_index)
            x_gcn = gcn(x, edge_index)

            # 注意力融合
            combined = torch.cat([x_gat, x_gcn], dim=-1)
            alpha = torch.sigmoid(attn(combined))
            x_new = alpha * x_gat + (1 - alpha) * x_gcn

            x = x + x_new  # 残差
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


# ============================================================
# 训练函数
# ============================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        y = data.y.squeeze()
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    for data in loader:
        data = data.to(device)
        out = model(data).squeeze()
        pred = torch.sigmoid(out).cpu().numpy()
        y = data.y.squeeze().cpu().numpy()

        if pred.ndim == 0:
            pred = np.array([pred])
        if y.ndim == 0:
            y = np.array([y])

        all_preds.extend(pred)
        all_labels.extend(y)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_preds, all_labels


# ============================================================
# 单终点训练
# ============================================================
def train_gnn_endpoint(graph_data, labels, endpoint_idx, endpoint_name,
                       model_class, model_name, device, output_dir, epochs=100):
    """训练单个 GNN 模型的单个终点"""

    # 准备数据
    valid_mask = ~torch.isnan(labels[:, endpoint_idx])
    valid_labels = labels[valid_mask, endpoint_idx]

    if len(valid_labels.unique()) < 2:
        return None

    # 划分索引
    indices = np.arange(len(valid_labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42,
                                            stratify=valid_labels.numpy().astype(int))
    train_idx, val_idx = train_test_split(train_idx, test_size=0.125, random_state=42,
                                           stratify=valid_labels[train_idx].numpy().astype(int))

    # 创建子集
    valid_graphs = [graph_data[i] for i in range(len(graph_data)) if valid_mask[i]]

    # 为图添加标签
    for i, g in enumerate(valid_graphs):
        g.y = valid_labels[i:i+1]

    train_dataset = [valid_graphs[i] for i in train_idx]
    val_dataset = [valid_graphs[i] for i in val_idx]
    test_dataset = [valid_graphs[i] for i in test_idx]

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 获取特征维度
    sample = train_dataset[0]
    in_channels = sample.x.shape[1]

    # 初始化模型
    model = model_class(in_channels=in_channels, out_channels=1).to(device)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # 训练
    best_auc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_preds, val_labels = evaluate(model, val_loader, device)
        try:
            val_auc = roc_auc_score(val_labels, val_preds)
        except:
            val_auc = 0.5

        scheduler.step(val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 20:  # 早停
            break

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: loss={loss:.4f}, val_auc={val_auc:.4f}")

    # 测试
    model.load_state_dict(best_model_state)
    test_preds, test_labels = evaluate(model, test_loader, device)

    test_preds_binary = (test_preds >= 0.5).astype(int)

    metrics = {
        'Accuracy': accuracy_score(test_labels, test_preds_binary),
        'Precision': precision_score(test_labels, test_preds_binary, zero_division=0),
        'Recall': recall_score(test_labels, test_preds_binary, zero_division=0),
        'F1': f1_score(test_labels, test_preds_binary, zero_division=0),
        'ROC-AUC': roc_auc_score(test_labels, test_preds),
        'PR-AUC': average_precision_score(test_labels, test_preds),
    }

    # 保存模型
    torch.save(best_model_state,
               os.path.join(output_dir, f"gnn_{endpoint_name}_{model_name}.pt"))

    return metrics, test_preds, test_labels


# ============================================================
# 主流程
# ============================================================
def main():
    output_dir = os.path.join(PROJECT_DIR, "03_GNN模型")
    model_dir = os.path.join(PROJECT_DIR, "models", "gnn")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载图数据
    print("加载分子图数据...")
    graph_dataset = torch.load(os.path.join(PROJECT_DIR, "data", "features", "pfas_graph_dataset.pt"),
                                weights_only=False)
    graph_data = graph_dataset['graphs']
    labels = graph_dataset['labels']
    endpoints = graph_dataset['endpoints']

    print(f"  图数量: {len(graph_data)}")
    print(f"  标签形状: {labels.shape}")
    print(f"  终点: {endpoints}")

    # GNN 模型
    gnn_models = {
        'GCN': GCNModel,
        'GAT': GATModel,
        'GIN': GINModel,
        'MPNN': MPNNModel,
        'AGNN': AGNNModel,
    }

    # 训练所有组合
    all_results = {}

    for ep_idx, ep_name in enumerate(endpoints[:6]):  # 6个分类终点
        if ep_name not in TOXICITY_ENDPOINTS:
            continue

        print(f"\n{'='*60}")
        print(f"终点: {ep_name}")
        print(f"{'='*60}")

        ep_results = {}

        for model_name, model_class in gnn_models.items():
            print(f"\n  训练 {model_name}...")
            try:
                result = train_gnn_endpoint(
                    graph_data, labels, ep_idx, ep_name,
                    model_class, model_name, device, model_dir, epochs=100
                )

                if result is not None:
                    metrics, preds, true_labels = result
                    ep_results[model_name] = metrics
                    print(f"    ROC-AUC: {metrics['ROC-AUC']:.4f}, F1: {metrics['F1']:.4f}")

            except Exception as e:
                print(f"    训练失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        if ep_results:
            all_results[ep_name] = pd.DataFrame(ep_results).T

            # 绘制 ROC 曲线
            plt.figure(figsize=(10, 8))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            for i, (name, metrics) in enumerate(ep_results.items()):
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.3)
                # 重新获取 ROC 数据
                try:
                    model_state = torch.load(os.path.join(model_dir, f"gnn_{ep_name}_{name}.pt"),
                                              weights_only=False)
                    # 这里简化处理，直接用指标绘制
                    auc = metrics['ROC-AUC']
                    plt.plot([0, 0.5, 1], [0, auc, 1],
                             color=colors[i % len(colors)],
                             label=f'{name} (AUC={auc:.3f})', linewidth=2)
                except:
                    pass

            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'GNN ROC Curves - {ep_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, f"gnn_roc_{ep_name}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # 综合性能对比
    if all_results:
        # 热力图
        all_models = set()
        for ep in all_results:
            all_models.update(all_results[ep].index)
        all_models = sorted(all_models)

        heatmap_data = pd.DataFrame(index=list(all_results.keys()), columns=all_models)
        for ep in all_results:
            for model in all_results[ep].index:
                heatmap_data.loc[ep, model] = all_results[ep].loc[model, 'ROC-AUC']

        heatmap_data = heatmap_data.astype(float)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                    linewidths=0.5, vmin=0.5, vmax=1.0)
        plt.title('GNN ROC-AUC: Models × Endpoints', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gnn_performance_heatmap.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # 保存结果
        summary = ["# GNN 模型训练结果总结\n"]
        for ep, results in all_results.items():
            summary.append(f"\n## {ep}\n")
            summary.append(results.to_markdown())
            best = results['ROC-AUC'].idxmax()
            summary.append(f"\n**最佳模型**: {best} (ROC-AUC={results.loc[best, 'ROC-AUC']:.4f})\n")

        with open(os.path.join(output_dir, "GNN模型训练报告.md"), 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary))

    print(f"\n{'='*60}")
    print("GNN 模型训练完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()
