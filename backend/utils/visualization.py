"""
可视化模块
"""
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def fig_to_base64(fig):
    """将matplotlib图表转换为base64字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def generate_bar_chart(predictions, compound_name):
    """生成柱状图"""
    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
    values = [predictions[ep]['prediction'] for ep in endpoints]
    labels = [predictions[ep]['name_cn'] for ep in endpoints]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#ff4444' if v > 0.6 else '#ffaa00' if v > 0.3 else '#44bb44' for v in values]

    bars = ax.bar(range(len(endpoints)), values, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    # 阈值线
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.5, label='高风险阈值')
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='中风险阈值')

    # 数值标签
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(endpoints)))
    ax.set_xticklabels([f"{ep}\n{labels[i]}" for i, ep in enumerate(endpoints)], fontsize=8)
    ax.set_ylabel('毒性概率')
    ax.set_title(f'{compound_name} 毒性预测结果')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    return fig_to_base64(fig)


def generate_radar_chart(predictions):
    """生成雷达图"""
    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
    values = [predictions[ep]['prediction'] for ep in endpoints]
    labels = [predictions[ep]['name_cn'] for ep in endpoints]

    angles = np.linspace(0, 2 * np.pi, len(endpoints), endpoint=False).tolist()
    angles += angles[:1]
    values_plot = values + values[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values_plot, alpha=0.25, color='#ff6b6b')
    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#ff6b6b')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f"{ep}\n{labels[i]}" for i, ep in enumerate(endpoints)], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title('毒性终点分布', fontsize=12, pad=20)
    ax.grid(True)

    return fig_to_base64(fig)


def generate_comparison_chart(preds1, preds2, name1, name2):
    """生成对比图"""
    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
    vals1 = [preds1[ep]['prediction'] for ep in endpoints]
    vals2 = [preds2[ep]['prediction'] for ep in endpoints]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(endpoints))
    width = 0.35

    bars1 = ax.bar(x - width/2, vals1, width, label=name1, color='#ff6b6b', edgecolor='white')
    bars2 = ax.bar(x + width/2, vals2, width, label=name2, color='#4ecdc4', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{ep}\n{ENDPOINT_CN.get(ep, ep)}" for ep in endpoints], fontsize=8)
    ax.set_ylabel('毒性概率')
    ax.set_title(f'{name1} vs {name2} 毒性对比')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.6, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    return fig_to_base64(fig)


def generate_risk_pie(predictions):
    """生成风险饼图"""
    endpoints = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']
    values = [predictions[ep]['prediction'] for ep in endpoints]

    high = sum(1 for v in values if v > 0.6)
    medium = sum(1 for v in values if 0.3 < v <= 0.6)
    low = sum(1 for v in values if v <= 0.3)

    labels = []
    sizes = []
    colors = []

    if high > 0:
        labels.append(f'高风险({high})')
        sizes.append(high)
        colors.append('#ff4444')
    if medium > 0:
        labels.append(f'中风险({medium})')
        sizes.append(medium)
        colors.append('#ffaa00')
    if low > 0:
        labels.append(f'低风险({low})')
        sizes.append(low)
        colors.append('#44bb44')

    if not sizes:
        labels = ['无数据']
        sizes = [1]
        colors = ['#cccccc']

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('风险终点占比')

    return fig_to_base64(fig)


# 常量
ENDPOINT_CN = {
    'NR-AR': '雄激素受体拮抗',
    'NR-AR-LBD': '配体结合域活性',
    'NR-AhR': '芳香烃受体激活',
    'SR-HSE': '热休克元件响应',
    'SR-MMP': '线粒体膜电位异常',
    'SR-p53': 'p53通路激活',
}
