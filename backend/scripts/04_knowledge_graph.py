"""
PFAS 知识图谱构建
8类实体 × 12类关系
"""
import sys, os, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# 实体与关系定义
# ============================================================
ENTITY_TYPES = {
    'compound': {'color': '#FF6B6B', 'size': 300, 'label': '化合物'},
    'functional_group': {'color': '#4ECDC4', 'size': 200, 'label': '官能团'},
    'toxicity_endpoint': {'color': '#45B7D1', 'size': 250, 'label': '毒性终点'},
    'toxicity_mechanism': {'color': '#96CEB4', 'size': 250, 'label': '毒性机制'},
    'environmental_medium': {'color': '#FFEAA7', 'size': 200, 'label': '环境介质'},
    'biological_species': {'color': '#DDA0DD', 'size': 200, 'label': '生物物种'},
    'health_effect': {'color': '#FF8C94', 'size': 250, 'label': '健康效应'},
    'regulation': {'color': '#A8D8EA', 'size': 200, 'label': '管控措施'},
}

RELATION_TYPES = {
    'contains': {'color': '#333333', 'style': 'solid', 'label': '包含'},
    'has_endpoint': {'color': '#FF6B6B', 'style': 'solid', 'label': '具有'},
    'via_mechanism': {'color': '#4ECDC4', 'style': 'dashed', 'label': '通过'},
    'exists_in': {'color': '#45B7D1', 'style': 'solid', 'label': '存在于'},
    'affects': {'color': '#96CEB4', 'style': 'solid', 'label': '影响'},
    'causes': {'color': '#FF8C94', 'style': 'solid', 'label': '导致'},
    'applies_to': {'color': '#A8D8EA', 'style': 'dotted', 'label': '适用'},
    'influences_endpoint': {'color': '#DDA0DD', 'style': 'dashed', 'label': '影响'},
    'leads_to_effect': {'color': '#FFEAA7', 'style': 'solid', 'label': '导致'},
    'medium_affects_species': {'color': '#96CEB4', 'style': 'dashed', 'label': '影响'},
    'protected_by': {'color': '#A8D8EA', 'style': 'dotted', 'label': '受保护'},
    'prevented_by': {'color': '#FF6B6B', 'style': 'dotted', 'label': '可预防'},
}

# ============================================================
# PFAS 知识库（基于文献和标准）
# ============================================================
def build_pfas_knowledge():
    """构建 PFAS 知识三元组"""

    triples = []

    # === 化合物 - 包含 - 官能团 ===
    pfas_compounds = {
        'PFOA': 'perfluorooctanoic acid',
        'PFOS': 'perfluorooctane sulfonic acid',
        'PFNA': 'perfluorononanoic acid',
        'PFDA': 'perfluorodecanoic acid',
        'PFHxA': 'perfluorohexanoic acid',
        'PFBS': 'perfluorobutane sulfonic acid',
        'PFHxS': 'perfluorohexane sulfonic acid',
        'GenX': 'hexafluoropropylene oxide dimer acid',
        'PFBA': 'perfluorobutanoic acid',
        'PFPeA': 'perfluoropentanoic acid',
        'PFUnDA': 'perfluoroundecanoic acid',
        'PFDoDA': 'perfluorododecanoic acid',
        'FOSA': 'perfluorooctane sulfonamide',
        'N-EtFOSE': 'N-ethyl perfluorooctane sulfonamidoethanol',
        'TFMS': 'trifluoromethanesulfonic acid',
        'TFA': 'trifluoroacetic acid',
        'ADONA': '4,8-dioxa-3H-perfluorononanoic acid',
        '6:2 FTCA': '6:2 fluorotelomer carboxylic acid',
        '8:2 FTCA': '8:2 fluorotelomer carboxylic acid',
        '9Cl-PF3ONS': '9-chlorohexadecafluoro-3-nonanesulfonic acid',
    }

    functional_groups = {
        'perfluoroalkyl chain': '全氟烷基链',
        'carboxylic acid group': '羧酸基团',
        'sulfonic acid group': '磺酸基团',
        'ether linkage': '醚键',
        'chlorine substituent': '氯取代基',
        'sulfonamide group': '磺酰胺基团',
        'fluorotelomer chain': '氟调聚物链',
    }

    # 化合物 - 包含 - 官能团
    for comp in pfas_compounds:
        if 'acid' in pfas_compounds[comp] and 'sulfonic' not in pfas_compounds[comp]:
            triples.append((comp, 'contains', 'carboxylic acid group'))
            triples.append((comp, 'contains', 'perfluoroalkyl chain'))
        elif 'sulfonic' in pfas_compounds[comp]:
            triples.append((comp, 'contains', 'sulfonic acid group'))
            triples.append((comp, 'contains', 'perfluoroalkyl chain'))
        elif 'sulfonamide' in pfas_compounds[comp]:
            triples.append((comp, 'contains', 'sulfonamide group'))
            triples.append((comp, 'contains', 'perfluoroalkyl chain'))

    if 'ether' in pfas_compounds.get('GenX', '') or 'GenX' in pfas_compounds:
        triples.append(('GenX', 'contains', 'ether linkage'))
    if 'chloro' in pfas_compounds.get('9Cl-PF3ONS', ''):
        triples.append(('9Cl-PF3ONS', 'contains', 'chlorine substituent'))
    if 'fluorotelomer' in pfas_compounds.get('6:2 FTCA', ''):
        triples.append(('6:2 FTCA', 'contains', 'fluorotelomer chain'))

    # === 化合物 - 具有 - 毒性终点 ===
    toxicity_endpoints = [
        'NR-AR (雄激素受体拮抗)',
        'NR-AR-LBD (配体结合域)',
        'NR-AhR (芳香烃受体激活)',
        'SR-HSE (热休克元件响应)',
        'SR-MMP (线粒体膜电位)',
        'SR-p53 (p53通路激活)',
        'BCF (生物富集系数)',
        'Biodegradability (生物降解性)',
    ]

    # 基于文献数据的毒性关联
    toxicity_data = {
        'PFOA': ['NR-AR', 'NR-AhR', 'SR-MMP', 'SR-p53', 'BCF'],
        'PFOS': ['NR-AR', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53', 'BCF'],
        'PFNA': ['NR-AR', 'NR-AhR', 'SR-MMP', 'BCF'],
        'PFDA': ['NR-AhR', 'SR-p53', 'BCF'],
        'PFHxA': ['NR-AhR', 'SR-MMP'],
        'PFBS': ['NR-AhR', 'SR-HSE'],
        'PFHxS': ['NR-AhR', 'SR-MMP', 'BCF'],
        'GenX': ['NR-AhR', 'SR-p53'],
        'PFBA': ['SR-HSE'],
        'PFUnDA': ['NR-AR', 'NR-AhR', 'SR-MMP', 'SR-p53', 'BCF'],
        'PFDoDA': ['NR-AR', 'BCF'],
        'FOSA': ['NR-AhR', 'SR-MMP'],
        'N-EtFOSE': ['NR-AhR', 'SR-HSE'],
        'TFA': ['SR-HSE'],
        'ADONA': ['NR-AhR'],
        '9Cl-PF3ONS': ['NR-AhR', 'SR-p53'],
    }

    for comp, endpoints in toxicity_data.items():
        for ep in endpoints:
            triples.append((comp, 'has_endpoint', ep))

    # === 化合物 - 通过 - 毒性机制 ===
    mechanisms = {
        'PFOA': ['PPARα激活', '氧化应激', '线粒体功能障碍'],
        'PFOS': ['PPARα激活', '甲状腺激素干扰', '免疫抑制', '氧化应激'],
        'PFNA': ['PPARα激活', '氧化应激'],
        'PFDA': ['PPARα激活', '脂质代谢干扰'],
        'GenX': ['肝脏毒性', '肾脏毒性'],
        'PFHxS': ['甲状腺激素干扰', '免疫抑制'],
        'PFBS': ['甲状腺激素干扰'],
    }

    for comp, mechs in mechanisms.items():
        for mech in mechs:
            triples.append((comp, 'via_mechanism', mech))

    # === 化合物 - 存在于 - 环境介质 ===
    media = ['饮用水', '地表水', '地下水', '土壤', '大气', '海洋', '沉积物', '生物体']
    pfas_media = {
        'PFOA': ['饮用水', '地表水', '地下水', '土壤', '海洋', '生物体'],
        'PFOS': ['饮用水', '地表水', '地下水', '土壤', '海洋', '沉积物', '生物体'],
        'PFHxA': ['饮用水', '地表水'],
        'PFBS': ['饮用水', '地表水'],
        'GenX': ['饮用水', '地表水'],
        'PFNA': ['地表水', '海洋', '生物体'],
        'PFDA': ['海洋', '生物体'],
        'TFA': ['饮用水', '大气'],
    }

    for comp, media_list in pfas_media.items():
        for medium in media_list:
            triples.append((comp, 'exists_in', medium))

    # === 化合物 - 影响 - 生物物种 ===
    species = ['鱼类', '甲壳类', '藻类', '鸟类', '哺乳动物', '人类', '两栖类']
    pfas_species = {
        'PFOA': ['鱼类', '哺乳动物', '人类'],
        'PFOS': ['鱼类', '甲壳类', '鸟类', '哺乳动物', '人类'],
        'PFNA': ['鱼类', '哺乳动物'],
        'PFHxA': ['鱼类'],
        'PFBS': ['鱼类', '甲壳类'],
        'GenX': ['哺乳动物'],
    }

    for comp, sp_list in pfas_species.items():
        for sp in sp_list:
            triples.append((comp, 'affects', sp))

    # === 化合物 - 导致 - 健康效应 ===
    health_effects = [
        '肝毒性', '肾毒性', '甲状腺功能异常', '免疫功能抑制',
        '发育毒性', '生殖毒性', '致癌性', '内分泌干扰',
        '代谢紊乱', '神经毒性', '心血管效应'
    ]

    pfas_health = {
        'PFOA': ['肝毒性', '肾毒性', '甲状腺功能异常', '免疫功能抑制', '发育毒性', '生殖毒性', '致癌性'],
        'PFOS': ['肝毒性', '甲状腺功能异常', '免疫功能抑制', '发育毒性', '生殖毒性', '内分泌干扰'],
        'PFNA': ['肝毒性', '内分泌干扰'],
        'PFDA': ['肝毒性'],
        'PFHxA': ['肝毒性'],
        'PFBS': ['甲状腺功能异常'],
        'GenX': ['肝毒性', '肾毒性'],
        'PFHxS': ['甲状腺功能异常', '免疫功能抑制'],
    }

    for comp, effects in pfas_health.items():
        for eff in effects:
            triples.append((comp, 'causes', eff))

    # === 化合物 - 适用 - 管控措施 ===
    regulations = [
        '中国新污染物治理行动方案',
        'GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)',
        'GB 3838-2002 地表水标准',
        'EPA PFAS 管控标准 (70 ng/L)',
        '欧盟 REACH 限制',
        '斯德哥尔摩公约 (POPs)',
        'WHO 饮用水指南',
        '各州级标准',
    ]

    regulated_compounds = ['PFOA', 'PFOS', 'PFHxS', 'PFDA', 'PFNA']
    for comp in regulated_compounds:
        triples.append((comp, 'applies_to', '中国新污染物治理行动方案'))
        triples.append((comp, 'applies_to', '斯德哥尔摩公约 (POPs)'))
        if comp in ['PFOA', 'PFOS']:
            triples.append((comp, 'applies_to', 'GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)'))
            triples.append((comp, 'applies_to', 'EPA PFAS 管控标准 (70 ng/L)'))
            triples.append((comp, 'applies_to', '欧盟 REACH 限制'))

    # === 官能团 - 影响 - 毒性终点 ===
    triples.append(('perfluoroalkyl chain length', 'influences_endpoint', 'BCF'))
    triples.append(('perfluoroalkyl chain length', 'influences_endpoint', 'bioaccumulation'))
    triples.append(('sulfonic acid group', 'influences_endpoint', 'persistence'))
    triples.append(('ether linkage', 'influences_endpoint', 'metabolic stability'))
    triples.append(('chlorine substituent', 'influences_endpoint', 'persistence'))

    # === 毒性机制 - 导致 - 健康效应 ===
    mechanism_effects = {
        'PPARα激活': ['肝毒性', '代谢紊乱'],
        '氧化应激': ['肝毒性', '肾毒性', '致癌性'],
        '线粒体功能障碍': ['肝毒性', '肾毒性'],
        '甲状腺激素干扰': ['甲状腺功能异常', '发育毒性'],
        '免疫抑制': ['免疫功能抑制'],
        '内分泌干扰': ['生殖毒性', '发育毒性'],
        '脂质代谢干扰': ['代谢紊乱'],
        '肝脏毒性': ['肝毒性'],
        '肾脏毒性': ['肾毒性'],
    }

    for mech, effects in mechanism_effects.items():
        for eff in effects:
            triples.append((mech, 'leads_to_effect', eff))

    # === 环境介质 - 影响 - 生物物种 ===
    medium_species = {
        '饮用水': ['人类'],
        '地表水': ['鱼类', '甲壳类', '藻类'],
        '海洋': ['鱼类', '甲壳类', '海洋哺乳动物', '鸟类'],
        '土壤': ['两栖类', '哺乳动物'],
        '大气': ['人类'],
    }

    for medium, sp_list in medium_species.items():
        for sp in sp_list:
            triples.append((medium, 'medium_affects_species', sp))

    # === 生物物种 - 受 - 管控措施保护 ===
    for sp in ['人类', '鱼类', '甲壳类']:
        triples.append((sp, 'protected_by', 'GB 5749-2022 生活饮用水标准 (40 ng/L PFOS+PFOA)'))
        triples.append((sp, 'protected_by', 'EPA PFAS 管控标准 (70 ng/L)'))

    # === 健康效应 - 可通过 - 管控措施预防 ===
    for eff in ['肝毒性', '肾毒性', '发育毒性', '免疫功能抑制']:
        triples.append((eff, 'prevented_by', '中国新污染物治理行动方案'))
        triples.append((eff, 'prevented_by', '斯德哥尔摩公约 (POPs)'))

    return triples

# ============================================================
# 构建 NetworkX 知识图谱
# ============================================================
def build_networkx_graph(triples):
    """构建 NetworkX 图"""
    G = nx.DiGraph()

    for h, r, t in triples:
        if not G.has_node(h):
            G.add_node(h, type='entity')
        if not G.has_node(t):
            G.add_node(t, type='entity')
        G.add_edge(h, t, relation=r)

    return G

# ============================================================
# 可视化
# ============================================================
def visualize_graph(G, title, output_path, max_nodes=80):
    """可视化知识图谱"""
    plt.figure(figsize=(20, 16))

    # 如果节点太多，取度数最高的
    if len(G.nodes) > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()

    # 布局
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # 节点颜色
    node_colors = []
    for node in G.nodes:
        node_lower = node.lower()
        if any(x in node_lower for x in ['pfoa', 'pfos', 'pfna', 'pfda', 'pfhxa', 'pfbs',
                                           'genx', 'pfba', 'pfpea', 'pfunda', 'pfdoda',
                                           'fosa', 'etfose', 'tfms', 'tfa', 'adona',
                                           'ftca', 'cl-pf']):
            node_colors.append('#FF6B6B')  # 化合物
        elif any(x in node_lower for x in ['nr-', 'sr-', 'bCF', 'biodeg']):
            node_colors.append('#45B7D1')  # 毒性终点
        elif any(x in node_lower for x in ['激活', '干扰', '抑制', '障碍', '毒性']):
            node_colors.append('#96CEB4')  # 机制
        elif any(x in node_lower for x in ['饮用水', '地表', '地下', '土壤', '大气', '海洋', '沉积']):
            node_colors.append('#FFEAA7')  # 环境介质
        elif any(x in node_lower for x in ['鱼', '甲壳', '鸟', '哺乳', '人类', '两栖', '藻']):
            node_colors.append('#DDA0DD')  # 生物
        elif any(x in node_lower for x in ['肝', '肾', '甲状腺', '免疫', '发育', '生殖', '致癌', '内分泌', '代谢', '神经']):
            node_colors.append('#FF8C94')  # 健康效应
        elif any(x in node_lower for x in ['gb', 'epa', 'reach', 'who', '斯德哥尔摩', '新污染物', '标准']):
            node_colors.append('#A8D8EA')  # 法规
        elif any(x in node_lower for x in ['酸', '链', '醚', '氯', '酰胺']):
            node_colors.append('#4ECDC4')  # 官能团
        else:
            node_colors.append('#CCCCCC')

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=800, alpha=0.9)

    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='#888888',
                           arrows=True, arrowsize=15,
                           alpha=0.5, width=1)

    # 标签
    nx.draw_networkx_labels(G, pos, font_size=8,
                            font_family='Microsoft YaHei')

    plt.title(title, fontsize=16, fontfamily='Microsoft YaHei')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================
# 子图可视化
# ============================================================
def visualize_subgraph(G, center_node, hops, title, output_path):
    """可视化以某节点为中心的子图"""
    nodes = {center_node}
    current = {center_node}

    for _ in range(hops):
        next_nodes = set()
        for n in current:
            for neighbor in G.neighbors(n):
                next_nodes.add(neighbor)
            for predecessor in G.predecessors(n):
                next_nodes.add(predecessor)
        nodes.update(next_nodes)
        current = next_nodes

    subG = G.subgraph(nodes).copy()
    visualize_graph(subG, title, output_path)

# ============================================================
# 保存三元组
# ============================================================
def save_triples(triples, output_path):
    """保存三元组为 CSV"""
    df = pd.DataFrame(triples, columns=['head', 'relation', 'tail'])
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df

# ============================================================
# 主流程
# ============================================================
def main():
    output_dir = os.path.join(PROJECT_DIR, "05_知识图谱")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("PFAS 知识图谱构建")
    print("=" * 60)

    # 1. 构建知识三元组
    print("\n第1步：构建知识三元组...")
    triples = build_pfas_knowledge()
    print(f"  三元组数量: {len(triples)}")

    # 保存三元组
    triples_path = os.path.join(output_dir, "pfas_kg_triples.csv")
    triples_df = save_triples(triples, triples_path)
    print(f"  三元组已保存: {triples_path}")

    # 统计
    print(f"\n  实体数量: {len(set(triples_df['head'].tolist() + triples_df['tail'].tolist()))}")
    print(f"  关系类型: {triples_df['relation'].nunique()}")
    print(f"\n  关系分布:")
    print(triples_df['relation'].value_counts().to_string())

    # 2. 构建 NetworkX 图
    print("\n第2步：构建知识图谱...")
    G = build_networkx_graph(triples)
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")

    # 3. 整体可视化
    print("\n第3步：生成可视化...")
    visualize_graph(G, "PFAS 知识图谱",
                    os.path.join(output_dir, "kg_overview.png"))

    # 4. 典型子图
    # PFOA 毒性机制子图
    if 'PFOA' in G.nodes:
        visualize_subgraph(G, 'PFOA', 2,
                           "PFOA 毒性机制与健康效应子图",
                           os.path.join(output_dir, "kg_pfoa_subgraph.png"))

    # PFOS 管控措施子图
    if 'PFOS' in G.nodes:
        visualize_subgraph(G, 'PFOS', 2,
                           "PFOS 管控措施子图",
                           os.path.join(output_dir, "kg_pfos_regulation.png"))

    # 毒性机制到健康效应
    mechanism_nodes = [n for n in G.nodes if any(x in n for x in ['激活', '干扰', '抑制', '障碍'])]
    if mechanism_nodes:
        subG = G.subgraph(mechanism_nodes[:15] +
                          [n for n in G.nodes if any(x in n for x in ['肝', '肾', '甲状腺', '免疫', '发育'])][:10]).copy()
        if subG.number_of_nodes() > 0:
            visualize_graph(subG, "毒性机制 → 健康效应",
                            os.path.join(output_dir, "kg_mechanism_effects.png"))

    # 5. 知识图谱统计报告
    report = f"""# PFAS 知识图谱构建报告

## 统计信息
- 三元组总数: {len(triples)}
- 实体总数: {G.number_of_nodes()}
- 关系总数: {G.number_of_edges()}
- 关系类型数: {triples_df['relation'].nunique()}

## 实体类型分布
- 化合物: {len([n for n in G.nodes if any(x in n.lower() for x in ['pfoa', 'pfos', 'pf', 'genx', 'ftca'])])}
- 毒性终点: {len([n for n in G.nodes if any(x in n for x in ['NR-', 'SR-', 'BCF'])])}
- 毒性机制: {len([n for n in G.nodes if any(x in n for x in ['激活', '干扰', '抑制', '障碍'])])}
- 健康效应: {len([n for n in G.nodes if any(x in n for x in ['肝', '肾', '甲状腺', '免疫'])])}
- 环境介质: {len([n for n in G.nodes if any(x in n for x in ['水', '土壤', '大气', '海洋'])])}
- 管控措施: {len([n for n in G.nodes if any(x in n for x in ['GB', 'EPA', 'REACH', '公约'])])}

## 关系类型分布
{triples_df['relation'].value_counts().to_string()}

## 输出文件
1. pfas_kg_triples.csv - 三元组数据
2. kg_overview.png - 知识图谱全景图
3. kg_pfoa_subgraph.png - PFOA 毒性机制子图
4. kg_pfos_regulation.png - PFOS 管控措施子图
5. kg_mechanism_effects.png - 毒性机制到健康效应图
"""

    with open(os.path.join(output_dir, "知识图谱构建报告.md"), 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n{'='*60}")
    print("知识图谱构建完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()
