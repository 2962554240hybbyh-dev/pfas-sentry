"""
修复校正表：从299篇论文中提取更多支持文献
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

PROJECT_DIR = r'E:\桌面\项目'

print("="*70)
print("  从299篇论文中提取更多支持文献")
print("="*70)

# 加载所有论文
papers_path = os.path.join(PROJECT_DIR, 'data', 'raw', 'pubmed_papers_500.csv')
if os.path.exists(papers_path):
    papers = pd.read_csv(papers_path)
else:
    papers = pd.read_csv(os.path.join(PROJECT_DIR, 'data', 'raw', 'pubmed_papers_100.csv'))
print(f"  论文数: {len(papers)}")

# 关键词映射
endpoint_keywords = {
    'NR-AR': ['androgen', 'AR ', 'antiandrogen', 'receptor'],
    'NR-AhR': ['AhR', 'aryl hydrocarbon', 'dioxin'],
    'SR-MMP': ['mitochondrial', 'membrane potential', 'MMP'],
    'SR-p53': ['p53', 'tumor suppressor'],
    'hepatotoxicity': ['liver', 'hepatotox', 'hepatocyte', 'ALT', 'AST'],
    'immunotoxicity': ['immune', 'antibody', 'immunotox', 'vaccine'],
}

pfas_compounds = {
    'PFOA': ['PFOA', 'perfluorooctanoic', 'perfluorooctanoate'],
    'PFOS': ['PFOS', 'perfluorooctane sulfon'],
    'GenX': ['GenX', 'HFPO-DA', 'hexafluoropropylene oxide'],
    'PFNA': ['PFNA', 'perfluorononanoic'],
    'PFHxS': ['PFHxS', 'perfluorohexane sulfon'],
}

# 提取数据
extracted = []
for _, paper in papers.iterrows():
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    for compound, aliases in pfas_compounds.items():
        found = False
        for alias in aliases:
            if alias.lower() in text:
                found = True
                break

        if not found:
            continue

        for endpoint, keywords in endpoint_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    extracted.append({
                        'compound': compound,
                        'endpoint': endpoint,
                        'pmid': str(paper.get('pmid', '')),
                    })
                    break

df = pd.DataFrame(extracted)
print(f"  提取数据: {len(df)} 条")

# 构建校正表
calibration = {}
for compound in df['compound'].unique():
    comp_data = df[df['compound'] == compound]
    calibration[compound] = {}

    for endpoint in comp_data['endpoint'].unique():
        ep_data = comp_data[comp_data['endpoint'] == endpoint]
        n_papers = len(ep_data)

        if n_papers >= 10:
            confidence = 0.90
        elif n_papers >= 5:
            confidence = 0.80
        elif n_papers >= 3:
            confidence = 0.70
        else:
            confidence = 0.60

        calibration[compound][endpoint] = {
            'value': 1.0,
            'confidence': confidence,
            'n_papers': n_papers,
            'pmids': ep_data['pmid'].tolist()[:10],
        }

# 保存
cal_path = os.path.join(PROJECT_DIR, 'models', 'qsar', 'calibration_table.json')
with open(cal_path, 'w', encoding='utf-8') as f:
    json.dump(calibration, f, ensure_ascii=False, indent=2)

print(f"  校正表已保存: {cal_path}")

# 显示结果
print("\n  最终校正表:")
for compound in ['PFOA', 'PFOS', 'GenX']:
    if compound in calibration:
        print(f"\n  {compound}:")
        for ep in ['NR-AR', 'NR-AhR', 'SR-MMP', 'SR-p53', 'hepatotoxicity', 'immunotoxicity']:
            if ep in calibration[compound]:
                data = calibration[compound][ep]
                print(f"    {ep:15s}: 值={data['value']:.1f}, 置信度={data['confidence']:.2f}, 文献={data['n_papers']}篇")
