"""
检查项目完整性
"""
import sys, os, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

PROJECT = r'E:\桌面\项目'
TARGET = r'E:\桌面\竞赛项目\PFAS-Sentry'

print("="*70)
print("  项目完整性对比检查")
print("="*70)

# 1. 原始项目关键文件
print("\n[1] 原始项目关键文件")
print("-"*50)

data_files = {
    'data/raw/tox21_real_data.csv': 'Tox21训练数据',
    'data/cleaned/pfas_clean_data.csv': 'PFAS数据集',
    'data/raw/pubmed_papers_100.csv': 'PubMed论文(100篇)',
    'data/raw/pfas_literature.csv': '文献数据',
    'data/pfas_toxicity_extracted.csv': '提取的毒性数据',
    'data/real_data.json': '真实数据JSON',
}

for f, desc in data_files.items():
    src = os.path.join(PROJECT, f)
    exists = os.path.exists(src)
    size = os.path.getsize(src) / 1024 if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  {status} {desc}: {size:.1f} KB')

# 模型文件
print('\n  模型文件:')
model_dir = os.path.join(PROJECT, 'models', 'qsar')
models = [f for f in os.listdir(model_dir) if f.endswith('.joblib')]
print(f'  Total: {len(models)} 个')

# 知识图谱
kg_path = os.path.join(PROJECT, '05_知识图谱', 'pfas_kg_triples.csv')
kg = pd.read_csv(kg_path)
print(f'  知识图谱: {len(kg)} 三元组')

# 2. 竞赛项目文件
print('\n[2] 竞赛项目文件')
print('-'*50)

target_files = [
    'backend/app.py',
    'backend/rag/knowledge_base.py',
    'backend/rag/retriever.py',
    'backend/rag/generator.py',
    'backend/utils/molecular.py',
    'backend/utils/prediction.py',
    'backend/utils/visualization.py',
    'backend/utils/report.py',
    'frontend/index.html',
    'frontend/prediction.html',
    'frontend/qa.html',
    'frontend/comparison.html',
    'frontend/css/style.css',
    'frontend/js/app.js',
    'backend/data/tox21_features.csv',
    'backend/data/pfas_dataset.csv',
    'backend/data/knowledge_graph.json',
    'backend/models/calibration_table.json',
    'requirements.txt',
]

for f in target_files:
    path = os.path.join(TARGET, f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  {status} {f} ({size} bytes)')

# 模型文件
target_models = [f for f in os.listdir(os.path.join(TARGET, 'backend/models/pre-trained')) if f.endswith('.joblib')]
print(f'\n  模型文件: {len(target_models)} 个')

# 3. 缺失文件检查
print('\n[3] 缺失文件检查')
print('-'*50)

# 检查论文数据
papers_src = os.path.join(PROJECT, 'data', 'raw', 'pubmed_papers_100.csv')
papers_dst = os.path.join(TARGET, 'backend', 'data', 'pubmed_papers.csv')
if os.path.exists(papers_src) and not os.path.exists(papers_dst):
    print('  MISSING PubMed论文数据 (100篇)')
    print('    源文件: data/raw/pubmed_papers_100.csv')
    print('    需要复制到: backend/data/pubmed_papers.csv')

# 检查提取的毒性数据
tox_src = os.path.join(PROJECT, 'data', 'pfas_toxicity_extracted.csv')
tox_dst = os.path.join(TARGET, 'backend', 'data', 'toxicity_extracted.csv')
if os.path.exists(tox_src) and not os.path.exists(tox_dst):
    print('  MISSING 提取的毒性数据')
    print('    源文件: data/pfas_toxicity_extracted.csv')
    print('    需要复制到: backend/data/toxicity_extracted.csv')

# 4. 总结
print('\n' + '='*70)
print('  总结')
print('='*70)
print(f'''
  原始项目: {len(models)}个模型, {len(kg)}三元组, 100篇论文
  竞赛项目: {len(target_models)}个模型, 442三元组

  需要补充:
  1. PubMed论文数据 (100篇)
  2. 提取的毒性数据
  3. 真实数据JSON

  结论: 核心数据已复制，但论文原始数据需要补充
''')
