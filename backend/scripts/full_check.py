"""
全面完整性检查
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

PROJECT = r'E:\桌面\项目'
TARGET = r'E:\桌面\竞赛项目\PFAS-Sentry'

print('='*70)
print('  全面完整性检查')
print('='*70)

# 1. 竞赛项目所有文件
print('\n[1] 竞赛项目所有文件')
print('-'*50)
total_size = 0
file_count = 0
for root, dirs, files in os.walk(TARGET):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
    for f in sorted(files):
        rel = os.path.relpath(os.path.join(root, f), TARGET)
        size = os.path.getsize(os.path.join(root, f))
        total_size += size
        file_count += 1
        print(f'  {rel} ({size:,} bytes)')

print(f'\n  总计: {file_count} 个文件, {total_size/1024/1024:.1f} MB')

# 2. 关键文件检查
print('\n[2] 关键文件检查')
print('-'*50)

critical = [
    ('backend/app.py', 'Flask主程序'),
    ('backend/rag/knowledge_base.py', '知识库'),
    ('backend/rag/retriever.py', '检索器'),
    ('backend/rag/generator.py', '生成器'),
    ('backend/utils/molecular.py', '分子解析'),
    ('backend/utils/prediction.py', '毒性预测'),
    ('backend/utils/visualization.py', '可视化'),
    ('backend/utils/report.py', '报告生成'),
    ('frontend/index.html', '首页'),
    ('frontend/prediction.html', '毒性预测页面'),
    ('frontend/qa.html', '智能问答页面'),
    ('frontend/comparison.html', '对比分析页面'),
    ('frontend/css/style.css', '样式'),
    ('frontend/js/app.js', 'JavaScript'),
    ('backend/data/tox21_features.csv', 'Tox21数据'),
    ('backend/data/pfas_dataset.csv', 'PFAS数据集'),
    ('backend/data/pubmed_papers.csv', 'PubMed论文'),
    ('backend/data/toxicity_extracted.csv', '毒性提取数据'),
    ('backend/data/real_data.json', '真实数据'),
    ('backend/data/knowledge_graph.json', '知识图谱'),
    ('backend/models/calibration_table.json', '校正表'),
    ('requirements.txt', '依赖包'),
    ('部署说明.md', '部署文档'),
    ('系统使用手册.md', '使用手册'),
]

all_ok = True
for f, desc in critical:
    path = os.path.join(TARGET, f)
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f'  OK {desc}: {f} ({size:,} bytes)')
    else:
        print(f'  MISSING {desc}: {f}')
        all_ok = False

# 3. 模型文件检查
print('\n[3] 模型文件检查')
print('-'*50)
model_dir = os.path.join(TARGET, 'backend/models/pre-trained')
models = sorted([f for f in os.listdir(model_dir) if f.endswith('.joblib')])
print(f'  总模型数: {len(models)}')
for ep in ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'SR-HSE', 'SR-MMP', 'SR-p53']:
    ep_models = [f for f in models if f.startswith(f'qsar_{ep}_')]
    print(f'  {ep}: {len(ep_models)} 个')

# 4. 结论
print('\n' + '='*70)
print('  结论')
print('='*70)
if all_ok:
    print('  所有关键文件都存在，项目完整')
else:
    print('  有缺失文件，需要补充')
print(f'  总文件数: {file_count}')
print(f'  总大小: {total_size/1024/1024:.1f} MB')
