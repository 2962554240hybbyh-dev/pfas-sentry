"""
从PubMed搜索更多PFAS毒理文献
"""
import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

PROJECT_DIR = r'E:\桌面\项目'

print("="*70)
print("  从PubMed搜索更多PFAS毒理文献")
print("="*70)

# 搜索与各终点相关的PFAS文献
queries = {
    'NR-AhR': [
        'PFOA aryl hydrocarbon receptor',
        'PFOS AhR activation',
        'PFAS dioxin-like toxicity',
        'perfluoroalkyl AhR',
    ],
    'SR-MMP': [
        'PFOA mitochondrial membrane potential',
        'PFOS mitochondrial toxicity',
        'PFAS MMP assay',
        'perfluoroalkyl mitochondrial',
    ],
    'SR-p53': [
        'PFOA p53 pathway',
        'PFOS tumor suppressor',
        'PFAS p53 activation',
    ],
    'NR-AR': [
        'PFOA androgen receptor',
        'PFOS antiandrogenic',
        'PFAS endocrine disruption',
    ],
}

all_papers = {}
for endpoint, query_list in queries.items():
    all_papers[endpoint] = []

    for query in query_list:
        print(f"\n  搜索: {query}")
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 20,
                'retmode': 'json',
                'sort': 'relevance',
            }
            full_url = url + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                pmids = data.get('esearchresult', {}).get('idlist', [])
                print(f"    找到 {len(pmids)} 篇")
                all_papers[endpoint].extend(pmids)
            time.sleep(0.3)
        except Exception as e:
            print(f"    搜索失败: {e}")

# 统计
print("\n" + "="*70)
print("  搜索结果统计")
print("="*70)

for endpoint, pmids in all_papers.items():
    unique_pmids = list(set(pmids))
    print(f"  {endpoint}: {len(unique_pmids)} 篇不重复文献")

# 更新校正表
print("\n" + "="*70)
print("  更新校正表")
print("="*70)

cal_path = os.path.join(PROJECT_DIR, 'models', 'qsar', 'calibration_table.json')
with open(cal_path, 'r', encoding='utf-8') as f:
    cal = json.load(f)

for endpoint, pmids in all_papers.items():
    count = len(set(pmids))

    if count >= 10:
        confidence = 0.85
    elif count >= 5:
        confidence = 0.75
    elif count >= 3:
        confidence = 0.65
    else:
        confidence = 0.5

    cal.setdefault('PFOA', {})[endpoint] = {
        'value': 1.0,
        'confidence': confidence,
        'n_papers': count,
    }
    print(f"  PFOA {endpoint}: {count}篇文献, confidence={confidence}")

with open(cal_path, 'w', encoding='utf-8') as f:
    json.dump(cal, f, ensure_ascii=False, indent=2)

print(f"\n  校正表已更新: {cal_path}")

# 显示最终结果
print("\n" + "="*70)
print("  最终校正表（PFOA）")
print("="*70)
for ep, data in cal.get('PFOA', {}).items():
    print(f"  {ep}: value={data.get('value')}, confidence={data.get('confidence')}, papers={data.get('n_papers')}")
