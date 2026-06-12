"""
从PubMed文献中提取真实PFAS毒理数据
高质量、可验证、有PMID来源
"""
import sys, os, json, time, re
sys.stdout.reconfigure(encoding='utf-8')

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd

PROJECT_DIR = r'E:\桌面\项目'

# ============================================================
# 1. 从PubMed搜索PFAS毒理文献
# ============================================================
def search_pfas_literature():
    """搜索PFAS毒理相关的高质量文献"""
    print("="*70)
    print("  步骤1：从PubMed搜索PFAS毒理文献")
    print("="*70)

    queries = [
        'PFOA toxicity NR-AhR activation',
        'PFOS toxicity NR-AR androgen receptor',
        'PFAS mitochondrial toxicity SR-MMP',
        'PFOA in vitro toxicity assay',
        'PFOS in vitro toxicity assay',
        'PFAS endocrine disruption assay',
        'PFOA PFOS ToxCast toxicity',
        'PFAS high throughput screening',
        'PFOA hepatotoxicity mechanism',
        'PFOS immunotoxicity mechanism',
    ]

    all_pmids = set()

    for query in queries:
        print(f"\n  搜索: {query}")
        try:
            # PubMed E-utilities search
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query + ' AND (pubmed[dp] >= 2015)',
                'retmax': 20,
                'retmode': 'json',
                'sort': 'relevance'
            }
            full_url = url + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                pmids = data.get('esearchresult', {}).get('idlist', [])
                print(f"    找到 {len(pmids)} 篇")
                all_pmids.update(pmids)
            time.sleep(0.5)
        except Exception as e:
            print(f"    搜索失败: {e}")

    print(f"\n  总共找到 {len(all_pmids)} 篇不重复文献")
    return list(all_pmids)


# ============================================================
# 2. 获取文献详细信息
# ============================================================
def fetch_paper_details(pmids):
    """获取文献的标题、摘要、作者等信息"""
    print("\n" + "="*70)
    print("  步骤2：获取文献详细信息")
    print("="*70)

    papers = []
    batch_size = 50

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i+batch_size]
        ids = ','.join(batch)

        try:
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids}&rettype=xml"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response:
                xml_data = response.read().decode('utf-8')

            root = ET.fromstring(xml_data)

            for article in root.findall('.//PubmedArticle'):
                paper = {}

                # PMID
                pmid_elem = article.find('.//PMID')
                paper['pmid'] = pmid_elem.text if pmid_elem is not None else ''

                # 标题
                title_elem = article.find('.//ArticleTitle')
                paper['title'] = title_elem.text if title_elem is not None else ''

                # 摘要
                abstract_parts = []
                for abstract_text in article.findall('.//AbstractText'):
                    if abstract_text.text:
                        abstract_parts.append(abstract_text.text)
                paper['abstract'] = ' '.join(abstract_parts)

                # 作者
                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None and first_name is not None:
                        authors.append(f"{last_name.text} {first_name.text}")
                paper['authors'] = '; '.join(authors[:5])

                # 期刊
                journal_elem = article.find('.//Journal/Title')
                paper['journal'] = journal_elem.text if journal_elem is not None else ''

                # 年份
                year_elem = article.find('.//PubDate/Year')
                paper['year'] = year_elem.text if year_elem is not None else ''

                # DOI
                for id_elem in article.findall('.//ArticleId'):
                    if id_elem.get('IdType') == 'doi':
                        paper['doi'] = id_elem.text
                        break
                else:
                    paper['doi'] = ''

                if paper['title']:
                    papers.append(paper)

            print(f"  已获取 {len(papers)} 篇文献详情")
            time.sleep(0.5)

        except Exception as e:
            print(f"  获取失败: {e}")

    return papers


# ============================================================
# 3. 从文献摘要中提取毒理数据
# ============================================================
def extract_toxicity_from_abstracts(papers):
    """从文献摘要中提取PFAS毒理数据"""
    print("\n" + "="*70)
    print("  步骤3：从文献摘要中提取毒理数据")
    print("="*70)

    # PFAS化合物
    pfas_compounds = {
        'PFOA': ['PFOA', 'perfluorooctanoic acid', 'C8HF15O2', '335-67-1'],
        'PFOS': ['PFOS', 'perfluorooctane sulfonic acid', 'C8HF17O3S', '1763-23-1'],
        'PFNA': ['PFNA', 'perfluorononanoic acid', 'C9HF17O2', '375-95-1'],
        'PFDA': ['PFDA', 'perfluorodecanoic acid', 'C10HF19O2', '335-76-2'],
        'PFHxA': ['PFHxA', 'perfluorohexanoic acid', 'C6HF11O2', '307-24-4'],
        'PFBA': ['PFBA', 'perfluorobutanoic acid', 'C4HF7O2', '375-22-4'],
        'PFBS': ['PFBS', 'perfluorobutane sulfonic acid', 'C4HF9O3S', '375-73-5'],
        'PFHxS': ['PFHxS', 'perfluorohexane sulfonic acid', 'C6HF13O3S', '355-46-4'],
        'GenX': ['GenX', 'HFPO-DA', 'hexafluoropropylene oxide', '13252-13-6'],
        'TFA': ['TFA', 'trifluoroacetic acid', 'C2HF3O2', '76-05-1'],
    }

    # 毒性关键词
    toxicity_keywords = {
        'NR-AhR': ['AhR', 'aryl hydrocarbon', 'AhR activation', 'AhR agonist'],
        'NR-AR': ['androgen receptor', 'AR antagonist', 'AR agonist', 'antiandrogenic'],
        'SR-MMP': ['mitochondrial', 'MMP', 'membrane potential', 'mitochondrial toxicity'],
        'hepatotoxicity': ['hepatotoxicity', 'liver', 'hepatocyte', 'ALT', 'AST', 'liver damage'],
        'immunotoxicity': ['immunotoxicity', 'immune', 'antibody', 'vaccine', 'immunosuppress'],
        'endocrine': ['endocrine', 'thyroid', 'hormone', 'disruption'],
        'developmental': ['developmental', 'fetal', 'birth weight', 'pregnancy'],
    }

    # 数值提取模式
    value_patterns = [
        r'(\d+\.?\d*)\s*[±]\s*(\d+\.?\d*)\s*(ng/mL|μg/mL|mg/L|μM|nM)',
        r'EC50[:\s]*(\d+\.?\d*)\s*(ng/mL|μg/mL|mg/L|μM|nM)',
        r'IC50[:\s]*(\d+\.?\d*)\s*(ng/mL|μg/mL|mg/L|μM|nM)',
        r'NOAEL[:\s]*(\d+\.?\d*)\s*(mg/kg/d|mg/kg/day)',
        r'LOAEL[:\s]*(\d+\.?\d*)\s*(mg/kg/d|mg/kg/day)',
        r'(\d+\.?\d*)\s*fold\s*(increase|decrease|induction)',
    ]

    extracted_data = []

    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        # 检查涉及哪些PFAS
        compounds_found = []
        for compound, aliases in pfas_compounds.items():
            for alias in aliases:
                if alias.lower() in text:
                    compounds_found.append(compound)
                    break

        if not compounds_found:
            continue

        # 检查涉及哪些毒性终点
        toxicity_found = []
        for endpoint, keywords in toxicity_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    toxicity_found.append(endpoint)
                    break

        if not toxicity_found:
            continue

        # 提取数值数据
        values_found = []
        for pattern in value_patterns:
            matches = re.findall(pattern, paper.get('abstract', ''), re.IGNORECASE)
            values_found.extend(matches)

        # 判断活性/非活性
        positive_keywords = ['active', 'positive', 'significant', 'increase', 'induction', 'agonist', 'effect']
        negative_keywords = ['inactive', 'negative', 'no effect', 'no significant', 'below detection']

        is_positive = any(kw in text for kw in positive_keywords)
        is_negative = any(kw in text for kw in negative_keywords)

        for compound in compounds_found:
            for endpoint in toxicity_found:
                # 判断活性
                if is_positive and not is_negative:
                    activity = 1
                elif is_negative and not is_positive:
                    activity = 0
                else:
                    activity = None  # 不确定

                extracted_data.append({
                    'pmid': paper['pmid'],
                    'title': paper['title'][:100],
                    'journal': paper['journal'],
                    'year': paper['year'],
                    'compound': compound,
                    'endpoint': endpoint,
                    'activity': activity,
                    'values': str(values_found[:3]) if values_found else 'N/A',
                    'confidence': 'high' if activity is not None else 'low',
                })

    df = pd.DataFrame(extracted_data)
    print(f"  提取数据条数: {len(df)}")
    print(f"  涉及化合物: {df['compound'].nunique() if len(df) > 0 else 0}")
    print(f"  涉及终点: {df['endpoint'].nunique() if len(df) > 0 else 0}")

    return df


# ============================================================
# 4. 构建校正表
# ============================================================
def build_calibration_table(extracted_df):
    """从提取的数据构建校正表"""
    print("\n" + "="*70)
    print("  步骤4：构建校正表")
    print("="*70)

    calibration = {}

    if len(extracted_df) == 0:
        print("  无提取数据，使用保守估计")
        return calibration

    # 按化合物和终点分组
    for compound in extracted_df['compound'].unique():
        compound_data = extracted_df[extracted_df['compound'] == compound]
        calibration[compound] = {}

        for endpoint in compound_data['endpoint'].unique():
            endpoint_data = compound_data[compound_data['endpoint'] == endpoint]

            # 计算活性比例
            valid_data = endpoint_data.dropna(subset=['activity'])
            if len(valid_data) > 0:
                active_ratio = valid_data['activity'].mean()
                n_papers = len(valid_data)

                # 基于证据数量设置置信度
                if n_papers >= 5:
                    confidence = 0.9
                elif n_papers >= 3:
                    confidence = 0.8
                elif n_papers >= 1:
                    confidence = 0.7
                else:
                    confidence = 0.5

                calibration[compound][endpoint] = {
                    'value': round(active_ratio, 2),
                    'confidence': confidence,
                    'n_papers': n_papers,
                    'pmids': valid_data['pmid'].tolist()[:5],
                }

                print(f"  {compound} - {endpoint}: 值={active_ratio:.2f}, 置信度={confidence}, 文献数={n_papers}")

    return calibration


# ============================================================
# 5. 主函数
# ============================================================
def main():
    print("\n" + "★"*70)
    print("  从PubMed文献提取真实PFAS毒理数据")
    print("★"*70)

    # 1. 搜索文献
    pmids = search_pfas_literature()

    if len(pmids) == 0:
        print("\n  未找到文献，使用保守估计数据")
        return

    # 2. 获取文献详情
    papers = fetch_paper_details(pmids[:100])  # 最多100篇

    # 3. 提取毒理数据
    extracted_df = extract_toxicity_from_abstracts(papers)

    # 4. 构建校正表
    calibration = build_calibration_table(extracted_df)

    # 5. 保存数据
    print("\n" + "="*70)
    print("  步骤5：保存数据")
    print("="*70)

    # 保存提取的数据
    if len(extracted_df) > 0:
        extracted_path = os.path.join(PROJECT_DIR, 'data', 'pfas_toxicity_extracted.csv')
        extracted_df.to_csv(extracted_path, index=False, encoding='utf-8-sig')
        print(f"  提取数据: {extracted_path} ({len(extracted_df)} 条)")

    # 保存校正表
    cal_path = os.path.join(PROJECT_DIR, 'models', 'qsar', 'calibration_table_real.json')
    with open(cal_path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    print(f"  校正表: {cal_path}")

    # 保存文献列表
    papers_df = pd.DataFrame(papers)
    papers_path = os.path.join(PROJECT_DIR, 'data', 'pfas_literature_toxicity.csv')
    papers_df.to_csv(papers_path, index=False, encoding='utf-8-sig')
    print(f"  文献列表: {papers_path} ({len(papers)} 篇)")

    # 总结
    print("\n" + "★"*70)
    print("  数据提取完成！")
    print("★"*70)
    print(f"\n  文献数: {len(papers)}")
    print(f"  提取数据: {len(extracted_df)} 条")
    print(f"  校正化合物: {len(calibration)} 种")

    if len(calibration) > 0:
        print("\n  校正表内容:")
        for compound, endpoints in calibration.items():
            print(f"    {compound}:")
            for ep, data in endpoints.items():
                print(f"      {ep}: 值={data['value']}, 置信度={data['confidence']}, 文献数={data['n_papers']}")


if __name__ == '__main__':
    main()
