"""
从权威数据库获取PFAS真实毒理数据
1. EPA CompTox Chemicals Dashboard
2. ECOTOX Knowledgebase
3. 扩大PubMed文献搜索
"""
import sys, os, json, time, re
sys.stdout.reconfigure(encoding='utf-8')

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import pandas as pd

PROJECT_DIR = r'E:\桌面\项目'

# ============================================================
# 1. 从EPA CompTox获取PFAS数据
# ============================================================
def fetch_comptox_data():
    """从EPA CompTox Chemicals Dashboard获取PFAS毒理数据"""
    print("="*70)
    print("  步骤1：从EPA CompTox获取PFAS数据")
    print("="*70)

    # EPA CompTox API endpoint
    # 文档: https://api-ccte.epa.gov/docs/
    base_url = "https://api-ccte.epa.gov/data"

    # PFAS相关的DTXSIDs（EPA内部标识符）
    # 这些是EPA CompTox中PFAS化合物的标识
    pfas_dtxsids = {
        'PFOA': 'DTXSID9031865',
        'PFOS': 'DTXSID9031864',
        'PFNA': 'DTXSID9040287',
        'PFDA': 'DTXSID6027446',
        'PFHxA': 'DTXSID9031866',
        'PFBA': 'DTXSID9040286',
        'PFBS': 'DTXSID2029681',
        'PFHxS': 'DTXSID9040285',
        'GenX': 'DTXSID70894987',
    }

    print("\n  注意: EPA CompTox API需要API密钥")
    print("  访问 https://api-ccte.epa.gov/ 获取API密钥")
    print("  或者手动下载: https://comptox.epa.gov/dashboard/")

    # 尝试获取公开数据
    results = {}

    # EPA CompTox提供的公开下载链接
    print("\n  尝试从EPA CompTox公开数据下载...")
    try:
        # PFAS Master List下载链接
        url = "https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER"
        print(f"  PFAS Master List: {url}")
        print("  包含12000+种PFAS化合物")
        print("  需要手动下载CSV文件")
    except Exception as e:
        print(f"  访问失败: {e}")

    return results


# ============================================================
# 2. 从ECOTOX获取水生生物毒性数据
# ============================================================
def fetch_ecotox_data():
    """从ECOTOX Knowledgebase获取水生生物毒性数据"""
    print("\n" + "="*70)
    print("  步骤2：从ECOTOX获取水生生物毒性数据")
    print("="*70)

    # ECOTOX API
    # 文档: https://cfpub.epa.gov/ecotox/api/
    base_url = "https://cfpub.epa.gov/ecotox/api/v1"

    pfas_names = [
        'perfluorooctanoic acid',
        'perfluorooctane sulfonic acid',
        'perfluorononanoic acid',
        'perfluorodecanoic acid',
        'perfluorohexanoic acid',
        'perfluorobutanoic acid',
        'perfluorobutane sulfonic acid',
        'perfluorohexane sulfonic acid',
    ]

    all_data = []

    for name in pfas_names:
        print(f"\n  搜索: {name}")
        try:
            # ECOTOX化学物质搜索
            url = f"{base_url}/chemical/search?search={urllib.parse.quote(name)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())

            if 'results' in data:
                chemicals = data['results']
                print(f"    找到 {len(chemicals)} 个匹配")

                for chem in chemicals[:3]:  # 取前3个
                    chem_id = chem.get('id', '')
                    chem_name = chem.get('name', '')

                    # 获取毒性数据
                    try:
                        tox_url = f"{base_url}/chemical/{chem_id}/results?limit=50"
                        req = urllib.request.Request(tox_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=30) as response:
                            tox_data = json.loads(response.read().decode())

                        if 'results' in tox_data:
                            for result in tox_data['results'][:10]:
                                all_data.append({
                                    'chemical': name,
                                    'species': result.get('species_name', ''),
                                    'endpoint': result.get('endpoint', ''),
                                    'value': result.get('value', ''),
                                    'unit': result.get('unit', ''),
                                    'duration': result.get('duration', ''),
                                    'reference': result.get('reference', ''),
                                })
                    except:
                        pass

            time.sleep(0.5)

        except Exception as e:
            print(f"    搜索失败: {e}")

    df = pd.DataFrame(all_data)
    print(f"\n  总共获取: {len(df)} 条毒性数据")
    return df


# ============================================================
# 3. 扩大PubMed文献搜索
# ============================================================
def search_more_papers(total_target=500):
    """扩大PubMed文献搜索到500篇"""
    print("\n" + "="*70)
    print(f"  步骤3：扩大PubMed文献搜索到{total_target}篇")
    print("="*70)

    # 更多搜索关键词
    queries = [
        # 毒性机制
        'PFOA toxicity mechanism',
        'PFOS toxicity mechanism',
        'PFAS endocrine disruption',
        'PFAS hepatotoxicity',
        'PFAS immunotoxicity',
        'PFAS developmental toxicity',
        'PFAS neurotoxicity',
        'PFAS reproductive toxicity',
        'PFAS carcinogenicity',

        # 环境归趋
        'PFAS environmental persistence',
        'PFAS bioaccumulation',
        'PFAS water contamination',
        'PFAS soil contamination',
        'PFAS atmospheric transport',

        # 健康效应
        'PFAS health effects human',
        'PFAS epidemiology cohort',
        'PFAS serum levels population',
        'PFAS thyroid disease',
        'PFAS cancer risk',
        'PFAS immune function children',

        # 风险评估
        'PFAS risk assessment',
        'PFAS drinking water standards',
        'PFAS regulation policy',
        'PFAS remediation treatment',

        # 具体化合物
        'GenX HFPO-DA toxicity',
        'PFBS short chain PFAS',
        'PFHxS toxicity',
        'PFNA health effects',
        'PFDA toxicology',

        # 新兴PFAS
        'emerging PFAS alternatives',
        'short chain PFAS toxicity',
        'PFAS replacement compounds',
    ]

    all_pmids = set()

    for query in queries:
        if len(all_pmids) >= total_target:
            break

        print(f"  搜索: {query}")
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': 30,
                'retmode': 'json',
                'sort': 'relevance',
                'datetype': 'pdat',
                'mindate': '2015',
                'maxdate': '2025',
            }
            full_url = url + '?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode())
                pmids = data.get('esearchresult', {}).get('idlist', [])
                all_pmids.update(pmids)
                print(f"    找到 {len(pmids)} 篇 (总计: {len(all_pmids)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"    搜索失败: {e}")

    print(f"\n  总共找到 {len(all_pmids)} 篇不重复文献")
    return list(all_pmids)


# ============================================================
# 4. 获取文献详情
# ============================================================
def fetch_papers_batch(pmids, batch_size=100):
    """批量获取文献详情"""
    print("\n" + "="*70)
    print("  步骤4：获取文献详细信息")
    print("="*70)

    papers = []

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

                pmid_elem = article.find('.//PMID')
                paper['pmid'] = pmid_elem.text if pmid_elem is not None else ''

                title_elem = article.find('.//ArticleTitle')
                paper['title'] = title_elem.text if title_elem is not None else ''

                abstract_parts = []
                for abstract_text in article.findall('.//AbstractText'):
                    if abstract_text.text:
                        abstract_parts.append(abstract_text.text)
                paper['abstract'] = ' '.join(abstract_parts)

                authors = []
                for author in article.findall('.//Author'):
                    last_name = author.find('LastName')
                    first_name = author.find('ForeName')
                    if last_name is not None and first_name is not None:
                        authors.append(f"{last_name.text} {first_name.text}")
                paper['authors'] = '; '.join(authors[:5])

                journal_elem = article.find('.//Journal/Title')
                paper['journal'] = journal_elem.text if journal_elem is not None else ''

                year_elem = article.find('.//PubDate/Year')
                paper['year'] = year_elem.text if year_elem is not None else ''

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
# 5. 提取毒理数据
# ============================================================
def extract_toxicity_data(papers):
    """从文献中提取毒理数据"""
    print("\n" + "="*70)
    print("  步骤5：从文献中提取毒理数据")
    print("="*70)

    pfas_aliases = {
        'PFOA': ['PFOA', 'perfluorooctanoic', 'C8HF15O2'],
        'PFOS': ['PFOS', 'perfluorooctane sulfon', 'C8HF17O3S'],
        'PFNA': ['PFNA', 'perfluorononanoic'],
        'PFDA': ['PFDA', 'perfluorodecanoic'],
        'PFHxA': ['PFHxA', 'perfluorohexanoic'],
        'PFBA': ['PFBA', 'perfluorobutanoic'],
        'PFBS': ['PFBS', 'perfluorobutane sulfon'],
        'PFHxS': ['PFHxS', 'perfluorohexane sulfon'],
        'GenX': ['GenX', 'HFPO-DA', 'hexafluoropropylene oxide'],
        'TFA': ['TFA', 'trifluoroacetic'],
    }

    toxicity_keywords = {
        'NR-AhR': ['AhR', 'aryl hydrocarbon'],
        'NR-AR': ['androgen receptor', 'antiandrogenic'],
        'SR-MMP': ['mitochondrial', 'membrane potential'],
        'hepatotoxicity': ['liver', 'hepatotox', 'ALT', 'AST'],
        'immunotoxicity': ['immune', 'immunotox', 'antibody'],
        'endocrine': ['endocrine', 'thyroid', 'hormone'],
        'developmental': ['developmental', 'fetal', 'birth'],
        'BCF': ['bioconcentration', 'BCF', 'bioaccumulation'],
    }

    extracted = []

    for paper in papers:
        text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

        compounds_found = []
        for compound, aliases in pfas_aliases.items():
            for alias in aliases:
                if alias.lower() in text:
                    compounds_found.append(compound)
                    break

        if not compounds_found:
            continue

        endpoints_found = []
        for endpoint, keywords in toxicity_keywords.items():
            for kw in keywords:
                if kw.lower() in text:
                    endpoints_found.append(endpoint)
                    break

        if not endpoints_found:
            continue

        positive_kw = ['active', 'positive', 'significant', 'increase', 'induction', 'toxic', 'adverse', 'effect', 'associated', 'elevated', 'higher']
        negative_kw = ['inactive', 'negative', 'no effect', 'no significant', 'no association', 'decrease', 'lower']

        pos_count = sum(1 for kw in positive_kw if kw in text)
        neg_count = sum(1 for kw in negative_kw if kw in text)

        for compound in compounds_found:
            for endpoint in endpoints_found:
                if pos_count > neg_count:
                    activity = 1
                elif neg_count > pos_count:
                    activity = 0
                else:
                    activity = None

                extracted.append({
                    'pmid': str(paper.get('pmid', '')),
                    'title': str(paper.get('title', ''))[:100],
                    'journal': str(paper.get('journal', '')),
                    'year': str(paper.get('year', '')),
                    'compound': compound,
                    'endpoint': endpoint,
                    'activity': activity,
                })

    df = pd.DataFrame(extracted)
    print(f"  提取数据: {len(df)} 条")
    print(f"  化合物数: {df['compound'].nunique() if len(df) > 0 else 0}")
    print(f"  终点数: {df['endpoint'].nunique() if len(df) > 0 else 0}")

    return df


# ============================================================
# 6. 构建校正表
# ============================================================
def build_calibration(extracted_df):
    """构建校正表"""
    print("\n" + "="*70)
    print("  步骤6：构建校正表")
    print("="*70)

    calibration = {}

    for compound in extracted_df['compound'].unique():
        comp_data = extracted_df[extracted_df['compound'] == compound]
        calibration[compound] = {}

        for endpoint in comp_data['endpoint'].unique():
            ep_data = comp_data[comp_data['endpoint'] == endpoint]
            valid = ep_data.dropna(subset=['activity'])

            if len(valid) > 0:
                active_ratio = valid['activity'].mean()
                n_papers = len(valid)

                if n_papers >= 10:
                    confidence = 0.9
                elif n_papers >= 5:
                    confidence = 0.8
                elif n_papers >= 3:
                    confidence = 0.7
                else:
                    confidence = 0.6

                calibration[compound][endpoint] = {
                    'value': round(float(active_ratio), 2),
                    'confidence': confidence,
                    'n_papers': n_papers,
                    'pmids': valid['pmid'].tolist()[:10],
                }

    return calibration


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "★"*70)
    print("  从权威数据库获取真实PFAS毒理数据")
    print("★"*70)

    # 1. EPA CompTox
    comptox = fetch_comptox_data()

    # 2. ECOTOX
    ecotox = fetch_ecotox_data()

    # 3. 扩大PubMed搜索
    pmids = search_more_papers(500)

    # 4. 获取文献详情
    papers = fetch_papers_batch(pmids[:300])  # 最多300篇

    # 5. 提取毒理数据
    extracted = extract_toxicity_data(papers)

    # 6. 构建校正表
    calibration = build_calibration(extracted)

    # 保存数据
    print("\n" + "="*70)
    print("  保存数据")
    print("="*70)

    # 保存文献
    if papers:
        papers_df = pd.DataFrame(papers)
        path = os.path.join(PROJECT_DIR, 'data', 'pubmed_papers_500.csv')
        papers_df.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  文献: {path} ({len(papers_df)} 篇)")

    # 保存提取数据
    if len(extracted) > 0:
        path = os.path.join(PROJECT_DIR, 'data', 'pfas_toxicity_extracted_v2.csv')
        extracted.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  提取数据: {path} ({len(extracted)} 条)")

    # 保存校正表
    path = os.path.join(PROJECT_DIR, 'models', 'qsar', 'calibration_table_v2.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(calibration, f, ensure_ascii=False, indent=2)
    print(f"  校正表: {path}")

    # 保存ECOTOX数据
    if len(ecotox) > 0:
        path = os.path.join(PROJECT_DIR, 'data', 'ecotox_pfas_data.csv')
        ecotox.to_csv(path, index=False, encoding='utf-8-sig')
        print(f"  ECOTOX数据: {path} ({len(ecotox)} 条)")

    # 总结
    print("\n" + "★"*70)
    print("  数据获取完成！")
    print("★"*70)
    print(f"\n  文献: {len(papers)} 篇")
    print(f"  提取数据: {len(extracted)} 条")
    print(f"  校正化合物: {len(calibration)} 种")
    print(f"  ECOTOX数据: {len(ecotox)} 条")

    # 显示校正表
    print("\n  校正表内容:")
    for compound, endpoints in calibration.items():
        print(f"    {compound}:")
        for ep, data in endpoints.items():
            print(f"      {ep}: 值={data['value']}, 置信度={data['confidence']}, 文献={data['n_papers']}")


if __name__ == '__main__':
    main()
