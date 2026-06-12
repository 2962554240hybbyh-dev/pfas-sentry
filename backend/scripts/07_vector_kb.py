"""
FAISS 多源向量知识库构建
包含：文献、法规、预测结果、知识图谱三元组
"""
import sys, os, json, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# 文本数据准备
# ============================================================
def prepare_text_data():
    """准备所有需要向量化的文本数据"""
    documents = []
    metadata = []

    # 1. PFAS 文献综述内容（基于权威文献整理）
    literature_texts = [
        {
            'title': 'PFAS毒理学综述',
            'content': '''全氟和多氟烷基物质(PFAS)是一类人工合成的有机化合物，因其独特的化学稳定性被广泛应用于工业和消费品。
毒理学研究表明，PFAS可通过多种机制产生毒性效应：
1) PPARα激活：PFAS可激活过氧化物酶体增殖物激活受体α(PPARα)，导致肝脏脂质代谢紊乱，引起肝细胞增殖和肝肿大。
2) 氧化应激：PFAS可诱导活性氧(ROS)产生，导致DNA氧化损伤和脂质过氧化。
3) 线粒体功能障碍：PFAS可干扰线粒体电子传递链，影响细胞能量代谢。
4) 甲状腺激素干扰：PFAS可竞争性结合甲状腺激素转运蛋白，干扰甲状腺激素的正常代谢。
5) 免疫抑制：PFAS可抑制免疫细胞功能，降低疫苗接种后的抗体反应。
PFOA和PFOS是最受关注的两种PFAS，已被证实具有肝毒性、发育毒性和潜在致癌性。''',
            'source': '毒理学综述',
            'type': 'literature'
        },
        {
            'title': 'PFAS环境归趋',
            'content': '''PFAS因其碳-氟键的极高稳定性，在环境中具有极强的持久性，被称为"永久化学品"。
环境归趋特征：
1) 水环境：PFAS广泛存在于地表水、地下水和饮用水中。全球多地饮用水检出PFAS，浓度范围从ng/L到μg/L。
2) 土壤：PFAS可通过污水灌溉、大气沉降和废弃物填埋进入土壤，在土壤中具有较强的吸附性和迁移性。
3) 大气：挥发性PFAS前体物质可在大气中远距离传输，降解生成持久性PFAS。
4) 生物富集：长链PFAS(C≥7)具有显著的生物富集能力，可在食物链中逐级放大。
5) 全球分布：PFAS已在极地地区、深海和偏远山区检出，表明其具有全球性传输特征。
PFAS的环境半衰期可达数十年甚至数百年，远超传统有机污染物。''',
            'source': '环境科学综述',
            'type': 'literature'
        },
        {
            'title': 'PFAS健康效应研究',
            'content': '''流行病学研究已揭示PFAS暴露与多种健康效应之间的关联：
1) 肝脏效应：血清PFAS水平升高与肝酶(ALT、AST)升高呈正相关，长期暴露可导致非酒精性脂肪肝。
2) 甲状腺疾病：PFAS暴露与甲状腺功能异常(包括甲亢和甲减)风险增加相关。
3) 免疫功能：PFAS暴露可降低儿童疫苗接种后的抗体反应，增加感染性疾病风险。
4) 生殖发育：PFAS暴露与生育力下降、低出生体重、早产等不良妊娠结局相关。
5) 肾脏效应：PFAS暴露与慢性肾病风险增加有关，可能通过肾小管损伤机制。
6) 代谢综合征：PFAS暴露与肥胖、2型糖尿病和血脂异常风险增加相关。
7) 癌症风险：PFOA被IARC列为2B类可能致癌物，与肾癌和睾丸癌风险增加有关。
这些健康效应的剂量-反应关系因PFAS种类和暴露时间而异。''',
            'source': '环境与健康研究',
            'type': 'literature'
        },
        {
            'title': 'PFAS毒性机制研究进展',
            'content': '''近年来PFAS毒性机制研究取得了重要进展：
1) 受体介导机制：PFAS可通过激活PPARα、PPARγ、CAR、PXR等核受体，调控下游基因表达，导致代谢紊乱。
2) 表观遗传学效应：PFAS可引起DNA甲基化模式改变，影响基因表达的长期调控。
3) 肠道菌群影响：PFAS暴露可改变肠道微生物组成，影响免疫功能和代谢稳态。
4) 蛋白质结合：PFAS可与血浆蛋白(如白蛋白)高度结合，影响蛋白质的正常功能。
5) 膜干扰效应：PFAS可插入细胞膜磷脂双分子层，影响膜流动性和信号转导。
6) 代际传递：PFAS可通过胎盘和母乳传递给下一代，对发育中的胎儿和婴儿产生毒性。
短链PFAS虽然生物富集性较低，但仍可通过上述机制产生毒性效应。''',
            'source': '分子毒理学研究',
            'type': 'literature'
        },
        {
            'title': 'PFAS环境风险评估方法',
            'content': '''PFAS环境风险评估通常包括以下步骤：
1) 危害识别：通过毒理学研究确定PFAS的毒性终点和剂量-反应关系。
2) 暴露评估：通过环境监测数据和人群生物监测数据评估PFAS暴露水平。
3) 风险表征：计算危害商数(HQ)和累积风险指数(HI)，评估风险水平。
4) 不确定性分析：考虑数据质量、模型假设和种属外推等因素。
常用的风险评估工具包括：
- QSAR模型：基于分子结构预测PFAS毒性
- 毒性当量因子法：评估PFAS混合物的累积毒性
- 概率风险评估：考虑参数变异性和不确定性
- 生物监测等效浓度(BEC)：将血液PFAS浓度转换为等效暴露浓度
最新的研究趋势是将机器学习和人工智能技术应用于PFAS风险评估。''',
            'source': '风险评估方法学',
            'type': 'literature'
        },
    ]

    # 2. 法规标准文本
    regulation_texts = [
        {
            'title': '中国新污染物治理行动方案',
            'content': '''2022年5月，国务院办公厅印发《新污染物治理行动方案》，将PFAS列为重点管控的新污染物。
主要措施包括：
1) 完善新污染物环境风险评估体系
2) 加强新污染物源头管控
3) 强化新污染物过程控制
4) 深化新污染物末端治理
5) 加强新污染物监测能力建设
该方案要求到2025年，完成重点区域PFAS环境风险评估，建立健全PFAS环境管控制度。''',
            'source': '中国法规',
            'type': 'regulation'
        },
        {
            'title': 'GB 5749-2022 生活饮用水卫生标准',
            'content': '''GB 5749-2022《生活饮用水卫生标准》于2023年4月1日正式实施。
PFAS相关限值：
- PFOS(全氟辛烷磺酸): 40 ng/L
- PFOA(全氟辛酸): 40 ng/L
该标准首次将PFAS纳入饮用水水质参考指标，标志着中国对PFAS污染的重视程度显著提升。
与国际标准相比：
- 美国EPA: PFOA和PFOS各4 ng/L (2023年更新)
- 欧盟: PFOA 100 ng/L (2020年指令)
- WHO: PFOA 100 ng/L, PFOS 40 ng/L (2022年临时指导值)''',
            'source': '中国标准',
            'type': 'regulation'
        },
        {
            'title': 'EPA PFAS管控标准',
            'content': '''美国环境保护局(EPA)对PFAS实施了全面的管控措施：
1) 饮用水标准：2023年发布PFAS国家一级饮用水法规(NPDWR)，设定PFOA和PFOS各4 ng/L的强制性标准。
2) PFAS报告规则：要求企业报告自2011年以来PFAS的制造、加工和使用情况。
3) 有毒物质控制法(TSCA)：将PFAS列入重点评估物质清单。
4) Superfund场地：将PFAS污染场地纳入国家优先清单(NPL)。
5) 国际合作：推动将PFAS列入斯德哥尔摩公约持久性有机污染物(POPs)清单。
EPA还发布了PFAS生态毒性筛选值和人体健康参考剂量(RfD)。''',
            'source': '美国法规',
            'type': 'regulation'
        },
        {
            'title': '欧盟REACH法规PFAS限制',
            'content': '''欧盟对PFAS实施了严格的管控措施：
1) PFOS限制：(EC) No 850/2004法规限制PFOS的使用和销售，限值为0.005%。
2) PFOA限制：2020年将PFOA及其盐类和相关物质列入REACH附件XVII限制清单。
3) PFHxS限制：2022年提议限制PFHxS及其盐类。
4) 整体PFAS限制提案：2023年由德国、荷兰、丹麦、挪威和瑞典联合提交了限制所有PFAS的提案，涵盖超过10000种物质。
5) 食品接触材料：(EU) 2022/1616法规限制PFAS在食品接触材料中的使用。
欧盟的目标是到2030年基本消除PFAS的非必要使用。''',
            'source': '欧盟法规',
            'type': 'regulation'
        },
        {
            'title': '斯德哥尔摩公约PFAS相关条款',
            'content': '''《斯德哥尔摩公约》是关于持久性有机污染物(POPs)的国际公约。
PFAS相关进展：
1) PFOS：2009年列入公约附件B（限制类），允许特定豁免用途。
2) PFOA：2019年列入公约附件A（消除类），允许特定豁免。
3) PFHxS：2022年列入公约附件A，全面禁止生产和使用。
4) 长链PFCAs：正在评估中，可能在近期列入公约。
公约要求各缔约方采取措施减少或消除PFAS的排放，并推动替代品的开发和应用。''',
            'source': '国际公约',
            'type': 'regulation'
        },
    ]

    # 3. PFAS预测结果文本
    prediction_texts = []
    try:
        pred_df = pd.read_csv(os.path.join(PROJECT_DIR, "04_模型融合与预测", "emerging_pfas_predictions.csv"))
        for _, row in pred_df.head(30).iterrows():
            text = f"新兴PFAS替代物 {row.get('ID', 'Unknown')} ({row.get('Name', '')}): "
            text += f"类别={row.get('Category', '')}, "
            for ep in ['NR-AR', 'NR-AhR', 'SR-MMP', 'SR-p53']:
                if ep in row.index:
                    text += f"{ep}={row[ep]:.3f}, "
            text += f"综合风险={row.get('综合风险分数', 0):.3f}, 等级={row.get('风险等级', 'N/A')}"
            prediction_texts.append({
                'title': f"预测_{row.get('ID', 'Unknown')}",
                'content': text,
                'source': 'QSAR预测',
                'type': 'prediction'
            })
    except:
        pass

    # 4. 知识图谱三元组文本
    kg_texts = []
    try:
        kg_df = pd.read_csv(os.path.join(PROJECT_DIR, "05_知识图谱", "pfas_kg_triples.csv"))
        # 按实体分组生成文本
        for head in kg_df['head'].unique()[:30]:
            related = kg_df[kg_df['head'] == head]
            text = f"关于{head}的知识：\n"
            for _, row in related.iterrows():
                text += f"- {head} {row['relation']} {row['tail']}\n"
            kg_texts.append({
                'title': f"KG_{head}",
                'content': text.strip(),
                'source': '知识图谱',
                'type': 'knowledge_graph'
            })
    except:
        pass

    all_docs = literature_texts + regulation_texts + prediction_texts + kg_texts
    return all_docs

# ============================================================
# FAISS 向量库构建
# ============================================================
def build_faiss_index(documents, output_dir):
    """构建 FAISS 向量索引"""
    import faiss

    print("构建向量嵌入...")

    # 使用简单的 TF-IDF 向量化（不依赖外部模型）
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [doc['content'] for doc in documents]

    # TF-IDF 向量化
    vectorizer = TfidfVectorizer(
        max_features=1024,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    vectors = vectorizer.fit_transform(texts).toarray().astype('float32')

    # 归一化
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    print(f"  向量维度: {vectors.shape[1]}")
    print(f"  文档数量: {vectors.shape[0]}")

    # 构建 FAISS 索引
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)  # 内积相似度
    index.add(vectors)

    # 保存索引（FAISS C++ 后端不支持中文路径，先存到临时路径再移动）
    import tempfile, shutil
    tmp_dir = tempfile.mkdtemp()
    tmp_index_path = os.path.join(tmp_dir, "pfas_faiss_index.index")
    faiss.write_index(index, tmp_index_path)

    index_path = os.path.join(output_dir, "pfas_faiss_index.index")
    shutil.move(tmp_index_path, index_path)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  FAISS 索引已保存: {index_path}")

    # 保存元数据
    metadata = []
    for i, doc in enumerate(documents):
        metadata.append({
            'id': i,
            'title': doc['title'],
            'source': doc['source'],
            'type': doc['type'],
            'content_preview': doc['content'][:200]
        })

    metadata_path = os.path.join(output_dir, "pfas_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  元数据已保存: {metadata_path}")

    # 保存向量化器
    import pickle
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    return index, vectorizer, metadata

# ============================================================
# 检索功能
# ============================================================
def search(query, index, vectorizer, metadata, documents, top_k=5):
    """检索最相关的文档"""
    # 向量化查询
    query_vec = vectorizer.transform([query]).toarray().astype('float32')
    norms = np.linalg.norm(query_vec, axis=1, keepdims=True)
    norms[norms == 0] = 1
    query_vec = query_vec / norms

    # FAISS 检索
    scores, indices = index.search(query_vec, min(top_k, len(documents)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and idx < len(documents):
            results.append({
                'score': float(score),
                'title': documents[idx]['title'],
                'content': documents[idx]['content'],
                'source': documents[idx]['source'],
                'type': documents[idx]['type']
            })

    return results

# ============================================================
# 主流程
# ============================================================
def main():
    # 使用原始字符串路径避免编码问题
    output_dir = r"E:\桌面\项目\06_RAG系统"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("FAISS 多源向量知识库构建")
    print("=" * 60)

    # 1. 准备文本数据
    print("\n第1步：准备文本数据...")
    documents = prepare_text_data()
    print(f"  文档总数: {len(documents)}")

    type_counts = {}
    for doc in documents:
        t = doc['type']
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"  文档类型分布: {type_counts}")

    # 2. 构建 FAISS 索引
    print("\n第2步：构建 FAISS 向量索引...")
    index, vectorizer, metadata = build_faiss_index(documents, output_dir)

    # 3. 测试检索
    print("\n第3步：测试检索功能...")
    test_queries = [
        "PFOA 的毒性机制是什么？",
        "PFAS 饮用水标准是多少？",
        "PFAS 对肝脏有什么影响？",
        "新兴PFAS替代物的风险如何？",
    ]

    for query in test_queries:
        print(f"\n  查询: {query}")
        results = search(query, index, vectorizer, metadata, documents, top_k=3)
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r['title']} (相似度={r['score']:.3f}, 来源={r['source']})")

    print(f"\n{'='*60}")
    print("向量知识库构建完成！")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

if __name__ == '__main__':
    main()
