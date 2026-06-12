"""
系统架构图与技术路线图生成
Mermaid 代码 + 渲染
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_DIR = r"E:\桌面\项目"

# ============================================================
# 系统架构图（Mermaid）
# ============================================================
SYSTEM_ARCHITECTURE = """```mermaid
graph TB
    subgraph 用户层["👤 用户交互层"]
        UI1["毒性预测查询"]
        UI2["毒性机制解释"]
        UI3["风险评估报告"]
    end

    subgraph RAG层["🧠 上层：RAG 增强风险评估系统"]
        direction TB
        RAG_Core["LangChain RAG 引擎"]
        Retriever["混合检索器<br/>(向量+关键词)"]
        Reranker["Cohere Rerank<br/>重排序模块"]
        Compressor["LLMLingua<br/>上下文压缩"]
        Citation["引用溯源模块"]

        RAG_Core --> Retriever
        Retriever --> Reranker
        Reranker --> Compressor
        Compressor --> Citation
    end

    subgraph 知识层["📚 中层：多源异构知识库"]
        direction LR
        FAISS[("FAISS 向量库<br/>文献+法规+预测")]
        Neo4j[("Neo4j 知识图谱<br/>8类实体×12类关系")]
        Docs["文献数据库<br/>100篇高被引论文"]
        Regs["法规标准库<br/>中美欧国际标准"]

        Docs --> FAISS
        Regs --> FAISS
        Neo4j --> FAISS
    end

    subgraph 预测层["🔬 底层：GNN+QSAR 双模型预测引擎"]
        direction LR
        subgraph QSAR["QSAR 传统模型"]
            XGBoost["XGBoost"]
            LightGBM["LightGBM"]
            RF["Random Forest"]
            SVM["SVM"]
        end
        subgraph GNN["GNN 图模型"]
            GIN["GIN"]
            GAT["GAT"]
            GCN["GCN"]
            MPNN["MPNN"]
        end
        Ensemble["加权集成模型"]
        SHAP_M["SHAP/GNNExplainer<br/>模型解释"]

        XGBoost --> Ensemble
        LightGBM --> Ensemble
        GIN --> Ensemble
        GAT --> Ensemble
        Ensemble --> SHAP_M
    end

    subgraph 数据层["💾 数据基础层"]
        direction LR
        Tox21["Tox21<br/>毒理数据"]
        PubChem["PubChem<br/>分子属性"]
        CompTox["EPA CompTox<br/>PFAS列表"]
        ECOTOX["ECOTOX<br/>生态毒理"]
        Literature["文献数据<br/>100篇论文"]
    end

    用户层 --> RAG层
    RAG层 --> 知识层
    知识层 --> 预测层
    预测层 --> 数据层

    UI1 -->|"SMILES/化合物名"| RAG_Core
    UI2 -->|"化合物名"| RAG_Core
    UI3 -->|"化合物名/SMILES"| RAG_Core

    RAG_Core -->|"向量检索"| FAISS
    RAG_Core -->|"图谱查询"| Neo4j
    RAG_Core -->|"调用模型"| Ensemble

    style 用户层 fill:#E3F2FD,stroke:#1565C0
    style RAG层 fill:#FFF3E0,stroke:#E65100
    style 知识层 fill:#E8F5E9,stroke:#2E7D32
    style 预测层 fill:#FCE4EC,stroke:#C62828
    style 数据层 fill:#F3E5F5,stroke:#6A1B9A
```

"""

# ============================================================
# 技术路线图（Mermaid）
# ============================================================
TECH_ROADMAP = """```mermaid
graph LR
    subgraph Phase1["阶段1：数据基础<br/>(2天)"]
        A1["多源数据库<br/>采集"] --> A2["数据清洗<br/>标准化"]
        A2 --> A3["分子描述符<br/>生成(500+)"]
        A2 --> A4["分子图<br/>数据集"]
    end

    subgraph Phase2["阶段2：模型构建<br/>(3天)"]
        B1["QSAR 6模型<br/>×8终点"] --> B3["模型解释<br/>(SHAP/LIME)"]
        B2["GNN 5模型<br/>×8终点"] --> B4["GNN解释<br/>(GNNExplainer)"]
        B3 --> B5["加权集成<br/>融合模型"]
        B4 --> B5
    end

    subgraph Phase3["阶段3：知识工程<br/>(2天)"]
        C1["100篇文献<br/>PDF解析"] --> C3["三元组<br/>抽取"]
        C2["法规标准<br/>采集"] --> C3
        C3 --> C4["Neo4j<br/>知识图谱"]
        C3 --> C5["FAISS<br/>向量库"]
    end

    subgraph Phase4["阶段4：RAG系统<br/>(2天)"]
        D1["LangChain<br/>RAG引擎"] --> D2["混合检索<br/>+重排序"]
        D2 --> D3["毒性预测<br/>查询功能"]
        D2 --> D4["毒性机制<br/>解释功能"]
        D2 --> D5["风险评估<br/>报告功能"]
    end

    subgraph Phase5["阶段5：集成测试<br/>(1天)"]
        E1["新兴PFAS<br/>大规模预测"] --> E3["系统演示<br/>界面"]
        E2["风险分级<br/>与报告"] --> E3
        E3 --> E4["竞赛文档<br/>准备"]
    end

    Phase1 --> Phase2
    Phase2 --> Phase3
    Phase3 --> Phase4
    Phase4 --> Phase5

    style Phase1 fill:#E3F2FD,stroke:#1565C0
    style Phase2 fill:#FCE4EC,stroke:#C62828
    style Phase3 fill:#E8F5E9,stroke:#2E7D32
    style Phase4 fill:#FFF3E0,stroke:#E65100
    style Phase5 fill:#F3E5F5,stroke:#6A1B9A
```

"""

# ============================================================
# 数据流图
# ============================================================
DATA_FLOW = """```mermaid
flowchart TD
    DB1[("Tox21\n毒理数据")] -->|筛选PFAS| Raw["原始PFAS\n数据集(500+)"]
    DB2[("PubChem\n分子属性")] -->|SMILES+属性| Raw
    DB3[("EPA CompTox\nPFAS列表")] -->|化合物列表| Raw

    Raw -->|去重+清洗| Clean["清洗后\n数据集"]
    Clean -->|RDKit| Desc["分子描述符\n(500+特征)"]
    Clean -->|PyG| Graph["分子图\n数据集"]
    Clean -->|8终点| Labels["毒性标签\n矩阵"]

    Desc -->|标准化| QSAR["QSAR模型\n(6种算法)"]
    Graph -->|图特征| GNN["GNN模型\n(5种架构)"]
    Labels --> QSAR
    Labels --> GNN

    QSAR -->|预测概率| Ensemble["集成模型"]
    GNN -->|预测概率| Ensemble

    Ensemble -->|预测100种| Emerging["新兴PFAS\n替代物预测"]

    PFAS_KG["PFAS知识图谱\n(Neo4j)"] --> VectorDB["FAISS向量库"]
    Literature["100篇文献"] -->|解析| VectorDB
    Regulations["法规标准"] -->|编码| VectorDB
    Emerging -->|结果| VectorDB

    VectorDB --> RAG["RAG系统"]
    PFAS_KG --> RAG

    RAG --> F1["毒性查询"]
    RAG --> F2["机制解释"]
    RAG --> F3["风险报告"]
```

"""

def main():
    output_dir = os.path.join(PROJECT_DIR, "07_可视化与报告")
    os.makedirs(output_dir, exist_ok=True)

    # 保存 Mermaid 源码
    with open(os.path.join(output_dir, "system_architecture.mmd"), 'w', encoding='utf-8') as f:
        f.write("graph TB\n" + SYSTEM_ARCHITECTURE.split("graph TB\n")[1].split("```")[0])

    with open(os.path.join(output_dir, "tech_roadmap.mmd"), 'w', encoding='utf-8') as f:
        f.write("graph LR\n" + TECH_ROADMAP.split("graph LR\n")[1].split("```")[0])

    with open(os.path.join(output_dir, "data_flow.mmd"), 'w', encoding='utf-8') as f:
        f.write("flowchart TD\n" + DATA_FLOW.split("flowchart TD\n")[1].split("```")[0])

    # 保存完整 Markdown（包含 Mermaid）
    architecture_md = f"""# 系统架构与技术路线图

## 1. 系统架构图

{SYSTEM_ARCHITECTURE}

## 2. 技术路线图

{TECH_ROADMAP}

## 3. 数据流图

{DATA_FLOW}

## 4. 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| **数据层** | Tox21, PubChem, EPA CompTox, ECOTOX | 多源PFAS数据采集 |
| **描述符** | RDKit, PaDEL-Descriptor | 500+分子描述符生成 |
| **图数据** | PyTorch Geometric | 分子图构建 |
| **QSAR** | XGBoost, LightGBM, SVM, RF, GBDT, LR | 传统ML预测 |
| **GNN** | GCN, GAT, GIN, MPNN, AGNN | 图神经网络预测 |
| **解释性** | SHAP, LIME, GNNExplainer | 模型解释 |
| **知识图谱** | Neo4j, NetworkX | PFAS知识图谱 |
| **向量库** | FAISS, bge-large-zh | 向量检索 |
| **RAG** | LangChain, Cohere Rerank, LLMLingua | 增强检索生成 |
| **可视化** | Matplotlib, Seaborn | 图表生成 |

## 5. 创新点

1. **双模型融合**: QSAR(传统描述符) + GNN(图特征) 互补，提升预测精度
2. **全面覆盖**: 8个毒理学终点，覆盖内分泌干扰、细胞毒性、环境归趋
3. **知识驱动**: 100篇文献 + 法规标准构建的PFAS知识图谱
4. **RAG增强**: 检索增强生成技术，实现可信、可溯源的风险评估
5. **大规模筛查**: 100种新兴PFAS替代物的风险预测与分级
"""

    with open(os.path.join(output_dir, "系统架构与技术路线.md"), 'w', encoding='utf-8') as f:
        f.write(architecture_md)

    print(f"系统架构图和技术路线图已保存到: {output_dir}")
    print("Mermaid 源码文件:")
    print("  - system_architecture.mmd")
    print("  - tech_roadmap.mmd")
    print("  - data_flow.mmd")
    print("\n在线渲染: https://mermaid.live")

if __name__ == '__main__':
    main()
