"""
backend/rag/rag_pipeline.py
LangChain RAG pipeline with ChromaDB vector store.
Indexes manufacturing knowledge base documents and retrieves context for LLM.
"""

import asyncio
import os
from typing import List, Tuple

import chromadb

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain.schema import Document

from backend.config import settings

# ── Singleton state ────────────────────────────────────────────────
_vectorstore: Chroma | None = None
_embeddings: GoogleGenerativeAIEmbeddings | None = None


def build_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.GOOGLE_AI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=0.2,
    )


# def build_llm() -> ChatOpenAI:
#     return ChatOpenAI(
#         model=settings.OPENAI_MODEL,
#         openai_api_key=settings.OPENAI_API_KEY,
#         temperature=0.2,
#     )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", google_api_key=settings.GOOGLE_API_KEY
        )
    return _embeddings


# ── Seed knowledge base ────────────────────────────────────────────

KNOWLEDGE_BASE = [
    {
        "content": """High temperature anomaly in manufacturing equipment typically indicates:
1. Coolant system failure or blockage
2. Excessive friction due to worn bearings
3. Overloaded motor running beyond rated capacity
4. Blocked heat exchangers or cooling fins
Immediate actions: reduce production speed by 20%, check coolant levels,
inspect bearing temperatures with infrared camera, schedule preventive maintenance.""",
        "metadata": {
            "category": "temperature",
            "severity": "high",
            "source": "ops_manual_v3.pdf",
        },
    },
    {
        "content": """Elevated pressure readings in pneumatic/hydraulic systems indicate:
1. Downstream blockage or valve closure
2. Filter clogging requiring replacement
3. System leak causing pump overcompensation
4. Regulator malfunction
Actions: check pressure relief valves, inspect filter differential pressure,
trace all downstream components for blockages.""",
        "metadata": {
            "category": "pressure",
            "severity": "high",
            "source": "maintenance_guide.pdf",
        },
    },
    {
        "content": """High vibration levels in rotating machinery suggest:
1. Bearing wear or failure (most common >6 mm/s)
2. Shaft misalignment post-maintenance
3. Imbalanced rotating components
4. Loose foundation bolts
Actions: perform vibration spectrum analysis, check alignment with laser tools,
inspect and lubricate bearings, tighten mounting hardware.""",
        "metadata": {
            "category": "vibration",
            "severity": "medium",
            "source": "predictive_maint.pdf",
        },
    },
    {
        "content": """Increased defect rate (>5%) root causes in production:
1. Tool wear — inspect cutting edges, replace at scheduled intervals
2. Raw material variation — check incoming material certificates
3. Machine calibration drift — perform gauge R&R study
4. Operator error — review SOP compliance, retrain if required
5. Environmental factors — check temperature and humidity in production area
Statistical process control: plot Xbar-R charts, identify out-of-control points.""",
        "metadata": {
            "category": "defect_rate",
            "severity": "high",
            "source": "quality_sop_v2.pdf",
        },
    },
    {
        "content": """Production speed reduction causes and corrective actions:
1. Upstream material starvation — check feeder and conveyor systems
2. Machine cycle time increase — inspect actuator response times
3. Operator-initiated slowdown for quality concerns — review quality alerts
4. Control system fault — check PLC error logs
Target OEE: maintain >85%. Speed reductions below 60% of target require
immediate supervisor notification and root cause documentation.""",
        "metadata": {
            "category": "production_speed",
            "severity": "medium",
            "source": "ops_manual_v3.pdf",
        },
    },
    {
        "content": """Critical multi-parameter anomaly protocol:
When two or more parameters exceed warning thresholds simultaneously:
1. Trigger immediate supervisor alert
2. Reduce production speed to 50% of nominal
3. Isolate affected machine if critical threshold reached
4. Document all readings in non-conformance report (NCR)
5. Do not resume full production until root cause identified and corrected
Common combined failures: high temperature + high vibration = bearing failure,
high temperature + low speed = motor overload.""",
        "metadata": {
            "category": "multi_anomaly",
            "severity": "critical",
            "source": "emergency_procedures.pdf",
        },
    },
    {
        "content": """Predictive maintenance schedule for CNC machines:
Daily: visual inspection, coolant level check, chip evacuation
Weekly: lubrication of linear guides, spindle warm-up cycle, tool offset verification
Monthly: geometric accuracy check, hydraulic fluid analysis, electrical connection inspection
Quarterly: spindle bearing vibration analysis, servo motor current analysis, full calibration
Annual: complete overhaul, replace wear items, recertification""",
        "metadata": {
            "category": "maintenance",
            "severity": "low",
            "source": "maint_schedule_v1.pdf",
        },
    },
    {
        "content": """Quality control statistical methods:
- Control charts (Xbar-R, P-chart, C-chart): monitor process stability
- Process capability (Cp, Cpk): measure conformance to specification
- FMEA: identify and prioritise failure modes before they occur
- 8D Problem Solving: structured root cause and corrective action process
- Six Sigma DMAIC: define-measure-analyse-improve-control framework
Acceptance quality limit (AQL) for defect rate: 1.5% standard, 4.0% tightened inspection.""",
        "metadata": {
            "category": "quality_methods",
            "severity": "low",
            "source": "quality_handbook.pdf",
        },
    },
]


def _seed_vectorstore(vectorstore: Chroma):
    """Add knowledge base documents to the vector store if empty."""
    existing = vectorstore.get()
    if existing and len(existing.get("ids", [])) > 0:
        return  # Already seeded

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = []
    for item in KNOWLEDGE_BASE:
        chunks = splitter.create_documents(
            texts=[item["content"]],
            metadatas=[item["metadata"]],
        )
        docs.extend(chunks)

    vectorstore.add_documents(docs)
    print(f"[RAG] Seeded {len(docs)} document chunks into ChromaDB.")


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        _vectorstore = Chroma(
            client=client,
            collection_name="qc_knowledge",
            embedding_function=get_embeddings(),
        )
        _seed_vectorstore(_vectorstore)
    return _vectorstore


async def retrieve_context(query: str, k: int = 4) -> Tuple[List[str], List[str]]:
    """
    Async RAG retrieval. Returns (doc_texts, source_names).
    Runs ChromaDB similarity search in a thread to avoid blocking.
    """

    def _search():
        vs = get_vectorstore()
        results = vs.similarity_search_with_score(query, k=k)
        texts = []
        sources = []
        for doc, score in results:
            if score < 1.5:  # Filter low-relevance results
                texts.append(doc.page_content)
                sources.append(doc.metadata.get("source", "unknown"))
        return texts, sources

    return await asyncio.to_thread(_search)


async def ingest_custom_document(text: str, metadata: dict):
    """Ingest a custom document into the knowledge base at runtime."""

    def _ingest():
        vs = get_vectorstore()
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = splitter.create_documents(texts=[text], metadatas=[metadata])
        vs.add_documents(docs)
        return len(docs)

    count = await asyncio.to_thread(_ingest)
    print(f"[RAG] Ingested {count} new chunks.")
    return count
