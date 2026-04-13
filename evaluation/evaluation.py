"""
evaluation/evaluation.py
RAGAS evaluation for RAG pipeline quality.
Arize Phoenix tracing for LangGraph observability.

Usage:
    python -m evaluation.evaluation
"""

import asyncio
import json
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field, asdict

from backend.config import settings
from backend.rag.rag_pipeline import retrieve_context, build_llm


# ── Data classes ───────────────────────────────────────────────────

@dataclass
class EvalSample:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class EvalResult:
    sample_id: int
    question: str
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Test questions ─────────────────────────────────────────────────

TEST_QUESTIONS = [
    {
        "question": "What causes high vibration in manufacturing equipment?",
        "ground_truth": "High vibration is typically caused by bearing wear or failure, shaft misalignment, imbalanced rotating components, or loose foundation bolts.",
    },
    {
        "question": "What should I do when defect rate exceeds 5%?",
        "ground_truth": "Inspect tools for wear, check raw material quality, verify machine calibration, review operator SOP compliance, and use statistical process control charts.",
    },
    {
        "question": "How do I handle a critical temperature anomaly?",
        "ground_truth": "Reduce production speed by 20%, check coolant levels, inspect bearings with infrared camera, and schedule preventive maintenance.",
    },
    {
        "question": "What does high pressure combined with low temperature indicate?",
        "ground_truth": "This combination may indicate a downstream blockage, filter clogging, or valve malfunction in the pneumatic or hydraulic system.",
    },
    {
        "question": "When should a machine be isolated during a multi-parameter anomaly?",
        "ground_truth": "Isolate the machine when critical thresholds are reached simultaneously across two or more parameters, per the multi-anomaly protocol.",
    },
]


# ── RAGAS evaluation ───────────────────────────────────────────────

async def generate_answer(question: str) -> tuple[str, List[str]]:
    """Generate an answer using the RAG pipeline."""
    contexts, sources = await retrieve_context(question, k=4)
    llm = build_llm()
    context_text = "\n\n".join(contexts) if contexts else "No context available."
    prompt = f"""Answer the following manufacturing QC question using the context below.

Context:
{context_text}

Question: {question}
Answer:"""
    response = await asyncio.to_thread(llm.invoke, prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, contexts


async def compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """
    Measure if claims in the answer are supported by the retrieved context.
    Simplified implementation — real RAGAS uses NLI models.
    """
    if not contexts:
        return 0.0
    llm = build_llm()
    ctx = "\n".join(contexts[:2])
    prompt = f"""Rate on a scale 0.0-1.0 how faithfully the answer is supported by the context.
Only output a number between 0.0 and 1.0.

Context: {ctx[:800]}
Answer: {answer[:400]}
Score:"""
    resp = await asyncio.to_thread(llm.invoke, prompt)
    txt = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    try:
        return min(1.0, max(0.0, float(txt.split()[0])))
    except (ValueError, IndexError):
        return 0.5


async def compute_answer_relevance(question: str, answer: str) -> float:
    """Measure how relevant the answer is to the question."""
    llm = build_llm()
    prompt = f"""Rate 0.0-1.0 how well this answer addresses the question. Output only a number.

Question: {question}
Answer: {answer[:400]}
Score:"""
    resp = await asyncio.to_thread(llm.invoke, prompt)
    txt = (resp.content if hasattr(resp, "content") else str(resp)).strip()
    try:
        return min(1.0, max(0.0, float(txt.split()[0])))
    except (ValueError, IndexError):
        return 0.5


async def compute_context_precision(question: str, contexts: List[str]) -> float:
    """Fraction of retrieved contexts that are actually relevant to the question."""
    if not contexts:
        return 0.0
    llm = build_llm()
    relevant = 0
    for ctx in contexts:
        prompt = f"""Is this context relevant to the question? Answer only yes or no.
Question: {question}
Context: {ctx[:400]}"""
        resp = await asyncio.to_thread(llm.invoke, prompt)
        answer_text = (resp.content if hasattr(resp, "content") else str(resp)).strip().lower()
        if "yes" in answer_text:
            relevant += 1
    return relevant / len(contexts)


async def evaluate_sample(idx: int, sample: dict) -> EvalResult:
    """Run full RAGAS-style evaluation on one question."""
    print(f"  [RAGAS] Evaluating Q{idx+1}: {sample['question'][:60]}...")
    question = sample["question"]

    answer, contexts = await generate_answer(question)

    faithfulness, relevance, precision = await asyncio.gather(
        compute_faithfulness(answer, contexts),
        compute_answer_relevance(question, answer),
        compute_context_precision(question, contexts),
    )

    overall = round((faithfulness + relevance + precision) / 3, 4)

    return EvalResult(
        sample_id=idx,
        question=question,
        faithfulness=round(faithfulness, 4),
        answer_relevance=round(relevance, 4),
        context_precision=round(precision, 4),
        overall_score=overall,
    )


async def run_ragas_evaluation() -> List[EvalResult]:
    """Run evaluation across all test questions."""
    print("\n[RAGAS] Starting evaluation suite...")
    tasks = [evaluate_sample(i, q) for i, q in enumerate(TEST_QUESTIONS)]
    results = await asyncio.gather(*tasks)
    return list(results)


def generate_eval_report(results: List[EvalResult]) -> dict:
    """Aggregate metrics and produce a summary report."""
    if not results:
        return {}

    avg_faithfulness   = sum(r.faithfulness for r in results) / len(results)
    avg_relevance      = sum(r.answer_relevance for r in results) / len(results)
    avg_precision      = sum(r.context_precision for r in results) / len(results)
    avg_overall        = sum(r.overall_score for r in results) / len(results)

    report = {
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "total_questions": len(results),
        "aggregate_metrics": {
            "avg_faithfulness":      round(avg_faithfulness, 4),
            "avg_answer_relevance":  round(avg_relevance, 4),
            "avg_context_precision": round(avg_precision, 4),
            "avg_overall_score":     round(avg_overall, 4),
        },
        "individual_results": [asdict(r) for r in results],
        "pass": avg_overall >= 0.70,
    }
    return report


def print_report(report: dict):
    print("\n" + "="*60)
    print("  RAGAS EVALUATION REPORT")
    print("="*60)
    agg = report["aggregate_metrics"]
    print(f"  Questions evaluated : {report['total_questions']}")
    print(f"  Faithfulness        : {agg['avg_faithfulness']:.2%}")
    print(f"  Answer Relevance    : {agg['avg_answer_relevance']:.2%}")
    print(f"  Context Precision   : {agg['avg_context_precision']:.2%}")
    print(f"  Overall Score       : {agg['avg_overall_score']:.2%}")
    print(f"  Status              : {'✓ PASS' if report['pass'] else '✗ FAIL'}")
    print("="*60)


# ── Arize Phoenix helpers ──────────────────────────────────────────

def setup_phoenix_tracer():
    """
    Initialise Arize Phoenix tracing for LangChain.
    Call once at application startup (already done in backend/main.py).
    """
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        session = px.launch_app()
        print(f"[Phoenix] Dashboard: {session.url}")

        provider = TracerProvider()
        exporter = OTLPSpanExporter(
            endpoint=f"http://{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}/v1/traces"
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument(tracer_provider=provider)
        print("[Phoenix] LangChain instrumentation active.")
        return session
    except ImportError:
        print("[Phoenix] Not installed. pip install arize-phoenix openinference-instrumentation-langchain")
        return None
    except Exception as e:
        print(f"[Phoenix] Error: {e}")
        return None


# ── Entry point ────────────────────────────────────────────────────

async def main():
    results = await run_ragas_evaluation()
    report = generate_eval_report(results)
    print_report(report)

    output_path = f"evaluation/eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[RAGAS] Report saved → {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
