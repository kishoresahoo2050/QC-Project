"""
backend/workflow/workflow.py
LangGraph state machines for:
  1. Sensor anomaly detection + root cause analysis (run_analysis_workflow)
  2. User chat with RAG + memory (run_chat_workflow)
"""

import asyncio
from typing import TypedDict, Optional, List, Any

import time
import json

from langgraph.graph import StateGraph, END

from backend.rag.rag_pipeline import retrieve_context, build_llm
from backend.config import settings


# ─────────────────────────────────────────────────────────────────
# State definitions
# ─────────────────────────────────────────────────────────────────


class AnalysisState(TypedDict):
    reading_id: str
    sensor_data: dict
    is_anomaly: bool
    severity: str
    anomaly_reasons: List[str]
    context_docs: List[str]
    root_cause: Optional[str]
    recommendation: Optional[str]
    confidence: Optional[float]
    retrieved_docs: int
    sources: List[str]


class ChatState(TypedDict):
    user_message: str
    history: List[dict]
    user_id: str
    context_docs: List[str]
    sources: List[str]
    response: Optional[str]


# ─────────────────────────────────────────────────────────────────
# Anomaly Detection Workflow
# ─────────────────────────────────────────────────────────────────

THRESHOLDS = {
    "temperature": {"warn": 80.0, "critical": 95.0},
    "pressure": {"warn": 150.0, "critical": 180.0},
    "vibration": {"warn": 5.0, "critical": 8.0},
    "defect_rate": {"warn": 0.05, "critical": 0.10},
    "production_speed": {"warn": 10.0, "critical": 5.0},  # low speed = warn
}


def detect_anomaly_node(state: AnalysisState) -> AnalysisState:
    """Rule-based anomaly detection; produces severity and reasons."""
    data = state["sensor_data"]
    reasons = []
    severity = "low"

    for metric, limits in THRESHOLDS.items():
        value = data.get(metric, 0)
        if metric == "production_speed":
            # Low production speed is the anomaly
            if value <= limits["critical"]:
                reasons.append(f"Production speed critically low ({value:.1f} u/min)")
                severity = "critical"
            elif value <= limits["warn"]:
                reasons.append(f"Production speed below threshold ({value:.1f} u/min)")
                if severity not in ("critical",):
                    severity = "high"
        else:
            if value >= limits["critical"]:
                reasons.append(f"{metric.capitalize()} critically high ({value:.1f})")
                severity = "critical"
            elif value >= limits["warn"]:
                reasons.append(f"{metric.capitalize()} elevated ({value:.1f})")
                if severity == "low":
                    severity = "medium"

    is_anomaly = len(reasons) > 0
    if is_anomaly and severity == "low":
        severity = "medium"

    return {
        **state,
        "is_anomaly": is_anomaly,
        "severity": severity,
        "anomaly_reasons": reasons,
    }


async def retrieve_context_node(state: AnalysisState) -> AnalysisState:
    """RAG retrieval: fetch relevant manufacturing knowledge."""
    if not state["is_anomaly"]:
        return {**state, "context_docs": [], "sources": [], "retrieved_docs": 0}

    query = f"QC issue: {', '.join(state['anomaly_reasons'][:3])}"
    print("Query: ", query)
    docs, sources = await retrieve_context(query, k=4)
    print("Docs: ", docs)
    print("Sources: ", sources)
    return {
        **state,
        "context_docs": docs,
        "sources": sources,
        "retrieved_docs": len(docs),
    }


async def analyze_root_cause_node(state: AnalysisState) -> AnalysisState:
    """LLM root cause analysis using retrieved context."""

    print("\n========== ENTER analyze_root_cause_node ==========")
    print("Incoming state:")
    print(json.dumps(state, indent=2, default=str))

    # Skip if no anomaly
    if not state["is_anomaly"]:
        print("No anomaly detected. Skipping LLM call.")
        return {
            **state,
            "root_cause": None,
            "recommendation": None,
            "confidence": None,
        }

    llm = build_llm()
    sensor = state["sensor_data"]

    reasons = "\n".join(f"  - {r}" for r in state["anomaly_reasons"])
    context = (
        "\n\n".join(state["context_docs"][:3])
        if state["context_docs"]
        else "No additional context available."
    )

    prompt = f"""You are an expert manufacturing quality control engineer.

Sensor Reading:
  Machine ID       : {sensor.get("machine_id", "Unknown")}
  Temperature      : {sensor.get("temperature", 0):.1f} °C
  Pressure         : {sensor.get("pressure", 0):.1f} kPa
  Vibration        : {sensor.get("vibration", 0):.2f} mm/s
  Defect Rate      : {sensor.get("defect_rate", 0) * 100:.2f}%
  Production Speed : {sensor.get("production_speed", 0):.1f} u/min

Detected Anomalies:
{reasons}

Relevant Knowledge Base Context:
{context}

Provide:
1. ROOT CAUSE: Concise diagnosis of the most likely cause (2-3 sentences).
2. RECOMMENDATION: Immediate corrective actions (2-3 bullet points).
3. CONFIDENCE: Your confidence level 0.0-1.0 as a number.

Format exactly:
ROOT CAUSE: <text>
RECOMMENDATION: <text>
CONFIDENCE: <number>"""

    # 🔍 Print prompt
    print("\n---------- LLM REQUEST (PROMPT) ----------")
    print(prompt)

    # ⏱️ Measure latency
    start = time.time()
    response = await asyncio.to_thread(llm.invoke, prompt)
    print(f"\n⏱️ LLM Latency: {time.time() - start:.2f} sec")

    # 🔍 Raw response
    print("\n---------- RAW LLM RESPONSE OBJECT ----------")
    print(response)

    text = response.content if hasattr(response, "content") else str(response)

    print("\n---------- LLM RESPONSE TEXT ----------")
    print(text)

    # =======================
    # ✅ ROBUST PARSER
    # =======================
    root_cause = None
    recommendation = None
    confidence = 0.7

    current_section = None
    buffer = []

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("ROOT CAUSE:"):
            if current_section == "recommendation":
                recommendation = "\n".join(buffer).strip()
            elif current_section == "root_cause":
                root_cause = "\n".join(buffer).strip()

            buffer = []
            current_section = "root_cause"

            content = line[len("ROOT CAUSE:") :].strip()
            if content:
                buffer.append(content)

        elif line.startswith("RECOMMENDATION:"):
            if current_section == "root_cause":
                root_cause = "\n".join(buffer).strip()
            elif current_section == "recommendation":
                recommendation = "\n".join(buffer).strip()

            buffer = []
            current_section = "recommendation"

            content = line[len("RECOMMENDATION:") :].strip()
            if content:
                buffer.append(content)

        elif line.startswith("CONFIDENCE:"):
            if current_section == "root_cause":
                root_cause = "\n".join(buffer).strip()
            elif current_section == "recommendation":
                recommendation = "\n".join(buffer).strip()

            buffer = []
            current_section = None

            try:
                confidence = float(line[len("CONFIDENCE:") :].strip())
            except ValueError:
                print("⚠️ Failed to parse confidence")

        else:
            if current_section:
                buffer.append(line)

    # Final flush
    if current_section == "root_cause":
        root_cause = "\n".join(buffer).strip()
    elif current_section == "recommendation":
        recommendation = "\n".join(buffer).strip()

    # 🔍 Final parsed output
    print("\n========== PARSED OUTPUT ==========")
    print("ROOT CAUSE:\n", root_cause)
    print("\nRECOMMENDATION:\n", recommendation)
    print("\nCONFIDENCE:", confidence)
    print("========== EXIT analyze_root_cause_node ==========\n")

    return {
        **state,
        "root_cause": root_cause,
        "recommendation": recommendation,
        "confidence": confidence,
    }


def build_analysis_workflow():
    g = StateGraph(AnalysisState)
    g.add_node("detect_anomaly", detect_anomaly_node)
    g.add_node("retrieve_context", retrieve_context_node)
    g.add_node("analyze_root_cause", analyze_root_cause_node)

    g.set_entry_point("detect_anomaly")
    g.add_edge("detect_anomaly", "retrieve_context")
    g.add_edge("retrieve_context", "analyze_root_cause")
    g.add_edge("analyze_root_cause", END)
    return g.compile()


_analysis_app = None


async def run_analysis_workflow(reading_id: str, sensor_data: dict) -> dict:
    global _analysis_app
    if _analysis_app is None:
        _analysis_app = build_analysis_workflow()

    initial: AnalysisState = {
        "reading_id": reading_id,
        "sensor_data": sensor_data,
        "is_anomaly": False,
        "severity": "low",
        "anomaly_reasons": [],
        "context_docs": [],
        "root_cause": None,
        "recommendation": None,
        "confidence": None,
        "retrieved_docs": 0,
        "sources": [],
    }
    result = await _analysis_app.ainvoke(initial)
    return result


# ─────────────────────────────────────────────────────────────────
# Chat Workflow
# ─────────────────────────────────────────────────────────────────


async def chat_retrieve_node(state: ChatState) -> ChatState:
    docs, sources = await retrieve_context(state["user_message"], k=3)
    return {**state, "context_docs": docs, "sources": sources}


async def chat_respond_node(state: ChatState) -> ChatState:
    llm = build_llm()

    history_text = ""
    for msg in state["history"][-10:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    context = "\n\n".join(state["context_docs"]) if state["context_docs"] else ""

    prompt = f"""You are an AI assistant specialising in manufacturing quality control.
Use the provided knowledge base context and conversation history to answer helpfully.

Knowledge Base Context:
{context or "No specific context retrieved."}

Conversation History:
{history_text or "No prior history."}

Current User Question: {state["user_message"]}

Answer:"""

    response = await asyncio.to_thread(llm.invoke, prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return {**state, "response": answer}


def build_chat_workflow():
    g = StateGraph(ChatState)
    g.add_node("retrieve", chat_retrieve_node)
    g.add_node("respond", chat_respond_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", END)
    return g.compile()


_chat_app = None


async def run_chat_workflow(
    user_message: str, history: List[dict], user_id: str
) -> dict:
    global _chat_app
    if _chat_app is None:
        _chat_app = build_chat_workflow()

    initial: ChatState = {
        "user_message": user_message,
        "history": history,
        "user_id": user_id,
        "context_docs": [],
        "sources": [],
        "response": None,
    }
    result = await _chat_app.ainvoke(initial)
    return result
