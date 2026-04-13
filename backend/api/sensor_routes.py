"""
backend/api/sensor_routes.py
Sensor data ingestion endpoint + WebSocket broadcaster for live dashboard.
"""

import asyncio
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.database import SensorReading, AnalysisResult, get_db
from backend.api.schemas import SensorPayload, AnalysisOut
from backend.workflow.workflow import run_analysis_workflow

router = APIRouter(prefix="/api", tags=["Sensors"])


# ── WebSocket Connection Manager ───────────────────────────────────


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        payload = json.dumps(data, default=str)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ── WebSocket Endpoint ─────────────────────────────────────────────


@router.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """Streamlit dashboard connects here to receive live sensor events."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive; server pushes data via broadcast()
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ── Ingest Endpoint (called by simulator) ─────────────────────────


@router.post("/ingest", response_model=AnalysisOut)
async def ingest_sensor_data(
    payload: SensorPayload,
    db: AsyncSession = Depends(get_db),
):
    ts = payload.timestamp or datetime.utcnow()

    # Persist raw reading
    reading = SensorReading(
        machine_id=payload.machine_id,
        temperature=payload.temperature,
        pressure=payload.pressure,
        vibration=payload.vibration,
        defect_rate=payload.defect_rate,
        production_speed=payload.production_speed,
        timestamp=ts,
    )
    db.add(reading)
    await db.flush()

    # Run LangGraph anomaly-detection + root cause workflow
    state = await run_analysis_workflow(
        reading_id=reading.id,
        sensor_data=payload.model_dump(),
    )

    reading.is_anomaly = state.get("is_anomaly", False)
    reading.severity = state.get("severity", "low")

    analysis = AnalysisResult(
        reading_id=reading.id,
        root_cause=state.get("root_cause"),
        recommendation=state.get("recommendation"),
        confidence=state.get("confidence"),
        retrieved_docs=state.get("retrieved_docs", 0),
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(reading)

    # Build broadcast event
    event = {
        "type": "sensor_update",
        "machine_id": reading.machine_id,
        "temperature": reading.temperature,
        "pressure": reading.pressure,
        "vibration": reading.vibration,
        "defect_rate": reading.defect_rate,
        "production_speed": reading.production_speed,
        "is_anomaly": reading.is_anomaly,
        "severity": reading.severity,
        "root_cause": state.get("root_cause"),
        "recommendation": state.get("recommendation"),
        "timestamp": ts.isoformat(),
    }
    await manager.broadcast(event)

    return AnalysisOut(
        reading_id=reading.id,
        machine_id=reading.machine_id,
        is_anomaly=reading.is_anomaly,
        severity=reading.severity,
        root_cause=state.get("root_cause"),
        recommendation=state.get("recommendation"),
        confidence=state.get("confidence"),
        timestamp=ts,
    )


# ── Recent Readings ────────────────────────────────────────────────


@router.get("/readings/recent")
async def recent_readings(limit: int = 50, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(SensorReading).order_by(SensorReading.timestamp.desc()).limit(limit)
    )
    readings = result.scalars().all()
    return [
        {
            "id": r.id,
            "machine_id": r.machine_id,
            "temperature": r.temperature,
            "pressure": r.pressure,
            "vibration": r.vibration,
            "defect_rate": r.defect_rate,
            "production_speed": r.production_speed,
            "is_anomaly": r.is_anomaly,
            "severity": r.severity,
            "timestamp": r.timestamp.isoformat(),
        }
        for r in readings
    ]
