"""
backend/main.py
FastAPI application entry point.
Registers all routers, initialises the database, and sets up Phoenix tracing.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models.database import init_db
from backend.api.auth_routes import router as auth_router, users_router
from backend.api.chat_routes import router as chat_router
from backend.api.sensor_routes import router as sensor_router
from backend.api.otp_routes import router as otp_router


# ── Lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    print(f"[DB] Tables initialised.")

    # Setup Arize Phoenix tracing (optional — skip if not configured)
    try:
        import phoenix as px
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        px.launch_app()
        provider = TracerProvider()
        exporter = OTLPSpanExporter(
            endpoint=f"http://{settings.PHOENIX_HOST}:{settings.PHOENIX_PORT}/v1/traces"
        )
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        LangChainInstrumentor().instrument(tracer_provider=provider)
        print("[Phoenix] Arize Phoenix tracing enabled.")
    except Exception as e:
        print(f"[Phoenix] Tracing not enabled: {e}")

    yield
    # Shutdown (cleanup if needed)


# ── App ────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="AI-driven real-time quality control insights for manufacturing.",
    lifespan=lifespan,
)

# CORS — allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(users_router)
app.include_router(chat_router)
app.include_router(sensor_router)
app.include_router(otp_router)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.APP_NAME}


# ── Run ────────────────────────────────────────────────────────────
# Start with: uvicorn backend.main:app --reload --port 8000

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
