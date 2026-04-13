# AI-Driven Quality Control Insights Generator
**From Reactive to Real-Time Manufacturing Intelligence**

---

## Architecture

```
Data Simulator (Python)
      ↓
FastAPI  (Real-Time Ingestion API + WebSocket)
      ↓
LangGraph (AI Workflow Orchestration)
      ↓
LangChain + ChromaDB (RAG Pipeline + Knowledge Retrieval)
      ↓
LLM Reasoning Engine (Root Cause Analysis)
      ↓
Streamlit (Live Dashboard + Auth + Chat)

Evaluation  → RAGAS   (answer quality & reliability)
Observability → Arize Phoenix (trace, debug, improve)
```

---

## Project Structure

```
ai_qc_project/
│
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # Settings via .env
│   ├── api/
│   │   ├── schemas.py           # Pydantic request/response models
│   │   ├── auth_routes.py       # Register, login, profile, admin
│   │   ├── chat_routes.py       # Chat sessions + message API
│   │   └── sensor_routes.py     # Ingest endpoint + WebSocket
│   ├── auth/
│   │   └── auth_utils.py        # JWT, bcrypt, FastAPI dependencies
│   ├── models/
│   │   └── database.py          # SQLAlchemy ORM + async engine
│   ├── workflow/
│   │   └── workflow.py          # LangGraph state machines
│   └── rag/
│       └── rag_pipeline.py      # ChromaDB + LangChain RAG
│
├── simulator/
│   └── data_simulator.py        # Synthetic sensor data stream
│
├── frontend/
│   └── app.py                   # Streamlit dashboard (all pages)
│
├── evaluation/
│   └── evaluation.py            # RAGAS + Arize Phoenix
│
├── data/
│   └── chroma_db/               # Persisted ChromaDB vectors
│
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd ai_qc_project
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY at minimum
```

### 3. Start the FastAPI backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start the Streamlit frontend

```bash
streamlit run frontend/app.py
```

### 5. Run the data simulator

```bash
# Single machine (normal):
python -m simulator.data_simulator

# Single machine with bearing failure:
python -m simulator.data_simulator --machine M-001 --fault bearing_failure

# All 4 machines simultaneously:
python -m simulator.data_simulator --all-machines
```

### 6. Run RAGAS evaluation

```bash
python -m evaluation.evaluation
```

---

## User Roles

| Role  | Capabilities |
|-------|-------------|
| `user`  | Dashboard, AI Chat, own profile |
| `admin` | All of above + view/deactivate all users |

**First admin:** Register normally, then manually set `role = 'admin'` in the DB, or add a seed script.

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/auth/register` | — | Register (consent required) |
| POST | `/auth/login` | — | Login → JWT token |
| GET | `/users/me` | ✓ | Own profile |
| PUT | `/users/me` | ✓ | Update profile |
| POST | `/users/me/change-password` | ✓ | Change password |
| DELETE | `/users/me` | ✓ | Delete account |
| GET | `/users/` | admin | List all users |
| PATCH | `/users/{id}/deactivate` | admin | Deactivate user |
| GET | `/chat/sessions` | ✓ | List chat sessions |
| GET | `/chat/sessions/{id}` | ✓ | Session + messages |
| POST | `/chat/message` | ✓ | Send message |
| DELETE | `/chat/sessions/{id}` | ✓ | Delete session |
| POST | `/api/ingest` | — | Ingest sensor reading |
| GET | `/api/readings/recent` | — | Recent readings |
| WS | `/ws/dashboard` | — | Real-time event stream |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `SECRET_KEY` | change-me | JWT signing key |
| `DATABASE_URL` | SQLite | Async SQLAlchemy URL |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage |
| `SIMULATOR_INTERVAL_SECONDS` | `2.0` | Reading frequency |
| `BACKEND_URL` | `http://localhost:8000` | For Streamlit |

---

## Fault Scenarios (Simulator)

| Scenario | Triggered By | Symptoms |
|----------|-------------|----------|
| `normal` | Default | Baseline readings with noise |
| `bearing_failure` | `--fault bearing_failure` | ↑ Vibration, ↑ Temp, ↓ Speed |
| `coolant_loss` | `--fault coolant_loss` | ↑↑ Temp, ↓ Pressure |
| `pressure_spike` | `--fault pressure_spike` | ↑↑ Pressure |
| `tool_wear` | `--fault tool_wear` | ↑↑ Defect Rate, ↓ Speed |

---

## Tech Stack

- **FastAPI** — async REST + WebSocket
- **LangGraph** — AI workflow state machine
- **LangChain** — RAG pipeline + memory
- **ChromaDB** — vector store for manufacturing knowledge
- **SQLAlchemy (async)** — ORM with SQLite/PostgreSQL
- **JWT + bcrypt** — secure authentication
- **Streamlit** — reactive dashboard frontend
- **Plotly** — real-time charts
- **RAGAS** — RAG evaluation metrics
- **Arize Phoenix** — LLM observability & tracing
