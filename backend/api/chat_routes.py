"""
backend/api/chat_routes.py
Chat endpoints — per-user sessions, history retrieval, AI response via LangGraph.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.auth.auth_utils import get_current_user
from backend.models.database import (
    User,
    ChatSession,
    ChatMessage,
    get_db,
)
from backend.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatSessionOut,
    ChatSessionDetail,
    ChatMessageOut,
)
from backend.workflow.workflow import run_chat_workflow

router = APIRouter(prefix="/chat", tags=["Chat"])


# ── List user's sessions ───────────────────────────────────────────


@router.get("/sessions", response_model=list[ChatSessionOut])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


# ── Get a session with full message history ────────────────────────


@router.get("/sessions/{session_id}", response_model=ChatSessionDetail)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession)
        .options(selectinload(ChatSession.messages))
        .where(ChatSession.id == session_id, ChatSession.user_id == current_user.id)
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


# ── Delete a session ───────────────────────────────────────────────


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    await db.delete(session)
    await db.commit()


# ── Send a message ─────────────────────────────────────────────────


@router.post("/message", response_model=ChatResponse)
async def send_message(
    body: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # Resolve or create session
    if body.session_id:
        result = await db.execute(
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(
                ChatSession.id == body.session_id,
                ChatSession.user_id == current_user.id,
            )
        )
        session = result.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
    else:
        # Auto-title from first ~40 chars of first message
        title = body.message[:40] + ("…" if len(body.message) > 40 else "")
        session = ChatSession(user_id=current_user.id, title=title)
        db.add(session)
        await db.flush()  # get session.id before messages reference it

    # Build history for LangGraph (last 10 exchanges)
    msg_result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(20)
    )

    messages = list(reversed(msg_result.scalars().all()))

    history = [{"role": m.role, "content": m.content} for m in messages]

    # Save user message
    user_msg = ChatMessage(session_id=session.id, role="user", content=body.message)
    db.add(user_msg)

    # Run AI workflow
    result_state = await run_chat_workflow(
        user_message=body.message,
        history=history,
        user_id=current_user.id,
    )

    answer = result_state.get("response", "I'm sorry, I could not generate a response.")
    sources = result_state.get("sources", [])

    # Save assistant message
    assistant_msg = ChatMessage(session_id=session.id, role="assistant", content=answer)
    db.add(assistant_msg)

    await db.commit()
    await db.refresh(session)

    return ChatResponse(
        session_id=session.id,
        session_title=session.title,
        answer=answer,
        sources=sources,
    )
