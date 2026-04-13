"""
models/database.py
SQLAlchemy ORM models for users, chat sessions, messages, and sensor readings.
"""

from datetime import datetime, timedelta
from sqlalchemy import (
    Column, String, Boolean, DateTime, Float,
    ForeignKey, Text, Enum as SAEnum, Integer
)
from sqlalchemy.orm import DeclarativeBase, relationship
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import enum
import uuid

from backend.config import settings


# ── Base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


def generate_uuid():
    return str(uuid.uuid4())


# ── Enums ─────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    admin = "admin"
    user = "user"


class SeverityLevel(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ── User ──────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id            = Column(String, primary_key=True, default=generate_uuid)
    email         = Column(String, unique=True, nullable=False, index=True)
    username      = Column(String, unique=True, nullable=False)
    full_name     = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    role          = Column(SAEnum(UserRole), default=UserRole.user, nullable=False)
    is_active        = Column(Boolean, default=True)
    is_email_verified = Column(Boolean, default=False)   # Must verify OTP before login
    consent_given    = Column(Boolean, default=False)    # GDPR consent at register
    created_at       = Column(DateTime, default=datetime.utcnow)
    updated_at       = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    otp_records   = relationship("OTPVerification", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"


# ── OTP Verification ───────────────────────────────────────────────

class OTPVerification(Base):
    __tablename__ = "otp_verifications"

    id         = Column(String, primary_key=True, default=generate_uuid)
    user_id    = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    otp_code   = Column(String(6), nullable=False)          # 6-digit static/generated OTP
    purpose    = Column(String, default="email_verify")     # email_verify | password_reset
    is_used    = Column(Boolean, default=False)
    expires_at = Column(DateTime, nullable=False)           # OTP valid for 10 minutes
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="otp_records")

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at

    @classmethod
    def new(cls, user_id: str, otp_code: str, purpose: str = "email_verify") -> "OTPVerification":
        return cls(
            user_id=user_id,
            otp_code=otp_code,
            purpose=purpose,
            expires_at=datetime.utcnow() + timedelta(minutes=10),
        )


# ── Chat Session ───────────────────────────────────────────────────

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id         = Column(String, primary_key=True, default=generate_uuid)
    user_id    = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title      = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user     = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


# ── Chat Message ───────────────────────────────────────────────────

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id         = Column(String, primary_key=True, default=generate_uuid)
    session_id = Column(String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False)
    role       = Column(String, nullable=False)   # "user" | "assistant"
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")


# ── Sensor Reading ─────────────────────────────────────────────────

class SensorReading(Base):
    __tablename__ = "sensor_readings"

    id               = Column(String, primary_key=True, default=generate_uuid)
    machine_id       = Column(String, nullable=False, index=True)
    temperature      = Column(Float, nullable=False)
    pressure         = Column(Float, nullable=False)
    vibration        = Column(Float, nullable=False)
    defect_rate      = Column(Float, nullable=False)
    production_speed = Column(Float, nullable=False)
    is_anomaly       = Column(Boolean, default=False)
    severity         = Column(SAEnum(SeverityLevel), default=SeverityLevel.low)
    timestamp        = Column(DateTime, default=datetime.utcnow, index=True)


# ── Analysis Result ────────────────────────────────────────────────

class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id              = Column(String, primary_key=True, default=generate_uuid)
    reading_id      = Column(String, ForeignKey("sensor_readings.id"), nullable=False)
    root_cause      = Column(Text, nullable=True)
    recommendation  = Column(Text, nullable=True)
    confidence      = Column(Float, nullable=True)
    retrieved_docs  = Column(Integer, default=0)
    ragas_score     = Column(Float, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

    reading = relationship("SensorReading")


# ── Database Engine ────────────────────────────────────────────────

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """Create all tables on startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    """FastAPI dependency — yields an async DB session."""
    async with AsyncSessionLocal() as session:
        yield session
