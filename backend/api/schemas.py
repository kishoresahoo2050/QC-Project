"""
backend/api/schemas.py
Pydantic v2 schemas for all request/response bodies.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, field_validator


# ── Auth Schemas ───────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    full_name: Optional[str] = None
    password: str = Field(..., min_length=8)
    consent_given: bool = Field(..., description="User must consent to data processing")

    @field_validator("consent_given")
    @classmethod
    def must_consent(cls, v):
        if not v:
            raise ValueError("You must accept the terms and privacy policy to register.")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── User Schemas ───────────────────────────────────────────────────

class UserOut(BaseModel):
    id: str
    email: EmailStr
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_email_verified: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = Field(None, min_length=3, max_length=50)


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


# ── Chat Schemas ───────────────────────────────────────────────────

class ChatMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionDetail(ChatSessionOut):
    messages: List[ChatMessageOut] = []


class ChatRequest(BaseModel):
    session_id: Optional[str] = None   # None → create new session
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    session_title: str
    answer: str
    sources: List[str] = []


# ── Sensor / Ingestion Schemas ─────────────────────────────────────

class SensorPayload(BaseModel):
    machine_id: str
    temperature: float
    pressure: float
    vibration: float
    defect_rate: float
    production_speed: float
    timestamp: Optional[datetime] = None


class AnalysisOut(BaseModel):
    reading_id: str
    machine_id: str
    is_anomaly: bool
    severity: str
    root_cause: Optional[str]
    recommendation: Optional[str]
    confidence: Optional[float]
    timestamp: datetime


# ── OTP Schemas ────────────────────────────────────────────────────

class OTPSendRequest(BaseModel):
    email: EmailStr


class OTPVerifyRequest(BaseModel):
    email: EmailStr
    otp_code: str = Field(..., min_length=6, max_length=6, pattern=r"^\d{6}$")


class OTPResponse(BaseModel):
    message: str
    otp_code: Optional[str] = None   # Returned in dev/demo mode only (static OTP display)
    expires_in_minutes: int = 10
