"""
backend/api/otp_routes.py
Email verification via OTP endpoints.

Flow:
  1. POST /auth/register          → user created, is_email_verified=False
  2. POST /otp/send               → generates OTP (static=123456), returns it in demo mode
  3. POST /otp/verify             → validates code, sets is_email_verified=True
  4. POST /auth/login             → blocked until verified

Additional:
  POST /otp/resend                → rate-limited resend (once per 60 s)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.database import User, get_db
from backend.auth.otp_utils import create_otp, verify_otp, can_resend_otp
from backend.api.schemas import OTPSendRequest, OTPVerifyRequest, OTPResponse
from backend.config import settings

router = APIRouter(prefix="/otp", tags=["OTP Verification"])


# ── Send / Generate OTP ────────────────────────────────────────────

@router.post("/send", response_model=OTPResponse)
async def send_otp(body: OTPSendRequest, db: AsyncSession = Depends(get_db)):
    """
    Generates (or re-generates) an OTP for the given email.
    In DEMO MODE the OTP code is returned directly in the response
    (static value from settings: '123456').
    In production: send via email and omit otp_code from response.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user:
        # Don't reveal whether email exists — return same message
        return OTPResponse(
            message="If that email is registered, an OTP has been sent.",
            expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
        )

    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email is already verified.")

    record = await create_otp(db, user_id=user.id, purpose="email_verify")

    # DEMO MODE: return OTP in response so Streamlit UI can display it
    return OTPResponse(
        message=f"OTP sent to {body.email}. Valid for {settings.OTP_EXPIRE_MINUTES} minutes.",
        otp_code=record.otp_code,   # Remove this line in production
        expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
    )


# ── Verify OTP ─────────────────────────────────────────────────────

@router.post("/verify", status_code=200)
async def verify_email_otp(body: OTPVerifyRequest, db: AsyncSession = Depends(get_db)):
    """
    Verifies the submitted OTP. On success marks the user as email-verified.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered.")

    if user.is_email_verified:
        return {"message": "Email is already verified. You can log in."}

    success, reason = await verify_otp(
        db, user_id=user.id, submitted_code=body.otp_code, purpose="email_verify"
    )
    if not success:
        raise HTTPException(status_code=400, detail=reason)

    user.is_email_verified = True
    await db.commit()
    return {"message": "Email verified successfully! You can now log in."}


# ── Resend OTP ─────────────────────────────────────────────────────

@router.post("/resend", response_model=OTPResponse)
async def resend_otp(body: OTPSendRequest, db: AsyncSession = Depends(get_db)):
    """
    Rate-limited resend — only allowed once every 60 seconds.
    """
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user:
        return OTPResponse(
            message="If that email is registered, an OTP has been sent.",
            expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
        )

    if user.is_email_verified:
        raise HTTPException(status_code=400, detail="Email is already verified.")

    allowed = await can_resend_otp(db, user_id=user.id, purpose="email_verify")
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Please wait 60 seconds before requesting a new OTP.",
        )

    record = await create_otp(db, user_id=user.id, purpose="email_verify")
    return OTPResponse(
        message=f"New OTP sent to {body.email}.",
        otp_code=record.otp_code,   # Remove in production
        expires_in_minutes=settings.OTP_EXPIRE_MINUTES,
    )


# ── Status check ───────────────────────────────────────────────────

@router.get("/status")
async def otp_status(email: str, db: AsyncSession = Depends(get_db)):
    """Check whether an email is verified (for Streamlit polling)."""
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="Email not registered.")
    return {
        "email": email,
        "is_email_verified": user.is_email_verified,
    }
