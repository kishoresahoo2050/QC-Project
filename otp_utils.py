"""
backend/auth/otp_utils.py
OTP generation and validation utilities.

In this implementation OTP is STATIC for demo/hackathon use:
  - Default static OTP: 123456  (configurable via settings.STATIC_OTP)
  - In a production system replace generate_otp() with secrets.randbelow
    and send the code via email (SendGrid, SES, etc.)
"""

import random
import string
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.database import OTPVerification, User
from backend.config import settings


# ── OTP Generation ─────────────────────────────────────────────────

def generate_otp() -> str:
    """
    Returns a 6-digit OTP.
    STATIC MODE (hackathon): always returns settings.STATIC_OTP ("123456").
    Switch to the random line below for real use.
    """
    # --- Production (random): ---
    # return "".join(random.choices(string.digits, k=6))

    # --- Static / demo mode: ---
    return settings.STATIC_OTP


# ── Store OTP ──────────────────────────────────────────────────────

async def create_otp(
    db: AsyncSession,
    user_id: str,
    purpose: str = "email_verify",
) -> OTPVerification:
    """
    Invalidate any existing unused OTPs for this user+purpose,
    then create a fresh one.
    """
    # Expire existing
    existing = await db.execute(
        select(OTPVerification).where(
            and_(
                OTPVerification.user_id == user_id,
                OTPVerification.purpose == purpose,
                OTPVerification.is_used == False,
            )
        )
    )
    for old in existing.scalars().all():
        old.is_used = True   # mark old OTPs as consumed

    otp_code = generate_otp()
    record = OTPVerification.new(user_id=user_id, otp_code=otp_code, purpose=purpose)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record


# ── Verify OTP ─────────────────────────────────────────────────────

async def verify_otp(
    db: AsyncSession,
    user_id: str,
    submitted_code: str,
    purpose: str = "email_verify",
) -> tuple[bool, str]:
    """
    Validate a submitted OTP code.

    Returns:
        (True, "ok")            — valid
        (False, reason_string)  — invalid with human-readable reason
    """
    result = await db.execute(
        select(OTPVerification).where(
            and_(
                OTPVerification.user_id == user_id,
                OTPVerification.purpose == purpose,
                OTPVerification.is_used == False,
            )
        ).order_by(OTPVerification.created_at.desc())
    )
    record = result.scalars().first()

    if not record:
        return False, "No active OTP found. Please request a new one."

    if record.is_expired:
        return False, "OTP has expired. Please request a new one."

    if record.otp_code != submitted_code.strip():
        return False, "Incorrect OTP code. Please try again."

    # Mark as used
    record.is_used = True
    await db.commit()
    return True, "ok"


# ── Resend / Request OTP (with rate limiting guard) ────────────────

async def can_resend_otp(db: AsyncSession, user_id: str, purpose: str = "email_verify") -> bool:
    """
    Simple rate-limit: allow resend only if the last OTP is >60 s old.
    """
    result = await db.execute(
        select(OTPVerification).where(
            and_(
                OTPVerification.user_id == user_id,
                OTPVerification.purpose == purpose,
            )
        ).order_by(OTPVerification.created_at.desc()).limit(1)
    )
    last = result.scalars().first()
    if not last:
        return True
    elapsed = (datetime.utcnow() - last.created_at).total_seconds()
    return elapsed >= 60
