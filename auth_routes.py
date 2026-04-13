"""
backend/api/auth_routes.py
Authentication and user profile endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.auth_utils import (
    hash_password, verify_password, create_access_token,
    get_current_user, require_admin,
)
from backend.models.database import User, UserRole, get_db
from backend.api.schemas import (
    RegisterRequest, LoginRequest, TokenResponse,
    UserOut, UpdateProfileRequest, ChangePasswordRequest,
)
from backend.auth.otp_utils import create_otp

router = APIRouter(prefix="/auth", tags=["Authentication"])
users_router = APIRouter(prefix="/users", tags=["Users"])


# ── Register ───────────────────────────────────────────────────────

@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Check duplicate email / username
    dup = await db.execute(
        select(User).where((User.email == body.email) | (User.username == body.username))
    )
    if dup.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email or username already registered.")

    user = User(
        email=body.email,
        username=body.username,
        full_name=body.full_name,
        hashed_password=hash_password(body.password),
        consent_given=body.consent_given,
        role=UserRole.user,
        is_email_verified=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    # Auto-generate OTP immediately after registration
    await create_otp(db, user_id=user.id, purpose="email_verify")

    return user


# ── Login ──────────────────────────────────────────────────────────

@router.post("/login", response_model=TokenResponse)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == body.email))
    user = result.scalar_one_or_none()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated.")
    if not user.is_email_verified:
        raise HTTPException(
            status_code=403,
            detail="Email not verified. Please verify your email with the OTP sent at registration.",
        )

    token = create_access_token({"sub": user.id, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}


# ── Current User Profile ───────────────────────────────────────────

@users_router.get("/me", response_model=UserOut)
async def get_profile(current_user: User = Depends(get_current_user)):
    return current_user


@users_router.put("/me", response_model=UserOut)
async def update_profile(
    body: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if body.username:
        dup = await db.execute(
            select(User).where(User.username == body.username, User.id != current_user.id)
        )
        if dup.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Username already taken.")
        current_user.username = body.username
    if body.full_name is not None:
        current_user.full_name = body.full_name

    await db.commit()
    await db.refresh(current_user)
    return current_user


@users_router.post("/me/change-password", status_code=200)
async def change_password(
    body: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    current_user.hashed_password = hash_password(body.new_password)
    await db.commit()
    return {"message": "Password updated successfully."}


@users_router.delete("/me", status_code=204)
async def delete_account(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await db.delete(current_user)
    await db.commit()


# ── Admin: All Users ───────────────────────────────────────────────

@users_router.get("/", response_model=list[UserOut])
async def list_all_users(
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return result.scalars().all()


@users_router.patch("/{user_id}/deactivate", response_model=UserOut)
async def deactivate_user(
    user_id: str,
    _: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.is_active = False
    await db.commit()
    await db.refresh(user)
    return user
