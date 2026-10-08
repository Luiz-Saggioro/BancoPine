"""Authentication: bcrypt password hashes, role checks, brute-force lockout."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import bcrypt
from sqlalchemy import and_, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from pine_config import ROLES
from pine_db import login_attempts, users

log = logging.getLogger(__name__)

USERNAME_RE = re.compile(r"^[a-zA-Z0-9._-]{3,64}$")
MIN_PASSWORD_LEN = 10


@dataclass(frozen=True)
class User:
    """Authenticated user (no secrets)."""

    username: str
    full_name: str
    role: str

    @property
    def can_write(self) -> bool:
        """Analysts and admins may save predictions."""
        return self.role in ("admin", "analyst")


class AuthError(Exception):
    """Login or user-management failure with a user-safe message."""


def validate_username(username: str) -> str:
    """Whitelist usernames: letters, digits, dot, dash, underscore."""
    username = (username or "").strip().lower()
    if not USERNAME_RE.match(username):
        raise AuthError("Username must be 3-64 characters: letters, digits, '.', '_' or '-'.")
    return username


def validate_password(password: str) -> None:
    """Minimum strength: length plus letters and digits."""
    if (
        len(password or "") < MIN_PASSWORD_LEN
        or not re.search(r"[A-Za-z]", password)
        or not re.search(r"\d", password)
    ):
        raise AuthError(f"Password needs at least {MIN_PASSWORD_LEN} characters, letters and digits.")


def hash_password(password: str) -> str:
    """bcrypt hash (salted, cost 12)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Constant-time bcrypt verification."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def create_user(engine: Engine, username: str, full_name: str, role: str, password: str) -> None:
    """Create a user after validating every field."""
    username = validate_username(username)
    validate_password(password)
    if role not in ROLES:
        raise AuthError("Unknown role.")
    full_name = (full_name or "").strip()[:128] or username
    try:
        with engine.begin() as conn:
            conn.execute(users.insert().values(
                username=username, full_name=full_name, role=role,
                password_hash=hash_password(password), active=True,
            ))
    except IntegrityError as exc:
        raise AuthError("That username already exists.") from exc


def set_active(engine: Engine, username: str, active: bool) -> None:
    """Enable or disable a user."""
    with engine.begin() as conn:
        conn.execute(users.update().where(users.c.username == username).values(active=active))


def reset_password(engine: Engine, username: str, password: str) -> None:
    """Set a new password for a user."""
    validate_password(password)
    with engine.begin() as conn:
        conn.execute(
            users.update().where(users.c.username == username)
            .values(password_hash=hash_password(password))
        )


def ensure_bootstrap_admin(engine: Engine, username: str | None, password: str | None,
                           full_name: str) -> bool:
    """Create the first admin from secrets when the users table is empty."""
    with engine.connect() as conn:
        count = conn.execute(select(func.count()).select_from(users)).scalar_one()
    if count or not username or not password:
        return False
    create_user(engine, username, full_name, "admin", password)
    log.info("Bootstrap admin created")
    return True


def _recent_failures(engine: Engine, username: str, window: timedelta) -> int:
    since = datetime.now(UTC) - window
    with engine.connect() as conn:
        last_ok = conn.execute(
            select(func.max(login_attempts.c.ts)).where(
                and_(login_attempts.c.username == username, login_attempts.c.success.is_(True))
            )
        ).scalar()
        if last_ok is not None:
            last_ok = last_ok if last_ok.tzinfo else last_ok.replace(tzinfo=UTC)
            since = max(since, last_ok)
        return conn.execute(
            select(func.count()).select_from(login_attempts).where(and_(
                login_attempts.c.username == username,
                login_attempts.c.success.is_(False),
                login_attempts.c.ts >= since,
            ))
        ).scalar_one()


def authenticate(engine: Engine, username: str, password: str, max_failed: int,
                 lockout_minutes: int) -> User:
    """Verify credentials with lockout after `max_failed` failures in the window.

    Raises:
        AuthError: generic message (does not reveal whether the user exists).
    """
    try:
        username = validate_username(username)
    except AuthError as exc:
        raise AuthError("Invalid username or password.") from exc
    if _recent_failures(engine, username, timedelta(minutes=lockout_minutes)) >= max_failed:
        raise AuthError(f"Too many failed attempts. Try again in {lockout_minutes} minutes.")
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.username == username)).mappings().first()
    ok = bool(row and row["active"] and verify_password(password or "", row["password_hash"]))
    now = datetime.now(UTC)
    with engine.begin() as conn:
        conn.execute(login_attempts.insert().values(username=username, success=ok, ts=now))
        if ok:
            conn.execute(users.update().where(users.c.username == username).values(last_login=now))
    if not ok:
        log.info("Failed login attempt")
        raise AuthError("Invalid username or password.")
    return User(row["username"], row["full_name"], row["role"])


def list_users(engine: Engine) -> list[dict]:
    """Users without password hashes."""
    cols = [users.c.username, users.c.full_name, users.c.role, users.c.active,
            users.c.created_at, users.c.last_login]
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(select(*cols).order_by(users.c.username)).mappings()]
