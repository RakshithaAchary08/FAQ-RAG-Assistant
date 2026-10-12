"""Signed authentication tokens for optional account sessions."""

import os
import smtplib
import time
from email.message import EmailMessage
from urllib.parse import urlencode

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

AUTH_TOKEN_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
AUTH_TOKEN_SALT = "faq-support-desk-auth"
AUTH_LOGIN_ERROR_MESSAGE = "Incorrect email or password."


def create_auth_token(user_id: int, email: str, secret: str) -> str:
    """Create a signed token identifying a user."""
    serializer = URLSafeTimedSerializer(secret, salt=AUTH_TOKEN_SALT)
    return serializer.dumps(
        {"user_id": user_id, "email": email, "issued_at": int(time.time())}
    )


def load_auth_token(
    token: str,
    secret: str,
    max_age: int = AUTH_TOKEN_MAX_AGE_SECONDS,
) -> dict[str, object] | None:
    """Validate a signed token and reject expired or malformed values."""
    serializer = URLSafeTimedSerializer(secret, salt=AUTH_TOKEN_SALT)
    try:
        payload = serializer.loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("user_id"), int)
        or not isinstance(payload.get("email"), str)
        or not isinstance(payload.get("issued_at"), int)
    ):
        return None
    return {
        "user_id": payload["user_id"],
        "email": payload["email"],
        "issued_at": payload["issued_at"],
    }


def create_password_reset_email(email: str, token: str) -> EmailMessage:
    """Build a reset email containing a single-use 30-minute token."""
    public_url = os.getenv("AUTH_PUBLIC_URL", "").strip().rstrip("/")
    link = (
        "\nOpen this link to reset your password: "
        f"{public_url}/?{urlencode({'reset_token': token})}\n"
        if public_url
        else ""
    )
    message = EmailMessage()
    message["Subject"] = "FAQ Chatbot password reset"
    message["From"] = (
        os.getenv("SMTP_FROM")
        or os.getenv("SMTP_USER")
        or "no-reply@localhost"
    )
    message["To"] = email
    message.set_content(
        "A password reset was requested for your FAQ Chatbot account.\n"
        f"{link}\n"
        f"Your one-time reset token is:\n{token}\n\n"
        "Enter it on the Forgot password page within 30 minutes. "
        "If you did not request this, you can ignore this email."
    )
    return message


def send_password_reset_email(email: str, token: str) -> None:
    """Send a reset email using SMTP settings from the environment."""
    host = os.getenv("SMTP_HOST", "").strip()
    if not host:
        raise RuntimeError("Password reset email is not configured.")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    message = create_password_reset_email(email, token)

    if os.getenv("SMTP_USE_SSL", "").strip().lower() in {"1", "true", "yes", "on"}:
        with smtplib.SMTP_SSL(host, port, timeout=10) as server:
            if username:
                server.login(username, password)
            server.send_message(message)
        return

    with smtplib.SMTP(host, port, timeout=10) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        if username:
            server.login(username, password)
        server.send_message(message)
