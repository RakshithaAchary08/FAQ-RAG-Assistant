from auth import (
    create_auth_token,
    create_password_reset_email,
    load_auth_token,
)


def test_auth_token_round_trip_and_expiry():
    token = create_auth_token(7, "person@example.com", "test-secret")

    token_data = load_auth_token(token, "test-secret")
    assert token_data is not None
    assert token_data["user_id"] == 7
    assert token_data["email"] == "person@example.com"
    assert isinstance(token_data["issued_at"], int)
    assert load_auth_token(token, "wrong-secret") is None
    assert load_auth_token(token, "test-secret", max_age=-1) is None


def test_password_reset_email_contains_token_and_optional_link(monkeypatch):
    monkeypatch.setenv("AUTH_PUBLIC_URL", "https://chat.example.com/")

    message = create_password_reset_email("person@example.com", "one-time-token")

    assert message["To"] == "person@example.com"
    assert "one-time-token" in message.get_content()
    assert (
        "https://chat.example.com/?reset_token=one-time-token"
        in message.get_content()
    )
