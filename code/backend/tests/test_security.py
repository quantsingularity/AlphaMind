"""Security tests for authentication: secret hardening and password hashing."""

from __future__ import annotations

import importlib

import pytest
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def _reload_auth(monkeypatch, env, secret):
    """Reload the auth module with a clean singleton and given environment."""
    import infrastructure.auth.authentication as auth_mod

    monkeypatch.setenv("ENVIRONMENT", env)
    if secret is None:
        monkeypatch.delenv("SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("SECRET_KEY", secret)
    importlib.reload(auth_mod)
    return auth_mod


def test_production_rejects_default_secret(monkeypatch):
    auth_mod = _reload_auth(monkeypatch, "production", None)
    with pytest.raises(RuntimeError):
        auth_mod.get_auth_system()


def test_production_rejects_short_secret(monkeypatch):
    auth_mod = _reload_auth(monkeypatch, "production", "too-short")
    with pytest.raises(RuntimeError):
        auth_mod.get_auth_system()


def test_production_accepts_strong_secret(monkeypatch):
    auth_mod = _reload_auth(monkeypatch, "production", "x" * 48)
    system = auth_mod.get_auth_system()
    assert system is not None


def test_development_allows_default_secret(monkeypatch):
    auth_mod = _reload_auth(monkeypatch, "development", None)
    # Should not raise in development.
    assert auth_mod.get_auth_system() is not None


def test_passwords_are_hashed_not_plaintext():
    """A registered user's stored password must be a bcrypt hash, never plaintext."""
    import infrastructure.auth.authentication as auth_mod

    importlib.reload(auth_mod)
    system = auth_mod.get_auth_system()
    plaintext = "PlaintextShouldNeverBeStored1!"
    system.register_user("hash_probe", plaintext, name="Probe", email="probe@x.io")
    stored = system.users_db["hash_probe"]["password"]
    assert stored != plaintext
    assert stored != plaintext.encode("utf-8")
    # bcrypt hashes start with the $2 marker.
    assert stored.startswith(b"$2"), "stored credential is not a bcrypt hash"


def test_wrong_password_is_rejected():
    """Login with an incorrect password must fail (401)."""
    email = "login_probe@example.com"
    client.post(
        "/api/auth/register",
        json={"email": email, "name": "Login Probe", "password": "Correct-Horse-42"},
    )
    bad = client.post(
        "/api/auth/login", json={"email": email, "password": "wrong-password"}
    )
    assert bad.status_code == 401, bad.text
    good = client.post(
        "/api/auth/login", json={"email": email, "password": "Correct-Horse-42"}
    )
    assert good.status_code == 200, good.text
