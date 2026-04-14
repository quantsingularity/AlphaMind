"""
Extended tests for AlphaMind Authentication System (FastAPI).
"""

import os
import sys
import time

import jwt
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from fastapi.testclient import TestClient
from infrastructure.auth.authentication import AuthenticationSystem

client = TestClient(app)

SECRET_KEY = "test-secret-key-pytest"


@pytest.fixture(scope="function")
def auth_system():
    """Creates an independent AuthenticationSystem instance for each test."""
    return AuthenticationSystem(secret_key=SECRET_KEY, token_expiration=24)


def test_register_success(auth_system):
    result = auth_system.register_user("user1", "password1")
    assert result["message"] == "User registered successfully"
    assert "user1" in auth_system.users_db


def test_register_duplicate(auth_system):
    from fastapi import HTTPException

    auth_system.register_user("dupuser", "pass")
    with pytest.raises(HTTPException) as exc_info:
        auth_system.register_user("dupuser", "pass")
    assert exc_info.value.status_code == 409


def test_login_success(auth_system):
    auth_system.register_user("loginuser", "mypass")
    result = auth_system.login_user("loginuser", "mypass")
    assert "token" in result
    assert result["username"] == "loginuser"


def test_login_wrong_password(auth_system):
    from fastapi import HTTPException

    auth_system.register_user("wpuser", "correct")
    with pytest.raises(HTTPException) as exc_info:
        auth_system.login_user("wpuser", "wrong")
    assert exc_info.value.status_code == 401


def test_login_nonexistent(auth_system):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        auth_system.login_user("nobody", "pass")
    assert exc_info.value.status_code == 401


def test_token_generation_and_verification(auth_system):
    token = auth_system.generate_token("myuser")
    assert isinstance(token, str)
    username = auth_system.verify_token(token)
    assert username == "myuser"


def test_token_expiration(auth_system):
    short_auth = AuthenticationSystem(secret_key=SECRET_KEY, token_expiration=1 / 3600)
    token = short_auth.generate_token("expuser")
    time.sleep(1.1)
    with pytest.raises(jwt.ExpiredSignatureError):
        short_auth.verify_token(token)


def test_api_register_endpoint():
    response = client.post(
        "/api/auth/register",
        json={"username": "apiuser_test", "password": "apipass"},
    )
    assert response.status_code == 201


def test_api_login_endpoint():
    client.post(
        "/api/auth/register",
        json={"username": "api_login_test", "password": "apipass"},
    )
    response = client.post(
        "/api/auth/login",
        json={"username": "api_login_test", "password": "apipass"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "token" in data
