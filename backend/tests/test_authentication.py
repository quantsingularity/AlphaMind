from typing import Any
from functools import wraps
import os
import sys
import time
from flask import Flask, jsonify, request
import jwt
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(backend_dir)
from backend.infrastructure.authentication import AuthenticationSystem


@pytest.fixture(scope="function")
def app_context() -> Any:
    """Creates independent app + auth system instance for each test."""
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "test-secret-key"
    app.config["TESTING"] = True
    auth = AuthenticationSystem(
        app, app.config["SECRET_KEY"], token_expiration=1 / 3600.0
    )

    def test_token_required(f):

        @wraps(f)
        def decorated(*args, **kwargs):
            token = None
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
            if not token:
                return (jsonify({"message": "Token is missing"}), 401)
            try:
                username = auth.verify_token(token)
            except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
                return (jsonify({"message": "Token is invalid or expired"}), 401)
            return f(username, *args, **kwargs)

        return decorated

    @app.route("/api/protected_test_unique")
    @test_token_required
    def protected_route(current_user):
        return (jsonify({"message": "Access granted", "user": current_user}), 200)

    with app.app_context():
        yield (app, auth)


@pytest.fixture
def app(app_context: Any) -> Any:
    return app_context[0]


@pytest.fixture
def auth_system(app_context: Any) -> Any:
    return app_context[1]


@pytest.fixture
def client(app: Any) -> Any:
    return app.test_client()


def test_register_success(client: Any) -> Any:
    response = client.post(
        "/api/auth/register", json={"username": "testuser", "password": "password123"}
    )
    assert response.status_code == 201
    assert response.json["message"] == "User registered successfully"


def test_register_missing_data(client: Any) -> Any:
    response = client.post("/api/auth/register", json={"username": "testuser"})
    assert response.status_code == 400
    assert response.json["message"] == "Missing username or password"


def test_register_existing_user(client: Any) -> Any:
    client.post(
        "/api/auth/register", json={"username": "testuser", "password": "password123"}
    )
    response = client.post(
        "/api/auth/register",
        json={"username": "testuser", "password": "anotherpassword"},
    )
    assert response.status_code == 409
    assert response.json["message"] == "User already exists"


def test_login_success(client: Any) -> Any:
    client.post(
        "/api/auth/register", json={"username": "loginuser", "password": "password123"}
    )
    response = client.post(
        "/api/auth/login", json={"username": "loginuser", "password": "password123"}
    )
    assert response.status_code == 200
    assert response.json["message"] == "Login successful"
    assert "token" in response.json
    assert response.json["username"] == "loginuser"


def test_login_invalid_credentials(client: Any) -> Any:
    client.post(
        "/api/auth/register", json={"username": "loginuser", "password": "password123"}
    )
    response = client.post(
        "/api/auth/login", json={"username": "loginuser", "password": "wrongpassword"}
    )
    assert response.status_code == 401
    assert response.json["message"] == "Invalid credentials"


def test_login_nonexistent_user(client: Any) -> Any:
    response = client.post(
        "/api/auth/login", json={"username": "nosuchuser", "password": "password123"}
    )
    assert response.status_code == 401
    assert response.json["message"] == "Invalid credentials"


def test_token_generation_verification(auth_system: Any) -> Any:
    username = "tokenuser"
    token = auth_system.generate_token(username)
    assert isinstance(token, str)
    verified_username = auth_system.verify_token(token)
    assert verified_username == username


def test_token_expiration(auth_system: Any) -> Any:
    username = "tokenuser_exp"
    token = auth_system.generate_token(username)
    time.sleep(1.1)
    with pytest.raises(jwt.ExpiredSignatureError):
        auth_system.verify_token(token)


def test_token_invalid(auth_system: Any) -> Any:
    invalid_token = "this.is.not.a.valid.token"
    with pytest.raises(jwt.InvalidTokenError):
        auth_system.verify_token(invalid_token)


def test_token_required_decorator_success(client: Any) -> Any:
    client.post(
        "/api/auth/register",
        json={"username": "protecteduser", "password": "password123"},
    )
    login_response = client.post(
        "/api/auth/login", json={"username": "protecteduser", "password": "password123"}
    )
    token = login_response.json["token"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/protected_test_unique", headers=headers)
    assert response.status_code == 200
    assert response.json["message"] == "Access granted"
    assert response.json["user"] == "protecteduser"


def test_token_required_decorator_missing(client: Any) -> Any:
    response = client.get("/api/protected_test_unique")
    assert response.status_code == 401
    assert response.json["message"] == "Token is missing"


def test_token_required_decorator_invalid(client: Any) -> Any:
    headers = {"Authorization": "Bearer invalid.token.string"}
    response = client.get("/api/protected_test_unique", headers=headers)
    assert response.status_code == 401
    assert "Token is invalid or expired" in response.json["message"]


def test_token_required_decorator_expired(client: Any, auth_system: Any) -> Any:
    client.post(
        "/api/auth/register",
        json={"username": "expireduser", "password": "password123"},
    )
    login_response = client.post(
        "/api/auth/login", json={"username": "expireduser", "password": "password123"}
    )
    token = login_response.json["token"]
    time.sleep(1.1)
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/api/protected_test_unique", headers=headers)
    assert response.status_code == 401
    assert "Token is invalid or expired" in response.json["message"]
