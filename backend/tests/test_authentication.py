# from functools import wraps  # Import wraps
# import json
# import os
# import sys
# import time

# from flask import Flask, jsonify, request  # Added request
# import jwt  # Import jwt for exception handling
# import pytest

# Add the backend directory to the path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# backend_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.append(backend_dir)

# from backend.infrastructure.authentication import AuthenticationSystem


# Using a single fixture to create app and auth context to avoid route overwriting
# @pytest.fixture(scope="function")
# def app_context():
#    """Create and configure a new app instance and auth system for each test."""
##     app = Flask(__name__)
##     app.config["SECRET_KEY"] = "test-secret-key"
##     app.config["TESTING"] = True
#    # Use a very short expiration for testing expiration scenarios
##     auth = AuthenticationSystem(
##         app,
##         app.config["SECRET_KEY"],
##         token_expiration=(1 / 3600.0),  # 1 second expiration
#    )
#
#    # --- Mocking the decorator slightly for testing ---
#    # The real decorator might fetch a user object. For testing the flow,
#    # we just pass the verified username string.
##     def test_token_required(f):
##         @wraps(f)
##         def decorated(*args, **kwargs):
##             token = None
#            # Get token from Authorization header
##             auth_header = request.headers.get("Authorization")
##             if auth_header and auth_header.startswith("Bearer "):
##                 token = auth_header.split(" ")[1]
##             if not token:
##                 return jsonify({"message": "Token is missing"}), 401
#            # Verify token
##             try:
##                 username = auth.verify_token(token)
##                 if not username:
#                    # This case might not be reachable if verify_token raises exceptions
##                     return jsonify({"message": "Token is invalid or expired"}), 401
##             except jwt.ExpiredSignatureError:
##                 return jsonify({"message": "Token is invalid or expired"}), 401
##             except jwt.InvalidTokenError:
##                 return jsonify({"message": "Token is invalid or expired"}), 401
#            # Pass the username to the route
##             return f(username, *args, **kwargs)  # Pass username as current_user
#
##         return decorated
#
#    # --- End Mock Decorator ---
#
#    # Add a dummy protected route using the test decorator
##     @app.route("/api/protected_test_unique")
##     @test_token_required  # Use the test decorator
##     def protected_test_route(current_user):  # Route expects the username string
##         return jsonify({"message": "Access granted", "user": current_user}), 200
#
#    # Provide the app context for the test
##     with app.app_context():
##         yield app, auth
#
#
## @pytest.fixture
## def app(app_context):
#    """Provides the Flask app instance from the app_context fixture."""
#     return app_context[0]


# @pytest.fixture
# def auth_system(app_context):
#    """Provides the AuthenticationSystem instance from the app_context fixture."""
##     return app_context[1]
#
#
## @pytest.fixture
## def client(app):
#    """A test client for the app."""
#     return app.test_client()


# --- Test Cases ---


# def test_register_success(client):
#    """Test successful user registration."""
##     response = client.post(
#        "/api/auth/register", json={"username": "testuser", "password": "password123"}
#    )
##     assert response.status_code == 201
##     assert response.json["message"] == "User registered successfully"
#
#
## def test_register_missing_data(client):
#    """Test registration with missing data."""
#     response = client.post("/api/auth/register", json={"username": "testuser"})
#     assert response.status_code == 400
#     assert response.json["message"] == "Missing username or password"


# def test_register_existing_user(client):
#    """Test registration with an existing username."""
##     client.post(
#        "/api/auth/register", json={"username": "testuser", "password": "password123"}
##     )  # Register first
##     response = client.post(
#        "/api/auth/register",
##         json={"username": "testuser", "password": "anotherpassword"},
#    )
##     assert response.status_code == 409
##     assert response.json["message"] == "User already exists"
#
#
## def test_login_success(client):
#    """Test successful user login."""
#     client.post(
#         "/api/auth/register", json={"username": "loginuser", "password": "password123"}
#     response = client.post(
#         "/api/auth/login", json={"username": "loginuser", "password": "password123"}
#     assert response.status_code == 200
#     assert response.json["message"] == "Login successful"
#     assert "token" in response.json
#     assert response.json["username"] == "loginuser"


# def test_login_invalid_credentials(client):
#    """Test login with incorrect password."""
##     client.post(
#        "/api/auth/register", json={"username": "loginuser", "password": "password123"}
#    )
##     response = client.post(
#        "/api/auth/login", json={"username": "loginuser", "password": "wrongpassword"}
#    )
##     assert response.status_code == 401
##     assert response.json["message"] == "Invalid credentials"
#
#
## def test_login_nonexistent_user(client):
#    """Test login for a user that does not exist."""
#     response = client.post(
#         "/api/auth/login", json={"username": "nosuchuser", "password": "password123"}
#     assert response.status_code == 401
#     assert response.json["message"] == "Invalid credentials"


# def test_token_generation_verification(auth_system):
#    """Test JWT token generation and verification."""
##     username = "tokenuser"
##     token = auth_system.generate_token(username)
##     assert isinstance(token, str)
#    # Assuming verify_token returns username on success, raises on failure
##     verified_username = auth_system.verify_token(token)
##     assert verified_username == username
#
#
## def test_token_expiration(auth_system):
#    """Test that the token expires correctly (verify_token raises error)."""
#     username = "tokenuser_exp"
#     token = auth_system.generate_token(username)
#     time.sleep(1.1)  # Wait for slightly longer than the 1-second expiration
#     # Expect ExpiredSignatureError when verifying an expired token
#     with pytest.raises(jwt.ExpiredSignatureError):
#         auth_system.verify_token(token)


# def test_token_invalid(auth_system):
#    """Test verification with an invalid token (verify_token raises error)."""
##     invalid_token = "this.is.not.a.valid.token"
#    # Expect DecodeError or InvalidTokenError for malformed tokens
##     with pytest.raises(jwt.InvalidTokenError):  # More general error
##         auth_system.verify_token(invalid_token)
#
#
## def test_token_required_decorator_success(client, auth_system):
#    """Test the @token_required decorator with a valid token."""
#     # Register and login to get a valid token
#     client.post(
#         "/api/auth/register",
#         json={"username": "protecteduser", "password": "password123"},
#     login_response = client.post(
#         "/api/auth/login", json={"username": "protecteduser", "password": "password123"}
#     token = login_response.json["token"]

#     headers = {"Authorization": f"Bearer {token}"}
#     response = client.get("/api/protected_test_unique", headers=headers)
#     assert response.status_code == 200
#     assert response.json["message"] == "Access granted"
#     assert (
#         response.json["user"] == "protecteduser"
#     )  # Check if username is passed correctly


# def test_token_required_decorator_missing(client):
#    """Test the @token_required decorator with no token."""
##     response = client.get("/api/protected_test_unique")
##     assert response.status_code == 401
##     assert response.json["message"] == "Token is missing"
#
#
## def test_token_required_decorator_invalid(client):
#    """Test the @token_required decorator with an invalid token."""
#     headers = {"Authorization": "Bearer invalid.token.string"}
#     response = client.get("/api/protected_test_unique", headers=headers)
#     assert response.status_code == 401
#     assert "Token is invalid or expired" in response.json["message"]


# def test_token_required_decorator_expired(client, auth_system):
#    """Test the @token_required decorator with an expired token."""
#    # Register and login to get a token
##     client.post(
#        "/api/auth/register",
##         json={"username": "expireduser", "password": "password123"},
#    )
##     login_response = client.post(
#        "/api/auth/login", json={"username": "expireduser", "password": "password123"}
#    )
##     token = login_response.json["token"]
#
##     time.sleep(1.1)  # Wait for token to expire (set to 1 second in fixture)
#
##     headers = {"Authorization": f"Bearer {token}"}
##     response = client.get("/api/protected_test_unique", headers=headers)
##     assert response.status_code == 401
##     assert "Token is invalid or expired" in response.json["message"]
