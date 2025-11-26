import datetime
import os
import sys
import unittest

# Correct the path to the backend directory within the project
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
)

from flask import Flask, jsonify, request
from infrastructure.authentication import AuthenticationSystem


class TestAuthenticationSystem(unittest.TestCase):
    """Comprehensive test suite for the AuthenticationSystem class"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = Flask(__name__)
        self.app.config["TESTING"] = True
        self.secret_key = "test-secret-key"
        self.token_expiration = 24  # hours

        # Create authentication system
        self.auth = AuthenticationSystem(
            app=self.app,
            secret_key=self.secret_key,
            token_expiration=self.token_expiration,
        )

        # Test client
        self.client = self.app.test_client()

        # Test user credentials
        self.test_username = "testuser"
        self.test_password = "Password123!"

    def test_initialization(self):
        """Test that the AuthenticationSystem initializes correctly"""
        self.assertEqual(self.auth.app, self.app)
        self.assertEqual(self.auth.secret_key, self.secret_key)
        self.assertEqual(self.auth.token_expiration, self.token_expiration)
        self.assertEqual(self.auth.users_db, {})

    def test_register_route(self):
        """Test the registration route"""
        with self.app.app_context():
            # Test successful registration
            response = self.client.post(
                "/api/auth/register",
                json={"username": self.test_username, "password": self.test_password},
            )
            self.assertEqual(response.status_code, 201)
            self.assertEqual(response.json["message"], "User registered successfully")

            # Verify user was added to the database
            self.assertIn(self.test_username, self.auth.users_db)

            # Test registration with existing username
            response = self.client.post(
                "/api/auth/register",
                json={"username": self.test_username, "password": "different_password"},
            )
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json["message"], "User already exists")

            # Test registration with missing username
            response = self.client.post(
                "/api/auth/register", json={"password": self.test_password}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

            # Test registration with missing password
            response = self.client.post(
                "/api/auth/register", json={"username": "newuser"}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

            # Test registration with empty JSON
            response = self.client.post("/api/auth/register", json={})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

    def test_login_route(self):
        """Test the login route"""
        with self.app.app_context():
            # Register a user first
            self.client.post(
                "/api/auth/register",
                json={"username": self.test_username, "password": self.test_password},
            )

            # Test successful login
            response = self.client.post(
                "/api/auth/login",
                json={"username": self.test_username, "password": self.test_password},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["message"], "Login successful")
            self.assertIn("token", response.json)
            self.assertEqual(response.json["username"], self.test_username)

            # Test login with incorrect password
            response = self.client.post(
                "/api/auth/login",
                json={"username": self.test_username, "password": "wrong_password"},
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json["message"], "Invalid credentials")

            # Test login with non-existent user
            response = self.client.post(
                "/api/auth/login",
                json={"username": "nonexistent_user", "password": self.test_password},
            )
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json["message"], "Invalid credentials")

            # Test login with missing username
            response = self.client.post(
                "/api/auth/login", json={"password": self.test_password}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

            # Test login with missing password
            response = self.client.post(
                "/api/auth/login", json={"username": self.test_username}
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

            # Test login with empty JSON
            response = self.client.post("/api/auth/login", json={})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json["message"], "Missing username or password")

    def test_token_generation_verification(self):
        """Test JWT token generation and verification"""
        # Generate token
        token = self.auth.generate_token(self.test_username)

        # Verify token is a string
        self.assertIsInstance(token, str)

        # Verify token can be decoded
        username = self.auth.verify_token(token)
        self.assertEqual(username, self.test_username)

    def test_token_required_decorator(self):
        """Test the token_required decorator"""
        # Create a separate app for this test to avoid conflicts
        test_app = Flask("decorator_test")
        test_app.config["TESTING"] = True
        test_auth = AuthenticationSystem(
            app=test_app,
            secret_key=self.secret_key,
            token_expiration=self.token_expiration,
        )

        # Register a user in the test auth system
        test_auth.users_db[self.test_username] = {
            "username": self.test_username,
            "password": b"dummy_hashed_password",
            "created_at": datetime.datetime.utcnow(),
        }

        # Create a protected route using the decorator
        @test_app.route("/api/test_protected")
        @test_auth.token_required
        def protected_route():
            return jsonify(
                {"message": "Access granted", "user": request.user["username"]}
            )

        # Create a test client for the new app
        test_client = test_app.test_client()

        with test_app.app_context():
            # Generate a token for the test user
            token = test_auth.generate_token(self.test_username)

            # Test access with valid token
            headers = {"Authorization": f"Bearer {token}"}
            response = test_client.get("/api/test_protected", headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json["message"], "Access granted")
            self.assertEqual(response.json["user"], self.test_username)

            # Test access without token
            response = test_client.get("/api/test_protected")
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.json["message"], "Token is missing")

            # Test access with invalid token
            headers = {"Authorization": "Bearer invalid.token.string"}
            response = test_client.get("/api/test_protected", headers=headers)
            self.assertEqual(response.status_code, 401)
            self.assertIn("Token is invalid", response.json["message"])

    def test_password_hashing(self):
        """Test that passwords are properly hashed"""
        with self.app.app_context():
            # Register a user
            self.client.post(
                "/api/auth/register",
                json={"username": self.test_username, "password": self.test_password},
            )

            # Check that the password is hashed
            stored_password = self.auth.users_db[self.test_username]["password"]
            self.assertNotEqual(stored_password, self.test_password.encode("utf-8"))

            # Verify the hash is valid
            import bcrypt

            self.assertTrue(
                bcrypt.checkpw(self.test_password.encode("utf-8"), stored_password)
            )

    def test_token_expiration(self):
        """Test token expiration"""
        # Create a separate app for this test to avoid conflicts
        test_app = Flask("expiration_test")
        test_app.config["TESTING"] = True

        # Create auth system with very short expiration
        short_auth = AuthenticationSystem(
            app=test_app,
            secret_key=self.secret_key,
            token_expiration=1 / 3600,  # 1 second
        )

        # Generate token
        token = short_auth.generate_token(self.test_username)

        # Verify token is valid initially
        username = short_auth.verify_token(token)
        self.assertEqual(username, self.test_username)

        # Wait for token to expire
        import time

        time.sleep(1.1)  # Wait slightly longer than expiration time

        # Verify token is now expired
        import jwt

        with self.assertRaises(jwt.ExpiredSignatureError):
            short_auth.verify_token(token)

    def test_security_headers(self):
        """Test that security headers are properly set"""
        with self.app.app_context():
            # Register a user
            response = self.client.post(
                "/api/auth/register",
                json={"username": self.test_username, "password": self.test_password},
            )

            # Check for security headers
            # Note: Flask's test client doesn't automatically add security headers
            # In a real app, these would be added via middleware or app configuration
            # This test is a placeholder for that verification

            # For example, check Content-Type
            self.assertEqual(response.content_type, "application/json")


if __name__ == "__main__":
    unittest.main()
