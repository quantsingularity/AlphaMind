"""
Tests for AlphaMind Authentication System (FastAPI).
"""

import os
import sys
import time
import unittest

import jwt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from fastapi.testclient import TestClient
from infrastructure.auth.authentication import AuthenticationSystem

client = TestClient(app)


class TestAuthenticationSystem(unittest.TestCase):
    """Comprehensive test suite for the AuthenticationSystem class."""

    def setUp(self):
        self.secret_key = "test-secret-key"
        self.token_expiration = 24
        self.auth = AuthenticationSystem(
            secret_key=self.secret_key,
            token_expiration=self.token_expiration,
        )
        self.test_username = "testuser"
        self.test_password = "Password123!"

    def test_initialization(self):
        self.assertEqual(self.auth.secret_key, self.secret_key)
        self.assertEqual(self.auth.token_expiration, self.token_expiration)
        self.assertEqual(self.auth.users_db, {})

    def test_register_endpoint(self):
        response = client.post(
            "/api/auth/register",
            json={"username": "newuser_reg", "password": "pass123"},
        )
        self.assertEqual(response.status_code, 201)
        self.assertIn("message", response.json())

    def test_register_duplicate(self):
        client.post(
            "/api/auth/register",
            json={"username": "dup_user", "password": "pass"},
        )
        response = client.post(
            "/api/auth/register",
            json={"username": "dup_user", "password": "pass"},
        )
        self.assertEqual(response.status_code, 409)

    def test_login_endpoint(self):
        client.post(
            "/api/auth/register",
            json={"username": "login_user", "password": "mypassword"},
        )
        response = client.post(
            "/api/auth/login",
            json={"username": "login_user", "password": "mypassword"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("token", data)
        self.assertEqual(data["username"], "login_user")

    def test_login_wrong_password(self):
        client.post(
            "/api/auth/register",
            json={"username": "wp_user", "password": "correct"},
        )
        response = client.post(
            "/api/auth/login",
            json={"username": "wp_user", "password": "wrong"},
        )
        self.assertEqual(response.status_code, 401)

    def test_login_nonexistent_user(self):
        response = client.post(
            "/api/auth/login",
            json={"username": "nobody", "password": "pass"},
        )
        self.assertEqual(response.status_code, 401)

    def test_token_generation_verification(self):
        token = self.auth.generate_token(self.test_username)
        self.assertIsInstance(token, str)
        username = self.auth.verify_token(token)
        self.assertEqual(username, self.test_username)

    def test_password_hashing(self):
        import bcrypt

        self.auth.register_user(self.test_username, self.test_password)
        stored_password = self.auth.users_db[self.test_username]["password"]
        self.assertNotEqual(stored_password, self.test_password.encode("utf-8"))
        self.assertTrue(
            bcrypt.checkpw(self.test_password.encode("utf-8"), stored_password)
        )

    def test_token_expiration(self):
        short_auth = AuthenticationSystem(
            secret_key=self.secret_key,
            token_expiration=1 / 3600,
        )
        token = short_auth.generate_token(self.test_username)
        username = short_auth.verify_token(token)
        self.assertEqual(username, self.test_username)
        time.sleep(1.1)
        with self.assertRaises(jwt.ExpiredSignatureError):
            short_auth.verify_token(token)


if __name__ == "__main__":
    unittest.main()
