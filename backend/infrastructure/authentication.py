import datetime
from functools import wraps

import bcrypt
from flask import Flask, jsonify, request
import jwt


class AuthenticationSystem:
    """
    Authentication system for AlphaMind API
    Provides user registration, login, and JWT token management
    """

    def __init__(self, app, secret_key, token_expiration=24):
        """
        Initialize the authentication system

        Args:
            app: Flask application instance
            secret_key: Secret key for JWT token signing
            token_expiration: Token expiration time in hours (default: 24)
        """
        self.app = app
        self.secret_key = secret_key
        self.token_expiration = token_expiration
        self.users_db = {}  # In a real app, this would be a database

        # Register routes
        self._register_routes()

    def _register_routes(self):
        """Register authentication routes with the Flask app"""

        @self.app.route("/api/auth/register", methods=["POST"])
        def register():
            data = request.get_json()

            # Validate input
            if not data or not data.get("username") or not data.get("password"):
                return jsonify({"message": "Missing username or password"}), 400

            username = data["username"]
            password = data["password"]

            # Check if user already exists
            if username in self.users_db:
                return jsonify({"message": "User already exists"}), 409

            # Hash password
            hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

            # Store user
            self.users_db[username] = {
                "username": username,
                "password": hashed_password,
                "created_at": datetime.datetime.utcnow(),
            }

            return jsonify({"message": "User registered successfully"}), 201

        @self.app.route("/api/auth/login", methods=["POST"])
        def login():
            data = request.get_json()

            # Validate input
            if not data or not data.get("username") or not data.get("password"):
                return jsonify({"message": "Missing username or password"}), 400

            username = data["username"]
            password = data["password"]

            # Check if user exists
            if username not in self.users_db:
                return jsonify({"message": "Invalid credentials"}), 401

            # Verify password
            if not bcrypt.checkpw(
                password.encode("utf-8"), self.users_db[username]["password"]
            ):
                return jsonify({"message": "Invalid credentials"}), 401

            # Generate token
            token = self.generate_token(username)

            return (
                jsonify(
                    {
                        "message": "Login successful",
                        "token": token,
                        "username": username,
                    }
                ),
                200,
            )

    def generate_token(self, username):
        """
        Generate a JWT token for the user

        Args:
            username: Username to include in the token

        Returns:
            JWT token string
        """
        payload = {
            "exp": datetime.datetime.utcnow()
            + datetime.timedelta(hours=self.token_expiration),
            "iat": datetime.datetime.utcnow(),
            "sub": username,
        }

        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token):
        """
        Verify a JWT token

        Args:
            token: JWT token to verify

        Returns:
            Username if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload["sub"]
        except jwt.ExpiredSignatureError:
            raise  # Re-raise the exception
        except jwt.InvalidTokenError:
            raise  # Re-raise the exception

    def token_required(self, f):
        """
        Decorator for routes that require authentication

        Usage:
            @auth.token_required
            def protected_route():
                # This route requires authentication
                pass
        """

        @wraps(f)
        def decorated(*args, **kwargs):
            token = None

            # Get token from Authorization header
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]

            if not token:
                return jsonify({"message": "Token is missing"}), 401

            # Verify token
            username = self.verify_token(token)
            if not username:
                return jsonify({"message": "Token is invalid or expired"}), 401

            # Add user to request context
            request.user = self.users_db[username]

            return f(*args, **kwargs)

        return decorated


# Example usage:
"""
app = Flask(__name__)
auth = AuthenticationSystem(app, 'your-secret-key')

@app.route('/api/protected', methods=['GET'])
@auth.token_required
def protected():
    return jsonify({'message': 'This is a protected route'})

if __name__ == '__main__':
    app.run(debug=True)
"""
