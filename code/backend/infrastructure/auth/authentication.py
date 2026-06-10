import datetime
import os
import uuid
from typing import Any, Dict, Optional

import bcrypt
import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

router = APIRouter(prefix="/api/auth", tags=["auth"])

security = HTTPBearer()


class RegisterRequest(BaseModel):
    # The mobile client registers with name/email; the original API used
    # username. Accept either so both clients work (email preferred as the id).
    username: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    password: str


class LoginRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class AuthenticationSystem:
    """
    Authentication system for AlphaMind API.
    Provides user registration, login, and JWT token management.
    Compatible with FastAPI.
    """

    def __init__(self, secret_key: str, token_expiration: int = 24) -> None:
        self.secret_key = secret_key
        self.token_expiration = token_expiration
        self.users_db: Dict[str, Any] = {}

    def register_user(
        self,
        username: str,
        password: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing username or password",
            )
        if username in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists",
            )
        hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
        self.users_db[username] = {
            "id": uuid.uuid4().hex,
            "username": username,
            "email": email or username,
            "name": name or username,
            "password": hashed_password,
            "created_at": datetime.datetime.now(datetime.timezone.utc),
        }
        token = self.generate_token(username)
        return {
            "message": "User registered successfully",
            "token": token,
            "username": username,
            "user": self._public_user(self.users_db[username]),
        }

    @staticmethod
    def _public_user(record: Dict[str, Any]) -> Dict[str, Any]:
        """Return a user record without sensitive fields."""
        return {
            "id": record.get("id"),
            "username": record.get("username"),
            "email": record.get("email"),
            "name": record.get("name"),
        }

    def login_user(self, username: str, password: str) -> Dict[str, Any]:
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing username or password",
            )
        if username not in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        if not bcrypt.checkpw(
            password.encode("utf-8"), self.users_db[username]["password"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        token = self.generate_token(username)
        return {
            "message": "Login successful",
            "token": token,
            "username": username,
            "user": self._public_user(self.users_db[username]),
        }

    def generate_token(self, username: str) -> str:
        payload = {
            "exp": datetime.datetime.now(datetime.timezone.utc)
            + datetime.timedelta(hours=self.token_expiration),
            "iat": datetime.datetime.now(datetime.timezone.utc),
            "sub": username,
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> str:
        payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
        return payload["sub"]

    def get_current_user(
        self, credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> Dict[str, Any]:
        token = credentials.credentials
        try:
            username = self.verify_token(token)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is invalid or expired",
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is invalid or expired",
            )
        if username not in self.users_db:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token is invalid or expired",
            )
        return self.users_db[username]


_auth_system: Optional[AuthenticationSystem] = None


def get_auth_system() -> AuthenticationSystem:
    global _auth_system
    if _auth_system is None:
        secret_key = os.getenv("SECRET_KEY", "change-this-secret-key-in-production")
        token_expiration = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        _auth_system = AuthenticationSystem(
            secret_key=secret_key,
            token_expiration=token_expiration,
        )
    return _auth_system


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(request: RegisterRequest) -> Dict[str, Any]:
    auth = get_auth_system()
    # Use email as the identifier when provided (mobile client), else username.
    identifier = request.email or request.username
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either email or username is required",
        )
    return auth.register_user(
        identifier, request.password, name=request.name, email=request.email
    )


@router.post("/login")
async def login(request: LoginRequest) -> Dict[str, Any]:
    auth = get_auth_system()
    identifier = request.email or request.username
    if not identifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either email or username is required",
        )
    return auth.login_user(identifier, request.password)


@router.get("/profile")
async def profile(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    auth = get_auth_system()
    user = auth.get_current_user(credentials)
    return auth._public_user(user)


@router.post("/refresh")
async def refresh(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Dict[str, Any]:
    auth = get_auth_system()
    user = auth.get_current_user(credentials)
    token = auth.generate_token(user["username"])
    return {"token": token, "user": auth._public_user(user)}


@router.post("/logout")
async def logout() -> Dict[str, Any]:
    # Tokens are stateless JWTs; the client discards them on logout. This
    # endpoint exists so clients can signal logout and is a no-op server-side.
    return {"message": "Logged out successfully"}
