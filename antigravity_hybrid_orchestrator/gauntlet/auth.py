import os
import hmac
import hashlib
from typing import Optional

def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret or len(secret) < 16:
        raise ValueError("JWT_SECRET environment variable is missing or insufficiently random (<16 chars)")
    return secret

def get_admin_api_key() -> str:
    key = os.environ.get("ADMIN_API_KEY")
    if not key or len(key) < 16:
        raise ValueError("ADMIN_API_KEY environment variable is missing or insufficiently random (<16 chars)")
    return key

def authenticate_api_key(api_key: str) -> bool:
    try:
        expected = get_admin_api_key()
        return hmac.compare_digest(api_key, expected)
    except ValueError:
        return False

def generate_token(user_id: str) -> str:
    secret = get_jwt_secret()
    return hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()

def verify_token(user_id: str, token: str) -> bool:
    try:
        expected = generate_token(user_id)
        return hmac.compare_digest(token, expected)
    except ValueError:
        return False
