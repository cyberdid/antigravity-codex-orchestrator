"""Test configuration and shared fixtures for Antigravity Hybrid Orchestrator."""

import os
import sys
import sqlite3
from pathlib import Path
from typing import Generator
import pytest

# Ensure project root is at the head of sys.path for reliable module imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root() -> Path:
    """Fixture returning the project root Path."""
    return PROJECT_ROOT


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Fixture ensuring a clean environment free of lingering secret keys."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    return monkeypatch


@pytest.fixture
def temp_db() -> Generator[sqlite3.Connection, None, None]:
    """Fixture providing an isolated in-memory SQLite database connection."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    yield conn
    conn.close()


@pytest.fixture
def sample_vulnerable_auth_code() -> str:
    """Baseline vulnerable auth.py code with hardcoded secrets and timing vulnerabilities."""
    return '''import os
import hmac
import hashlib
from typing import Optional

# VULNERABILITY: Hardcoded plaintext secrets
JWT_SECRET = "super-secret-hardcoded-key-12345"
ADMIN_API_KEY = "admin-secret-token-xyz789"

def authenticate_api_key(api_key: str) -> bool:
    # VULNERABILITY: Non-constant time comparison vulnerable to timing side-channels
    return api_key == ADMIN_API_KEY

def generate_token(user_id: str) -> str:
    return hmac.new(JWT_SECRET.encode(), user_id.encode(), hashlib.sha256).hexdigest()

def verify_token(user_id: str, token: str) -> bool:
    expected = generate_token(user_id)
    # VULNERABILITY: Non-constant time comparison
    return token == expected
'''


@pytest.fixture
def sample_vulnerable_billing_code() -> str:
    """Baseline vulnerable billing.py code with check-then-act race conditions."""
    return '''import time
from typing import Dict

class BillingSystem:
    def __init__(self):
        self.balances: Dict[str, float] = {}

    def create_account(self, account_id: str, initial_balance: float = 0.0) -> None:
        self.balances[account_id] = initial_balance

    def get_balance(self, account_id: str) -> float:
        return self.balances.get(account_id, 0.0)

    def withdraw(self, account_id: str, amount: float) -> bool:
        current_balance = self.get_balance(account_id)
        if current_balance >= amount:
            # Latency simulates I/O or database access, guaranteeing context switch
            time.sleep(0.005)
            self.balances[account_id] = current_balance - amount
            return True
        return False
'''


@pytest.fixture
def sample_vulnerable_database_code() -> str:
    """Baseline vulnerable database.py code with string formatted SQL injection."""
    return '''import sqlite3
from typing import List, Tuple, Optional

class DatabaseManager:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                role TEXT,
                email TEXT
            )
        """)
        self.conn.commit()

    def add_user(self, username: str, role: str, email: str) -> None:
        cursor = self.conn.cursor()
        # VULNERABILITY: Raw string concatenation / format
        query = f"INSERT INTO users (username, role, email) VALUES ('{username}', '{role}', '{email}')"
        cursor.execute(query)
        self.conn.commit()

    def get_user_by_username(self, username: str) -> Optional[Tuple]:
        cursor = self.conn.cursor()
        # VULNERABILITY: Raw f-string interpolation allows SQL injection
        query = f"SELECT id, username, role, email FROM users WHERE username = '{username}'"
        cursor.execute(query)
        return cursor.fetchone()

    def get_users_by_role(self, role: str) -> List[Tuple]:
        cursor = self.conn.cursor()
        # VULNERABILITY: Raw f-string interpolation allows SQL injection
        query = f"SELECT id, username, role, email FROM users WHERE role = '{role}'"
        cursor.execute(query)
        return cursor.fetchall()
'''
