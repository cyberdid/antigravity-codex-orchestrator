"""Vulnerable security baselines and environment reset utilities for the Security Gauntlet."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

AUTH_VULNERABLE = '''import os
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

BILLING_VULNERABLE = '''import time
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

DATABASE_VULNERABLE = '''import sqlite3
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

AUTH_SECURE = '''import os
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
'''

BILLING_SECURE = '''import time
import threading
from typing import Dict

class BillingSystem:
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self._lock = threading.Lock()

    def create_account(self, account_id: str, initial_balance: float = 0.0) -> None:
        with self._lock:
            self.balances[account_id] = initial_balance

    def get_balance(self, account_id: str) -> float:
        with self._lock:
            return self.balances.get(account_id, 0.0)

    def withdraw(self, account_id: str, amount: float) -> bool:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        with self._lock:
            current_balance = self.balances.get(account_id, 0.0)
            if current_balance >= amount:
                time.sleep(0.005)
                self.balances[account_id] = current_balance - amount
                return True
            return False
'''

DATABASE_SECURE = '''import sqlite3
from typing import List, Tuple, Optional

class DatabaseManager:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
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
        cursor.execute(
            "INSERT INTO users (username, role, email) VALUES (?, ?, ?)",
            (username, role, email),
        )
        self.conn.commit()

    def get_user_by_username(self, username: str) -> Optional[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE username = ?",
            (username,),
        )
        return cursor.fetchone()

    def get_users_by_role(self, role: str) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE role = ?",
            (role,),
        )
        return cursor.fetchall()

    def close(self) -> None:
        self.conn.close()
'''

VULNERABLE_BASELINES: Dict[str, str] = {
    "auth.py": AUTH_VULNERABLE,
    "billing.py": BILLING_VULNERABLE,
    "database.py": DATABASE_VULNERABLE,
}

SECURE_IMPLEMENTATIONS: Dict[str, str] = {
    "auth.py": AUTH_SECURE,
    "billing.py": BILLING_SECURE,
    "database.py": DATABASE_SECURE,
}


def _normalize_name(name: str) -> str:
    base = os.path.basename(name)
    if not base.endswith(".py"):
        base = f"{base}.py"
    return base


def get_baseline(module_name: str) -> str:
    """Returns the vulnerable baseline source code for a module."""
    key = _normalize_name(module_name)
    if key not in VULNERABLE_BASELINES:
        raise KeyError(f"Unknown gauntlet module: {module_name}. Valid: {list(VULNERABLE_BASELINES.keys())}")
    return VULNERABLE_BASELINES[key]


def get_secure(module_name: str) -> str:
    """Returns the secure implementation source code for a module."""
    key = _normalize_name(module_name)
    if key not in SECURE_IMPLEMENTATIONS:
        raise KeyError(f"Unknown gauntlet module: {module_name}. Valid: {list(SECURE_IMPLEMENTATIONS.keys())}")
    return SECURE_IMPLEMENTATIONS[key]


def reset_gauntlet(
    target_dir: Optional[Union[str, Path]] = None,
    targets: Optional[Union[str, List[str]]] = None,
) -> None:
    """Restores the vulnerable baseline files into target_dir.

    Args:
        target_dir: Directory to reset. Defaults to the directory containing this module.
        targets: Optional target module name or list of module names to reset.
    """
    dest = Path(target_dir) if target_dir is not None else Path(__file__).parent.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if targets is not None:
        if isinstance(targets, str):
            selected = [_normalize_name(targets)]
        else:
            selected = [_normalize_name(t) for t in targets]
    else:
        selected = list(VULNERABLE_BASELINES.keys())

    for filename in selected:
        if filename in VULNERABLE_BASELINES:
            file_path = dest / filename
            file_path.write_text(VULNERABLE_BASELINES[filename], encoding="utf-8")


def restore_secure(
    target_dir: Optional[Union[str, Path]] = None,
    targets: Optional[Union[str, List[str]]] = None,
) -> None:
    """Restores the secure patched files into target_dir.

    Args:
        target_dir: Directory to update. Defaults to the directory containing this module.
        targets: Optional target module name or list of module names to restore.
    """
    dest = Path(target_dir) if target_dir is not None else Path(__file__).parent.resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if targets is not None:
        if isinstance(targets, str):
            selected = [_normalize_name(targets)]
        else:
            selected = [_normalize_name(t) for t in targets]
    else:
        selected = list(SECURE_IMPLEMENTATIONS.keys())

    for filename in selected:
        if filename in SECURE_IMPLEMENTATIONS:
            file_path = dest / filename
            file_path.write_text(SECURE_IMPLEMENTATIONS[filename], encoding="utf-8")
