"""Deterministic mocks for On-Device Workforce agents (Builder A, Builder B, Blind Critic)."""

import random
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

from antigravity_hybrid_orchestrator.architect import StrategyResult
from antigravity_hybrid_orchestrator.workforce import CriticResult, PatchResult
from antigravity_hybrid_orchestrator.workforce.blind_critic import anonymize_code


AUTH_PATCH_BUILDER_A = '''import os
import hmac
import hashlib
from typing import Optional

def get_admin_api_key() -> str:
    key = os.environ.get("ADMIN_API_KEY")
    if not key or len(key) < 16:
        raise ValueError("ADMIN_API_KEY environment variable missing or too short")
    return key

def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret or len(secret) < 16:
        raise ValueError("JWT_SECRET environment variable missing or too short")
    return secret

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

AUTH_PATCH_BUILDER_B = '''import os
import hmac
import hashlib
from typing import Optional

def get_admin_api_key() -> str:
    key = os.environ.get("ADMIN_API_KEY")
    if not key or len(key) < 16:
        raise ValueError("ADMIN_API_KEY environment variable missing or too short")
    return key

def get_jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET")
    if not secret or len(secret) < 16:
        raise ValueError("JWT_SECRET environment variable missing or too short")
    return secret

def authenticate_api_key(api_key: str) -> bool:
    if not isinstance(api_key, str) or not api_key:
        return False
    try:
        expected = get_admin_api_key()
        return hmac.compare_digest(api_key, expected)
    except ValueError:
        return False

def generate_token(user_id: str) -> str:
    if not user_id:
        raise ValueError("user_id cannot be empty")
    secret = get_jwt_secret()
    return hmac.new(secret.encode(), user_id.encode(), hashlib.sha256).hexdigest()

def verify_token(user_id: str, token: str) -> bool:
    if not isinstance(token, str) or not token:
        return False
    try:
        expected = generate_token(user_id)
        return hmac.compare_digest(token, expected)
    except ValueError:
        return False
'''

BILLING_PATCH_BUILDER_A = '''import time
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

BILLING_PATCH_BUILDER_B = '''import time
import threading
from typing import Dict

class BillingSystem:
    def __init__(self):
        self.balances: Dict[str, float] = {}
        self._lock = threading.RLock()

    def create_account(self, account_id: str, initial_balance: float = 0.0) -> None:
        with self._lock:
            self.balances[account_id] = max(0.0, float(initial_balance))

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

DATABASE_PATCH_BUILDER_A = '''import sqlite3
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

DATABASE_PATCH_BUILDER_B = '''import sqlite3
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
        if not username:
            raise ValueError("Username cannot be empty")
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, role, email) VALUES (?, ?, ?)",
            (str(username), str(role), str(email)),
        )
        self.conn.commit()

    def get_user_by_username(self, username: str) -> Optional[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE username = ?",
            (str(username),),
        )
        return cursor.fetchone()

    def get_users_by_role(self, role: str) -> List[Tuple]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, username, role, email FROM users WHERE role = ?",
            (str(role),),
        )
        return cursor.fetchall()

    def close(self) -> None:
        self.conn.close()
'''


class MockWorkforceClient:
    """Mock client simulating token streaming from on-device Ollama."""

    def __init__(self, token_chunk_size: int = 15) -> None:
        self.token_chunk_size = token_chunk_size

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> Generator[Tuple[str, str, Dict[str, Any]], None, None]:
        accumulated = ""
        mock_response = "```python\n# Patched module\npass\n```\n"

        words = mock_response.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            accumulated += chunk
            yield chunk, accumulated, {"done": False}

        yield "", accumulated, {
            "done": True,
            "eval_count": 420,
            "eval_duration_s": 1.1,
            "prompt_eval_count": 150,
            "total_duration_s": 1.25,
        }


class MockBuilderAAgent:
    """Mock Builder A: Returns deterministic surgical minimal patches."""

    NAME = "Builder A"

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    def generate_patch(
        self,
        vulnerable_code: str,
        strategy: StrategyResult,
        on_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> PatchResult:
        code_lower = vulnerable_code.lower()
        if "billing" in code_lower or "balances" in code_lower:
            patch = BILLING_PATCH_BUILDER_A
        elif "database" in code_lower or "sqlite3" in code_lower or "get_user_by_username" in code_lower:
            patch = DATABASE_PATCH_BUILDER_A
        else:
            patch = AUTH_PATCH_BUILDER_A

        if on_chunk:
            on_chunk(patch, patch)

        return PatchResult(
            builder_name=self.NAME,
            patch_code=patch,
            eval_tokens=420,
            eval_duration_s=1.1,
        )


class MockBuilderBAgent:
    """Mock Builder B: Returns deterministic defensive / hardened patches."""

    NAME = "Builder B"

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    def generate_patch(
        self,
        vulnerable_code: str,
        strategy: StrategyResult,
        on_chunk: Optional[Callable[[str, str], None]] = None,
    ) -> PatchResult:
        code_lower = vulnerable_code.lower()
        if "billing" in code_lower or "balances" in code_lower:
            patch = BILLING_PATCH_BUILDER_B
        elif "database" in code_lower or "sqlite3" in code_lower or "get_user_by_username" in code_lower:
            patch = DATABASE_PATCH_BUILDER_B
        else:
            patch = AUTH_PATCH_BUILDER_B

        if on_chunk:
            on_chunk(patch, patch)

        return PatchResult(
            builder_name=self.NAME,
            patch_code=patch,
            eval_tokens=510,
            eval_duration_s=1.3,
        )


class MockBlindCriticAgent:
    """Mock Blind Critic: Anonymously evaluates and selects winning patch."""

    def __init__(self, client: Optional[Any] = None) -> None:
        self.client = client

    def evaluate(
        self,
        candidate_1: Union[PatchResult, str],
        candidate_2: Union[PatchResult, str],
        strategy: StrategyResult,
        on_chunk: Optional[Callable[[str, str], None]] = None,
        builder_1_name: str = "Builder A",
        builder_2_name: str = "Builder B",
        shuffle: bool = True,
    ) -> CriticResult:
        name_1 = candidate_1.builder_name if isinstance(candidate_1, PatchResult) else builder_1_name
        name_2 = candidate_2.builder_name if isinstance(candidate_2, PatchResult) else builder_2_name

        # Shuffle Alpha / Beta
        is_swapped = shuffle and random.random() < 0.5
        if is_swapped:
            alpha_builder, beta_builder = name_2, name_1
        else:
            alpha_builder, beta_builder = name_1, name_2

        # In mock mode, we prefer Builder B's defense-in-depth, or Beta
        if beta_builder == "Builder B":
            winner = "Candidate Beta"
            winning_builder = beta_builder
            alpha_score = 8.0
            beta_score = 9.5
        elif alpha_builder == "Builder B":
            winner = "Candidate Alpha"
            winning_builder = alpha_builder
            alpha_score = 9.5
            beta_score = 8.0
        else:
            winner = "Candidate Beta"
            winning_builder = beta_builder
            alpha_score = 8.0
            beta_score = 9.2

        critique = (
            f"{winner} ({winning_builder}) successfully remediated all identified vulnerabilities "
            "with superior defensive hardening, input validation, and fail-safe handling "
            "while strictly preserving public API signatures and contracts."
        )

        if on_chunk:
            on_chunk(critique, critique)

        return CriticResult(
            winner=winner,
            winning_builder=winning_builder,
            alpha_score=alpha_score,
            beta_score=beta_score,
            critique=critique,
            eval_tokens=220,
            eval_duration_s=0.6,
        )
