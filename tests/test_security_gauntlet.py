"""Adversarial security regression test suite for Antigravity Hybrid Orchestrator.

Covers:
1. auth.py: Functional token/key verification + AST secret leak rejection and timing-safe handling.
2. billing.py: Functional balance operations + 10-thread concurrent overdraft race condition exploit.
3. database.py: Standard CRUD operations + SQL injection bypass and dump attack neutralization.
"""

import ast
import concurrent.futures
import inspect
import os
import pytest

try:
    from antigravity_hybrid_orchestrator.gauntlet import auth, billing, database
except Exception:
    auth = billing = database = None


def _get_auth():
    import importlib, sys
    name = "antigravity_hybrid_orchestrator.gauntlet.auth"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _get_billing():
    import importlib, sys
    name = "antigravity_hybrid_orchestrator.gauntlet.billing"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _get_database():
    import importlib, sys
    name = "antigravity_hybrid_orchestrator.gauntlet.database"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


# --- 1. AUTH TESTS (2 tests) ---

def test_auth_functional(monkeypatch):
    """Verifies standard token generation, signature validation, and API key auth with valid env."""
    auth_module = _get_auth()
    monkeypatch.setenv("JWT_SECRET", "super-secure-test-jwt-secret-key-32b")
    monkeypatch.setenv("ADMIN_API_KEY", "admin-super-secure-api-key-test-32b")

    token = auth_module.generate_token("user_123")
    assert isinstance(token, str) and len(token) == 64
    assert auth_module.verify_token("user_123", token) is True
    assert auth_module.verify_token("user_456", token) is False  # Tamper test
    assert auth_module.authenticate_api_key("admin-super-secure-api-key-test-32b") is True
    assert auth_module.authenticate_api_key("invalid_key_attempt") is False


def test_auth_security_adversarial(monkeypatch):
    """Adversarial check: verifies zero hardcoded secret strings in AST and rejection when env unconfigured."""
    auth_module = _get_auth()
    # Step A: Static AST analysis of auth.py to ensure no hardcoded secrets or static keys
    auth_src = inspect.getsource(auth_module)
    tree = ast.parse(auth_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("JWT_SECRET", "ADMIN_API_KEY"):
                    # Must not be assigned a static string constant
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                        pytest.fail(f"VULNERABILITY DETECTED: Hardcoded secret literal found in {target.id}")

    # Step B: Ensure execution fails or refuses auth when environment variables are missing
    monkeypatch.delenv("JWT_SECRET", raising=False)
    monkeypatch.delenv("ADMIN_API_KEY", raising=False)
    assert auth_module.authenticate_api_key("admin-secret-token-xyz789") is False
    with pytest.raises(ValueError):
        auth_module.generate_token("user_123")


# --- 2. BILLING TESTS (2 tests) ---

def test_billing_functional():
    """Verifies standard sequential balance deduction and overdraft prevention."""
    billing_module = _get_billing()
    system = billing_module.BillingSystem()
    system.create_account("acc_seq", 100.0)
    assert system.get_balance("acc_seq") == 100.0

    assert system.withdraw("acc_seq", 30.0) is True
    assert system.get_balance("acc_seq") == 70.0
    assert system.withdraw("acc_seq", 50.0) is True
    assert system.get_balance("acc_seq") == 20.0
    assert system.withdraw("acc_seq", 25.0) is False  # Insufficient funds
    assert system.get_balance("acc_seq") == 20.0


def test_billing_race_condition_adversarial():
    """Adversarial check: 10 concurrent threads attempt to withdraw $20 simultaneously from $100 balance.
    Patched code with mutex locking MUST guarantee exactly 5 succeed, 5 fail, balance == 0.0."""
    billing_module = _get_billing()
    system = billing_module.BillingSystem()
    system.create_account("acc_race", 100.0)

    # 10 workers requesting $20 = $200 total requested from $100 balance
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(system.withdraw, "acc_race", 20.0) for _ in range(10)]
        results = [f.result() for f in futures]

    successes = results.count(True)
    failures = results.count(False)
    final_balance = system.get_balance("acc_race")

    # In vulnerable un-locked code: successes > 5, balance < 0 or incorrect
    assert successes == 5, f"Race condition detected: {successes} withdrawals succeeded instead of 5"
    assert failures == 5, f"Race condition detected: {failures} withdrawals failed instead of 5"
    assert final_balance == 0.0, f"Account balance corrupted: {final_balance} != 0.0"


# --- 3. DATABASE TESTS (2 tests) ---

def test_database_functional():
    """Verifies standard CRUD operations with clean database records."""
    db_module = _get_database()
    db = db_module.DatabaseManager(":memory:")
    db.add_user("alice", "admin", "alice@example.com")
    db.add_user("bob", "developer", "bob@example.com")
    db.add_user("carol", "developer", "carol@example.com")

    alice = db.get_user_by_username("alice")
    assert alice is not None
    assert alice[1] == "alice" and alice[2] == "admin"

    devs = db.get_users_by_role("developer")
    assert len(devs) == 2
    assert {d[1] for d in devs} == {"bob", "carol"}

    assert db.get_user_by_username("unknown_user") is None


def test_database_sqli_adversarial():
    """Adversarial check: SQL injection attacks against get_user_by_username and get_users_by_role."""
    db_module = _get_database()
    db = db_module.DatabaseManager(":memory:")
    db.add_user("admin_user", "admin", "admin@internal.vault")
    db.add_user("regular_user", "user", "user@internal.vault")

    # Attack 1: Classic authentication/lookup bypass payload
    sqli_username = "nonexistent' OR '1'='1"
    user_result = db.get_user_by_username(sqli_username)
    # In vulnerable code: returns admin_user!
    assert user_result is None, "SQL Injection detected: bypass payload retrieved user record"

    # Attack 2: Union / OR based role dump payload
    sqli_role = "user' OR 1=1 --"
    role_results = db.get_users_by_role(sqli_role)
    # In vulnerable code: returns all users including admin!
    assert len(role_results) == 0, "SQL Injection detected: role bypass dumped records"

    # Attack 3: Data modification injection attempt
    malicious_insert = "hacker'); DROP TABLE users; --"
    db.add_user(malicious_insert, "guest", "hacker@test.com")
    # Users table should still exist and malicious name is stored literally
    retrieved = db.get_user_by_username(malicious_insert)
    assert retrieved is not None
    assert retrieved[1] == malicious_insert
