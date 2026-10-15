"""Tests for AST privacy stripping and zero-leak cloud payload verification.

Covers:
- ASTPrivacyStripper / strip_code_for_cloud
- verify_cloud_payload / SecurityInvariantViolationError
- Edge cases: docstrings, type annotations, nested classes, async defs, forbidden AST nodes, secret literal leakage
"""

import ast
import pytest

try:
    from antigravity_hybrid_orchestrator.privacy import (
        strip_code_for_cloud,
        verify_cloud_payload,
        SecurityInvariantViolationError,
    )
except ImportError:
    try:
        from antigravity_hybrid_orchestrator.privacy.ast_stripper import strip_code_for_cloud
        from antigravity_hybrid_orchestrator.privacy.verifier import (
            verify_cloud_payload,
            SecurityInvariantViolationError,
        )
    except ImportError:
        strip_code_for_cloud = None
        verify_cloud_payload = None
        SecurityInvariantViolationError = Exception


def _get_privacy_components():
    global strip_code_for_cloud, verify_cloud_payload, SecurityInvariantViolationError
    if strip_code_for_cloud is None or verify_cloud_payload is None:
        try:
            from antigravity_hybrid_orchestrator.privacy import (
                strip_code_for_cloud as _strip,
                verify_cloud_payload as _verify,
                SecurityInvariantViolationError as _err,
            )
        except ImportError:
            from antigravity_hybrid_orchestrator.privacy.ast_stripper import (
                strip_code_for_cloud as _strip,
            )
            from antigravity_hybrid_orchestrator.privacy.verifier import (
                verify_cloud_payload as _verify,
                SecurityInvariantViolationError as _err,
            )
        strip_code_for_cloud = _strip
        verify_cloud_payload = _verify
        SecurityInvariantViolationError = _err
    return strip_code_for_cloud, verify_cloud_payload, SecurityInvariantViolationError


def test_strip_code_basic_function_body_removal():
    """Validates that function bodies and executable statements are replaced with Ellipsis (...)."""
    strip_fn, _, _ = _get_privacy_components()
    original_code = '''
def calculate_hash(data: str) -> str:
    salt = "secret_salt"
    computed = data + salt
    return hashlib.sha256(computed.encode()).hexdigest()
'''
    stripped = strip_fn(original_code)
    # Body must be replaced with ...
    assert "salt = \"secret_salt\"" not in stripped
    assert "secret_salt" not in stripped
    assert "computed = data + salt" not in stripped
    assert "return" not in stripped
    assert "def calculate_hash(data: str) -> str:" in stripped
    assert "..." in stripped


def test_strip_code_docstring_removal():
    """Validates that module, class, and function docstrings are completely eliminated."""
    strip_fn, _, _ = _get_privacy_components()
    original_code = '''"""Module level secret documentation."""

class SecureVault:
    """Class level secret documentation."""
    def open_vault(self, key: str) -> bool:
        """Function level secret documentation."""
        return key == "master"
'''
    stripped = strip_fn(original_code)
    assert "Module level secret documentation." not in stripped
    assert "Class level secret documentation." not in stripped
    assert "Function level secret documentation." not in stripped
    assert "master" not in stripped
    assert "class SecureVault:" in stripped
    assert "def open_vault(self, key: str) -> bool:" in stripped


def test_strip_code_preserves_imports_and_types():
    """Validates that imports and complex parameter/return type annotations are strictly preserved."""
    strip_fn, _, _ = _get_privacy_components()
    original_code = '''import os
import hmac
from typing import Optional, List, Dict, Any

def process_batch(items: List[Dict[str, Any]], timeout: Optional[int] = 30) -> Dict[str, Any]:
    results = {}
    for item in items:
        results[item["id"]] = True
    return results
'''
    stripped = strip_fn(original_code)
    assert "import os" in stripped
    assert "import hmac" in stripped
    assert "from typing import Optional, List, Dict, Any" in stripped
    assert "def process_batch(items: List[Dict[str, Any]], timeout: Optional[int]=30) -> Dict[str, Any]:" in stripped or \
           "def process_batch(items: List[Dict[str, Any]], timeout: Optional[int] = 30) -> Dict[str, Any]:" in stripped
    assert "for item in items:" not in stripped
    assert "return results" not in stripped


def test_strip_code_nested_classes_and_async_defs():
    """Validates that nested classes and async def functions are preserved with stripped bodies."""
    strip_fn, _, _ = _get_privacy_components()
    original_code = '''
class ServiceClient:
    class NestedHelper:
        async def fetch_async(self, endpoint: str) -> bytes:
            payload = await client.get(endpoint)
            return payload.content

    async def run(self) -> None:
        await self.NestedHelper().fetch_async("/api")
'''
    stripped = strip_fn(original_code)
    tree = ast.parse(stripped)
    classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    async_funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]

    assert "ServiceClient" in classes
    assert "NestedHelper" in classes
    assert "fetch_async" in async_funcs
    assert "run" in async_funcs
    assert "payload.content" not in stripped


def test_strip_code_global_literal_replacement():
    """Validates that global variable assignment values are replaced with Ellipsis to prevent secret leakage."""
    strip_fn, _, _ = _get_privacy_components()
    original_code = '''
JWT_SECRET = "super-secret-hardcoded-key-12345"
ADMIN_API_KEY = "admin-secret-token-xyz789"
DEBUG_MODE: bool = True
'''
    stripped = strip_fn(original_code)
    assert "super-secret-hardcoded-key-12345" not in stripped
    assert "admin-secret-token-xyz789" not in stripped
    assert "JWT_SECRET" in stripped
    assert "ADMIN_API_KEY" in stripped


def test_verify_cloud_payload_valid_clean():
    """Validates that stripped code generated from valid sources passes verify_cloud_payload."""
    strip_fn, verify_fn, _ = _get_privacy_components()
    original_code = '''import os
from typing import List

API_URL = "https://internal.vault/v1"

def fetch_data(query: str) -> List[str]:
    # Proprietary business logic
    conn = connect(API_URL)
    return conn.query(query)
'''
    stripped = strip_fn(original_code)
    assert verify_fn(stripped, original_code) is True


def test_verify_cloud_payload_rejects_non_ellipsis_body():
    """Validates that any non-Ellipsis / non-Pass expression in a function body raises SecurityInvariantViolationError."""
    _, verify_fn, violation_err = _get_privacy_components()
    original_code = "def foo():\n    return 42\n"
    # Construct stripped code that leaks an executable return or constant
    leaked_body = "def foo():\n    return 42\n"

    with pytest.raises(violation_err) as exc_info:
        verify_fn(leaked_body, original_code)
    assert "Forbidden logic node" in str(exc_info.value) or "body" in str(exc_info.value)


def test_verify_cloud_payload_rejects_multi_statement_body():
    """Validates that a function body with length > 1 raises SecurityInvariantViolationError."""
    _, verify_fn, violation_err = _get_privacy_components()
    original_code = "def compute():\n    pass\n"
    multi_stmt = "def compute():\n    ...\n    pass\n"

    with pytest.raises(violation_err) as exc_info:
        verify_fn(multi_stmt, original_code)
    assert "body length > 1" in str(exc_info.value) or "Forbidden logic node" in str(exc_info.value)


@pytest.mark.parametrize("bad_stmt", [
    "if x > 0: ...",
    "for x in y: ...",
    "while True: ...",
    "try:\n    ...\nexcept:\n    ...",
    "with lock: ...",
    "return x",
    "yield x",
    "x = a + b",
])
def test_verify_cloud_payload_rejects_forbidden_logic_nodes(bad_stmt):
    """Validates that forbidden control-flow and executable logic nodes raise SecurityInvariantViolationError."""
    _, verify_fn, violation_err = _get_privacy_components()
    original_code = "def sample():\n    pass\n"
    leaked_code = f"def sample():\n    {bad_stmt}\n"

    with pytest.raises(violation_err):
        verify_fn(leaked_code, original_code)


def test_verify_cloud_payload_detects_string_literal_leaks():
    """Validates that original string literals (>= 5 chars) present in stripped code trigger violation."""
    _, verify_fn, violation_err = _get_privacy_components()
    original_code = '''
TOP_SECRET_CREDENTIAL = "CONFIDENTIAL_ORGANIZATION_KEY_98765"

def login():
    return TOP_SECRET_CREDENTIAL
'''
    # Stripped code that failed to strip the secret literal
    faulty_stripped = '''
TOP_SECRET_CREDENTIAL = "CONFIDENTIAL_ORGANIZATION_KEY_98765"

def login():
    ...
'''
    with pytest.raises(violation_err) as exc_info:
        verify_fn(faulty_stripped, original_code)
    assert "CONFIDENTIAL_ORGANIZATION_KEY_98765" in str(exc_info.value) or "literal" in str(exc_info.value).lower()


def test_privacy_pipeline_on_security_gauntlet_targets(
    sample_vulnerable_auth_code,
    sample_vulnerable_billing_code,
    sample_vulnerable_database_code,
):
    """End-to-end verification of the privacy engine across all 3 security gauntlet baseline targets."""
    strip_fn, verify_fn, _ = _get_privacy_components()

    for name, code in [
        ("auth", sample_vulnerable_auth_code),
        ("billing", sample_vulnerable_billing_code),
        ("database", sample_vulnerable_database_code),
    ]:
        stripped = strip_fn(code)

        # 1. verify_cloud_payload must pass cleanly
        assert verify_fn(stripped, code) is True, f"Privacy verification failed for {name}"

        # 2. Assert zero bytes of function bodies or sensitive literals leaked
        if name == "auth":
            assert "super-secret-hardcoded-key-12345" not in stripped
            assert "admin-secret-token-xyz789" not in stripped
            assert "hmac.new" not in stripped
            assert "return" not in stripped
        elif name == "billing":
            assert "self.balances.get" not in stripped
            assert "time.sleep(0.005)" not in stripped
            assert "current_balance - amount" not in stripped
            assert "return" not in stripped
        elif name == "database":
            assert "CREATE TABLE IF NOT EXISTS" not in stripped
            assert "INSERT INTO users" not in stripped
            assert "SELECT id, username" not in stripped
            assert "fetchone()" not in stripped
