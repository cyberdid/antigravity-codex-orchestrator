"""Adversarial stress tests authored by Challenger 1 (teamwork_preview_challenger).

Empirically challenges:
1. AST privacy stripping against sneak leaks (default args, AugAssign, decorators, match statements, annotations, lambdas, type aliases).
2. Billing system against high-concurrency race condition exploits (50-100 threads, barrier synchronization).
"""

import ast
import concurrent.futures
import threading
import pytest

from antigravity_hybrid_orchestrator.privacy.ast_stripper import strip_code_for_cloud
from antigravity_hybrid_orchestrator.privacy.verifier import (
    verify_cloud_payload,
    SecurityInvariantViolationError,
)
from antigravity_hybrid_orchestrator.gauntlet.billing import BillingSystem


# ==============================================================================
# 1. BILLING HIGH-CONCURRENCY RACE CONDITION STRESS TESTS (PASSING / ROBUST)
# ==============================================================================

def test_billing_high_concurrency_100_threads_simultaneous_withdraw():
    """Stress test: 100 concurrent threads competing for 50 valid withdrawal slots."""
    system = BillingSystem()
    system.create_account("acc_stress_100", 1000.0)

    num_threads = 100
    amount = 20.0  # 100 * 20 = $2000 total requested from $1000 balance

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(system.withdraw, "acc_stress_100", amount) for _ in range(num_threads)]
        results = [f.result() for f in futures]

    successes = results.count(True)
    failures = results.count(False)
    final_balance = system.get_balance("acc_stress_100")

    assert successes == 50, f"Race condition: {successes} succeeded instead of 50"
    assert failures == 50, f"Race condition: {failures} failed instead of 50"
    assert final_balance == 0.0, f"Balance corruption: {final_balance} != 0.0"


def test_billing_high_concurrency_barrier_synchronized_tight_balance():
    """Stress test: 100 threads synchronized via Barrier to hit withdraw at the exact same instant."""
    system = BillingSystem()
    system.create_account("acc_barrier_100", 100.0)

    num_threads = 100
    amount = 50.0  # Exactly 2 withdrawals can succeed from 100.0 balance
    barrier = threading.Barrier(num_threads)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        res = system.withdraw("acc_barrier_100", amount)
        with lock:
            results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = results.count(True)
    failures = results.count(False)
    final_balance = system.get_balance("acc_barrier_100")

    assert successes == 2, f"Race condition under barrier: {successes} succeeded instead of 2"
    assert failures == 98, f"Race condition under barrier: {failures} failed instead of 98"
    assert final_balance == 0.0, f"Balance corrupted under barrier: {final_balance} != 0.0"


# ==============================================================================
# 2. AST PRIVACY STRIPPING ADVERSARIAL SNEAK LEAK TESTS (VULNERABILITIES FOUND)
# ==============================================================================

def test_ast_privacy_leak_via_function_default_argument():
    """Adversarial check: secrets embedded in function default parameters must not reach cloud."""
    secret_token = "SUPER_SECRET_LEAK_TOKEN_99999"
    code = f'''
def connect_service(endpoint: str, api_key: str = "{secret_token}") -> None:
    pass
'''
    stripped = strip_code_for_cloud(code)

    # 1. Stripper must strip the default argument value
    assert secret_token not in stripped, f"Secret leaked in stripped code: {stripped}"
    assert verify_cloud_payload(stripped, code) is True

    # 2. Verifier must catch the leak if unstripped
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_augmented_assignment():
    """Adversarial check: augmented assignments (+=) must be stripped and not leaked."""
    secret_part = "_SNEAKY_APPENDED_SECRET_987654321"
    code = f'''
API_SECRET = "prefix"
API_SECRET += "{secret_part}"

def run():
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_part not in stripped, f"Secret leaked in AugAssign: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_decorator_literal():
    """Adversarial check: decorator literal expressions must be stripped."""
    secret_dec = "SNEAKY_DECORATOR_SECRET_STRING_12345"
    code = f'''
@("{secret_dec}")
def protected_handler():
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_dec not in stripped, f"Secret leaked in decorator: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_nested_decorators():
    """Adversarial check: nested decorators containing subscripts/literals must be stripped."""
    secret_key = "NESTED_SECRET_KEY_REGISTRY_99999"
    code = f'''
@decorator_a
@decorator_b
@REGISTRY["{secret_key}"]
def protected_endpoint():
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_key not in stripped, f"Secret leaked in nested decorator: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_lambda_default_expression():
    """Adversarial check: executable lambda with literal comparison in default arg must be stripped."""
    secret_token = "SECRET_LAMBDA_COMPARE_TOKEN_12345"
    code = f'''
def execute(action: str, validator = lambda x: x == "{secret_token}"):
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_token not in stripped, f"Secret leaked in lambda default: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_match_statement():
    """Adversarial check: match-case control flow statements must be stripped."""
    secret_match = "SECRET_MATCH_KEY_12345"
    code = f'''
match status:
    case "{secret_match}":
        pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_match not in stripped, f"Secret leaked in match-case: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_type_annotation_literal():
    """Adversarial check: string literals in type annotations must not leak secrets."""
    secret_type = "SECRET_ANNOTATION_TYPE_KEY_12345"
    code = f'''
def authenticate(token: "{secret_type}") -> bool:
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_type not in stripped, f"Secret leaked in type annotation: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_privacy_leak_via_type_alias_statement():
    """Adversarial check: Python 3.12+ type alias statement containing literal secret must be stripped."""
    secret_alias = "SECRET_TYPE_ALIAS_VALUE_12345"
    code = f'''
type SecretToken = "{secret_alias}"

def handler():
    pass
'''
    stripped = strip_code_for_cloud(code)

    assert secret_alias not in stripped, f"Secret leaked in type alias: {stripped}"
    assert verify_cloud_payload(stripped, code) is True
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)


def test_ast_verifier_blocks_callable_in_default_arg():
    """Adversarial check: verify_cloud_payload must catch ast.Call if unstripped code has callable default."""
    code = '''
def calculate(value: int, default = eval("1+1")):
    pass
'''
    stripped = strip_code_for_cloud(code)
    assert verify_cloud_payload(stripped, code) is True
    # The verifier must detect ast.Call if unstripped code with callable default is passed
    with pytest.raises(SecurityInvariantViolationError):
        verify_cloud_payload(code, code)
