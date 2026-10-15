"""Programmatic test runner for executing security gauntlet test suites."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Optional, Tuple


class GauntletTestRunner:
    """Runs pytest suites programmatically targeting gauntlet modules."""

    DEFAULT_PYTEST_PATHS = [
        str(Path(__file__).resolve().parent.parent.parent / ".venv" / "bin" / "pytest"),
        shutil.which("pytest"),
        os.path.expanduser("~/.local/bin/pytest"),
        "pytest",
    ]

    TARGET_TEST_MAP = {
        "auth.py": ("test_auth_functional", "test_auth_security_adversarial"),
        "billing.py": ("test_billing_functional", "test_billing_race_condition_adversarial"),
        "database.py": ("test_database_functional", "test_database_sqli_adversarial"),
    }

    def __init__(
        self,
        project_root: Optional[Path] = None,
        pytest_bin: Optional[str] = None,
    ) -> None:
        if project_root is not None:
            self.project_root = Path(project_root).resolve()
        else:
            # Locate root directory containing pyproject.toml / tests/
            self.project_root = Path(__file__).resolve().parent.parent.parent

        if pytest_bin:
            self.pytest_bin = pytest_bin
        else:
            self.pytest_bin = None
            for candidate in self.DEFAULT_PYTEST_PATHS:
                if candidate and shutil.which(candidate):
                    self.pytest_bin = candidate
                    break
            if not self.pytest_bin:
                self.pytest_bin = os.path.expanduser("~/.local/bin/pytest")

    def _normalize_target(self, target: str) -> str:
        base = os.path.basename(target)
        if not base.endswith(".py"):
            base = f"{base}.py"
        return base

    def run_single_test(self, test_name: str) -> bool:
        """Runs a single test function by name and returns whether it passed."""
        test_file = self.project_root / "tests" / "test_security_gauntlet.py"
        cmd = [
            self.pytest_bin,
            str(test_file),
            "-k",
            test_name,
            "-q",
        ]
        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=30.0,
            )
            self.last_output = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
            return res.returncode == 0
        except Exception as e:
            self.last_output = f"Execution error: {e}"
            return False

    def run_tests_for_target(self, target: str) -> Tuple[bool, bool]:
        """Runs functional and adversarial tests for a given target file.

        Returns:
            Tuple of (functional_passed, adversarial_passed)
        """
        norm_target = self._normalize_target(target)
        if norm_target not in self.TARGET_TEST_MAP:
            return False, False

        func_test, adv_test = self.TARGET_TEST_MAP[norm_target]
        func_ok = self.run_single_test(func_test)
        adv_ok = self.run_single_test(adv_test)
        return func_ok, adv_ok

    def run_all_tests(self) -> Dict[str, Tuple[bool, bool]]:
        """Runs tests for all 3 gauntlet modules."""
        results = {}
        for target in self.TARGET_TEST_MAP:
            results[target] = self.run_tests_for_target(target)
        return results
