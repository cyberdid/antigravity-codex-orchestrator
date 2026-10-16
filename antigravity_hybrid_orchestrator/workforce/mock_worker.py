"""Deterministic Mock Autonomous Worker for fast, offline testing and CI (Milestone 3, Feature F10).

Provides:
- MockAutonomousWorker: High-speed (<0.05s) deterministic simulation of autonomous worker
  executing SubTasks using real WorkspaceToolSuite, with failure injection, realistic code
  synthesis, and HUD event emission.
- Legacy gauntlet workforce client compatibility (MockLocalWorkforceClient, MockWorkforceClient).
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple

from antigravity_hybrid_orchestrator.planner.dag import SubTask
from antigravity_hybrid_orchestrator.workforce.tools import ToolResult, WorkspaceToolSuite
from antigravity_hybrid_orchestrator.workforce.worker import SubtaskResult


class MockAutonomousWorker:
    """Fast, deterministic mock autonomous worker for offline testing and CI.

    Attributes:
        fail_subtasks: Set of subtask IDs that should fail execution.
        error_on_tool: Mapping of tool_name -> error message to inject on tool calls.
        custom_file_contents: Mapping of relative path -> custom content to write.
        max_turns: Maximum simulated ReAct turns before completion.
        event_callback: Callable receiving (event_type, subtask_id, payload).
        token_chunk_size: Compatibility setting for token streaming simulation.
    """

    _EVAL_TEMPLATES: Dict[str, str] = {
        "matrix.py": (
            "def multiply(a, b):\n"
            "    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]\n\n"
            "def transpose(a):\n"
            "    return [list(col) for col in zip(*a)]\n"
        ),
        "test_matrix.py": (
            "import unittest\n"
            "import matrix as m\n\n"
            "class TestMatrix(unittest.TestCase):\n"
            "    def test_multiply_and_transpose(self):\n"
            "        self.assertEqual(m.multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19, 22], [43, 50]])\n"
            "        self.assertEqual(m.transpose([[1, 2, 3], [4, 5, 6]]), [[1, 4], [2, 5], [3, 6]])\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "cli.py": (
            "import argparse\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--count', type=int, default=3)\n"
            "    parser.add_argument('--prefix', type=str, default='item')\n"
            "    args = parser.parse_args()\n"
            "    for i in range(1, args.count + 1):\n"
            "        print(f'{args.prefix}-{i}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "test_cli.py": (
            "import subprocess\n"
            "import sys\n"
            "import unittest\n\n"
            "class TestCli(unittest.TestCase):\n"
            "    def test_cli_flags(self):\n"
            "        out = subprocess.run([sys.executable, 'cli.py', '--count', '2', '--prefix', 'x'], capture_output=True, text=True, check=True).stdout.split()\n"
            "        self.assertEqual(out, ['x-1', 'x-2'])\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "lru_cache.py": (
            "from collections import OrderedDict\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int):\n"
            "        self.capacity = capacity\n"
            "        self._store = OrderedDict()\n\n"
            "    def get(self, key):\n"
            "        if key not in self._store:\n"
            "            return None\n"
            "        self._store.move_to_end(key)\n"
            "        return self._store[key]\n\n"
            "    def put(self, key, value) -> None:\n"
            "        if key in self._store:\n"
            "            self._store.move_to_end(key)\n"
            "        self._store[key] = value\n"
            "        if len(self._store) > self.capacity:\n"
            "            self._store.popitem(last=False)\n"
        ),
        "test_lru_cache.py": (
            "import unittest\n"
            "from lru_cache import LRUCache\n\n"
            "class TestLRUCache(unittest.TestCase):\n"
            "    def test_eviction(self):\n"
            "        c = LRUCache(2)\n"
            "        c.put('a', 1)\n"
            "        c.put('b', 2)\n"
            "        self.assertEqual(c.get('a'), 1)\n"
            "        c.put('c', 3)\n"
            "        self.assertIsNone(c.get('b'))\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "tokensign.py": (
            "import hashlib\n"
            "import hmac\n\n"
            "def sign(payload: str, secret: str) -> str:\n"
            "    mac = hmac.new(secret.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()\n"
            "    return f'{payload}.{mac}'\n\n"
            "def verify(token: str, secret: str) -> bool:\n"
            "    if '.' not in token:\n"
            "        return False\n"
            "    payload, sig = token.rsplit('.', 1)\n"
            "    expected = hmac.new(secret.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()\n"
            "    return hmac.compare_digest(sig, expected)\n"
        ),
        "test_token.py": (
            "import unittest\n"
            "import tokensign as t\n\n"
            "class TestTokenSign(unittest.TestCase):\n"
            "    def test_sign_and_verify(self):\n"
            "        tok = t.sign('user=ada', 'k1')\n"
            "        self.assertTrue(t.verify(tok, 'k1'))\n"
            "        self.assertFalse(t.verify(tok, 'k2'))\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "repo.py": (
            "from contextlib import contextmanager\n"
            "import sqlite3\n\n"
            "class Repo:\n"
            "    def __init__(self, path: str):\n"
            "        self.conn = sqlite3.connect(path)\n"
            "        self._in_tx = False\n"
            "        self.conn.execute(\n"
            "            'CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL)'\n"
            "        )\n"
            "        self.conn.commit()\n\n"
            "    def add(self, name: str, email: str) -> int:\n"
            "        cur = self.conn.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))\n"
            "        if not self._in_tx:\n"
            "            self.conn.commit()\n"
            "        return int(cur.lastrowid)\n\n"
            "    def get(self, user_id: int):\n"
            "        row = self.conn.execute('SELECT id, name, email FROM users WHERE id = ?', (user_id,)).fetchone()\n"
            "        if row is None:\n"
            "            return None\n"
            "        return {'id': row[0], 'name': row[1], 'email': row[2]}\n\n"
            "    def list_all(self):\n"
            "        rows = self.conn.execute('SELECT id, name, email FROM users ORDER BY id').fetchall()\n"
            "        return [{'id': r[0], 'name': r[1], 'email': r[2]} for r in rows]\n\n"
            "    def delete(self, user_id: int) -> bool:\n"
            "        cur = self.conn.execute('DELETE FROM users WHERE id = ?', (user_id,))\n"
            "        if not self._in_tx:\n"
            "            self.conn.commit()\n"
            "        return cur.rowcount > 0\n\n"
            "    @contextmanager\n"
            "    def transaction(self):\n"
            "        self._in_tx = True\n"
            "        try:\n"
            "            yield\n"
            "            self.conn.commit()\n"
            "        except Exception:\n"
            "            self.conn.rollback()\n"
            "            raise\n"
            "        finally:\n"
            "            self._in_tx = False\n"
        ),
        "test_repo.py": (
            "import os\n"
            "import tempfile\n"
            "import unittest\n"
            "from repo import Repo\n\n"
            "class TestRepo(unittest.TestCase):\n"
            "    def test_crud_and_rollback(self):\n"
            "        d = tempfile.mkdtemp()\n"
            "        r = Repo(os.path.join(d, 'r.db'))\n"
            "        uid = r.add('ada', 'ada@x.io')\n"
            "        self.assertEqual(r.get(uid)['name'], 'ada')\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
        "inventory.py": (
            "import argparse\n"
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n\n"
            "def main():\n"
            "    parser = argparse.ArgumentParser()\n"
            "    parser.add_argument('--db', default='inventory.json')\n"
            "    sub = parser.add_subparsers(dest='cmd', required=True)\n"
            "    p_add = sub.add_parser('add')\n"
            "    p_add.add_argument('name')\n"
            "    p_add.add_argument('qty', type=int)\n"
            "    p_rem = sub.add_parser('remove')\n"
            "    p_rem.add_argument('name')\n"
            "    sub.add_parser('list')\n"
            "    args = parser.parse_args()\n"
            "    db_path = Path(args.db)\n"
            "    data = json.loads(db_path.read_text(encoding='utf-8')) if db_path.exists() else {}\n"
            "    if args.cmd == 'add':\n"
            "        data[args.name] = int(data.get(args.name, 0)) + args.qty\n"
            "        db_path.write_text(json.dumps(data), encoding='utf-8')\n"
            "    elif args.cmd == 'remove':\n"
            "        if args.name not in data:\n"
            "            sys.exit(1)\n"
            "        del data[args.name]\n"
            "        db_path.write_text(json.dumps(data), encoding='utf-8')\n"
            "    elif args.cmd == 'list':\n"
            "        for name in sorted(data):\n"
            "            print(f'{name} {data[name]}')\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        ),
        "test_inventory.py": (
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "import tempfile\n"
            "import unittest\n\n"
            "class TestInventory(unittest.TestCase):\n"
            "    def test_inventory_cli(self):\n"
            "        d = tempfile.mkdtemp()\n"
            "        db = os.path.join(d, 'inv.json')\n"
            "        res = subprocess.run([sys.executable, 'inventory.py', '--db', db, 'add', 'bolt', '5'])\n"
            "        self.assertEqual(res.returncode, 0)\n\n"
            "if __name__ == '__main__':\n"
            "    unittest.main()\n"
        ),
    }

    def __init__(
        self,
        fail_subtasks: Optional[Set[str]] = None,
        error_on_tool: Optional[Dict[str, str]] = None,
        custom_file_contents: Optional[Dict[str, str]] = None,
        max_turns: int = 10,
        event_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        token_chunk_size: int = 15,
        **kwargs: Any,
    ) -> None:
        self.fail_subtasks: Set[str] = set(fail_subtasks or [])
        self.error_on_tool: Dict[str, str] = dict(error_on_tool or {})
        self.custom_file_contents: Dict[str, str] = dict(custom_file_contents or {})
        self.max_turns = max_turns
        self.event_callback = event_callback
        self.token_chunk_size = token_chunk_size

    def _emit(self, event_type: str, subtask_id: str, payload: Dict[str, Any]) -> None:
        """Safely dispatches HUD event without failing if callback raises."""
        if self.event_callback:
            try:
                self.event_callback(event_type, subtask_id, payload)
            except Exception:
                pass

    def execute_subtask(
        self,
        subtask: SubTask,
        tool_suite: WorkspaceToolSuite,
        feedback_context: Optional[str] = None,
        deadline: Optional[float] = None,  # accepted for interface parity; the mock finishes instantly
    ) -> SubtaskResult:
        """Executes a SubTask deterministically using the provided tool_suite (<0.05s)."""
        turn = 1
        tool_calls_count = 0

        # 1. Emit subtask_start
        self._emit(
            "subtask_start",
            subtask.id,
            {"subtask_id": subtask.id, "title": subtask.title},
        )

        # 2. Emit thought
        thought_msg = f"Planning implementation for subtask '{subtask.id}': {subtask.title}"
        if feedback_context:
            thought_msg += f". Addressing feedback: {feedback_context}"
        self._emit(
            "thought",
            subtask.id,
            {"thought": thought_msg, "turn": turn, "feedback_context": feedback_context},
        )

        # 3. Check injected failure
        if subtask.id in self.fail_subtasks:
            err = f"Injected failure for subtask {subtask.id}"
            self._emit(
                "subtask_failed",
                subtask.id,
                {"subtask_id": subtask.id, "error": err},
            )
            return SubtaskResult(
                subtask_id=subtask.id,
                success=False,
                turns=1,
                tool_calls_count=0,
                error=err,
                metadata={"injected_failure": True},
            )

        # 4. Process affected files
        for file_path in subtask.affected_files:
            # Check existing file read
            candidate = tool_suite.workspace_root / file_path
            if candidate.exists() and candidate.is_file():
                turn += 1
                tool_calls_count += 1
                self._emit(
                    "tool_call",
                    subtask.id,
                    {"tool": "read_file", "args": {"path": file_path}, "turn": turn},
                )

                if "read_file" in self.error_on_tool:
                    injected_err = self.error_on_tool["read_file"]
                    read_res = ToolResult(success=False, output="", error=injected_err)
                else:
                    read_res = tool_suite.read_file(file_path)

                self._emit(
                    "tool_result",
                    subtask.id,
                    {
                        "tool": "read_file",
                        "success": read_res.success,
                        "output": read_res.output,
                        "error": read_res.error,
                    },
                )

            # Determine content to write
            if file_path in self.custom_file_contents:
                content = self.custom_file_contents[file_path]
            elif Path(file_path).name in self._EVAL_TEMPLATES:
                content = self._EVAL_TEMPLATES[Path(file_path).name]
            elif file_path.endswith(".py"):
                file_name = Path(file_path).name.lower()
                is_test_file = (
                    file_name.startswith("test_")
                    or file_name.endswith("_test.py")
                    or "tests/" in file_path.replace("\\", "/").lower()
                )
                if is_test_file:
                    content = "def test_example():\n    assert True\n\ndef test_solution():\n    assert True\n"
                else:
                    combined_desc = f"{subtask.title} {subtask.description} {subtask.success_criteria}".lower()
                    import re
                    words = set(re.findall(r"\w+", combined_desc))
                    if any(k in words for k in ("add", "math", "calc", "calculator", "addition")):
                        content = (
                            "def add(a: int, b: int) -> int:\n"
                            "    return a + b\n\n"
                            "def execute() -> bool:\n"
                            "    return True\n"
                        )
                    else:
                        content = (
                            f'"""Generated implementation for {subtask.title}."""\n\n'
                            "def execute() -> bool:\n"
                            "    return True\n\n"
                            'if __name__ == "__main__":\n'
                            '    print("Success")\n'
                        )

            elif file_path.endswith(".json"):
                content = json.dumps({"status": "ok", "task_id": subtask.id}, indent=2)
            elif file_path.endswith((".md", ".txt")):
                content = f"# {subtask.title}\n\nCompleted successfully.\n"
            else:
                content = f"# Content for {file_path}\n"

            # Execute write_file
            turn += 1
            tool_calls_count += 1
            self._emit(
                "tool_call",
                subtask.id,
                {"tool": "write_file", "args": {"path": file_path, "content": content}, "turn": turn},
            )

            if "write_file" in self.error_on_tool:
                injected_err = self.error_on_tool["write_file"]
                self._emit(
                    "tool_result",
                    subtask.id,
                    {"tool": "write_file", "success": False, "output": "", "error": injected_err},
                )
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": injected_err},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=injected_err,
                    metadata={"tool_error": injected_err},
                )

            write_res = tool_suite.write_file(file_path, content)
            self._emit(
                "tool_result",
                subtask.id,
                {
                    "tool": "write_file",
                    "success": write_res.success,
                    "output": write_res.output,
                    "error": write_res.error,
                },
            )

            if not write_res.success:
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": write_res.error},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=write_res.error,
                )

        # 5. Execute verification command if provided
        if subtask.verification_command:
            turn += 1
            tool_calls_count += 1
            self._emit(
                "tool_call",
                subtask.id,
                {"tool": "run_command", "args": {"command": subtask.verification_command}, "turn": turn},
            )

            if "run_command" in self.error_on_tool:
                injected_err = self.error_on_tool["run_command"]
                self._emit(
                    "tool_result",
                    subtask.id,
                    {"tool": "run_command", "success": False, "output": "", "error": injected_err},
                )
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": injected_err},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=injected_err,
                    metadata={"tool_error": injected_err},
                )

            cmd_res = tool_suite.run_command(subtask.verification_command)
            self._emit(
                "tool_result",
                subtask.id,
                {
                    "tool": "run_command",
                    "success": cmd_res.success,
                    "output": cmd_res.output,
                    "error": cmd_res.error,
                },
            )

            if not cmd_res.success:
                err_msg = cmd_res.error or f"Verification command failed with exit code {cmd_res.metadata.get('exit_code')}"
                self._emit(
                    "subtask_failed",
                    subtask.id,
                    {"subtask_id": subtask.id, "error": err_msg},
                )
                return SubtaskResult(
                    subtask_id=subtask.id,
                    success=False,
                    turns=turn,
                    tool_calls_count=tool_calls_count,
                    error=err_msg,
                    metadata={"exit_code": cmd_res.metadata.get("exit_code")},
                )

        # 6. Complete subtask successfully
        summary = f"Subtask '{subtask.id}' completed successfully."
        self._emit(
            "subtask_complete",
            subtask.id,
            {"subtask_id": subtask.id, "summary": summary},
        )
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            turns=turn,
            tool_calls_count=tool_calls_count,
            summary=summary,
            artifacts=list(subtask.affected_files),
        )

    # Legacy gauntlet workforce client compatibility
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
            "eval_duration_s": 0.05,
            "prompt_eval_count": 150,
            "total_duration_s": 0.06,
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        for _, accumulated, meta in self.generate_stream(prompt, system_prompt, temperature):
            if meta.get("done"):
                return accumulated
        return "```python\n# Patched module\npass\n```\n"


# Re-exports for legacy client aliases
MockLocalWorkforceClient = MockAutonomousWorker
MockWorkforceClient = MockAutonomousWorker
