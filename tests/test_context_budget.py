import json
import pytest
from antigravity_hybrid_orchestrator.workforce.context import compact_messages


def serialized(messages):
    return len(json.dumps(messages, ensure_ascii=False).encode())


def test_many_small_turns_have_total_bound_and_complete_recent_exchanges():
    messages = [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'TASK'}]
    for i in range(200):
        messages.extend([{'role': 'assistant', 'tool_calls': [{'function': {'name': 'read', 'arguments': {'path': str(i)}}}]},
                         {'role': 'tool', 'tool_name': 'read', 'content': 'é' * 100}])
    compact_messages(messages, max_bytes=4000)
    assert serialized(messages) <= 4000
    assert messages[:2] == [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'TASK'}]
    assert messages[-2]['tool_calls'][0]['function']['arguments']['path'] == '199'
    assert all(m['role'] != 'tool' or messages[i-1]['role'] in ('assistant', 'tool') for i, m in enumerate(messages))


def test_irreducible_prompt_fails_clearly():
    messages = [{'role': 'system', 'content': 'x' * 3000}, {'role': 'user', 'content': 'task'}]
    with pytest.raises(ValueError, match='context budget'):
        compact_messages(messages, max_bytes=1000)


def test_nested_and_serialized_arguments_are_bounded():
    messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'},
                {'role': 'assistant', 'tool_calls': [{'function': {'name': 'write', 'arguments': {'nested': ['x' * 100000]}}}]},
                {'role': 'tool', 'tool_name': 'write', 'content': 'ok'}]
    compact_messages(messages, max_bytes=3000)
    assert serialized(messages) < 3000
    assert messages[-1]['content'] == 'ok'


def test_small_model_window_reserves_room_for_output(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
    monkeypatch.setenv('OLLAMA_NUM_CTX', '2048')
    captured = {}
    client = WorkforceClient()
    def post(url, payload):
        captured.update(payload)
        return {'message': {'content': 'ok'}}
    monkeypatch.setattr(client, '_post_json', post)
    client.chat([{'role': 'user', 'content': 'task'}])
    assert captured['options']['num_predict'] <= 1024


def test_tool_overhead_and_context_budget_alignment(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
    from antigravity_hybrid_orchestrator.workforce.tools import WorkspaceToolSuite
    monkeypatch.delenv('OLLAMA_NUM_CTX', raising=False)
    monkeypatch.delenv('OLLAMA_NUM_PREDICT', raising=False)
    tools = WorkspaceToolSuite(workspace_root='.').get_tool_definitions()
    assert len(tools) >= 6
    tools_bytes = len(json.dumps(tools, ensure_ascii=False).encode())
    assert tools_bytes > 5000  # realistic tools overhead ~6 KB

    client = WorkforceClient()
    captured_payload = {}
    def post(url, payload):
        captured_payload.update(payload)
        return {'message': {'content': 'ok'}}
    monkeypatch.setattr(client, '_post_json', post)

    # Prompt matching the size that previously crashed (2178+ bytes)
    system_prompt = 'You are an autonomous R&D engineer operating in a sandbox. ' * 30
    task_prompt = 'Execute scripts/run_protocol.py --output orion2-run --profile full and verify all study artifacts. ' * 10
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': task_prompt},
    ]
    total_prompt_bytes = len(json.dumps(messages, ensure_ascii=False).encode())
    assert total_prompt_bytes > 2200

    # Under the bug, this threw ValueError("Required system/task prompt exceeds context budget")
    # With unit alignment, this must succeed cleanly
    resp = client.chat(messages=messages, tools=tools)
    assert resp['content'] == 'ok'
    assert captured_payload['options']['num_predict'] == 2048


def test_tool_call_output_cap_vs_plain_chat(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.ollama_client import WorkforceClient
    monkeypatch.delenv('OLLAMA_NUM_CTX', raising=False)
    monkeypatch.delenv('OLLAMA_NUM_PREDICT', raising=False)
    captured = {}
    client = WorkforceClient()
    def post(url, payload):
        captured.clear()
        captured.update(payload)
        return {'message': {'content': 'ok'}}
    monkeypatch.setattr(client, '_post_json', post)

    # Without tools: full prediction cap (8192)
    client.chat([{'role': 'user', 'content': 'hello'}])
    assert captured['options']['num_predict'] == 8192

    # With tools: limited to tool_call_reserve (2048)
    client.chat([{'role': 'user', 'content': 'hello'}], tools=[{'type': 'function', 'function': {'name': 'test'}}])
    assert captured['options']['num_predict'] == 2048


def test_bytes_per_token_and_tool_reserve_env_overrides(monkeypatch):
    from antigravity_hybrid_orchestrator.workforce.context import bytes_per_token, tool_call_reserve
    monkeypatch.setenv('OLLAMA_BYTES_PER_TOKEN', '3.5')
    assert bytes_per_token() == 3.5
    monkeypatch.setenv('OLLAMA_BYTES_PER_TOKEN', 'bad')
    assert bytes_per_token() == 3.0

    monkeypatch.setenv('OLLAMA_TOOL_RESERVE', '1024')
    assert tool_call_reserve() == 1024
    monkeypatch.setenv('OLLAMA_TOOL_RESERVE', 'bad')
    assert tool_call_reserve() == 2048

