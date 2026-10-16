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
