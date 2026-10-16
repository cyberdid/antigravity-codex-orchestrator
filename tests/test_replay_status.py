import json
import pytest
from antigravity_hybrid_orchestrator.hud.replay import replay_run


@pytest.mark.parametrize('footer,expected', [({'success': False, 'status': 'paused'}, 'PAUSED'), ({'success': False}, 'FAILED'), ({'success': True, 'status': 'completed'}, 'COMPLETE'), ({'success': False, 'status': 'completed'}, 'FAILED')])
def test_replay_footer_status(tmp_path, footer, expected):
    path = tmp_path / 'run.jsonl'
    path.write_text(json.dumps({'kind': 'footer', **footer}) + '\n')
    assert replay_run(str(path), speed=0, no_hud=True).state.pipeline_status == expected
