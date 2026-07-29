import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('relative', ['sim/a3c.py', 'test/a3c.py'])
def test_actor_and_critic_sixth_branches_read_row_five(relative):
    source = (REPOSITORY / relative).read_text(encoding='utf-8')
    assert source.count(
        'fully_connected(inputs[:, 5:6, -1], 128'
    ) == 2
    assert 'fully_connected(inputs[:, 4:5, -1], 128' not in source


@pytest.mark.parametrize('relative', ['sim/a3c.py', 'test/a3c.py'])
def test_row_four_tail_is_excluded_from_all_network_branches(relative):
    source = (REPOSITORY / relative).read_text(encoding='utf-8')
    assert source.count(
        'conv_1d(inputs[:, 4:5, :A_DIM], 128, 4'
    ) == 2
    assert 'inputs[:, 4:5, -1]' not in source


@pytest.mark.parametrize('relative', ['sim/a3c.py', 'test/a3c.py'])
def test_real_actor_and_critic_ignore_row_four_tail_and_use_row_five(
        relative, monkeypatch):
    tf = pytest.importorskip('tensorflow')
    pytest.importorskip('tflearn')
    monkeypatch.syspath_prepend(str(REPOSITORY / Path(relative).parent))
    tf.reset_default_graph()
    tf.set_random_seed(42)
    module = load_module(
        'graph_' + relative.replace('/', '_'), REPOSITORY / relative
    )
    with tf.Session() as session:
        actor = module.ActorNetwork(
            session, state_dim=[6, 8], action_dim=6, learning_rate=0.0001
        )
        critic = module.CriticNetwork(
            session, state_dim=[6, 8], learning_rate=0.001
        )
        session.run(tf.global_variables_initializer())
        baseline = np.zeros((1, 6, 8))
        row_four_tail = baseline.copy()
        row_four_tail[0, 4, -1] = 1000.0
        row_five = baseline.copy()
        row_five[0, 5, -1] = 1000.0

        actor_baseline = actor.predict(baseline)
        critic_baseline = critic.predict(baseline)
        np.testing.assert_allclose(
            actor.predict(row_four_tail), actor_baseline, rtol=0, atol=0
        )
        np.testing.assert_allclose(
            critic.predict(row_four_tail), critic_baseline, rtol=0, atol=0
        )
        assert not np.allclose(actor.predict(row_five), actor_baseline)
        assert not np.allclose(critic.predict(row_five), critic_baseline)


def test_trace_loading_is_sorted_and_ignores_directories(tmp_path):
    trace_dir = tmp_path / 'traces'
    trace_dir.mkdir()
    (trace_dir / 'z-trace').write_text('0 1\n1 2\n', encoding='utf-8')
    (trace_dir / 'a-trace').write_text('0 3\n1 4\n', encoding='utf-8')
    (trace_dir / 'ignored').mkdir()
    loader = load_module(
        'retrain_load_trace', REPOSITORY / 'sim' / 'load_trace.py'
    )
    _, bandwidth, names = loader.load_trace(str(trace_dir))
    assert names == ['a-trace', 'z-trace']
    assert bandwidth == [[3.0, 4.0], [1.0, 2.0]]


def test_environment_uses_local_rng():
    environment = load_module(
        'retrain_env', REPOSITORY / 'sim' / 'env.py'
    )
    times = [[0.0, 1.0, 2.0, 3.0]]
    bandwidth = [[100.0, 100.0, 100.0, 100.0]]
    np.random.seed(12345)
    expected = np.random.RandomState(12345).rand()
    instance = environment.Environment(
        times, bandwidth, random_seed=42,
        video_size_prefix=str(REPOSITORY / 'sim' / 'video_size_'),
    )
    instance.get_video_chunk(1)
    assert np.random.rand() == expected


def test_entropy_reaches_point_one_at_update_100000(monkeypatch):
    pytest.importorskip('tensorflow')
    pytest.importorskip('tflearn')
    monkeypatch.syspath_prepend(str(REPOSITORY / 'sim'))
    module = load_module(
        'retrain_multi_agent', REPOSITORY / 'sim' / 'multi_agent.py'
    )
    for beta in range(1, 6):
        assert module.calculate_entropy_weight(99999, beta) == pytest.approx(
            0.1
        )
        assert module.calculate_entropy_weight(100000, beta) == pytest.approx(
            0.1
        )


def test_normalization_only_scales_throughput_and_next_sizes(monkeypatch):
    pytest.importorskip('tensorflow')
    pytest.importorskip('tflearn')
    monkeypatch.syspath_prepend(str(REPOSITORY / 'sim'))
    module = load_module(
        'retrain_multi_agent_state', REPOSITORY / 'sim' / 'multi_agent.py'
    )
    state = np.zeros((6, 8))
    arguments = (
        state, 2, 8.0, 1000.0, 5000000.0,
        [1000000.0] * 6, 24,
    )
    raw = module.update_state(*arguments, False)
    normalized = module.update_state(*arguments, True)
    assert normalized[2, -1] == pytest.approx(raw[2, -1] / 10.0)
    np.testing.assert_allclose(
        normalized[4, :6], raw[4, :6] / 10.0
    )
    np.testing.assert_allclose(normalized[[0, 1, 3, 5]], raw[[0, 1, 3, 5]])


def test_training_uses_exact_requested_update_count():
    source = (REPOSITORY / 'sim' / 'multi_agent.py').read_text(
        encoding='utf-8'
    )
    assert "while epoch < config['max_epochs']:" in source
    assert 'while epoch <= config' not in source


def test_train_and_heldout_receive_same_normalization_value():
    source = (REPOSITORY / 'sim' / 'multi_agent.py').read_text(
        encoding='utf-8'
    )
    assert "'--normalized', str(config['normalized']).lower()" in source
    assert "config['normalized']," in source
