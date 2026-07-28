"""Deterministic, Windows-compatible synchronous A3C training for Pensieve."""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import a3c
import env
import load_trace


S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 40
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RAND_RANGE = 1000
MAX_EPOCHS = 110000
RANDOM_SEED = 42


def parse_bool(value):
    normalized = str(value).strip().lower()
    if normalized in ('1', 'true', 'yes', 'on'):
        return True
    if normalized in ('0', 'false', 'no', 'off'):
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=int, choices=range(1, 6), required=True)
    parser.add_argument('--normalized', type=parse_bool, required=True)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--train-traces', required=True)
    parser.add_argument('--test-traces', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--num-agents', type=int, default=NUM_AGENTS)
    parser.add_argument('--save-interval', type=int,
                        default=MODEL_SAVE_INTERVAL)
    parser.add_argument('--train-seq-len', type=int, default=TRAIN_SEQ_LEN)
    parser.add_argument('--audit-trajectories', action='store_true')
    parser.add_argument(
        '--video-size-prefix',
        default=os.path.join(script_dir, 'video_size_'),
    )
    return parser.parse_args()


def make_config(args):
    config = vars(args).copy()
    for key in ('train_traces', 'test_traces', 'output_dir',
                'video_size_prefix'):
        config[key] = os.path.abspath(config[key])
    config['script_dir'] = os.path.dirname(os.path.abspath(__file__))
    config['python_executable'] = sys.executable
    return config


def calculate_entropy_weight(epoch, beta):
    """Return the stair-step entropy weight for a zero-based update index."""
    entropy_weight = (
        beta - int((epoch + 1) / 10000) * (beta - 0.1) / 10.0
    )
    return max(0.1, entropy_weight)


def model_label(config):
    suffix = '_normalized' if config['normalized'] else ''
    return 'beta-{}{}'.format(config['beta'], suffix)


def write_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def test_checkpoint(epoch, checkpoint, test_log_file, config):
    test_output = os.path.join(config['output_dir'], 'heldout', 'latest')
    shutil.rmtree(test_output, ignore_errors=True)
    os.makedirs(test_output)
    command = [
        config['python_executable'],
        os.path.join(config['script_dir'], 'rl_test.py'),
        '--model', checkpoint,
        '--test-traces', config['test_traces'],
        '--output-dir', test_output,
        '--normalized', str(config['normalized']).lower(),
        '--seed', str(config['seed']),
        '--video-size-prefix', config['video_size_prefix'],
    ]
    subprocess.run(command, cwd=config['script_dir'], check=True)

    rewards = []
    for name in sorted(os.listdir(test_output)):
        path = os.path.join(test_output, name)
        if not os.path.isfile(path):
            continue
        reward = []
        with open(path, 'r') as handle:
            for line in handle:
                fields = line.split()
                if fields:
                    reward.append(float(fields[-1]))
        rewards.append(np.sum(reward[1:]))
    if not rewards:
        raise RuntimeError('held-out evaluation produced no logs')

    rewards = np.asarray(rewards)
    if not np.all(np.isfinite(rewards)):
        raise FloatingPointError('held-out rewards contain NaN or Inf')
    columns = [
        epoch, np.min(rewards), np.percentile(rewards, 5),
        np.mean(rewards), np.percentile(rewards, 50),
        np.percentile(rewards, 95), np.max(rewards),
    ]
    test_log_file.write('\t'.join(str(value) for value in columns) + '\n')
    test_log_file.flush()


def assert_finite_network(actor, critic):
    params = actor.get_network_params() + critic.get_network_params()
    if not all(np.all(np.isfinite(value)) for value in params):
        raise FloatingPointError('network parameters contain NaN or Inf')


def central_agent(net_params_queues, exp_queues, config):
    np.random.seed(config['seed'])
    tf.set_random_seed(config['seed'])

    central_log_path = os.path.join(config['output_dir'], 'central.log')
    logger = logging.getLogger('pensieve.central.{}'.format(os.getpid()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(central_log_path, mode='w')
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)

    test_summary_path = os.path.join(
        config['output_dir'], 'heldout_summary.tsv'
    )
    tensorboard_dir = os.path.join(config['output_dir'], 'tensorboard')
    checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with tf.Session() as sess, open(
            test_summary_path, 'w', newline='\n') as test_log_file:
        test_log_file.write(
            'epoch\tmin\tp05\tmean\tmedian\tp95\tmax\n'
        )
        actor = a3c.ActorNetwork(
            sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE,
        )
        critic = a3c.CriticNetwork(
            sess, state_dim=[S_INFO, S_LEN],
            learning_rate=CRITIC_LR_RATE,
        )
        summary_ops, summary_vars = a3c.build_summaries()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
        saver = tf.train.Saver(max_to_keep=2)
        epoch = 0

        try:
            while epoch < config['max_epochs']:
                actor_net_params = actor.get_network_params()
                critic_net_params = critic.get_network_params()
                for queue in net_params_queues:
                    queue.put([actor_net_params, critic_net_params])

                total_batch_len = 0.0
                total_reward = 0.0
                total_td_loss = 0.0
                total_entropy = 0.0
                actor_gradient_batch = []
                critic_gradient_batch = []
                entropy_weight = calculate_entropy_weight(
                    epoch, config['beta']
                )

                for queue in exp_queues:
                    s_batch, a_batch, r_batch, terminal, info = queue.get()
                    actor_gradient, critic_gradient, td_batch = \
                        a3c.compute_gradients(
                            s_batch=np.stack(s_batch, axis=0),
                            a_batch=np.vstack(a_batch),
                            r_batch=np.vstack(r_batch),
                            terminal=terminal,
                            actor=actor,
                            critic=critic,
                            entropy_weight=entropy_weight,
                        )
                    actor_gradient_batch.append(actor_gradient)
                    critic_gradient_batch.append(critic_gradient)
                    total_reward += np.sum(r_batch)
                    total_td_loss += np.sum(td_batch)
                    total_batch_len += len(r_batch)
                    total_entropy += np.sum(info['entropy'])

                for actor_gradient, critic_gradient in zip(
                        actor_gradient_batch, critic_gradient_batch):
                    actor.apply_gradients(actor_gradient)
                    critic.apply_gradients(critic_gradient)

                epoch += 1
                avg_reward = total_reward / config['num_agents']
                avg_td_loss = total_td_loss / total_batch_len
                avg_entropy = total_entropy / total_batch_len
                metrics = np.asarray(
                    [avg_reward, avg_td_loss, avg_entropy, entropy_weight]
                )
                if not np.all(np.isfinite(metrics)):
                    raise FloatingPointError(
                        'non-finite metric at epoch {}'.format(epoch)
                    )
                logger.info(
                    'Epoch: %d TD_loss: %.17g Avg_reward: %.17g '
                    'Avg_entropy: %.17g Entropy_weight: %.17g',
                    epoch, avg_td_loss, avg_reward, avg_entropy,
                    entropy_weight,
                )
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: avg_td_loss,
                    summary_vars[1]: avg_reward,
                    summary_vars[2]: avg_entropy,
                })
                writer.add_summary(summary_str, epoch)
                writer.flush()

                if epoch % config['save_interval'] == 0:
                    assert_finite_network(actor, critic)
                    latest = saver.save(
                        sess, os.path.join(checkpoint_dir, 'latest.ckpt')
                    )
                    test_checkpoint(epoch, latest, test_log_file, config)
                    write_json(
                        os.path.join(config['output_dir'], 'status.json'),
                        {
                            'completed_epochs': epoch,
                            'entropy_weight': entropy_weight,
                            'label': model_label(config),
                            'state': 'running',
                            'updated_at': time.strftime(
                                '%Y-%m-%dT%H:%M:%S%z'
                            ),
                        },
                    )
                    print(
                        '[{} epoch {}] entropy={}'.format(
                            model_label(config), epoch, entropy_weight
                        ),
                        flush=True,
                    )

            assert_finite_network(actor, critic)
            final_checkpoint = saver.save(
                sess, os.path.join(checkpoint_dir, 'final.ckpt')
            )
            if epoch % config['save_interval'] != 0:
                test_checkpoint(epoch, final_checkpoint, test_log_file, config)
            write_json(
                os.path.join(config['output_dir'], 'status.json'),
                {
                    'completed_epochs': epoch,
                    'entropy_weight': calculate_entropy_weight(
                        max(0, epoch - 1), config['beta']
                    ),
                    'final_checkpoint': final_checkpoint,
                    'label': model_label(config),
                    'state': 'complete',
                    'updated_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                },
            )
        finally:
            writer.close()
            handler.close()


def update_state(state, bit_rate, buffer_size, delay, video_chunk_size,
                 next_video_chunk_sizes, video_chunk_remain, normalized):
    state = np.roll(np.array(state, copy=True), -1, axis=1)
    state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(max(VIDEO_BIT_RATE))
    state[1, -1] = buffer_size / BUFFER_NORM_FACTOR
    scale = 10.0 if normalized else 1.0
    state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K / scale
    state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR
    state[4, :A_DIM] = (
        np.asarray(next_video_chunk_sizes) / M_IN_K / M_IN_K / scale
    )
    state[5, -1] = (
        min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP)
        / CHUNK_TIL_VIDEO_END_CAP
    )
    return state


def agent(agent_id, net_params_queue, exp_queue, config):
    agent_seed = config['seed'] + agent_id
    np.random.seed(agent_seed)
    tf.set_random_seed(agent_seed)
    action_rng = np.random.RandomState(agent_seed + 10000)
    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(
        config['train_traces']
    )
    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        random_seed=agent_seed,
        video_size_prefix=config['video_size_prefix'],
    )

    audit_file = None
    if config['audit_trajectories']:
        audit_file = open(
            os.path.join(
                config['output_dir'],
                'agent_{:02d}_trajectory.tsv'.format(agent_id),
            ),
            'w',
            buffering=1,
            newline='\n',
        )
        audit_file.write(
            'trace_index\tchunk_index\tselected_action\tstate_sha256\n'
        )

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(
            sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE,
        )
        critic = a3c.CriticNetwork(
            sess, state_dim=[S_INFO, S_LEN],
            learning_rate=CRITIC_LR_RATE,
        )
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1
        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        while True:
            trace_index = net_env.trace_idx
            chunk_index = net_env.video_chunk_counter
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
                next_video_chunk_sizes, end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)
            reward = (
                VIDEO_BIT_RATE[bit_rate] / M_IN_K
                - REBUF_PENALTY * rebuf
                - SMOOTH_PENALTY
                * abs(VIDEO_BIT_RATE[bit_rate]
                      - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            )
            if config['normalized']:
                reward /= 10.0
            r_batch.append(reward)
            last_bit_rate = bit_rate

            state = update_state(
                s_batch[-1] if s_batch else np.zeros((S_INFO, S_LEN)),
                bit_rate, buffer_size, delay, video_chunk_size,
                next_video_chunk_sizes, video_chunk_remain,
                config['normalized'],
            )
            action_prob = actor.predict(
                np.reshape(state, (1, S_INFO, S_LEN))
            )
            if not np.all(np.isfinite(action_prob)):
                raise FloatingPointError('actor emitted non-finite probability')
            action_cumsum = np.cumsum(action_prob)
            sample = action_rng.randint(1, RAND_RANGE) / float(RAND_RANGE)
            bit_rate = int((action_cumsum > sample).argmax())
            entropy_record.append(a3c.compute_entropy(action_prob[0]))
            if audit_file is not None:
                audit_file.write(
                    '{}\t{}\t{}\t{}\n'.format(
                        trace_index, chunk_index, bit_rate,
                        hashlib.sha256(state.tobytes()).hexdigest(),
                    )
                )
                audit_file.flush()

            if len(r_batch) >= config['train_seq_len'] or end_of_video:
                exp_queue.put([
                    s_batch[1:], a_batch[1:], r_batch[1:], end_of_video,
                    {'entropy': entropy_record},
                ])
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)
                s_batch[:] = []
                a_batch[:] = []
                r_batch[:] = []
                entropy_record[:] = []

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY
                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
            else:
                s_batch.append(state)
                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)


def validate_config(config):
    if config['max_epochs'] <= 0:
        raise ValueError('max-epochs must be positive')
    if config['num_agents'] <= 0:
        raise ValueError('num-agents must be positive')
    if config['save_interval'] <= 0:
        raise ValueError('save-interval must be positive')
    for key in ('train_traces', 'test_traces'):
        if not os.path.isdir(config[key]):
            raise ValueError('{} is not a directory: {}'.format(
                key, config[key]
            ))
    for bitrate in range(A_DIM):
        path = config['video_size_prefix'] + str(bitrate)
        if not os.path.isfile(path):
            raise ValueError('missing video-size file: {}'.format(path))


def main():
    config = make_config(parse_args())
    validate_config(config)
    os.makedirs(config['output_dir'], exist_ok=False)
    config_to_record = config.copy()
    config_to_record['started_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    config_to_record['tensorflow_version'] = tf.__version__
    config_to_record['numpy_version'] = np.__version__
    write_json(
        os.path.join(config['output_dir'], 'run_config.json'),
        config_to_record,
    )

    context = mp.get_context('spawn')
    net_params_queues = [
        context.Queue(1) for _ in range(config['num_agents'])
    ]
    exp_queues = [
        context.Queue(1) for _ in range(config['num_agents'])
    ]
    coordinator = context.Process(
        target=central_agent,
        args=(net_params_queues, exp_queues, config),
        name='pensieve-central',
    )
    agents = [
        context.Process(
            target=agent,
            args=(i, net_params_queues[i], exp_queues[i], config),
            name='pensieve-agent-{}'.format(i),
        )
        for i in range(config['num_agents'])
    ]

    started = time.time()
    coordinator.start()
    for process in agents:
        process.start()
    coordinator.join()
    for process in agents:
        process.terminate()
        process.join()
    for queue in net_params_queues + exp_queues:
        queue.close()

    if coordinator.exitcode != 0:
        raise RuntimeError(
            'central agent exited with code {}'.format(coordinator.exitcode)
        )
    elapsed_minutes = round((time.time() - started) / 60.0, 2)
    print(
        'Training {} completed in {} minutes'.format(
            model_label(config), elapsed_minutes
        )
    )


if __name__ == '__main__':
    mp.freeze_support()
    main()
