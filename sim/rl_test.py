"""Evaluate a Pensieve checkpoint on a deterministic, held-out trace set."""

import argparse
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import a3c
import fixed_env as env
import load_trace


S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [1000, 2500, 5000, 8000, 16000, 40000]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 40
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1
RAND_RANGE = 1000


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
    parser.add_argument('--model', required=True)
    parser.add_argument('--test-traces', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--normalized', type=parse_bool, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--video-size-prefix',
        default=os.path.join(script_dir, 'video_size_'),
    )
    return parser.parse_args()


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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    action_rng = np.random.RandomState(args.seed)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        os.path.abspath(args.test_traces)
    )
    if not all_file_names:
        raise ValueError('no held-out trace files found')

    net_env = env.Environment(
        all_cooked_time=all_cooked_time,
        all_cooked_bw=all_cooked_bw,
        random_seed=args.seed,
        video_size_prefix=os.path.abspath(args.video_size_prefix),
    )

    with tf.Session() as sess:
        actor = a3c.ActorNetwork(
            sess, state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
            learning_rate=ACTOR_LR_RATE,
        )
        a3c.CriticNetwork(
            sess, state_dim=[S_INFO, S_LEN], learning_rate=CRITIC_LR_RATE,
        )
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(sess, os.path.abspath(args.model))

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        state = np.zeros((S_INFO, S_LEN))
        time_stamp = 0.0
        video_count = 0
        log_file = None

        try:
            while video_count < len(all_file_names):
                if log_file is None:
                    log_path = os.path.join(
                        args.output_dir,
                        'log_sim_rl_' + all_file_names[net_env.trace_idx],
                    )
                    log_file = open(log_path, 'w', newline='\n')

                delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
                    next_video_chunk_sizes, end_of_video, video_chunk_remain = \
                    net_env.get_video_chunk(bit_rate)
                time_stamp += delay + sleep_time

                reward = (
                    VIDEO_BIT_RATE[bit_rate] / M_IN_K
                    - REBUF_PENALTY * rebuf
                    - SMOOTH_PENALTY
                    * abs(VIDEO_BIT_RATE[bit_rate]
                          - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
                )
                if args.normalized:
                    reward /= 10.0

                log_file.write(
                    '{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        time_stamp / M_IN_K, VIDEO_BIT_RATE[bit_rate],
                        buffer_size, rebuf, video_chunk_size, delay, reward,
                    )
                )
                last_bit_rate = bit_rate
                state = update_state(
                    state, bit_rate, buffer_size, delay, video_chunk_size,
                    next_video_chunk_sizes, video_chunk_remain,
                    args.normalized,
                )
                action_prob = actor.predict(
                    np.reshape(state, (1, S_INFO, S_LEN))
                )
                if not np.all(np.isfinite(action_prob)):
                    raise FloatingPointError('actor emitted non-finite probability')
                action_cumsum = np.cumsum(action_prob)
                sample = action_rng.randint(1, RAND_RANGE) / float(RAND_RANGE)
                bit_rate = int((action_cumsum > sample).argmax())

                if end_of_video:
                    log_file.write('\n')
                    log_file.close()
                    log_file = None
                    video_count += 1
                    last_bit_rate = DEFAULT_QUALITY
                    bit_rate = DEFAULT_QUALITY
                    state = np.zeros((S_INFO, S_LEN))
                    time_stamp = 0.0
        finally:
            if log_file is not None:
                log_file.close()


if __name__ == '__main__':
    main()
