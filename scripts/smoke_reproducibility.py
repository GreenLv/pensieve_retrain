"""Run two short spawn-based trainings and compare deterministic evidence."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def copy_subset(source, target, count):
    os.makedirs(target)
    names = sorted(
        name for name in os.listdir(source)
        if os.path.isfile(os.path.join(source, name))
    )[:count]
    for name in names:
        shutil.copy2(os.path.join(source, name), os.path.join(target, name))


def compared_hashes(run_dir):
    paths = [
        os.path.join(run_dir, 'central.log'),
        os.path.join(run_dir, 'heldout_summary.tsv'),
    ]
    paths.extend(
        os.path.join(run_dir, name)
        for name in sorted(os.listdir(run_dir))
        if name.startswith('agent_') and name.endswith('_trajectory.tsv')
    )
    return {
        os.path.relpath(path, run_dir).replace(os.sep, '/'): sha256_file(path)
        for path in paths
    }


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace-root', default=os.path.join(repository, 'work', 'traces')
    )
    parser.add_argument(
        '--output-root',
        default=os.path.join(repository, 'artifacts', 'runs', 'smoke'),
    )
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--agents', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_root = os.path.abspath(args.output_root)
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    smoke_traces = os.path.join(output_root, 'traces')
    copy_subset(
        os.path.join(os.path.abspath(args.trace_root), 'train_traces'),
        os.path.join(smoke_traces, 'train_traces'), 8,
    )
    copy_subset(
        os.path.join(os.path.abspath(args.trace_root), 'test_traces'),
        os.path.join(smoke_traces, 'test_traces'), 3,
    )

    for repetition in ('run-a', 'run-b'):
        command = [
            sys.executable, os.path.join(repository, 'sim', 'multi_agent.py'),
            '--beta', '1', '--normalized', 'false',
            '--seed', str(args.seed),
            '--train-traces', os.path.join(smoke_traces, 'train_traces'),
            '--test-traces', os.path.join(smoke_traces, 'test_traces'),
            '--output-dir', os.path.join(output_root, repetition),
            '--max-epochs', str(args.epochs),
            '--num-agents', str(args.agents),
            '--save-interval', str(args.epochs),
            '--train-seq-len', '8',
            '--audit-trajectories',
        ]
        subprocess.run(command, cwd=repository, check=True)

    first = compared_hashes(os.path.join(output_root, 'run-a'))
    second = compared_hashes(os.path.join(output_root, 'run-b'))
    if first != second:
        differing = sorted(
            key for key in set(first) | set(second)
            if first.get(key) != second.get(key)
        )
        raise AssertionError(
            'short trainings are not reproducible: {}'.format(differing)
        )
    print('Reproducibility smoke passed:')
    for name in sorted(first):
        print('{}  {}'.format(first[name], name))


if __name__ == '__main__':
    main()
