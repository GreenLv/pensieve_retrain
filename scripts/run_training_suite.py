"""Run the six frozen Pensieve training configurations sequentially."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time


MODEL_SPECS = [
    ('beta-1', 1, False),
    ('beta-2', 2, False),
    ('beta-3', 3, False),
    ('beta-4', 4, False),
    ('beta-5', 5, False),
    ('beta-1_normalized', 1, True),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path, root):
    return {
        'path': os.path.relpath(path, root).replace(os.sep, '/'),
        'size': os.path.getsize(path),
        'sha256': sha256_file(path),
    }


def write_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(temporary, path)


def run_one(command, log_path, cwd):
    with open(log_path, 'w', buffering=1, newline='\n') as log:
        process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1,
        )
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return process.wait()


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--trace-root', default=os.path.join(repository, 'work', 'traces')
    )
    parser.add_argument(
        '--runs-root',
        default=os.path.join(repository, 'artifacts', 'runs', 'formal'),
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-epochs', type=int, default=110000)
    parser.add_argument('--num-agents', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=100)
    parser.add_argument(
        '--models',
        nargs='+',
        choices=[spec[0] for spec in MODEL_SPECS],
        default=[spec[0] for spec in MODEL_SPECS],
    )
    args = parser.parse_args()

    runs_root = os.path.abspath(args.runs_root)
    os.makedirs(runs_root, exist_ok=True)
    code_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=repository,
        universal_newlines=True,
    ).strip()
    dirty = subprocess.check_output(
        ['git', 'status', '--porcelain'], cwd=repository,
        universal_newlines=True,
    ).strip()
    if dirty:
        raise ValueError(
            'formal training requires a clean committed tree:\n{}'.format(
                dirty
            )
        )
    trace_manifest_path = os.path.join(
        os.path.abspath(args.trace_root), 'trace_manifest.json'
    )
    if not os.path.isfile(trace_manifest_path):
        raise ValueError('missing trace manifest: {}'.format(
            trace_manifest_path
        ))
    manifest_path = os.path.join(runs_root, 'suite_manifest.json')
    manifest = {
        'protocol': {
            'seed': args.seed,
            'max_epochs': args.max_epochs,
            'num_agents': args.num_agents,
            'save_interval': args.save_interval,
            'from_scratch': True,
            'code_commit': code_commit,
            'trace_manifest_sha256': sha256_file(trace_manifest_path),
        },
        'models': [],
        'state': 'running',
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    write_json(manifest_path, manifest)

    specs = {spec[0]: spec for spec in MODEL_SPECS}
    for requested in args.models:
        label, beta, normalized = specs[requested]
        output_dir = os.path.join(runs_root, label)
        if os.path.exists(output_dir):
            raise ValueError(
                'refusing to overwrite existing run: {}'.format(output_dir)
            )
        record = {
            'label': label,
            'beta': beta,
            'normalized': normalized,
            'seed': args.seed,
            'state': 'running',
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }
        manifest['models'].append(record)
        write_json(manifest_path, manifest)

        command = [
            sys.executable, os.path.join(repository, 'sim', 'multi_agent.py'),
            '--beta', str(beta),
            '--normalized', str(normalized).lower(),
            '--seed', str(args.seed),
            '--train-traces',
            os.path.join(os.path.abspath(args.trace_root), 'train_traces'),
            '--test-traces',
            os.path.join(os.path.abspath(args.trace_root), 'test_traces'),
            '--output-dir', output_dir,
            '--max-epochs', str(args.max_epochs),
            '--num-agents', str(args.num_agents),
            '--save-interval', str(args.save_interval),
        ]
        exit_code = run_one(
            command, output_dir + '.runner.log', repository
        )
        if exit_code != 0:
            record['state'] = 'failed'
            record['exit_code'] = exit_code
            record['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
            manifest['state'] = 'failed'
            write_json(manifest_path, manifest)
            raise SystemExit(exit_code)

        status_path = os.path.join(output_dir, 'status.json')
        with open(status_path, 'r') as handle:
            status = json.load(handle)
        if status.get('completed_epochs') != args.max_epochs:
            raise ValueError('{} stopped at wrong epoch'.format(label))
        artifacts = []
        for relative in (
            'central.log', 'heldout_summary.tsv', 'run_config.json',
            'status.json', 'checkpoints/final.ckpt.data-00000-of-00001',
            'checkpoints/final.ckpt.index', 'checkpoints/final.ckpt.meta',
        ):
            path = os.path.join(output_dir, relative)
            artifacts.append(artifact_record(path, runs_root))
        runner_log = output_dir + '.runner.log'
        artifacts.append(artifact_record(runner_log, runs_root))
        record.update({
            'state': 'complete',
            'ended_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'completed_epochs': status['completed_epochs'],
            'artifacts': artifacts,
        })
        write_json(manifest_path, manifest)

    manifest['state'] = 'complete'
    manifest['ended_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write_json(manifest_path, manifest)
    print('Suite complete: {}'.format(manifest_path))


if __name__ == '__main__':
    main()
