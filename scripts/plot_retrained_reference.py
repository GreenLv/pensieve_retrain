"""Render README figures for the corrected normalized Pensieve reference.

The figures are derived only from the formal normalized beta=1 run. The
training plot shows every update reward plus an explicitly labelled trailing
mean; the held-out plot shows the mean and the descriptive 5th--95th
percentile envelope across the frozen held-out trace sessions.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile


os.environ.setdefault(
    'MPLCONFIGDIR', os.path.join(tempfile.gettempdir(), 'pensieve-matplotlib')
)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


CENTRAL_PATTERN = re.compile(
    r'^Epoch:\s+(?P<epoch>\d+)\s+TD_loss:\s+\S+\s+'
    r'Avg_reward:\s+(?P<reward>\S+)'
)
WINDOW = 1000
FIGURE_WIDTH = 6.5
FIGURE_HEIGHT = 3.4


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_central_log(path):
    epochs = []
    rewards = []
    with open(path, 'r') as handle:
        for line in handle:
            match = CENTRAL_PATTERN.match(line)
            if match is None:
                continue
            epochs.append(int(match.group('epoch')))
            rewards.append(float(match.group('reward')))
    if not epochs:
        raise ValueError('no training records found in {}'.format(path))
    return np.asarray(epochs), np.asarray(rewards)


def read_heldout_summary(path):
    columns = {'epoch': [], 'p05': [], 'mean': [], 'p95': []}
    with open(path, 'r', newline='') as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            for name in columns:
                columns[name].append(float(row[name]))
    if not columns['epoch']:
        raise ValueError('no held-out records found in {}'.format(path))
    return {name: np.asarray(values) for name, values in columns.items()}


def configure_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def polish_axis(axis):
    axis.grid(True, axis='y', color='#d9d9d9', linewidth=0.7)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_linewidth(0.8)


def save_with_sidecar(figure, stem, source_paths, contract):
    outputs = []
    for suffix in ('pdf', 'png'):
        path = stem + '.' + suffix
        kwargs = {'bbox_inches': 'tight', 'pad_inches': 0.02, 'facecolor': 'white'}
        if suffix == 'png':
            kwargs['dpi'] = 300
        figure.savefig(path, **kwargs)
        outputs.append({
            'path': os.path.basename(path),
            'sha256': sha256_file(path),
            'size': os.path.getsize(path),
        })
    sidecar = {
        'schema': 'pensieve.figure.v1',
        'matplotlib_version': matplotlib.__version__,
        'figure_size_inches': [FIGURE_WIDTH, FIGURE_HEIGHT],
        'raster_dpi': 300,
        'contract': contract,
        'sources': [
            {
                'path': os.path.relpath(path, os.getcwd()).replace(os.sep, '/'),
                'sha256': sha256_file(path),
                'size': os.path.getsize(path),
            }
            for path in source_paths
        ],
        'outputs': outputs,
    }
    with open(stem + '.figure.json', 'w', newline='\n') as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write('\n')
    plt.close(figure)


def plot_training(epochs, rewards, central_log, output_dir):
    weights = np.ones(WINDOW, dtype=float) / WINDOW
    rolling = np.convolve(rewards, weights, mode='valid')
    rolling_epochs = epochs[WINDOW - 1:]

    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.plot(epochs, rewards, color='#9aa7b2', linewidth=0.35, alpha=0.35,
              label='Per-update reward')
    axis.plot(rolling_epochs, rolling, color='#c52f36', linewidth=1.5,
              label='Trailing mean (1,000 updates)')
    axis.set(
        xlabel='Training update',
        ylabel='Training reward (symlog scale)',
    )
    axis.set_xlim(epochs[0], epochs[-1])
    axis.set_yscale('symlog', linthresh=10)
    polish_axis(axis)
    axis.legend(loc='lower right')
    figure.tight_layout()
    save_with_sidecar(
        figure,
        os.path.join(output_dir, 'beta-1_normalized_training_reward'),
        [central_log],
        {
            'claim': 'The plot reports the corrected normalized beta=1 training trajectory.',
            'experimental_unit': 'one synchronous training update aggregated across 16 agents',
            'uncertainty': 'none; the red line is a trailing arithmetic mean, not an uncertainty interval',
        },
    )


def plot_heldout(summary, heldout_log, output_dir):
    figure, axis = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    axis.fill_between(
        summary['epoch'], summary['p05'], summary['p95'],
        color='#4c78a8', alpha=0.18,
        label='5th--95th percentile across held-out traces',
    )
    axis.plot(summary['epoch'], summary['mean'], color='#c52f36', linewidth=1.5,
              label='Mean across held-out traces')
    axis.set(
        xlabel='Training update',
        ylabel='Held-out total reward per session (symlog scale)',
    )
    axis.set_xlim(summary['epoch'][0], summary['epoch'][-1])
    axis.set_yscale('symlog', linthresh=10)
    polish_axis(axis)
    axis.legend(loc='lower right')
    figure.tight_layout()
    save_with_sidecar(
        figure,
        os.path.join(output_dir, 'beta-1_normalized_heldout_reward'),
        [heldout_log],
        {
            'claim': 'The plot reports held-out reward during the corrected normalized beta=1 training run.',
            'experimental_unit': 'one held-out trace session; 100 sessions at each evaluation',
            'uncertainty': 'descriptive 5th--95th percentile envelope across held-out trace sessions',
        },
    )


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run-dir',
        default=os.path.join(
            repository, 'artifacts', 'runs', 'formal', 'beta-1_normalized'
        ),
    )
    parser.add_argument(
        '--output-dir',
        default=os.path.join(repository, 'retrained_info', 'training_info'),
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output_dir = os.path.abspath(args.output_dir)
    central_log = os.path.join(run_dir, 'central.log')
    heldout_log = os.path.join(run_dir, 'heldout_summary.tsv')
    os.makedirs(output_dir, exist_ok=True)
    configure_style()
    epochs, rewards = read_central_log(central_log)
    heldout = read_heldout_summary(heldout_log)
    plot_training(epochs, rewards, central_log, output_dir)
    plot_heldout(heldout, heldout_log, output_dir)


if __name__ == '__main__':
    main()
