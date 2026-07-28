"""Capture the exact interpreter and installed package set as JSON."""

import argparse
import json
import os
import platform
import sys

import pkg_resources


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output',
        default=os.path.join(
            repository, 'artifacts', 'reproducibility',
            'environment.json',
        ),
    )
    args = parser.parse_args()
    packages = {
        distribution.project_name: distribution.version
        for distribution in pkg_resources.working_set
    }
    record = {
        'executable': sys.executable,
        'python_version': sys.version,
        'platform': platform.platform(),
        'packages': dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w', newline='\n') as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write('\n')
    print(output)


if __name__ == '__main__':
    main()
