"""Launch the formal suite detached while preserving stdout/stderr logs."""

import os
import subprocess
import sys


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(repository, 'artifacts', 'runs')
    os.makedirs(log_dir, exist_ok=True)
    stdout_path = os.path.join(log_dir, 'formal.stdout.log')
    stderr_path = os.path.join(log_dir, 'formal.stderr.log')
    creation_flags = 0
    if os.name == 'nt':
        creation_flags = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NO_WINDOW
        )
    with open(stdout_path, 'w', buffering=1) as stdout, open(
            stderr_path, 'w', buffering=1) as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                os.path.join(repository, 'scripts', 'run_training_suite.py'),
            ],
            cwd=repository,
            stdout=stdout,
            stderr=stderr,
            creationflags=creation_flags,
            close_fds=True,
        )
    print(process.pid)


if __name__ == '__main__':
    main()
