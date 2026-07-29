"""Safely extract and freeze the train/test trace corpus."""

import argparse
import hashlib
import json
import os
import shutil
import zipfile


EXPECTED_TRAIN_COUNT = 410
EXPECTED_TEST_COUNT = 100


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive, output_dir):
    root = os.path.abspath(output_dir)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            target = os.path.abspath(os.path.join(root, info.filename))
            if os.path.commonpath([root, target]) != root:
                raise ValueError('unsafe archive member: {}'.format(
                    info.filename
                ))
        bundle.extractall(root)


def inventory(folder, root):
    records = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        records.append({
            'path': os.path.relpath(path, root).replace(os.sep, '/'),
            'size': os.path.getsize(path),
            'sha256': sha256_file(path),
        })
    return records


def write_json(path, value):
    with open(path, 'w', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    repository = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--archive',
        default=os.path.join(
            repository, 'retrained_info', 'data_preprocess',
            'network_traces.zip',
        ),
    )
    parser.add_argument(
        '--output-dir', default=os.path.join(repository, 'work', 'traces')
    )
    parser.add_argument(
        '--manifest',
        default=os.path.join(
            repository, 'artifacts', 'reproducibility',
            'trace_manifest.json',
        ),
    )
    args = parser.parse_args()

    archive = os.path.abspath(args.archive)
    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    safe_extract(archive, output_dir)

    train_dir = os.path.join(output_dir, 'train_traces')
    test_dir = os.path.join(output_dir, 'test_traces')
    train = inventory(train_dir, output_dir)
    test = inventory(test_dir, output_dir)
    if len(train) != EXPECTED_TRAIN_COUNT:
        raise ValueError('expected 410 training traces, got {}'.format(
            len(train)
        ))
    if len(test) != EXPECTED_TEST_COUNT:
        raise ValueError('expected 100 test traces, got {}'.format(len(test)))

    train_names = {os.path.basename(item['path']) for item in train}
    test_names = {os.path.basename(item['path']) for item in test}
    train_hashes = {item['sha256'] for item in train}
    test_hashes = {item['sha256'] for item in test}
    name_overlap = sorted(train_names & test_names)
    content_overlap = sorted(train_hashes & test_hashes)
    if name_overlap or content_overlap:
        raise ValueError(
            'train/test overlap: names={}, contents={}'.format(
                len(name_overlap), len(content_overlap)
            )
        )

    manifest = {
        'archive': os.path.relpath(archive, repository).replace(os.sep, '/'),
        'archive_sha256': sha256_file(archive),
        'train_count': len(train),
        'test_count': len(test),
        'name_overlap': name_overlap,
        'content_sha256_overlap': content_overlap,
        'train': train,
        'test': test,
    }
    local_manifest_path = os.path.join(output_dir, 'trace_manifest.json')
    write_json(local_manifest_path, manifest)
    manifest_path = os.path.abspath(args.manifest)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    write_json(manifest_path, manifest)
    print(manifest_path)
    print('train=410 test=100 overlap=0')


if __name__ == '__main__':
    main()
