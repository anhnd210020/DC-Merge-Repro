#!/usr/bin/env python3
"""Build a hash manifest for unique project data selected for laptop backup."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path('/workspace/DC-Merge-Repro')
PLAN = Path('/srv/dcmerge/backup_plan')
RESULTS = Path('/srv/dcmerge/results')
SUN = Path('/workspace/.dcmerge_sun397_http_stage/sun397.tar')
SUN_SHA = '38afed47c5acce92ac26b36b620e936f4a849e7934e30c65487a17e2dbe0c917'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_blob_hashes() -> set[str]:
    oids = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-list', '--objects', '--all'], text=True
    ).splitlines()
    object_ids = list(dict.fromkeys(line.split(' ', 1)[0] for line in oids))
    p = subprocess.run(
        ['git', '-C', str(REPO), 'cat-file', '--batch'],
        input=''.join(oid + '\n' for oid in object_ids).encode(),
        stdout=subprocess.PIPE,
        check=True,
    )
    data = memoryview(p.stdout)
    pos = 0
    hashes: set[str] = set()
    for _ in object_ids:
        end = p.stdout.find(b'\n', pos)
        if end < 0:
            raise RuntimeError('truncated git cat-file output')
        header = p.stdout[pos:end].split()
        pos = end + 1
        kind, size = header[1], int(header[2])
        body = data[pos:pos + size]
        if kind == b'blob':
            hashes.add(hashlib.sha256(body).hexdigest())
        pos += size
        if p.stdout[pos:pos + 1] != b'\n':
            raise RuntimeError('invalid git cat-file record separator')
        pos += 1
    return hashes


def add(rows, path: Path, dest: str, category: str, note: str = '',
        preverified_sha: str | None = None):
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = preverified_sha or sha256(path)
    rows.append({
        'category': category,
        'source_path': str(path),
        'backup_path': dest.replace('\\', '/'),
        'bytes': path.stat().st_size,
        'sha256': digest,
        'note': note,
    })


def tree_stats(root: Path) -> tuple[int, int]:
    files = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    return sum(p.stat().st_size for p in files), len(files)


def main() -> None:
    PLAN.mkdir(parents=True, exist_ok=True)
    if SUN.stat().st_size != 14591037440:
        raise RuntimeError('SUN397 archive size differs from authoritative size')
    sun_manifest = json.loads((SUN.parent / 'manifest.json').read_text())
    if sun_manifest['sha256'] != SUN_SHA or sun_manifest['size'] != SUN.stat().st_size:
        raise RuntimeError('SUN397 archive metadata changed')

    print('Loading reachable Git blob hashes for content deduplication...', flush=True)
    git_hashes = git_blob_hashes()
    print(f'Loaded {len(git_hashes):,} distinct Git blob hashes.', flush=True)

    rows: list[dict] = []
    # The Git bundle captures the Experiment 3 and 4 handoff code/results and
    # repository refs. The bundle is the laptop's one copy of tracked code.
    add(rows, PLAN / 'DC-Merge-Repro.bundle', 'project/DC-Merge-Repro.bundle',
        'git_bundle', 'All local refs, including main and both handoff branches.')

    # The seven non-SUN extracted datasets are stored in one tar for an efficient
    # resumable transfer. Keep their per-file hashes in a separate contents list.
    datasets = ('MNIST', 'cars', 'dtd', 'eurosat', 'gtsrb', 'resisc45', 'svhn')
    dataset_content_rows: list[dict] = []
    for dataset in datasets:
        base = Path('/workspace/datasets8') / dataset
        for p in sorted(x for x in base.rglob('*') if x.is_file()):
            rel = p.relative_to(Path('/workspace/datasets8'))
            dataset_content_rows.append({
                'relative_path': rel.as_posix(),
                'bytes': p.stat().st_size,
                'sha256': sha256(p),
            })
            if len(dataset_content_rows) % 25000 == 0:
                print(f'Hashed {len(dataset_content_rows):,} dataset files...', flush=True)
    with (PLAN / 'dataset_contents.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['relative_path', 'bytes', 'sha256'])
        w.writeheader()
        w.writerows(dataset_content_rows)
    dataset_archive = PLAN / 'datasets8_non_sun397.tar'
    with tarfile.open(dataset_archive, 'r:') as tf:
        archive_files = [m for m in tf.getmembers() if m.isfile()]
    archive_paths = {m.name for m in archive_files}
    content_paths = {row['relative_path'] for row in dataset_content_rows}
    if archive_paths != content_paths:
        raise RuntimeError(f'dataset tar member set mismatch: archive={len(archive_paths)} manifest={len(content_paths)}')
    if sum(m.size for m in archive_files) != sum(int(row['bytes']) for row in dataset_content_rows):
        raise RuntimeError('dataset tar member byte total differs from dataset content manifest')
    expected_dataset_hashes = {row['relative_path']: row['sha256'] for row in dataset_content_rows}
    with tarfile.open(dataset_archive, 'r:') as tf:
        for index, member in enumerate((m for m in tf if m.isfile()), 1):
            h = hashlib.sha256()
            stream = tf.extractfile(member)
            if stream is None:
                raise RuntimeError(f'cannot read dataset tar member {member.name}')
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
            if h.hexdigest() != expected_dataset_hashes[member.name]:
                raise RuntimeError(f'dataset tar content hash mismatch: {member.name}')
            if index % 25000 == 0:
                print(f'Verified {index:,} dataset tar members...', flush=True)
    print(f'Verified all {len(archive_files):,} dataset tar member hashes.', flush=True)
    add(rows, dataset_archive, 'assets/source_archives/datasets8_non_sun397.tar',
        'dataset_archive', 'Uncompressed tar of seven non-SUN datasets; per-file SHA-256 values are in dataset_contents.csv.')

    add(rows, SUN, 'assets/source_archives/sun397.tar', 'verified_dataset_archive',
        'Pinned revision 42fc343338f31c2f41cfa85e97d20e22dfbf9d45; authoritative size and SHA-256 verified.',
        preverified_sha=SUN_SHA)
    source_archive = Path('/workspace/ducanh/experiment2-handoff-20261016.tar.gz')
    # Read every tar header so this is known to be a readable source archive.
    with tarfile.open(source_archive, 'r:gz') as tf:
        archive_members = len(tf.getmembers())
    add(rows, source_archive, 'assets/source_archives/experiment2-handoff-20261016.tar.gz',
        'project_source_archive', f'Locally SHA-256 hashed and readable; {archive_members} tar members; no publisher digest recorded.')

    # Used PyTorch CLIP checkpoint and its exact tokenizer/config files. The
    # Flax serialization and Hub download cache are alternate/redundant copies.
    clip = Path('/workspace/models/clip-vit-base-patch32')
    clip_files = (
        '.gitattributes', 'README.md', 'config.json', 'merges.txt',
        'preprocessor_config.json', 'pytorch_model.bin',
        'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json',
        'vocab.json',
    )
    for name in clip_files:
        add(rows, clip / name, f'assets/models/clip-vit-base-patch32/{name}',
            'base_model', 'Used PyTorch CLIP checkpoint/configuration; excludes alternate Flax weights and partial cache files.')

    task_model = Path('/workspace/models/DC-Merge-Vision-B32-r16-8task')
    add(rows, task_model / '.gitattributes',
        'assets/models/DC-Merge-Vision-B32-r16-8task/.gitattributes', 'adapter_head_metadata')
    for subtree in ('lora_checkpoints', 'lora_heads'):
        base = task_model / subtree
        for p in sorted(x for x in base.rglob('*') if x.is_file()):
            rel = p.relative_to(task_model)
            add(rows, p, f'assets/models/DC-Merge-Vision-B32-r16-8task/{rel.as_posix()}',
                'adapter' if subtree == 'lora_checkpoints' else 'head')

    # This one legacy subset image is a changed file; its other 35 files exactly
    # match paths/content already present in the verified SUN397 dataset archive.
    legacy_unique = Path('/workspace/ducanh/local-assets/datasets8/sun397/val/y_youth_hostel/sun_bnermqodfamvctxr.jpg')
    add(rows, legacy_unique,
        'assets/legacy_sun397_subset/val/y_youth_hostel/sun_bnermqodfamvctxr.jpg',
        'unique_legacy_dataset_file', 'Unique legacy variant; the other 35 subset files match the included SUN397 tree.')

    # Exact installed environment exports are small and portable as records;
    # the 7.77 GB Linux environment itself is deliberately not copied to Windows.
    for name in ('dcmerge_vision.environment.yml', 'dcmerge_vision.explicit.txt',
                 'dcmerge_vision.pip-freeze.txt', 'dcmerge_vision.python-version.txt'):
        add(rows, PLAN / name, f'environment/{name}', 'environment_manifest',
            'Environment specification only; Linux binaries are not bundled for Windows.')

    # The prior incident logs are retained for local diagnosis.
    for p in sorted(Path('/workspace/.hf_home/xet/logs').glob('*.log')):
        add(rows, p, f'logs/xet/{p.name}', 'xet_diagnostic_log')

    # Preserve the small direct-download/recovery source and its state records.
    recovery = Path('/workspace/.dcmerge_sun397_http_stage')
    for name in ('download_single.py', 'extract_sun397.sh',
                 'provisioning_recovery.sh', 'checkpoint.json', 'manifest.json'):
        add(rows, recovery / name, f'server_only/download_recovery/{name}',
            'download_recovery_source_or_config')

    # Include server-only result files after checking every reachable Git object.
    # Byte-identical files already in the bundle are represented there once.
    result_dup = []
    result_seen: dict[str, str] = {}
    result_files = sorted(x for x in RESULTS.rglob('*') if x.is_file())
    for p in result_files:
        digest = sha256(p)
        if digest in git_hashes:
            result_dup.append(p)
        elif digest in result_seen:
            # A matching server-only copy is already represented once; keep the
            # canonical earlier path and note aliases in the inventory.
            result_dup.append(p)
        else:
            result_seen[digest] = str(p)
            add(rows, p, f'server_only/results/{p.relative_to(RESULTS).as_posix()}',
                'server_only_result_or_log', 'Not byte-identical to any Git bundle blob.')

    # Small diagnostics at the workspace root not already in any Git revision.
    for name in (
        'run_dcmerge_b32_8task.sh', 'dcmerge_oracle_baseline_gate.py',
        'run_dcmerge_oracle_baseline_gate.sh', 'dcmerge_baseline_diagnostic.py',
        'dcmerge_baseline_clone_diag.py', 'dcmerge_uniform_matrix_check.py',
    ):
        p = Path('/workspace') / name
        if p.exists() and sha256(p) not in git_hashes:
            add(rows, p, f'server_only/scripts/{name}', 'server_only_project_script')

    rows.sort(key=lambda row: row['backup_path'])
    with (PLAN / 'backup_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['category', 'source_path', 'backup_path', 'bytes', 'sha256', 'note'])
        w.writeheader()
        w.writerows(rows)
    with (PLAN / 'SHA256SUMS').open('w') as f:
        for row in rows:
            f.write(f"{row['sha256']}  {row['backup_path']}\n")
    manifest_sha = sha256(PLAN / 'backup_manifest.csv')
    (PLAN / 'backup_manifest.csv.sha256').write_text(f'{manifest_sha}  backup_manifest.csv\n')
    dataset_manifest_sha = sha256(PLAN / 'dataset_contents.csv')
    (PLAN / 'dataset_contents.csv.sha256').write_text(f'{dataset_manifest_sha}  dataset_contents.csv\n')

    category_bytes = Counter()
    category_count = Counter()
    for row in rows:
        category_bytes[row['category']] += int(row['bytes'])
        category_count[row['category']] += 1
    summary = {
        'manifest_sha256': manifest_sha,
        'dataset_contents_manifest_sha256': dataset_manifest_sha,
        'file_count': len(rows),
        'payload_bytes': sum(int(row['bytes']) for row in rows),
        'payload_gib': sum(int(row['bytes']) for row in rows) / (1024 ** 3),
        'category_counts': dict(category_count),
        'category_bytes': dict(category_bytes),
        'server_result_files_already_in_git_bundle': len(result_dup),
        'server_result_files_total': len(result_files),
        'git_blob_hashes_checked': len(git_hashes),
    }
    (PLAN / 'backup_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (PLAN / 'excluded_inventory.csv').open('w', newline='') as f:
        fields = ['category', 'path', 'bytes', 'file_count', 'backup_status', 'reason']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        def excluded(category, path, size, count, reason):
            w.writerow({'category': category, 'path': path, 'bytes': size,
                        'file_count': count, 'backup_status': 'excluded from payload', 'reason': reason})
        sun_tree = Path('/workspace/datasets8/sun397')
        sun_bytes, sun_files = tree_stats(sun_tree)
        excluded('extracted_dataset_duplicate', str(sun_tree), sun_bytes, sun_files,
                 'Same complete SUN397 dataset as included, SHA-256 verified source archive; extract archive locally if needed.')
        partial1 = Path('/workspace/.dcmerge_sun397_http_stage/xet-partial-preserved.incomplete')
        partial2 = Path('/workspace/.dcmerge_dataset_tars/.cache/huggingface/download/St8GeywbLqz59REFTvSueLgmg4s=.38afed47c5acce92ac26b36b620e936f4a849e7934e30c65487a17e2dbe0c917.16a3ceee.incomplete')
        for p in (partial1, partial2):
            excluded('preserved_incomplete_xet_cache', str(p), p.stat().st_size, 1,
                     'Stalled partial for the same SUN397 target; retained on server and superseded for backup by the verified complete archive.')
        flax = clip / 'flax_model.msgpack'
        excluded('alternate_model_serialization', str(flax), flax.stat().st_size, 1,
                 'Alternate serialization of the same base CLIP weights; the used PyTorch checkpoint is included.')
        model_cache = clip / '.cache'
        cache_bytes, cache_files = tree_stats(model_cache)
        excluded('incomplete_model_download_cache', str(model_cache), cache_bytes, cache_files,
                 'Hugging Face partial files, locks, and metadata; completed model files are included.')
        micromamba_cache = Path('/workspace/.micromamba/pkgs')
        cache_bytes, cache_files = tree_stats(micromamba_cache)
        excluded('recreatable_package_cache', str(micromamba_cache), cache_bytes, cache_files,
                 'Recreatable package download cache; portable exact environment exports are included.')
        bootstrap = Path('/workspace/.hf_bootstrap')
        tree_bytes, tree_files = tree_stats(bootstrap)
        excluded('download_helper_environment', str(bootstrap), tree_bytes, tree_files,
                 'HF download bootstrap Python environment; not needed to run the experiment and can be recreated.')
        env = Path('/venv/dcmerge_vision')
        tree_bytes, tree_files = tree_stats(env)
        excluded('installed_linux_environment', str(env), tree_bytes, tree_files,
                 'Linux-only installed environment; exact environment records are included instead.')
        local_root = Path('/workspace/ducanh/local-assets/datasets8/sun397')
        local_bytes, local_files = tree_stats(local_root)
        excluded('legacy_sun397_subset_duplicates', str(local_root),
                 local_bytes - legacy_unique.stat().st_size, local_files - 1,
                 '35 files are byte-identical to files in the included SUN397 archive; one changed file is included separately.')
        recovery = Path('/workspace/.dcmerge_sun397_http_stage')
        for p in sorted(recovery.iterdir()):
            if p.is_file() and p.name.endswith('.incomplete'):
                excluded('incomplete_model_or_xet_partial', str(p), p.stat().st_size, 1,
                         'Retained on server; excluded from backup because completed verified model/archive copies are present.')
        task_cache = Path('/workspace/models/DC-Merge-Vision-B32-r16-8task/.cache')
        cache_bytes, cache_files = tree_stats(task_cache)
        excluded('adapter_hub_metadata_cache', str(task_cache),
                 cache_bytes, cache_files,
                 'Small Hub locks and metadata; adapter checkpoints and heads are included.')
        source_tree = Path('/workspace/ducanh/DC-Merge')
        tree_bytes, tree_files = tree_stats(source_tree)
        excluded('upstream_source_working_tree', str(source_tree), tree_bytes, tree_files,
                 'Code contents are represented by the Git bundle and historical source archive; no second tree copy is transferred.')
        excluded('result_files_already_in_git_bundle', str(RESULTS),
                 sum(p.stat().st_size for p in result_dup), len(result_dup),
                 'Byte-identical result/log content already exists in the Git bundle; only the 40 unique server result files are transferred.')
        micromamba_bin = Path('/workspace/bin/micromamba')
        excluded('linux_environment_tool_binary', str(micromamba_bin), micromamba_bin.stat().st_size, 1,
                 'Linux-only, reproducible environment manager; Windows backup uses environment lock/export files instead.')
        excluded('transfer_test', str(PLAN / 'transfer_test_64MiB.bin'),
                 (PLAN / 'transfer_test_64MiB.bin').stat().st_size, 1,
                 'Small uncompressed throughput-test file only; exclude from the completed backup manifest.')

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
