#!/usr/bin/env python3
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi, get_token, hf_hub_url

REPO = 'anhnd210020/DC-Merge-Vision-8Task'
FILENAME = 'sun397.tar'
REVISION = '42fc343338f31c2f41cfa85e97d20e22dfbf9d45'
EXPECTED_SIZE = 14591037440
EXPECTED_SHA256 = '38afed47c5acce92ac26b36b620e936f4a849e7934e30c65487a17e2dbe0c917'
STAGE = Path('/workspace/.dcmerge_sun397_http_stage')
PARTIAL = STAGE / 'sun397.tar.partial'
FINAL = STAGE / 'sun397.tar'
MANIFEST = STAGE / 'manifest.json'
CHECKPOINT = STAGE / 'checkpoint.json'
FLUSH_BYTES = 16 * 1024 * 1024
PROGRESS_BYTES = 128 * 1024 * 1024


def log(message):
    print(f'[{time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}] {message}', flush=True)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dirfd = os.open(str(path.parent), os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)


def checkpoint(offset):
    atomic_json(CHECKPOINT, {'revision': REVISION, 'bytes': offset})


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    token = get_token()
    if not token:
        log('BLOCKED: Hugging Face authentication is unavailable.')
        return 2

    try:
        info = HfApi(token=token).dataset_info(REPO, revision=REVISION, files_metadata=True)
    except Exception as exc:
        response = getattr(exc, 'response', None)
        status = getattr(response, 'status_code', None)
        log(f'BLOCKED: metadata request failed ({type(exc).__name__}, status={status}).')
        return 3
    if info.sha != REVISION:
        log('BLOCKED: resolved revision differs from the pinned revision.')
        return 4
    file_info = next((item for item in info.siblings if item.rfilename == FILENAME), None)
    lfs = getattr(file_info, 'lfs', None) if file_info else None
    remote_sha = getattr(lfs, 'sha256', None) if lfs else None
    if not file_info or file_info.size != EXPECTED_SIZE or remote_sha != EXPECTED_SHA256:
        log('BLOCKED: authoritative file metadata does not match pinned size and SHA-256.')
        return 5

    manifest = {
        'repo': REPO,
        'filename': FILENAME,
        'revision': REVISION,
        'size': EXPECTED_SIZE,
        'sha256': EXPECTED_SHA256,
        'mode': 'single-stream HTTP Range with resume from fsynced byte offset',
    }
    if MANIFEST.exists():
        try:
            existing = json.loads(MANIFEST.read_text())
        except Exception:
            log('BLOCKED: staging manifest is unreadable; preserving staging files.')
            return 6
        if existing != manifest:
            log('BLOCKED: staging manifest does not match the pinned asset; preserving staging files.')
            return 7
    else:
        atomic_json(MANIFEST, manifest)

    if FINAL.exists():
        actual_size = FINAL.stat().st_size
        actual_sha = hash_file(FINAL) if actual_size == EXPECTED_SIZE else ''
        if actual_size == EXPECTED_SIZE and actual_sha == EXPECTED_SHA256:
            log(f'VERIFIED existing staged archive size={actual_size} sha256={actual_sha}')
            return 0
        log('BLOCKED: a staged final file exists but failed verification; preserving it.')
        return 8

    if CHECKPOINT.exists():
        try:
            state = json.loads(CHECKPOINT.read_text())
            offset = int(state['bytes'])
            if state.get('revision') != REVISION or offset < 0 or offset > EXPECTED_SIZE:
                raise ValueError('bad checkpoint')
        except Exception:
            log('BLOCKED: checkpoint is invalid; preserving staging files.')
            return 9
    else:
        offset = 0

    if PARTIAL.exists():
        actual = PARTIAL.stat().st_size
        if actual < offset:
            log('BLOCKED: staged partial is shorter than its durable checkpoint.')
            return 10
        if actual != offset:
            with PARTIAL.open('r+b') as f:
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())
    elif offset != 0:
        log('BLOCKED: checkpoint exists without a partial file.')
        return 11
    else:
        with PARTIAL.open('wb') as f:
            f.flush()
            os.fsync(f.fileno())

    url = hf_hub_url(REPO, FILENAME, repo_type='dataset', revision=REVISION)
    headers = {'Authorization': 'Bearer ' + token, 'Accept-Encoding': 'identity'}
    started = time.monotonic()
    session_bytes = 0
    last_flushed = offset
    last_reported = offset
    retry = 0

    with PARTIAL.open('r+b', buffering=0) as output:
        output.seek(offset)
        while offset < EXPECTED_SIZE:
            end = EXPECTED_SIZE - 1
            expected_range = f'bytes {offset}-{end}/{EXPECTED_SIZE}'
            expected_count = end - offset + 1
            got = 0
            try:
                with httpx.stream(
                    'GET', url,
                    headers={**headers, 'Range': f'bytes={offset}-{end}'},
                    follow_redirects=True,
                    timeout=httpx.Timeout(180, connect=30),
                ) as response:
                    status = response.status_code
                    content_range = response.headers.get('Content-Range', '')
                    if status in (401, 403):
                        log(f'BLOCKED: HTTP {status} for pinned range starting at byte {offset}.')
                        return 12
                    if status != 206 or content_range != expected_range:
                        log(f'BLOCKED: expected HTTP 206 Content-Range {expected_range}; received status={status}, Content-Range={content_range[:100]!r}.')
                        return 13
                    content_length = response.headers.get('Content-Length')
                    if content_length is not None and int(content_length) != expected_count:
                        log(f'BLOCKED: Content-Length mismatch at byte {offset}.')
                        return 14
                    for chunk in response.iter_bytes(4 * 1024 * 1024):
                        if not chunk:
                            continue
                        if got + len(chunk) > expected_count:
                            log(f'BLOCKED: response body exceeded requested range at byte {offset}.')
                            return 15
                        output.write(chunk)
                        got += len(chunk)
                        session_bytes += len(chunk)
                        if output.tell() - last_flushed >= FLUSH_BYTES:
                            output.flush()
                            os.fsync(output.fileno())
                            offset = output.tell()
                            checkpoint(offset)
                            last_flushed = offset
                        if output.tell() - last_reported >= PROGRESS_BYTES:
                            offset = output.tell()
                            elapsed = max(time.monotonic() - started, 0.001)
                            rate = session_bytes / 1048576 / elapsed
                            remaining = max(EXPECTED_SIZE - offset, 0)
                            eta = remaining / max(rate * 1048576, 1)
                            log(f'progress bytes={offset}/{EXPECTED_SIZE} rate={rate:.2f} MiB/s eta_min={eta/60:.1f}')
                            last_reported = offset
                    output.flush()
                    os.fsync(output.fileno())
                    offset = output.tell()
                    checkpoint(offset)
                    last_flushed = offset
                    if got != expected_count:
                        retry += 1
                        log(f'short response: received={got}/{expected_count}; resuming at byte {offset}.')
                        time.sleep(min(2 ** min(retry, 5), 30))
                        continue
                    retry = 0
            except Exception as exc:
                output.flush()
                os.fsync(output.fileno())
                offset = output.tell()
                checkpoint(offset)
                retry += 1
                elapsed = max(time.monotonic() - started, 0.001)
                rate = session_bytes / 1048576 / elapsed
                log(f'connection interrupted ({type(exc).__name__}); durable_bytes={offset}; rate={rate:.2f} MiB/s; retry={retry}.')
                time.sleep(min(2 ** min(retry, 5), 30))

    final_size = PARTIAL.stat().st_size
    if final_size != EXPECTED_SIZE:
        log(f'BLOCKED: completed stream has size={final_size}, expected={EXPECTED_SIZE}.')
        return 16
    final_sha = hash_file(PARTIAL)
    if final_sha != EXPECTED_SHA256:
        log(f'BLOCKED: SHA-256 mismatch; actual={final_sha}. Partial retained at {PARTIAL}.')
        return 17
    os.replace(PARTIAL, FINAL)
    dirfd = os.open(str(STAGE), os.O_DIRECTORY)
    try:
        os.fsync(dirfd)
    finally:
        os.close(dirfd)
    log(f'VERIFIED staged archive size={EXPECTED_SIZE} sha256={final_sha} path={FINAL}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
