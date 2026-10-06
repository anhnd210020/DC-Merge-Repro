"""Atomic JSON/CSV artifacts and content seals."""
from __future__ import annotations
import csv, hashlib, json, os, tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping
from .design import canonical_json

def hash_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def seal(value: Any) -> dict[str,Any]:
    return {"payload":value,"sha256":hashlib.sha256(canonical_json(value).encode()).hexdigest()}

def unseal(value: Mapping[str,Any]) -> Any:
    payload=value.get('payload')
    if value.get('sha256') != hashlib.sha256(canonical_json(payload).encode()).hexdigest(): raise ValueError('manifest checksum mismatch')
    return payload

@contextmanager
def output_lock(output: Path):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); stream=(output/'.workflow.lock').open('a+b')
    try:
        if os.name=='nt':
            import msvcrt
            stream.seek(0)
            try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as e: raise RuntimeError(f'another workflow holds {output}') from e
        else:
            import fcntl
            try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as e: raise RuntimeError(f'another workflow holds {output}') from e
        yield
    finally:
        try:
            if os.name=='nt':
                import msvcrt
                stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        except OSError: pass
        stream.close()

def atomic_json(path: Path, value: Any) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    _atomic(path,data)

def atomic_text(path: Path, value: str) -> None:
    _atomic(Path(path),value.encode('utf-8'))

def atomic_csv(path: Path, rows: Iterable[Mapping[str,Any]]) -> None:
    rows=list(rows); fields=list(dict.fromkeys(k for row in rows for k in row))
    from io import StringIO
    s=StringIO(newline=''); w=csv.DictWriter(s,fieldnames=fields,extrasaction='ignore');
    if fields: w.writeheader(); w.writerows(rows)
    _atomic(Path(path),s.getvalue().encode())

def _atomic(path: Path,data:bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    except BaseException:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise
