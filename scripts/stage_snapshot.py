#!/usr/bin/env python3
"""Archive an MDMT research phase, then push and verify its exact remote HEAD."""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from datetime import timezone, timedelta
    class ZoneInfo:
        def __new__(cls, name):
            if name != 'Asia/Shanghai':
                raise ValueError(name)
            return timezone(timedelta(hours=8))

ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT = 2 * 1024 * 1024
SECRETS = re.compile(rb'(?:-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{24,})')


def git(*args, check=True, data=None):
    return subprocess.run(['git', *args], cwd=ROOT, input=data,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--message', required=True)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,100}', args.stage):
        raise SystemExit('Stage must be a simple ASCII identifier.')
    if git('rev-parse', '--show-toplevel').stdout.decode().strip() != str(ROOT):
        raise SystemExit('Wrong Git root.')
    if git('diff', '--cached', '--name-only').stdout.strip():
        raise SystemExit('Existing staged changes: inspect them before archiving this phase.')
    if git('branch', '--show-current').stdout.decode().strip() != 'main':
        raise SystemExit('Expected main; do not change another branch automatically.')

    archive = ROOT/'versioning'/'stages'/args.stage
    if archive.exists():
        raise SystemExit('Stage already archived; use a new stage identifier.')
    now = datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    artifacts = []
    # No external symlinks are followed; checkpoint hashes stay within this root.
    for path in sorted(ROOT.rglob('*')):
        if '.git' in path.parts or path.is_symlink() or not path.is_file():
            continue
        if path.suffix in {'.pt', '.pth', '.ckpt'} or path.stat().st_size > MAX_TEXT:
            row = {'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                   'stored_in_git': False}
            if path.suffix in {'.pt', '.pth', '.ckpt'}:
                row['sha256'] = sha(path)
            artifacts.append(row)
    json_write(archive/'LOCAL_ARTIFACT_INDEX.json', {
        'stage': args.stage, 'time_asia_shanghai': now, 'local_root': str(ROOT),
        'scope': 'local path/size index; SHA256 for checkpoints only',
        'local_files_deleted': False, 'artifacts': artifacts,
    })
    # Compact digests for changed and newly added allowed files, avoiding self-hash.
    candidates = set()
    for cmd in [('ls-files', '-z'), ('ls-files', '--others', '--exclude-standard', '-z')]:
        candidates.update(p.decode() for p in git(*cmd).stdout.split(b'\0') if p)
    allowed, excluded, digests = [], [], []
    for name in sorted(candidates):
        path = ROOT/name
        if path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_size > MAX_TEXT:
            excluded.append(name)
            continue
        blob = path.read_bytes()
        if b'\x00' in blob:
            raise SystemExit('Unexpected binary among allowed files: ' + name)
        if SECRETS.search(blob):
            raise SystemExit('Potential credential in allowed file; inspect locally: ' + name)
        allowed.append(name)
        digests.append({'path': name, 'bytes': len(blob), 'sha256': hashlib.sha256(blob).hexdigest()})
    json_write(archive/'TRACKED_EVIDENCE_INDEX.json', {
        'stage': args.stage, 'time_asia_shanghai': now, 'files': digests,
        'excluded_large_text': excluded,
        'self_index_hash_omitted': True,
    })
    allowed.append(str((archive/'TRACKED_EVIDENCE_INDEX.json').relative_to(ROOT)))
    if allowed:
        git('add', '--pathspec-from-file=-', '--pathspec-file-nul',
            data=b'\0'.join(s.encode() for s in allowed) + b'\0')
    # Track deletions only for files previously included in this research repo.
    git('add', '-u')
    oversized = []
    for line in git('ls-files', '-s').stdout.decode().splitlines():
        _, name = line.split('\t', 1)
        oid = line.split()[1]
        if int(git('cat-file', '-s', oid).stdout) > MAX_TEXT:
            oversized.append(name)
    if oversized:
        raise SystemExit('Oversized staged files: ' + ', '.join(oversized))
    if not git('diff', '--cached', '--quiet', check=False).returncode:
        raise SystemExit('No phase changes to commit.')
    git('commit', '-m', args.message)
    head = git('rev-parse', 'HEAD').stdout.decode().strip()
    print(json.dumps({'status': 'LOCAL_PHASE_COMMITTED', 'stage': args.stage,
                      'head': head, 'allowed_files': len(allowed),
                      'large_text_excluded': len(excluded)}, ensure_ascii=False), flush=True)
    if args.local_only:
        return
    if git('remote', 'get-url', 'origin', check=False).returncode:
        raise SystemExit('Local phase committed; origin missing, remote push NOT completed.')
    git('push', '--set-upstream', 'origin', 'refs/heads/main:refs/heads/main')
    remote = git('ls-remote', '--exit-code', 'origin', 'refs/heads/main').stdout.decode().split()[0]
    if remote != head:
        raise SystemExit('Push verification failed: remote main differs from local HEAD.')
    print(json.dumps({'status': 'REMOTE_PHASE_VERIFIED', 'stage': args.stage,
                      'head': head, 'remote_main': remote}), flush=True)


if __name__ == '__main__':
    main()
