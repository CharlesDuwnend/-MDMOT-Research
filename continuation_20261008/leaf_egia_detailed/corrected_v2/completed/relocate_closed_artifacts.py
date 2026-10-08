"""Relocate closed SRC research artifacts intact, preserving logical paths."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/home/chenhc/leaf_egia_detailed_20261008_v2')
SOURCE=Path('/home/chenhc/leaf_joint_consequence_20260922/artifacts')
TARGET=Path('/raid/datasets/chc_data/research_artifact_storage/leaf_egia_20261008/leaf_joint_consequence_20260922_artifacts')

def inventory(root):
    rows={}
    for p in sorted(root.rglob('*')):
        name=str(p.relative_to(root))
        if p.is_symlink(): rows[name]={'symlink':os.readlink(p)}
        elif p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''):h.update(block)
            rows[name]={'sha256':h.hexdigest(),'size':p.stat().st_size,'mode':p.stat().st_mode & 0o777}
        elif p.is_dir():rows[name]={'directory':True,'mode':p.stat().st_mode & 0o777}
        else:raise RuntimeError('UNEXPECTED_FILE_TYPE: '+str(p))
    return rows

assert SOURCE.is_dir() and not SOURCE.is_symlink() and not TARGET.exists()
probe=subprocess.run(['lsof','-w','+D',str(SOURCE)],capture_output=True,text=True)
assert probe.returncode==1 and not probe.stdout.strip(),'ARTIFACTS_HAVE_OPEN_FILE_HANDLES'
before=inventory(SOURCE)
free_before=shutil.disk_usage(ROOT).free
TARGET.parent.mkdir(parents=True,exist_ok=True)
shutil.copytree(SOURCE,TARGET,symlinks=True,copy_function=shutil.copy2)
assert inventory(TARGET)==before,'COPIED_BYTES_OR_METADATA_DIFFER'
assert inventory(SOURCE)==before,'SOURCE_CHANGED_DURING_COPY'
backup=SOURCE.with_name('artifacts_storage_verified_copy_20261008')
assert not backup.exists()
SOURCE.rename(backup)
try:
    SOURCE.symlink_to(TARGET,target_is_directory=True)
    assert inventory(SOURCE)==before
    frozen=json.loads((ROOT/'manifest.json').read_text())['frozen_files_sha256']
    for name,expected in frozen.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
except BaseException:
    if SOURCE.is_symlink():SOURCE.unlink()
    backup.rename(SOURCE)
    raise
encoded=json.dumps(before,sort_keys=True).encode()
(TARGET.parent/'inventory.json').write_bytes(encoded+b'\n')
shutil.rmtree(backup) # Only the verified duplicate; every artifact remains at TARGET.
receipt=dict(status='INTACT_CLOSED_ARTIFACT_RELOCATION_VERIFIED',
    completed_utc=datetime.datetime.utcnow().isoformat()+'Z',source_logical_path=str(SOURCE),
    storage_target=str(TARGET),all_source_bytes_preserved=True,logical_path_preserved_by_symlink=True,
    no_active_file_handles=True,frozen_dependency_hashes_unchanged=True,
    inventory_sha256=hashlib.sha256(encoded+b'\n').hexdigest(),entries=len(before),
    preserved_file_bytes=sum(v.get('size',0) for v in before.values()),
    free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(ROOT).free)
(ROOT/'review/STORAGE_RELOCATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
