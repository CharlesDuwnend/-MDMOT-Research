"""Scoped phase snapshot, preserving unrelated staged research changes."""
import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
REPO=Path('/home/chenhc/claude_try_MDMOT')
def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=REPO,env=env)
def main():
    prefix,stage,message=sys.argv[1:]
    scope={p.decode() for p in git('ls-files','--cached','--others','--exclude-standard','-z','--',prefix).split(b'\0') if p}
    assert scope
    expected={str(p.relative_to(REPO)) for p in (REPO/prefix).rglob('*') if p.is_file()}
    assert scope==expected,(scope^expected)
    before=git('diff','--cached','--binary');old=git('rev-parse','HEAD').decode().strip()
    remote_available=False;dependency=None
    try:
        probe=subprocess.run(['git','ls-remote','--exit-code','origin','refs/heads/main'],cwd=REPO,capture_output=True,text=True,timeout=25)
        remote_available=probe.returncode==0
        if not remote_available:dependency=probe.stderr.strip()
    except subprocess.TimeoutExpired:dependency='origin probe exceeded 25 seconds'
    fd,index=tempfile.mkstemp(prefix='leaf_reduced_phase_index_');os.close(fd);os.unlink(index)
    env=os.environ.copy();env['GIT_INDEX_FILE']=index
    result=None
    try:
        git('read-tree','HEAD',env=env)
        command=['python3','scripts/stage_snapshot.py','--stage',stage,'--message',message]
        if not remote_available:command.append('--local-only')
        for name in sorted(scope):command+=['--include',name]
        result=subprocess.run(command,cwd=REPO,env=env,capture_output=True,text=True,timeout=120)
    finally:
        head=git('rev-parse','HEAD').decode().strip()
        if head!=old:
            names=git('diff','--name-only',old,head).decode().splitlines()
            assert all(p.startswith(prefix+'/') or p.startswith('versioning/stages/'+stage+'/') for p in names)
            git('restore','--staged','--source=HEAD','--',*names)
        assert git('diff','--cached','--binary')==before
        if Path(index).exists():Path(index).unlink()
    print(json.dumps(dict(local_head=head,local_committed=head!=old,remote_verified=bool(remote_available and result and result.returncode==0),
        dependency=dependency,helper_returncode=result.returncode if result else None,
        stdout=result.stdout if result else None,stderr=result.stderr if result else None,
        protected_staged_diff_sha256=hashlib.sha256(before).hexdigest()),ensure_ascii=False))
    assert result is not None and head!=old
if __name__=='__main__':main()
