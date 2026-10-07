"""Keep annotation-file reads out of the actual online inference process."""
import builtins
import io
import json
import os
from pathlib import Path
import runpy
import sys

blocked_root = Path(os.environ['LEAF_GT_GUARD_ROOT']).resolve()
receipt = Path(os.environ['LEAF_GT_GUARD_RECEIPT'])
counts = {'checked_file_open_calls': 0, 'blocked_gt_attempts': 0}


def guarded(function):
    def call(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)):
            counts['checked_file_open_calls'] += 1
            path = Path(os.fsdecode(file)).resolve()
            if path == blocked_root or blocked_root in path.parents:
                counts['blocked_gt_attempts'] += 1
                receipt.write_text(json.dumps(dict(status='STOP_GT_READ_ATTEMPT', attempted_path=str(path), **counts), indent=2)+'\n')
                raise RuntimeError('ONLINE_INFERENCE_ATTEMPTED_GT_READ: '+str(path))
        return function(file, *args, **kwargs)
    return call


builtins.open = guarded(builtins.open)
io.open = guarded(io.open)
os.open = guarded(os.open)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
receipt.write_text(json.dumps(dict(status='PASS_NO_GT_READ', guarded_annotation_root=str(blocked_root), **counts), indent=2)+'\n')
