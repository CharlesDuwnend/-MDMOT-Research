#!/usr/bin/env python3
"""Add absolute path spelling without changing sealed predictions or inputs.

Python 3.8 retains a relative __file__ for a relatively invoked main script.
run_calibration.sh explicitly cd's to HERE. The original receipt is preserved.
"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

source=HERE/'PREDICTION_RECEIPT.json'
target=HERE/'PREDICTION_RECEIPT_CANONICAL.json'
assert not target.exists()
receipt=json.loads(source.read_text())
relative={k:v for k,v in receipt['inputs'].items() if not Path(k).is_absolute()}
assert set(relative)=={'predict_calibration.py'}
key='predict_calibration.py';absolute=HERE/key
assert sha(absolute)==relative[key]
assert 'cd "$P28_DIR"' in (HERE/'run_calibration.sh').read_text()
assert 'P28_DIR='+str(HERE) in (HERE/'run_calibration.sh').read_text()
del receipt['inputs'][key]
assert receipt['inputs'].get(str(absolute),relative[key])==relative[key]
receipt['inputs'][str(absolute)]=relative[key]
for p in [source,HERE/'run_calibration.sh',Path(__file__).resolve()]:receipt['inputs'][str(p)]=sha(p)
receipt['canonicalization']=dict(original_receipt_sha256=sha(source),working_directory=str(HERE),
    relative_input=key,absolute_input=str(absolute),reason='Python3.8 main __file__ relative under explicit launcher cwd',
    prediction_or_checkpoint_bytes_changed=False,original_receipt_preserved=True,calibration_XML_opened=False)
target.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(status='PASS_PATH_CANONICALIZATION_ONLY',path=str(target),sha256=sha(target))))
