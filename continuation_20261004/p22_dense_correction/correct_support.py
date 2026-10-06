"""Materialize composed-path validity for frozen P22, leaving sources intact."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'p22_dense_motion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def composed_support(step_valid):
    assert step_valid.dtype==np.bool_ and step_valid.shape[-3]==2
    current=step_valid[...,1,:,:]
    historical=step_valid[...,0,:,:]&current
    return historical, current


def main():
    raw=json.loads((SOURCE/'FEATURE_RECEIPT.json').read_text())
    for e in raw['files']:assert sha(e['path'])==e['sha256']
    rows=[];per_pair={}
    for entry in raw['files']:
        path=Path(entry['path'])
        if path.suffix!='.npz':continue
        data=np.load(path,allow_pickle=False)
        report=json.loads(path.with_suffix('.json').read_text())
        historical,current=composed_support(data['valid'])
        target=HERE/path.name
        np.savez_compressed(target,keys=data['keys'],historical_path_valid=historical,
                            current_edge_valid=current,common_two_step_valid=historical)
        counts=per_pair.setdefault(report['pair'],dict(objects=0,common_half_grid=0,common_spatial_above_original_bg_diagnostic=0))
        for i,obj in enumerate(report['objects']):
            counts['objects']+=1
            common=historical[i]
            if common.mean()>=.5:
                counts['common_half_grid']+=1
                std=float(np.sqrt(np.var(data['residual_fields'][i,1][common],axis=0).sum()))
                counts['common_spatial_above_original_bg_diagnostic']+=int(std>obj['background_holdout_p90_pixels'])
        rows.append(dict(source=str(path),source_sha256=sha(path),output=str(target),sha256=sha(target)))
    receipt={'status':'COMPOSED_MASK_CORRECTION_MATERIALIZED','source_receipt_sha256':sha(SOURCE/'FEATURE_RECEIPT.json'),
             'source_sha256':sha(__file__),'files':rows,'per_pair':per_pair,
             'marginal_masks_preserved_in_original':True,'residual_fields_unchanged':True,
             'gate_numeric_count_holds_in_pairs':[p for p,v in per_pair.items() if v['common_spatial_above_original_bg_diagnostic']>=20],
             'vector_basis':'Raw residual channels remain in t-2 and t-1 respectively; use exact endpoint pullback before vector comparison.',
             'camera_intervention':'Original lacks warped fields and real-pixel masks; correction does not repair or validate camera invariance.',
             'not_identity_information_or_method_pass':True}
    (HERE/'CORRECTED_SUPPORT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(per_pair))


if __name__=='__main__':main()
