#!/usr/bin/env python3
"""Read-only archive audit after FPN export; no images, GT, or model inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--role',choices=('fit','calibration'),required=True);args=p.parse_args()
    out=HERE/args.role;start=time.monotonic()
    assert (out/'fpn.exit').read_text().strip()=='0'
    receipt=json.loads((out/'FEATURE_RECEIPT.json').read_text())
    assert receipt['status']=='PASS_FROZEN_NATIVE_FPN_EXPORT' and receipt['role']==args.role
    assert (out/'FEATURE_RECEIPT.sha256').read_text().split()[0]==sha(out/'FEATURE_RECEIPT.json')
    for name,expected in receipt['outputs'].items():assert sha(out/name)==expected,name
    index=json.loads((out/'FEATURE_INDEX.json').read_text());entries=index['entries']
    assert len(entries)==receipt['frame_count']==receipt['archive_count']
    assert len(index['groups'])==receipt['group_count']
    assert index['by_key']=={x['key']:i for i,x in enumerate(entries)}
    native_count=0;byte_count=0;shapes=set();min_score=1.;max_score=0.
    for entry in entries:
        archive=Path(entry['archive_path'])
        assert archive.resolve()==(out/'features'/f"{entry['key']}.npz").resolve()
        digest=sha(archive)
        assert digest==entry['archive_sha256']==receipt['archive_hashes'][str(archive)]
        assert archive.stat().st_size==entry['archive_bytes']
        byte_count+=archive.stat().st_size
        metadata=entry['metadata'];height,width=metadata['pad_shape'][:2]
        with np.load(archive,allow_pickle=False) as z:
            assert set(z.files)=={f'f{i}' for i in range(5)}|{f'det{i}' for i in range(3)}
            for i,stride in enumerate((8,16,32,64,128)):
                value=z[f'f{i}'];shape=(1,256,math.ceil(height/stride),math.ceil(width/stride))
                assert value.dtype==np.float32 and value.shape==shape==tuple(entry['fpn_shapes'][i])
                assert np.isfinite(value).all();shapes.add((i,shape))
            n=0
            for i in range(3):
                value=z[f'det{i}'];assert value.dtype==np.float32 and value.shape==(entry['det_counts'][i],5)
                assert np.isfinite(value).all()
                if len(value):
                    min_score=min(min_score,float(value[:,4].min()));max_score=max(max_score,float(value[:,4].max()))
                n+=len(value)
            assert n<=100
            native_count+=n
    assert byte_count==receipt['archive_total_bytes'] and native_count==receipt['total_native_detections']
    # Native BaseDenseHead thresholds class scores before multiplying its score
    # factor, so final native scores may legitimately be slightly below .05.
    # Preserve returned arrays; never introduce an extra cache postfilter.
    assert min_score>=0. and max_score<=1.
    mismatch=[]
    for group in index['groups']:
        selected=[entries[index['by_key'][key]] for key in group['image_keys']]
        assert [group['frame']-x['frame'] for x in selected]==[0,1,4,8]
        assert all((x['role'],x['pair'],x['view'])==(args.role,group['pair'],group['view']) for x in selected)
        assert group['current_metadata_key']==selected[0]['key']
        if any(x['metadata']['ori_shape']!=selected[0]['metadata']['ori_shape'] or x['fpn_shapes']!=selected[0]['fpn_shapes'] for x in selected[1:]):
            mismatch.append(group['key'])
    result=dict(status='PASS_NATIVE_FPN_ARCHIVE_VERIFICATION',role=args.role,
        receipt_sha256=sha(out/'FEATURE_RECEIPT.json'),feature_index_sha256=sha(out/'FEATURE_INDEX.json'),verification_code_sha256=sha(__file__),
        archive_count=len(entries),archive_total_bytes=byte_count,group_count=len(index['groups']),total_native_detections=native_count,
        minimum_native_score=min_score,maximum_native_score=max_score,
        native_score_threshold_semantics='cfg.score_thr=.05 applied before score_factors multiplication; output range [0,1], no extra postfilter',
        distinct_level_shapes=[{'level':i,'shape':list(shape)} for i,shape in sorted(shapes)],
        within_group_image_or_feature_shape_mismatch_keys=mismatch,all_archive_hashes_match=True,
        all_feature_arrays_float32_finite=True,all_native_detection_arrays_float32_finite=True,
        exact_plan_coverage_and_fixed_past_lags=True,images_GT_or_model_read=False,
        elapsed_seconds=round(time.monotonic()-start,3))
    (out/'FEATURE_VERIFY.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (out/'FEATURE_VERIFY.sha256').write_text(sha(out/'FEATURE_VERIFY.json')+'  FEATURE_VERIFY.json\n')
    print(json.dumps(result))


if __name__=='__main__':main()
