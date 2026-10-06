"""Test the rigid-motion plus spatial-support explanation on frozen fields.

Oracle support coefficients use observed flow; no deployable segmenter is trained.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import csv
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'p22_dense_motion'
sys.path.insert(0,str(HERE.parent/'p22_dense_analysis'))
from photometric_analysis import sha, project, sample, roi, bin_name


def rank_one(values, valid, offset):
    """Least-squares affine line or principal nonnegative ray through zero.

    Pixel coefficients are oracle projections of observed values. With an
    offset, coefficients can be rescaled to [0,1] by shifting along the line;
    the offset itself is NOT independently observed background.
    """
    x=values[valid].astype(np.float64)
    center=x.mean(0) if offset else np.zeros(2)
    _,_,v=np.linalg.svd(x-center,full_matrices=False)
    direction=v[0]
    coeff=(values-center)@direction
    if not offset:
        selected=coeff[valid]
        if np.square(selected[selected<0]).sum()>np.square(selected[selected>=0]).sum():
            direction=-direction;coeff=-coeff
        coeff=np.maximum(coeff,0)
    result=center+coeff[...,None]*direction
    squared=np.square(x).sum() if not offset else np.square(x-center).sum()
    error=np.square(result[valid]-x).sum()
    explained=1-error/squared if squared>1e-15 else None
    return result.astype(np.float32), explained


def summarize(rows):
    out={'objects':len(rows)}
    if not rows:return out
    keys=['ray_energy_explained','line_shape_energy_explained','dense_l1','constant_l1','ray_l1','line_l1',
          'ray_minus_dense_l1','line_minus_dense_l1','constant_minus_dense_l1','common_fraction']
    for k in keys:
        vals=np.array([r[k] for r in rows if r[k] is not None],float)
        out[k]={'n':len(vals),'mean':float(vals.mean()) if len(vals) else None,
                'median':float(np.median(vals)) if len(vals) else None,
                'p10':float(np.percentile(vals,10)) if len(vals) else None,
                'p90':float(np.percentile(vals,90)) if len(vals) else None}
    gain=out['constant_minus_dense_l1']['mean']
    out['dense_over_constant_gain_explained_by_ray']=1-out['ray_minus_dense_l1']['mean']/gain if gain>1e-12 else None
    out['dense_over_constant_gain_explained_by_line']=1-out['line_minus_dense_l1']['mean']/gain if gain>1e-12 else None
    return out


def main():
    cv2.setNumThreads(1)
    # Verify recovery of a spatially varying scalar support, and non-recovery
    # of a field with two independent directions.
    yy,xx=np.indices((16,16));mask=np.ones((16,16),bool)
    alpha=xx/15; mix=np.array([.2,-.1])+alpha[...,None]*np.array([3.,-2.])
    assert np.max(np.abs(rank_one(mix,mask,True)[0]-mix))<1e-6
    assert rank_one(np.stack([xx,yy],-1),mask,True)[1] < .51
    receipt=json.loads((SOURCE/'FEATURE_RECEIPT.json').read_text())
    for e in receipt['files']:assert sha(e['path'])==e['sha256'],e['path']
    paths=[Path(e['path']) for e in receipt['files'] if Path(e['path']).suffix=='.json' and '_t' in Path(e['path']).name]
    assert len(paths)==20
    protocol={'type':'exploratory competing-explanation test after P22 photometric/temporal inspection',
              'hypothesis':'Spatial fields mostly represent one rigid displacement modulated by foreground/background occupancy, not independently rich vector dynamics.',
              'controls':['H plus median vector','H plus rank-one nonnegative ray through residual zero',
                          'H plus best affine line of residual vectors (offset and oracle scalar support)', 'H plus full dense residual'],
              'support':'Same current visibility and endpoint validity for all arms; temporal descriptive subset additionally uses both saved step masks.',
              'no_threshold_sweep':True,'identity_labels':False,'new_flow_inference':False,'training':False,
              'source_sha256':sha(__file__),'feature_receipt_sha256':sha(SOURCE/'FEATURE_RECEIPT.json')}
    (HERE/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];image_refs={}
    for path in paths:
        report=json.loads(path.read_text()); archive=np.load(path.with_suffix('.npz'),allow_pickle=False)
        assert list(archive['keys'])==[o['key'] for o in report['objects']]
        images=[]
        for ref in report['images'][-2:]:
            assert sha(ref['path'])==ref['sha256']
            image_refs[ref['path']]=ref['sha256']
            images.append(cv2.imread(ref['path']).astype(np.float32)/255.)
        previous,current=images;h,w=current.shape[:2]
        H=np.array(report['background_matrices'][str(report['anchor'])])
        for i,obj in enumerate(report['objects']):
            values=archive['residual_fields'][i,1];valid=archive['valid'][i,1]
            if valid.sum()<6:continue
            joint=valid & archive['valid'][i,0]
            joint_std=float(np.sqrt(np.var(values[joint],axis=0).sum())) if joint.any() else 0.
            above=bool(joint.mean()>=.5 and joint_std>obj['background_holdout_p90_pixels'])
            ray,ray_exp=rank_one(values,valid,False);line,line_exp=rank_one(values,valid,True)
            p=roi(obj['bbox']);bg=project(H,p)
            q={'constant':bg+np.median(values[valid],axis=0),'ray':bg+ray,'line':bg+line,'dense':bg+values}
            common=valid.copy()
            for point in q.values():common&=(point[...,0]>=0)&(point[...,0]<w-1)&(point[...,1]>=0)&(point[...,1]<h-1)
            observed=sample(current,p)
            errors={k:np.abs(sample(previous,point)-observed).mean(-1) for k,point in q.items()}
            for region in ['all','inner_half']:
                mask=common.copy()
                if region=='inner_half':mask&=(xx>=4)&(xx<12)&(yy>=4)&(yy<12)
                if mask.sum()<6:continue
                row={'pair':report['pair'],'key':obj['key'],'region':region,'size_bin':bin_name(obj['native_min_side']),
                     'joint_above_original_background_diagnostic':above,'joint_fraction':float(joint.mean()),
                     'ray_energy_explained':ray_exp,'line_shape_energy_explained':line_exp,
                     'common_fraction':float(mask.sum()/(256 if region=='all' else 64))}
                row.update({k+'_l1':float(error[mask].mean()) for k,error in errors.items()})
                row.update({k+'_minus_dense_l1':row[k+'_l1']-row['dense_l1'] for k in ['constant','ray','line']})
                rows.append(row)
    groups={}
    for region in ['all','inner_half']:
        selected=[r for r in rows if r['region']==region]
        groups[region]={'all':summarize(selected),'per_pair':{p:summarize([r for r in selected if r['pair']==p]) for p in receipt['per_pair']},
                        'joint_above_original_background_diagnostic':summarize([r for r in selected if r['joint_above_original_background_diagnostic']]),
                        'per_size':{b:summarize([r for r in selected if r['size_bin']==b]) for b in ['lt16','16to32','ge32']}}
    output={'status':'OBSERVED_FIELD_ORACLE_COMPLEXITY_CONTROL_COMPLETE','groups':groups,
            'source_sha256':sha(__file__),'protocol_sha256':sha(HERE/'PROTOCOL.json'),'feature_receipt_sha256':sha(SOURCE/'FEATURE_RECEIPT.json'),
            'verified_feature_files':len(receipt['files']),'image_hashes':image_refs,
            'limits':['Per-pixel scalar coefficients are fitted to the same observed flow; reconstruction is in-sample, not independent prediction.',
                      'Offset-line control has a freely fitted object-specific offset, so it is an oracle competing explanation, not physically recovered background.',
                      'Explained vector energy is about low-dimensional residual directions; arbitrary scalar spatial support can still contain spatial information.',
                      'Neither high nor low explained energy establishes identity information, camera invariance, or a novel method.',
                      'The subset repeats the prior uncalibrated std-vs-p90 diagnostic with corrected joint masks, not a tuned threshold or scientific significance gate.']}
    (HERE/'RIGID_SUPPORT.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    with (HERE/'objects.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({r:{g:{k:v for k,v in s.items() if k in ['objects','line_shape_energy_explained','dense_over_constant_gain_explained_by_line','dense_over_constant_gain_explained_by_ray']} for g,s in groups[r].items() if g in ['all','joint_above_original_background_diagnostic']} for r in groups}))


if __name__=='__main__':main()
