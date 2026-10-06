"""Label-free diagnostics of frozen P22 fields; no new flow or model fitting.

All arms are observed-interval reconstructions, not held-out predictions.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib
import json
import csv
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'p22_dense_motion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(image, points):
    return cv2.remap(image.astype(np.float32), points[..., 0].astype(np.float32),
                     points[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def project(matrix, points):
    h = np.concatenate([points, np.ones(points.shape[:-1] + (1,))], -1) @ matrix.T
    assert np.all(np.abs(h[..., 2]) > 1e-8)
    return h[..., :2] / h[..., 2:3]


def roi(box):
    x1, y1, x2, y2 = box
    xx, yy = np.meshgrid(x1+(np.arange(16)+.5)/16*(x2-x1),
                         y1+(np.arange(16)+.5)/16*(y2-y1))
    return np.stack([xx, yy], -1)


def bin_name(side):
    return 'lt16' if side < 16 else '16to32' if side < 32 else 'ge32'


def summary(rows):
    if not rows:
        return {'objects': 0}
    out = {'objects': len(rows)}
    for key in ['bg_l1', 'constant_l1', 'affine_l1', 'dense_l1', 'shuffle_l1',
                'constant_minus_dense_l1', 'affine_minus_dense_l1',
                'shuffle_minus_dense_l1', 'common_fraction']:
        values = [r[key] for r in rows]
        out[key + '_mean'] = float(np.mean(values))
        out[key + '_median'] = float(np.median(values))
    out['dense_better_than_constant_fraction'] = float(np.mean([r['constant_minus_dense_l1'] > 0 for r in rows]))
    out['dense_better_than_affine_fraction'] = float(np.mean([r['affine_minus_dense_l1'] > 0 for r in rows]))
    return out


def main():
    cv2.setNumThreads(1)
    receipt_path = SOURCE / 'FEATURE_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    for entry in receipt['files']:
        assert sha(entry['path']) == entry['sha256'], entry['path']
    paths = [Path(e['path']) for e in receipt['files']
             if Path(e['path']).suffix == '.json' and '_t' in Path(e['path']).name]
    assert len(paths) == 20
    image_refs, rows, camera_rows, geom = {}, [], [], []
    for path in paths:
        report = json.loads(path.read_text())
        assert report['pair'] in ['23', '25', '29', '69', '78']
        assert report['PID_read'] is False and report['raw_XML_read'] is False
        images = []
        for ref in report['images'][-2:]:
            assert '/MDMT/train/' in ref['path']
            assert sha(ref['path']) == ref['sha256']
            image_refs[ref['path']] = ref['sha256']
            value = cv2.imread(ref['path'])
            assert value is not None
            images.append(value.astype(np.float32) / 255.)
        previous, current = images
        height, width = current.shape[:2]
        matrix = np.asarray(report['background_matrices'][str(report['anchor'])])
        archive = np.load(path.with_suffix('.npz'), allow_pickle=False)
        assert list(archive['keys']) == [r['key'] for r in report['objects']]
        geom.extend([dict(pair=report['pair'], camera=report['camera'], anchor=report['anchor'],
                          time=int(t), **v) for t,v in report['geometry'].items()])
        for i, obj in enumerate(report['objects']):
            values, visible = archive['residual_fields'][i,1], archive['valid'][i,1]
            p = roi(obj['bbox'])
            bg = project(matrix, p)
            yy, xx = np.indices((16,16))
            design = np.stack([np.ones_like(xx), (xx-7.5)/8, (yy-7.5)/8], -1)
            if visible.sum() < 6:
                continue
            constant = np.median(values[visible], axis=0)
            coeff = np.linalg.lstsq(design[visible], values[visible], rcond=None)[0]
            affine = design @ coeff
            rng = np.random.default_rng(int(hashlib.sha256(obj['key'].encode()).hexdigest()[:8],16))
            shuffled = values.copy()
            shuffled[visible] = values[visible][rng.permutation(visible.sum())]
            endpoints = dict(bg=bg, constant=bg+constant, affine=bg+affine,
                             dense=bg+values, shuffle=bg+shuffled)
            common = visible.copy()
            for q in endpoints.values():
                common &= (q[...,0]>=0)&(q[...,0]<width-1)&(q[...,1]>=0)&(q[...,1]<height-1)
            target = sample(current, p)
            errors = {k: np.abs(sample(previous, q)-target).mean(-1) for k,q in endpoints.items()}
            for region in ['all', 'inner_half']:
                mask = common.copy()
                if region == 'inner_half':
                    mask &= (xx>=4)&(xx<12)&(yy>=4)&(yy<12)
                if mask.sum() < 6:
                    continue
                row = dict(pair=report['pair'], camera=report['camera'], anchor=report['anchor'],
                           key=obj['key'], region=region, native_min_side=obj['native_min_side'],
                           size_bin=bin_name(obj['native_min_side']),
                           common_fraction=float(mask.sum() / (256 if region == 'all' else 64)))
                row.update({k+'_l1':float(e[mask].mean()) for k,e in errors.items()})
                row.update({k+'_minus_dense_l1':row[k+'_l1']-row['dense_l1'] for k in ['constant','affine','shuffle']})
                rows.append(row)
            if 'camera_intervention_residual_delta_pixels' in obj:
                camera_rows.append(dict(pair=report['pair'], size_bin=bin_name(obj['native_min_side']), **obj))
    grouped = {}
    for region in ['all','inner_half']:
        selected = [r for r in rows if r['region']==region]
        grouped[region] = {'pooled_descriptive':summary(selected),
                          'per_pair':{p:summary([r for r in selected if r['pair']==p]) for p in receipt['per_pair']},
                          'per_native_size':{b:summary([r for r in selected if r['size_bin']==b]) for b in ['lt16','16to32','ge32']}}
    def camera_summary(selected):
        selected = [o for o in selected if o['camera_intervention_residual_delta_pixels'] is not None]
        def med(key): return float(np.median([o[key] for o in selected])) if selected else None
        return {'objects':len(selected),
                **{key+'_median':med(key) for key in ['camera_intervention_common_fraction',
                   'camera_intervention_residual_delta_pixels','current_median_residual_pixels',
                   'current_spatial_std_pixels','background_holdout_p90_pixels']}}
    camera = {p:camera_summary([o for o in camera_rows if o['pair']==p]) for p in receipt['per_pair']}
    camera_sizes = {b:camera_summary([o for o in camera_rows if o['size_bin']==b]) for b in ['lt16','16to32','ge32']}
    output = {'status':'DESCRIPTIVE_OBSERVED_INTERVAL_INPUT_ANALYSIS',
              'feature_receipt_sha256':sha(receipt_path), 'source_sha256':sha(__file__),
              'verified_feature_files':len(receipt['files']), 'unique_images_read':len(image_refs),
              'arms':'background H; H+componentwise median residual; H+least-squares affine residual; H+dense residual; H+spatially permuted residual',
              'error':'RGB channel mean absolute difference in [0,1], evaluated on identical valid endpoints per object and region',
              'limits':['RAFT and all fitted residual controls see both reconstruction images: no held-out prediction or optical-flow ground truth.',
                        'Boxes are predicted observation regions, not foreground segmentation; inner-half is a diagnostic, not guaranteed foreground.',
                        '16x16 grid values and objects in the same scene are dependent; five pairs are the outer sampling units. No statistical significance claim.',
                        'Camera delta uses common masks but magnitude/std metadata use original visible masks; do not interpret their ratio as exact.',
                        'No identity labels, model training, new flow inference, calibration/dev or official val/test.'],
              'photometric':grouped, 'camera_intervention_per_pair':camera,
              'camera_intervention_per_native_size':camera_sizes,
              'background_per_interval':geom, 'image_hashes':image_refs}
    (HERE/'PHOTOMETRIC.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    with (HERE/'photometric_objects.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'photometric':{k:v['per_pair'] for k,v in grouped.items()},'camera':camera}))


if __name__ == '__main__':
    main()
