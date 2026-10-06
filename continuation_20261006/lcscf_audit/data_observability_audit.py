#!/usr/bin/env python3
import gzip,json,glob,hashlib,datetime
from pathlib import Path
root=Path('/home/chenhc/mdmot_sci_id_20260906/data/train_episodes_v1')
rows=[]
for f in sorted(root.glob('*.jsonl.gz')):
    with gzip.open(f,'rt') as h:
        for line in h:
            if line.strip(): rows.append(json.loads(line))
counts={
 'events':len(rows),
 'supervision_valid':sum(bool(r.get('supervision_valid')) for r in rows),
 'positive_events':sum(bool(r.get('same_id_candidate_indices')) for r in rows),
 'positive_candidate_members':sum(len(r.get('same_id_candidate_indices',[])) for r in rows),
 'multi_positive_events':sum(len(r.get('same_id_candidate_indices',[]))>1 for r in rows),
 'zero_candidate_events':sum(len(r.get('candidates',[]))==0 for r in rows),
 'known_negative_events':sum(bool(r.get('known_negative_indices')) for r in rows),
 'unmapped_events':sum(bool(r.get('unmapped_candidate_indices')) for r in rows),
 'dense_correspondence_fields':sum(any(k in r for k in ('homography','pixel_correspondence','flow','keypoints','visibility_mask','occlusion_mask')) for r in rows),
 'source_target_image_pairs':len(set((r['target_image'],r['source_image']) for r in rows)),
}
fields=sorted(set().union(*(r.keys() for r in rows)))
payload={
 'status':'HOLD_LCSCF_DENSE_SUPERVISION_NOT_OBSERVABLE',
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'manifest':str(root/'manifest.json'),
 'manifest_sha256':hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
 'counts':counts,
 'available_fields':fields,
 'missing_contracts':[
  'no dense target-source correspondence labels',
  'no homography/pose/flow field in episode rows',
  'no per-pixel occlusion/visibility mask',
 ],
 'legal_supervision':[
  'pair-level same-ID candidate indices',
  'known negatives and no-match rows',
  'target/source bounding boxes and frame paths',
 ],
 'interpretation':'A learned offset field can be implemented, but its intended local correspondence/cycle/occlusion effect is not directly observable from the frozen train episode contract. Pair-level association loss alone would reduce to ordinary local matching and is insufficient authorization for a paper method.'
}
Path('continuation_20261006/lcscf_audit/LCSCF_DATA_OBSERVABILITY_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
