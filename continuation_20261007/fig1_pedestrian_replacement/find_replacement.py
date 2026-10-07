#!/usr/bin/env python3
"""Inspect real pedestrian duplicate-output alternatives for Figure 1."""
from pathlib import Path
import importlib.util
import json
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/chenhc/src_egia_icme_paper')
OUT = PAPER / 'output/pdf/fig1_pedestrian_replacement_20261007'
SCRIPT = PAPER / 'scripts/build_paper_case_figures.py'


def main():
    spec = importlib.util.spec_from_file_location('real_figure_cases', SCRIPT)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    catalogue = HERE.parent / 'motivation_duplicate_analysis/native_duplicate_candidates.json'
    candidates = json.loads(catalogue.read_text())
    runs = [c for c in candidates if c['category'] == 'pedestrian'
            and not c['sequence'].startswith('uav0000297_')
            and c['sequence'] not in ['uav0000073_00600_v', 'uav0000120_04775_v']]
    qualified = []
    sources = {str(catalogue): base.sha(catalogue), str(SCRIPT): base.sha(SCRIPT)}
    cache = {}
    for seq in sorted(set(c['sequence'] for c in runs)):
        paths = base.inputs(seq)
        ours, native, gt = map(base.load, paths)
        cache[seq] = tuple(base.framed(x) for x in [ours, native, gt])
        sources.update({str(p): base.sha(p) for p in paths})
    for run in runs:
        of, nf, gf = cache[run['sequence']]
        for frame in range(run['start_frame'], run['end_frame'] + 1):
            g = gf[frame]
            target = g[g[:, 1] == run['gt_id']][0]
            if int(target[6]) != 1 or int(target[7]) != 1:
                continue
            matching = []
            other = g[(g[:, 1] != run['gt_id']) & (g[:, 7] > 0)]
            for rows in [nf.get(frame, np.empty((0, 10))), of.get(frame, np.empty((0, 10)))]:
                ov = base.iou(target[None, 2:6], rows[:, 2:6])[0]
                rr = rows[ov >= .5]
                if len(rr) == 0:
                    break
                matching.append((rr, ov[ov >= .5]))
            if len(matching) != 2 or [len(v[0]) for v in matching] != [2, 1]:
                continue
            all_rows = np.vstack([m[0] for m in matching])
            minimum = float(min(v.min() for _, v in matching))
            other_max = float(base.iou(all_rows[:, 2:6], other[:, 2:6]).max(initial=0))
            if minimum < .6 or other_max >= .3 or not np.all(all_rows[:, 7] == 1):
                continue
            photo = base.DATA / 'sequences' / run['sequence'] / f'{frame:07d}.jpg'
            crop = base.crop_target(photo, target, 220, 236)
            if not all(r[2] >= crop[0] and r[3] >= crop[1]
                       and r[2]+r[4] <= crop[2] and r[3]+r[5] <= crop[3] for r in all_rows):
                continue
            qualified.append({**run, 'frame': frame, 'target_row': target.tolist(),
                              'native_ids': sorted(int(r[1]) for r in matching[0][0]),
                              'leaf_ids': sorted(int(r[1]) for r in matching[1][0]),
                              'native_rows': matching[0][0].tolist(), 'leaf_rows': matching[1][0].tolist(),
                              'minimum_target_iou': minimum, 'max_other_object_iou': other_max,
                              'crop_xyxy': list(crop), 'photo': str(photo), 'photo_sha256': base.sha(photo),
                              'score': float(target[4]*target[5]*minimum)})
    qualified.sort(key=lambda c: c['score'], reverse=True)
    # Diversify target identities, rather than showing adjacent frames of one target.
    distinct = []
    seen = set()
    for c in qualified:
        key = (c['sequence'], c['gt_id'])
        if key not in seen:
            seen.add(key)
            distinct.append(c)
        if len(distinct) == 12:
            break
    assert distinct, 'No verified replacement candidates.'
    OUT.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    w, h = 444, 280
    sheet = Image.new('RGB', (3*w, ((len(distinct)+2)//3)*h), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, c in enumerate(distinct):
        x, y = (index % 3)*w, (index // 3)*h
        draw.text((x+3, y+3), f"#{index+1} {c['sequence']} / GT {c['gt_id']} / frame {c['frame']}", fill='black', font=font)
        draw.text((x+3, y+21), f"Native {c['native_ids']} / Ours {c['leaf_ids']} / min IoU {c['minimum_target_iou']:.3f}", fill='black', font=font)
        with Image.open(c['photo']) as im:
            crop = im.convert('RGB').crop(c['crop_xyxy'])
        for method, rows in enumerate([c['native_rows'], c['leaf_rows']]):
            panel = crop.copy()
            painter = ImageDraw.Draw(panel)
            for i, r in enumerate(rows):
                b = [r[2]-c['crop_xyxy'][0], r[3]-c['crop_xyxy'][1],
                     r[2]+r[4]-c['crop_xyxy'][0], r[3]+r[5]-c['crop_xyxy'][1]]
                painter.rectangle(b, outline=['#547A9A','#9383B5'][i] if method == 0 else '#CE8167', width=2)
            sheet.paste(panel, (x+method*224, y+42))
    sheet.save(OUT / 'candidate_review.png')
    (HERE / 'CANDIDATE_AUDIT.json').write_text(json.dumps({
        'status': 'CANDIDATES_PENDING_VISUAL_SELECTION',
        'existing_catalogue_runs_considered': len(runs), 'qualified_frames': len(qualified),
        'source_sha256': sources, 'preview_candidates': distinct,
        'selection_contract': 'Real test-dev outputs: Native two / Ours one, pedestrian class, all target IoUs >= 0.6, other-object IoU < 0.3, same original crop, boxes fully contained. Visual selection checks background text.'
    }, indent=2)+'\n')
    print(json.dumps({'qualified_frames': len(qualified), 'preview_candidates': len(distinct),
                      'preview': str(OUT/'candidate_review.png')}, indent=2))


if __name__ == '__main__':
    main()
