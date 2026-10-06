#!/usr/bin/env python3
"""P23 visual-arm calibration readout.

head_only vs visual_ft against the sealed P14 frozen_adapter reference, scored
with the frozen P4 exact-tie positive-present retrieval readout on cal5 only.

Progression rule (FROZEN.json):
  visual_ft cal pair-macro r1 >= head_only + 0.01 and >=3/5 pair wins
  -> keep as a stronger visual baseline; otherwise HOLD + implementation look-back.

Also runs a byte-consistency probe: re-encode the cal5 crops with the *pretrained*
backbone and compare to the frozen `unit_dino` the P4 cache already stores. If the
probe fails, the crop/preprocess pipeline differs and the visual_ft number is void.
"""
import os
GPU = 'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['XFORMERS_DISABLED'] = '1'
import hashlib
import json
import sys
sys.dont_write_bytecode = True
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
sys.path[:0] = [str(OLD / 'p2/src'), str(OLD / 'p4_diagnosis'), str(OLD / 'p14_fusion'), str(HERE)]
from data import split_assignments, sha256            # noqa: E402
import compare_crop32 as p4                            # noqa: E402
from extract_dino_crop32 import decode_image           # noqa: E402
import train_visual as TV                              # noqa: E402

import torch                                           # noqa: E402

CAL = ['27', '32', '42', '64', '65']
CACHE = OLD / 'p4_diagnosis/crop32'
EVAL_V2 = OLD / 'p14_fusion/evaluation_v2'
ARMS = ('frozen_adapter', 'head_only', 'visual_ft')
R1 = ('all_candidates', 'known_positive_r1_tie_averaged')


def ref(path):
    return {'path': str(path), 'sha256': sha256(path)}


def build_crops(pair):
    with np.load(CACHE / (pair + '.npz'), allow_pickle=False) as a:
        keys = a['keys']
        box = a['crop_box_xyxy']
        iref = a['image_ref']
        valid = a['crop_valid']
    images = {r['image_ref']: r for r in json.loads((CACHE / (pair + '_images.json')).read_text())}
    n = len(keys)
    crops = np.zeros((n, 3, 224, 224), np.uint8)
    order = np.argsort(iref, kind='stable')
    tasks, cur, idx, rows = [], None, [], []
    for i in order:
        r = str(iref[i])
        if r != cur:
            if idx:
                tasks.append((images[cur], idx, rows))
            cur, idx, rows = r, [], []
        idx.append(int(i))
        rows.append({'bbox': [float(x) for x in box[i]]})
    if idx:
        tasks.append((images[cur], idx, rows))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for indices, vals, bounds, oks, src in pool.map(decode_image, tasks):
            for k, v, ok in zip(indices, vals, oks):
                if not ok:
                    raise ValueError('cal5 crop unexpectedly invalid: ' + pair)
                crops[k] = v
    if not valid.all():
        raise ValueError('crop_valid false on cal5: ' + pair)
    return crops


def load_models():
    models = {}
    for arm in ('head_only', 'visual_ft'):
        ckpt_path = HERE / 'runs' / arm / 'fixed_step_001200.pt'
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['arm'] != arm or ckpt['step'] != 1200:
            raise ValueError('checkpoint lineage differs: ' + arm)
        model = TV.VisualModel(arm, with_backbone=True)
        model.head.load_state_dict(ckpt['head_state'], strict=True)
        if ckpt['backbone_state'] is not None:
            model.backbone.load_state_dict(ckpt['backbone_state'], strict=True)
        models[arm] = model.eval().cuda()
    return models


def encode(model, fpn, rgb):
    out = np.zeros((len(fpn), 128), np.float32)
    with torch.inference_mode():
        for s in range(0, len(fpn), 2048):
            sel = slice(s, s + 2048)
            e = model(torch.from_numpy(fpn[sel]).cuda(),
                      rgb=torch.from_numpy(rgb[sel]).cuda())
            out[sel] = e.float().cpu().numpy()
    return out


def main():
    torch.set_num_threads(4)
    models = load_models()
    cache_manifest = json.loads((OLD / 'p2/cache/MANIFEST.json').read_text())
    inputs = {r['sequence']: r for r in cache_manifest['sequences']}
    frame_manifest = json.loads((CACHE / 'FRAME_MANIFEST.json').read_text())
    records = [r for r in frame_manifest['pairs'] if r['pair'] in CAL and r['role'] == 'calibration']
    if sorted(r['pair'] for r in records) != CAL:
        raise ValueError('cal5 frame manifest mismatch')

    per_pair, probe = {}, {}
    for rec in records:
        pair = rec['pair']
        row, features, sources, coverage = p4.align_pair(rec, CACHE, inputs)
        with np.load(EVAL_V2 / (pair + '_embeddings.npz'), allow_pickle=False) as ev:
            if not np.array_equal(row['row_key'], ev['row_key']):
                raise ValueError('row_key differs from sealed P14 evaluation: ' + pair)
            frozen_adapter = ev['frozen_adapter']
        common = row['row_common_support']
        crops = build_crops(pair)
        # byte-consistency probe: the pretrained backbone must reproduce the frozen DINO table
        with torch.inference_mode():
            sel = np.flatnonzero(common)
            dino_repro = np.zeros((len(row['row_key']), 384), np.float32)
            for s in range(0, len(sel), 2048):
                chunk = sel[s:s + 2048]
                dino_repro[chunk] = models['head_only'].encode_rgb(
                    torch.from_numpy(crops[chunk]).cuda()).float().cpu().numpy()
        cos = (dino_repro[common] * features['dino384'][common]).sum(1)
        probe[pair] = {'min_cos': float(cos.min()), 'mean_cos': float(cos.mean()),
                       'n': int(common.sum())}
        head_only = encode(models['head_only'], features['fpn256'], crops)
        visual_ft = encode(models['visual_ft'], features['fpn256'], crops)
        feats = {'frozen_adapter': frozen_adapter, 'head_only': head_only, 'visual_ft': visual_ft}
        p4.METHODS = ARMS
        arrays = p4.evaluate_arrays(row, feats)
        rep = p4.summaries(arrays)
        per_pair[pair] = {'role': rec['role'], 'coverage': coverage,
                          'r1': {a: rep['all']['methods'][a][R1[0]][R1[1]] for a in ARMS}}
        print(json.dumps({'pair': pair, 'r1': per_pair[pair]['r1'],
                          'probe_min_cos': probe[pair]['min_cos']}), flush=True)

    macro = {a: float(np.mean([per_pair[p]['r1'][a] for p in CAL])) for a in ARMS}
    wins = sum(1 for p in CAL if per_pair[p]['r1']['visual_ft'] > per_pair[p]['r1']['head_only'])
    result = {
        'status': 'P23_VISUAL_CAL5_READOUT',
        'readout': 'P4 exact-tie positive-present r1 on cal5 only',
        'arms': list(ARMS), 'cal5': CAL,
        'pair_macro_r1': macro,
        'visual_ft_vs_head_only_lift': macro['visual_ft'] - macro['head_only'],
        'visual_ft_vs_head_only_pair_wins_of_5': wins,
        'progression_rule': 'visual_ft >= head_only+0.01 and >=3/5 wins -> keep; else HOLD+look-back',
        'progression_verdict': ('KEEP_STRONGER_VISUAL_BASELINE'
                                if macro['visual_ft'] >= macro['head_only'] + 0.01 and wins >= 3
                                else 'HOLD_VISUAL_FT_NOT_ABOVE_HEAD_ONLY'),
        'dino_reproduction_probe': probe,
        'per_pair': per_pair,
        'frozen_adapter_reference_is_sealed_p14_evaluation_v2': True,
        'dev_val_test_read': False,
        'GPU2_used': False,
        'sources': {'p23_head_only': ref(HERE / 'runs/head_only/fixed_step_001200.pt'),
                    'p23_visual_ft': ref(HERE / 'runs/visual_ft/fixed_step_001200.pt'),
                    'readout': ref(OLD / 'p4_diagnosis/compare_crop32.py'),
                    'frozen_p14_eval_dir': str(EVAL_V2)},
    }
    out = HERE / 'CAL5_READOUT.json'
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('pair_macro_r1', 'visual_ft_vs_head_only_lift',
                                             'visual_ft_vs_head_only_pair_wins_of_5',
                                             'progression_verdict')}, indent=1))


if __name__ == '__main__':
    main()
