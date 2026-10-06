#!/usr/bin/env python3
"""Render the sealed FIT motion diagnostic as readable tables."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = json.loads((HERE / 'SUMMARY.json').read_text())
v = json.loads((HERE / 'VERIFY.json').read_text())
b = s['strata']['by_lag']
lines = [
    '**P28 FIT-only cached RAFT motion lookback — complete.**', '',
    'The cached current-to-past RAFT fields contain useful geometric motion on the fixed FIT cohort: '
    'center prediction is better than zero flow at lags 1, 4 and 8. Source direction, per-axis resize, '
    'halfpixel coordinates, border sampling and saved-result checks pass. This supports further '
    'implementation investigation; it does not explain the failed FGFA pilot or validate a new method.', '',
    '**Scope and cohort.** The diagnostic reads only FIT15 XML and the existing 240 FIT groups, each '
    'with cached fields at lags 1/4/8. No images, GPU, model loading, new inference or non-FIT GT. '
    'No P28 model, feature, flow, prediction, trainer, P26 or reference-workspace artifact is changed. '
    'Identity is `(pair, view, raw_local_id, class)`; the same supported class/local track must be annotated '
    'at both endpoints and every intermediate frame. This is a same-view annotation correspondence, '
    'not a cross-view/global identity assumption.', '',
    'FIT pairs: ' + ', '.join(s['fit_pairs']) + '. The 240 current frames contain 13,663 supported boxes; '
    'the three lags produce 40,420 comparisons. Occluded annotations are retained. '
    'The parser, manifest and FIT XML hashes match the pinned training inputs.', '',
    '| Lag | Eligible comparisons | Past endpoint absent | Interior gap |',
    '|---:|---:|---:|---:|',
]
for lag in ('1', '4', '8'):
    lines.append(f"| {lag} | {b[lag]['all']['n']:,} | {s['exclusions'].get('lag'+lag+'_past_endpoint_absent',0)} | {s['exclusions'].get('lag'+lag+'_interior_annotation_gap',0)} |")
lines += [
    '', '**Geometry contract.** Export calls `RAFT(current, past)`; torchvision correlates image1 '
    'against image2 and returns `coords1 - coords0`, in `(dx,dy)` order. Every flow archive is '
    '`[3,2,360,640]` for original images of `1080×1920`, with no padding. For an original continuous-edge '
    'box center `c`, the flow raster pixel index is `c*(Fw/W,Fh/H) - 0.5`. The predicted past center is '
    '`c + sampled_flow*(W/Fw,H/Fh)`, using bilinear sampling, border extension and `align_corners=False`.', '',
    'The known-translation test supplies a constant synthetic flow field and verifies projection sign '
    'and resize scale; it does not run RAFT on synthetic images. Zero flow, border extension, spatial '
    'ramps, an independent CPU `grid_sample` comparison and the gap/class guard also pass. At all real '
    'GT sample locations, NumPy64 versus Torch32 sampling differs by at most '
    f"{s['max_numpy64_vs_grid_sample32_displacement_component_difference_original_px']:.9f} original pixels per component "
    '(fixed numerical gate: ≤0.01 px). GT continuous-edge coordinates must remain distinct from '
    'feature/pixel-center indices. These checks do not establish feature-content alignment.', '',
    '**Observed center errors in original pixels.** Comparisons are annotation weighted. “Win” '
    'means strictly smaller error than zero flow for the same observation; it is not an AP/recall metric.', '',
    '| Lag | n | RAFT median | Zero median | RAFT mean | Zero mean | RAFT p90 | Zero p90 | Win |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for lag in ('1', '4', '8'):
    q = b[lag]['all']; r = q['raft_error_px']; z = q['zero_error_px']
    lines.append(f"| {lag} | {q['n']:,} | {r['median']:.3f} | {z['median']:.3f} | {r['mean']:.3f} | {z['mean']:.3f} | {r['p90']:.3f} | {z['p90']:.3f} | {100*q['raft_better_than_zero_fraction']:.2f}% |")
lines += [
    '', '**Two separate size normalizations.** Each error is divided by the square root of current '
    'box area or past box area, respectively. The denominators are never pooled or substituted.', '',
    '| Lag | Denominator | RAFT median | Zero median | RAFT p90 | Zero p90 |',
    '|---:|---|---:|---:|---:|---:|',
]
for lag in ('1', '4', '8'):
    for endpoint in ('current', 'past'):
        q = b[lag]['all']; r = q[f'raft_error_over_{endpoint}_sqrt_area']; z = q[f'zero_error_over_{endpoint}_sqrt_area']
        lines.append(f"| {lag} | {endpoint} sqrt(area) | {r['median']:.4f} | {z['median']:.4f} | {r['p90']:.4f} | {z['p90']:.4f} |")
lines += [
    '', '**Size strata, defined on the current box.** Min-side <16 and sqrt(area) <16 are different '
    'cohorts. Normalized columns below use current sqrt(area); the CSV also contains past normalization, '
    'mean errors and all complements/intersections.', '',
    '| Criterion | Lag | n | RAFT median px | RAFT norm median | RAFT norm p90 | Zero norm median | Win |',
    '|---|---:|---:|---:|---:|---:|---:|---:|',
]
for field, label in [('current_min_side_lt16', 'min-side'), ('current_sqrt_area_lt16', 'sqrt(area)')]:
    for small in ('True', 'False'):
        for lag in ('1', '4', '8'):
            q=b[lag][field][small]; r=q['raft_error_over_current_sqrt_area']; z=q['zero_error_over_current_sqrt_area']
            criterion=label + (' <16' if small=='True' else ' ≥16')
            lines.append(f"| {criterion} | {lag} | {q['n']:,} | {q['raft_error_px']['median']:.3f} | {r['median']:.4f} | {r['p90']:.4f} | {z['median']:.4f} | {100*q['raft_better_than_zero_fraction']:.2f}% |")
lines += [
    '', '**Current occlusion strata.** The annotation flag is descriptive. The occluded and unoccluded '
    'cohorts differ in size, scene and motion; these aggregates do not establish an occlusion effect. '
    'Past-endpoint and any-path occlusion plus size×occlusion intersections are in `STRATA.csv`.', '',
    '| Current occluded | Lag | n | RAFT median px | Zero median px | RAFT current-area norm p90 | Win |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for occ in ('0','1'):
    for lag in ('1','4','8'):
        q=b[lag]['current_occluded'][occ]
        lines.append(f"| {'yes' if occ=='1' else 'no'} | {lag} | {q['n']:,} | {q['raft_error_px']['median']:.3f} | {q['zero_error_px']['median']:.3f} | {q['raft_error_over_current_sqrt_area']['p90']:.4f} | {100*q['raft_better_than_zero_fraction']:.2f}% |")
lines += [
    '', '**Pair heterogeneity and domain flags.** All 15 FIT pairs have lower mean and median '
    'RAFT error than zero flow at each lag. This does not imply that every observation benefits: '
    'pair 78 at lag 1 has only 42.33% wins and a mean error reduction of 0.0348 px. At lags 4/8 all '
    '15 pairs have win fraction >50%. Pair/lag and class tables are retained in the CSV.', '',
    'Every current GT center is inside the image. One past GT center at lag 4 is outside; it is '
    'retained and flagged. RAFT endpoints outside the image number 0/22/54 at lags 1/4/8; they are '
    'also retained and flagged. No outlier or error-based exclusion is applied.', '',
    '**Bounded interpretation.** There is no evidence here for a gross exporter direction, original-pixel '
    'scale or halfpixel sampling error. The useful FIT geometric signal does not support a blanket '
    'STOP for temporal motion. However, long-lag tails remain material: for both small-object definitions, '
    'lag-8 normalized p90 is about 0.94 current sqrt(area). A GT bounding-box center is not necessarily '
    'a tracked physical surface point under deformation or occlusion. The analysis does not verify '
    'FPN feature lattice/content, resampling blur, correspondence under occlusion, learned weighting '
    'or optimization, nor causally explain the failed calibration pilot. It provides no new '
    'detection/tracking performance, validation selection, threshold/window recommendation or authorization '
    'to alter the trained system.', '',
    '**Verification and artifacts.** The producer completed with exit 0 in 23.754 s. Independent '
    f"saved-row verification completed with exit 0 in {v['elapsed_seconds']:.3f} s: {v['rows']:,} rows, "
    f"{v['groups']} groups and all {v['strata_checked']} published strata agree; all "
    f"{v['input_hashes_rechecked']} bound input hashes remain unchanged. Maximum independently "
    f"recomputed pixel-error difference is {v['max_abs_recompute_errors']['raft_error_px']:.3g} px. "
    'This second check reconstructs result arithmetic and count conservation; annotation eligibility '
    'is enforced by the producer and its contract test, not re-parsed by the saved-row verifier.', '',
    '- `PROTOCOL.md`, `audit_motion.py`: scope, eligibility and producer.',
    '- `DIRECTION_AUDIT.json`, `TESTS.json`: source contract and synthetic/numerical evidence.',
    '- `observations.npz`, `SUMMARY.json`, `STRATA.csv`: complete rows and all descriptive strata.',
    '- `verify_saved.py`, `VERIFY.json`, `verify.log`: independent saved-row checks.',
    '- `INPUTS.json`, `run.log`, `EXECUTION.json`, `RECEIPT.json`, `SHA256SUMS`: lineage and execution seal.',
    '',
]
(HERE/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps({'status':'REPORT_RENDERED','path':str(HERE/'REPORT.md')}))
