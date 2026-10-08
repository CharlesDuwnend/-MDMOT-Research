"""Plot complete, verified conditional contrasts; no sequence rankings."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/completed_mechanisms'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(fig, name):
    for extension in ['pdf', 'svg', 'png']:
        fig.savefig(OUT / (name + '.' + extension), dpi=300,
                    facecolor='white', metadata={'Creator': 'Matplotlib'}
                    if extension == 'pdf' else None)
    with Image.open(OUT / (name + '.png')) as im:
        ImageOps.grayscale(im).save(OUT / (name + '_grayscale.png'))
    plt.close(fig)


def score_grid(ax, names, metrics, title, horizontal, vertical, captions):
    ax.set_xlim(-.55, 6.05)
    ax.set_ylim(.05, 6.05)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=9, pad=11)
    # Neutral cream/blue panels echo manuscript Fig.4; color encodes no metric.
    for row in range(2):
        for col in range(2):
            cx, cy = [1.15, 4.45][col], [1.35, 4.45][row]
            name = names[row][col]
            m = metrics[name]
            ax.add_patch(FancyBboxPatch(
                (cx - 1.10, cy - .75), 2.20, 1.50,
                boxstyle='round,pad=0.02,rounding_size=0.06',
                facecolor='#FBF8F4' if not row else '#DBDCEA',
                edgecolor='#90A9C2', linewidth=.75))
            ax.text(cx, cy + .43, captions[row][col], ha='center',
                    va='center', fontsize=7.8, color='#58636F')
            ax.text(cx, cy - .05, '%.4f / %.4f' %
                    (100 * m['mota'], 100 * m['idf1']),
                    ha='center', va='center', fontsize=8.2)
    for row in range(2):
        cy = [1.35, 4.45][row]
        a, b = metrics[names[row][1]], metrics[names[row][0]]
        ax.annotate('', xy=(3.33, cy), xytext=(2.28, cy),
                    arrowprops={'arrowstyle': '->', 'lw': .8, 'color': '#58636F'})
        ax.text(2.8, cy + .31, '%+.4f\n%+.4f' %
                (100 * (a['mota'] - b['mota']),
                 100 * (a['idf1'] - b['idf1'])),
                ha='center', va='bottom', fontsize=7.2)
    for col in range(2):
        cx = [1.15, 4.45][col]
        a, b = metrics[names[1][col]], metrics[names[0][col]]
        ax.annotate('', xy=(cx, 3.63), xytext=(cx, 2.17),
                    arrowprops={'arrowstyle': '->', 'lw': .8, 'color': '#58636F'})
        ax.text(cx, 2.89, '%+.4f\n%+.4f' %
                (100 * (a['mota'] - b['mota']),
                 100 * (a['idf1'] - b['idf1'])),
                ha='center', va='center', fontsize=7.2,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2})
    for col in range(2):
        ax.text([1.15, 4.45][col], .30,
                horizontal + (': off' if not col else ': on'),
                ha='center', va='center', fontsize=7.7)
    for row in range(2):
        ax.text(-.34, [1.35, 4.45][row],
                vertical + (': off' if not row else ': on'),
                ha='center', va='center', rotation=90, fontsize=7.7)
    ax.text(2.8, 5.72, 'Cells: MOTA / IDF1 (%)', ha='center', fontsize=7.8)


def main():
    s = json.loads((ROOT / 'artifacts/mechanism_summary.json').read_text())
    assert s['status'] == 'COMPLETE_VERIFIED_CORRECTED_METADATA_MECHANISM_ANALYSIS'
    assert not s['selection_performed'] and len(s['corrected_contrasts']) == 10
    m = {}
    for arm, evidence in s['corrected_provenance'].items():
        assert sha(evidence['path']) == evidence['sha256']
        m[arm] = json.loads(Path(evidence['path']).read_text())['metrics']['OVERALL']
    assert len(m) == 9 and {x['num_objects'] for x in m.values()} == {229506}
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8, 'text.color': '#111111',
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6,
        'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.32))
    fig.subplots_adjust(left=.035, right=.995, top=.88, bottom=.07, wspace=.13)
    score_grid(axes[0],
               [['C00_nativecost_reference', 'C10_mask_only'],
                ['C01_penalty_only', 'C11_full_reference']], m,
               '(a) SRC assignment operators', 'Appearance mask', 'Penalty',
               [['Neither', 'Mask only'], ['Penalty only', 'Mask + penalty']])
    score_grid(axes[1],
               [['E00_fusion_off_no_selective', 'E10_fusion_no_selective'],
                ['E01_fusion_off_selective', 'C11_full_reference']], m,
               '(b) EGIA birth policies', 'Fusion', 'Selective',
               [['Neither', 'Fusion only'], ['Selective only', 'Fusion + selective']])
    fig.text(.5, .04, 'Arrows: conditional change in MOTA / IDF1 (percentage points)',
             ha='center', fontsize=7.8, color='#58636F')
    export(fig, 'conditional_operator_grids')

    rows = [
        ('mask_without_penalty', 'Mask | penalty off'),
        ('mask_with_penalty', 'Mask | penalty on'),
        ('penalty_without_mask', 'Penalty | mask off'),
        ('penalty_with_mask', 'Penalty | mask on'),
        ('fusion_without_selective', 'Fusion | selective off'),
        ('fusion_with_selective', 'Fusion | selective on'),
        ('selective_without_fusion', 'Selective | fusion off'),
        ('selective_with_fusion', 'Selective | fusion on'),
        ('paired_hierarchy_minus_flat', 'Hierarchical − flat'),
    ]
    counts = ['num_false_positives', 'num_misses', 'num_switches']
    values, annotations, csv_rows = [], [], []
    for key, label in rows:
        d = s['corrected_contrasts'][key]
        v = [-100 * d[k] / 229506 for k in counts] + [d['mota_pp'], d['idf1_pp']]
        assert np.isclose(sum(v[:3]), v[3], rtol=0, atol=2e-14)
        values.append(v)
        annotations.append(['%+.3f\n(%+d)' % (v[i], -d[k])
                            for i, k in enumerate(counts)] +
                           ['%+.4f' % x for x in v[3:]])
        csv_rows.append({'contrast': key, **{k: v[i] for i, k in enumerate(
            ['fp_contribution_pp', 'fn_contribution_pp', 'ids_contribution_pp',
             'mota_delta_pp', 'idf1_delta_pp'])}})
    with (OUT / 'error_contribution_data.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0])); w.writeheader(); w.writerows(csv_rows)
    # Blue/white/ochre diverging map; signed numbers also remain legible in gray.
    cmap = LinearSegmentedColormap.from_list('signed_effects',
                                            ['#DCC0A8', '#FFFFFF', '#ACC0D4'])
    limit = max(abs(x) for row in values for x in row)
    fig, ax = plt.subplots(figsize=(7.16, 4.85))
    fig.subplots_adjust(left=.265, right=.995, bottom=.23, top=.85)
    im = ax.imshow(values, cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
                   aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels(['FP term', 'FN term', 'IDs term',
                                                'ΔMOTA', 'ΔIDF1'])
    ax.set_yticks(range(9)); ax.set_yticklabels([label for _, label in rows])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=7)
    for i in range(9):
        for j in range(5):
            ax.text(j, i, annotations[i][j], ha='center', va='center', fontsize=7.5)
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 9, 1), minor=True)
    ax.grid(which='minor', color='#C8CDD3', linewidth=.45)
    ax.tick_params(which='minor', length=0)
    for y in [3.5, 7.5]: ax.axhline(y, color='#58636F', linewidth=1.05)
    fig.text(.265, .96, 'Error tradeoffs of the conditional interventions',
             ha='left', va='top', fontsize=9)
    cax = fig.add_axes([.44, .16, .46, .016])
    bar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-2, 0, 2])
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    bar.outline.set_linewidth(.5)
    fig.text(.5, .055, 'All values: percentage points; positive means improvement.\n'
             'FP / FN / IDs cells: MOTA contribution (errors removed).',
             ha='center', fontsize=7.8)
    export(fig, 'conditional_error_tradeoffs')
    caption = """# Completed mechanism figure captions

**Conditional operator grids.** (a) Appearance-class masking and semantic
penalty are toggled while SRC memory, its original responsibility feature
definition and EGIA remain fixed. (b) Fusion and selective birth policies are
toggled while source heads, SRC, coherence and other settings remain fixed.
The top-right cells share C11. Cells show pooled MOTA/IDF1; arrows show all
conditional differences in percentage points. Background colors are decorative,
not a cost or performance scale. All nine runs cover the same 17 sequences and
6635 frames with identical ordered detector/ReID inputs. These are conditional
closed-loop effects at the frozen operating point, without significance claims.
C00 is a native-assignment-cost control retaining SRC memory and EGIA.

**Conditional error tradeoffs.** All four conditional contrasts in each
factorial and the paired hierarchy-minus-flat comparison are retained. FP, FN
and ID-switch cells show favorable MOTA contributions
100*(errors_control-errors_treatment)/229506 and errors removed in parentheses.
The three terms sum to ΔMOTA; IDF1 has its separate definition. Blue/ochre denote
positive/negative values on one shared symmetric scale. Zeros and negative
results are retained. The paired heads share the predeclared fit rows, scaler,
event weights and optimizer settings; their parameterization and regularization
geometry differ. Historical metadata-contaminated runs enter neither figure.

The first figure is the compact main-text candidate; the second explains its
error tradeoffs and retains the negative head result. These are author-review
artifacts, exported at IEEE double-column width; the protected manuscript has
not been modified. No actual per-edge cost matrix can be reconstructed from
the current aggregate runtime logs, so these plots do not invent such a case.
"""
    (OUT / 'FIGURE_CAPTIONS.md').write_text(caption)
    paths = [p for p in OUT.iterdir() if p.is_file()
             and p.name != 'FIGURE_RECEIPT.json' and '_pdf_render' not in p.name]
    receipt = {'status': 'GENERATED_FROM_COMPLETE_VERIFIED_NINE_ARM_RESULTS',
               'manifest_sha256': s['manifest_sha256'],
               'summary_sha256': sha(ROOT / 'artifacts/mechanism_summary.json'),
               'builder_sha256': sha(__file__), 'all_conditional_contrasts_retained': True,
               'paper_modified': False, 'files_sha256': {str(p): sha(p) for p in sorted(paths)}}
    (OUT / 'FIGURE_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt['status'])


if __name__ == '__main__':
    main()
