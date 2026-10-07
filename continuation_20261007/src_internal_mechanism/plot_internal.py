#!/usr/bin/env python3
"""Render internal SRC evidence, not sequence effects or headline MOT scores."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
OUT = Path('/home/chenhc/src_egia_icme_paper/figures/src_internal_mechanism_v3')
sys.path.insert(0, '/home/chenhc/.codex/skills/scipilot-figure-skill/scripts')
from visual_qa import audit_layout
INK, MUTED, BLUE, ORANGE, TEAL, GRAY = '#223645', '#627182', '#0072B2', '#D55E00', '#009E73', '#737373'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.labelsize': 7.5, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
    'axes.linewidth': .6, 'text.color': INK, 'axes.labelcolor': INK,
    'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})


def load_csv(name):
    return list(csv.DictReader((HERE / name).open()))


def axes_style(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for name in ['bottom', 'left']:
        ax.spines[name].set_color('#8795A2')
    ax.tick_params(length=2.5, width=.6)
    ax.grid(axis='y', color='#E3E8ED', linewidth=.5)
    ax.set_axisbelow(True)


def make_cost(r):
    fig = plt.figure(figsize=(3.5, 5.3))
    fig.text(.065, .973, 'Measured selectivity of SRC costs', fontsize=9, fontweight='bold', va='top')
    fig.text(.065, .940, 'Frozen calibration states; identical candidate edges', fontsize=7.1, color=MUTED, va='top')
    styles = {
        'same_owner': (BLUE, '-', 'Same owner'),
        'different_owner_same_group': (GRAY, '--', 'Different owner, same group'),
        'different_owner_cross_group': (ORANGE, ':', 'Different owner, cross group')}
    handles = [Line2D([0], [0], color=color, ls=style, lw=1.5,
        label=f"{label}  (n = {r['cost_summary'][kind]['n']:,})")
        for kind, (color, style, label) in styles.items()]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.14, .918),
        frameon=False, fontsize=7.0, handlelength=2.2, labelspacing=.4, borderaxespad=0)
    data = load_csv('COST_CDF.csv')
    for y, operator, title in [(.555, 'base', '(a) Before SRC'), (.225, 'src', '(b) After SRC')]:
        ax = fig.add_axes([.18, y, .77, .230])
        axes_style(ax)
        fig.text(.18, y + .245, title, fontsize=8, fontweight='bold')
        for kind, (color, style, label) in styles.items():
            rows = [d for d in data if d['kind'] == kind and d['operator'] == operator]
            ax.step([float(d['cost']) for d in rows], [float(d['cdf']) for d in rows],
                where='post', color=color, ls=style, lw=1.3)
        ax.axvline(.8, color=INK, ls=(0, (4, 3)), lw=.75)
        ax.set_xlim(0, 1.01); ax.set_ylim(-.015, 1.025)
        ax.set_xticks([0, .2, .4, .6, .8, 1]); ax.set_yticks([0, .5, 1])
        ax.set_ylabel('Fraction at or below cost', labelpad=3)
        ax.set_xlabel('Association cost', labelpad=3)
        ax.text(.78, .08, r'$\tau = 0.80$', ha='right', va='bottom', fontsize=7.0,
            transform=ax.get_xaxis_transform(), color=MUTED)
    kept = r['cost_summary']['same_owner']['remaining_eligible_fraction'] * 100
    pruned = r['cost_summary']['different_owner_cross_group']['ineligible_fraction'] * 100
    fig.text(.065, .124, f'{kept:.1f}% of same-owner edges remain eligible.', fontsize=7.6, color=BLUE)
    fig.text(.065, .091, f'{pruned:.1f}% of cross-group mismatches become ineligible.', fontsize=7.6, color=ORANGE)
    fig.text(.065, .045, 'All edges were legal before SRC; groups are human / vehicle.', fontsize=7.0, color=MUTED)
    fig.text(.065, .018, 'Operator replay on recorded states; no new assignment is executed.', fontsize=6.6, color=MUTED)
    return fig


def make_calibration(r):
    fig = plt.figure(figsize=(3.5, 3.7))
    fig.text(.065, .973, 'Reliability of SRC semantic evidence', fontsize=9, fontweight='bold', va='top')
    fig.text(.065, .925, '26,088 frozen calibration observations', fontsize=7.4, color=MUTED, va='top')
    ax = fig.add_axes([.18, .29, .77, .53]); axes_style(ax)
    ax.plot([0, 1], [0, 1], color=GRAY, ls=(0, (3, 3)), lw=.9)
    data = load_csv('SEMANTIC_RELIABILITY.csv')
    styles = [('raw', GRAY, 's', 'Raw group confidence'), ('learned', TEAL, 'o', 'Learned SRC evidence')]
    for model, color, marker, name in styles:
        rows = [d for d in data if d['model'] == model]
        count = np.array([int(d['n']) for d in rows])
        # Every nonempty bin is retained; marker area is proportional to sqrt(n).
        ax.scatter([float(d['mean_predicted_human']) for d in rows],
            [float(d['observed_human_fraction']) for d in rows],
            s=5 + .35 * np.sqrt(count), color=color, marker=marker,
            edgecolors='white', linewidths=.4, zorder=3)
    ax.set_xlim(-.015, 1.025); ax.set_ylim(-.015, 1.025)
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlabel('Predicted human-group probability', labelpad=3)
    ax.set_ylabel('Observed human-group fraction', labelpad=3)
    handles = [Line2D([0], [0], color=color, marker=marker, ls='none', markersize=4, label=name)
        for model, color, marker, name in styles]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.165, .900),
        frameon=False, fontsize=7.2, ncol=1, labelspacing=.35, borderaxespad=0)
    m = r['calibration_observation_metrics']
    fig.text(.065, .175, f"NLL: {m['raw']['nll']:.3f} → {m['learned']['nll']:.3f}", fontsize=8, color=TEAL)
    fig.text(.53, .175, f"Brier: {m['raw']['brier_two_class']:.3f} → {m['learned']['brier_two_class']:.3f}", fontsize=8, color=TEAL)
    fig.text(.065, .108, 'Ten fixed probability bins; marker size reflects bin support.', fontsize=6.9, color=MUTED)
    fig.text(.065, .066, 'Every nonempty bin is shown; estimates are descriptive.', fontsize=6.9, color=MUTED)
    fig.text(.065, .025, 'Same frozen SRC head used for cost evaluation; no refitting.', fontsize=6.9, color=MUTED)
    return fig


def text_audit(fig):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside = []
    for t in texts:
        b = t.get_window_extent(renderer)
        if b.x0 < canvas.x0 - 1 or b.y0 < canvas.y0 - 1 or b.x1 > canvas.x1 + 1 or b.y1 > canvas.y1 + 1:
            outside.append(t.get_text())
    # Axis tick labels include duplicates that belong to hidden tick sides;
    # only visible labels above are considered. Check every positive overlap.
    overlapping = []
    for i, a in enumerate(texts):
        aa = a.get_window_extent(renderer)
        for b in texts[i+1:]:
            bb = b.get_window_extent(renderer)
            if min(aa.x1, bb.x1) - max(aa.x0, bb.x0) > 1 and min(aa.y1, bb.y1) - max(aa.y0, bb.y0) > 1:
                overlapping.append([a.get_text(), b.get_text()])
    return {'outside_canvas': outside, 'overlapping_text': overlapping,
        'minimum_visible_font_pt': min(t.get_fontsize() for t in texts), 'text_count': len(texts)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--export', action='store_true'); a = ap.parse_args()
    r = json.loads((HERE / 'ANALYSIS_RECEIPT.json').read_text())
    audit = json.loads((HERE / 'INDEPENDENT_AUDIT.json').read_text())
    assert r['status'] == 'COMPLETE_FROZEN_CALIBRATION_INTERNAL_DIAGNOSTIC'
    assert audit['status'] == 'PASS_INDEPENDENT_INTERNAL_MECHANISM_AUDIT'
    assert audit['analysis_receipt_sha256'] == hashlib.sha256((HERE / 'ANALYSIS_RECEIPT.json').read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, maker in [('src_cost_selectivity', make_cost), ('src_semantic_evidence_reliability', make_calibration)]:
        fig = maker(r)
        fig.savefig(OUT / (name + '_preview.png'), dpi=300, facecolor='white')
        with Image.open(OUT / (name + '_preview.png')) as im:
            ImageOps.grayscale(im).save(OUT / (name + '_grayscale.png'), dpi=(300, 300))
        checked = text_audit(fig)
        checked['skill_audit'] = audit_layout(fig)
        results[name] = checked
        if a.export:
            assert not checked['outside_canvas'], checked
            assert not checked['overlapping_text'], checked
            assert checked['minimum_visible_font_pt'] >= 6.5
            for suffix in ['pdf', 'svg', 'png']:
                fig.savefig(OUT / (name + '.' + suffix), dpi=600, facecolor='white')
        plt.close(fig)
    record = {'status': 'PASS_VECTOR_EXPORT' if a.export else 'PREVIEW_ONLY',
        'figures': results, 'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'analysis_receipt_sha256': hashlib.sha256((HERE / 'ANALYSIS_RECEIPT.json').read_bytes()).hexdigest(),
        'output_root': str(OUT)}
    (HERE / 'FIGURE_QA.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
