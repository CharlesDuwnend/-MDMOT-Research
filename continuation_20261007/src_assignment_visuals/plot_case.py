#!/usr/bin/env python3
"""A semantics-to-cost triptych and an assignment-link view of one real case."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle, Circle, PathPatch
from matplotlib.path import Path as MPath
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
OUT = Path('/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v4')
sys.path.insert(0, '/home/chenhc/.codex/skills/scipilot-figure-skill/scripts')
from visual_qa import audit_layout
INK, MUTED, HUMAN, VEHICLE, GOOD, CONFLICT = '#172B3A', '#607181', '#0072B2', '#E69F00', '#009E73', '#D55E00'
CMAP = plt.get_cmap('cividis')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.labelsize': 7.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .6})


def matching_matrix(fig, rect, matrix, pairs=None, case=None, forbidden=False):
    ax = fig.add_axes(rect)
    ax.set_xlim(-.5, 2.5); ax.set_ylim(2.5, -.5); ax.set_aspect('equal')
    ax.set_xticks([0, 1, 2], [r'$D_1$', r'$D_2$', r'$D_3$'])
    ax.set_yticks([0, 1, 2], [r'$T_1$', r'$T_2$', r'$T_3$'])
    ax.xaxis.tick_top(); ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(3):
        for j in range(3):
            value = float(matrix[i, j])
            ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, facecolor=CMAP(value),
                edgecolor='white', linewidth=.8))
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', fontsize=8,
                color='white' if value < .45 else INK)
            if forbidden and value > case['threshold']:
                ax.text(j+.34, i-.32, '×', ha='center', va='center', fontsize=6.8, color=INK)
    if pairs is not None:
        for i, j in pairs:
            same = case['track_labels'][i]['gt_id'] == case['detection_labels'][j]['gt_id']
            ax.add_patch(Rectangle((j-.445, i-.445), .89, .89, fill=False,
                edgecolor='white', linewidth=3.1, zorder=4))
            ax.add_patch(Rectangle((j-.445, i-.445), .89, .89, fill=False,
                edgecolor=GOOD if same else CONFLICT, linewidth=1.7,
                linestyle='-' if same else (0, (3, 1.5)), zorder=5))
    return ax


def semantic_glyphs(fig, case):
    ax = fig.add_axes([.038, .32, .235, .40])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.text(.265, 1.055, r'Stored $b_i$', ha='center', fontsize=7.4)
    ax.text(.775, 1.055, r'Incoming $z_j$', ha='center', fontsize=7.4)
    for i, y in enumerate([.82, .50, .18]):
        for x, labelx, label, prob in [(.10, .008, rf'$T_{i+1}$', case['beliefs'][i]),
                                      (.61, .515, rf'$D_{i+1}$', case['observation_probabilities'][i])]:
            ax.text(labelx, y, label, fontsize=7.5, va='center')
            ax.add_patch(Rectangle((x, y-.055), .33*prob[0], .11, facecolor=HUMAN, edgecolor='none'))
            ax.add_patch(Rectangle((x+.33*prob[0], y-.055), .33*prob[1], .11, facecolor=VEHICLE, edgecolor='none'))
            ax.add_patch(Rectangle((x, y-.055), .33, .11, fill=False, edgecolor='#8997A1', linewidth=.45))
            ax.text(x+.165, y-.10, f'H={prob[0]:.3f}', fontsize=6.8, ha='center', va='top')
    ax.add_patch(Rectangle((.09, -.10), .045, .055, facecolor=HUMAN, edgecolor='none', clip_on=False))
    ax.text(.15, -.074, 'Human', va='center', fontsize=7.0)
    ax.add_patch(Rectangle((.55, -.10), .045, .055, facecolor=VEHICLE, edgecolor='none', clip_on=False))
    ax.text(.61, -.074, 'Vehicle', va='center', fontsize=7.0)
    return ax


def triptych(record):
    c = record['case']
    fig = plt.figure(figsize=(7.16, 2.95))
    fig.text(.025, .983, 'SRC resolves ambiguity in a real candidate group', fontsize=9, va='top', fontweight='bold')
    fig.text(.145, .89, '(a) Semantic evidence', ha='center', fontsize=8.2)
    fig.text(.411, .89, '(b) Semantic disagreement', ha='center', fontsize=8.2)
    fig.text(.769, .89, '(c) Cost refinement', ha='center', fontsize=8.2)
    semantic_glyphs(fig, c)
    matching_matrix(fig, [.345, .32, .15, .37], np.array(c['semantic_penalty']))
    matching_matrix(fig, [.575, .32, .15, .37], np.array(c['base_cost']), c['base_assignment'], c)
    matching_matrix(fig, [.810, .32, .15, .37], np.array(c['src_cost']), c['src_assignment'], c, forbidden=True)
    fig.text(.420, .795, r'$u_{ij}=\mathrm{clip}(1-b_i^\top(z_j/\pi),0,1)$', fontsize=7.1, ha='center')
    fig.text(.650, .795, r'Base $C$', ha='center', fontsize=7.6)
    fig.text(.885, .795, r'Refined $\widetilde C$', ha='center', fontsize=7.6)
    fig.text(.763, .505, '→', fontsize=13, ha='center', va='center', color=MUTED)
    scalar = plt.cm.ScalarMappable(norm=colors.Normalize(0, 1), cmap=CMAP)
    for x, w, label in [(.345, .15, 'Disagreement'), (.575, .385, 'Cost; × means above 0.80')]:
        cb = fig.colorbar(scalar, cax=fig.add_axes([x, .226, w, .024]), orientation='horizontal', ticks=[0, .5, 1])
        cb.ax.tick_params(labelsize=6.8, length=2, pad=1)
        cb.outline.set_linewidth(.45); cb.set_label(label, fontsize=6.8, labelpad=1)
    fig.text(.042, .198, r'$\widetilde C=C+(1-C)u$', fontsize=8.0)
    fig.text(.042, .144, r'$\pi(H)=0.332;\ \pi(V)=0.668$', fontsize=6.8, color=MUTED)
    handles = [Line2D([0], [0], color=GOOD, lw=1.6, label='Selected, same owner'),
               Line2D([0], [0], color=CONFLICT, lw=1.6, ls='--', label='Selected, different owner')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.50, .075), frameon=False,
        fontsize=7.0, ncol=2, handlelength=1.8, columnspacing=1.8)
    fig.text(.025, .075, 'Two tied base optima → one unique SRC optimum', fontsize=8.0, color=GOOD)
    fig.text(.025, .023, 'Real calibration example; same candidates and fixed threshold.', fontsize=6.8, color=MUTED)
    return fig


def curve(ax, y0, y1, color, width, style='-', alpha=1):
    path = MPath([(0.24, y0), (.44, y0), (.56, y1), (.76, y1)],
                 [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(path, fill=False, color=color, linewidth=width, linestyle=style, alpha=alpha))


def assignment_panel(fig, position, case, pairs, matrix, show_alternative=False):
    ax = fig.add_axes(position); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    yy = [.81, .50, .19]
    if show_alternative:
        for i, j in case['src_assignment']:
            if [i, j] not in pairs:
                curve(ax, yy[i], yy[j], '#9AA5AE', .7, style=(0, (2, 2)), alpha=.7)
    for i, j in pairs:
        same = case['track_labels'][i]['gt_id'] == case['detection_labels'][j]['gt_id']
        curve(ax, yy[i], yy[j], GOOD if same else CONFLICT, 1.8, style='-' if same else (0, (3, 2)))
        # Place labels near track endpoints to avoid crossing-line congestion.
        ax.text(.34, yy[i]+.06, f'{matrix[i,j]:.3f}', fontsize=7.4, color=GOOD if same else CONFLICT,
            ha='center', bbox={'facecolor':'white','edgecolor':'none','pad':1.0})
    for i, y in enumerate(yy):
        ax.add_patch(Rectangle((.09, y-.06), .14, .12, facecolor='#F5F7F9', edgecolor=INK, linewidth=.65))
        ax.add_patch(Circle((.83, y), .07, facecolor='#F5F7F9', edgecolor=INK, linewidth=.65))
        ax.text(.16, y, rf'$T_{i+1}$', fontsize=8, ha='center', va='center')
        ax.text(.83, y, rf'$D_{i+1}$', fontsize=8, ha='center', va='center')
        tgroup = 'H' if case['beliefs'][i][0] >= .5 else 'V'
        dgroup = 'H' if case['observation_probabilities'][i][0] >= .5 else 'V'
        ax.text(.045, y, tgroup, color=HUMAN if tgroup == 'H' else VEHICLE, fontsize=7.2, ha='center', va='center')
        ax.text(.952, y, dgroup, color=HUMAN if dgroup == 'H' else VEHICLE, fontsize=7.2, ha='center', va='center')
    ax.text(.16, .985, 'Tracks', fontsize=7.3, ha='center')
    ax.text(.83, .985, 'Detections', fontsize=7.3, ha='center')
    return ax


def links(record):
    c = record['case']
    fig = plt.figure(figsize=(3.5, 4.75))
    fig.text(.06, .975, 'Semantic evidence resolves assignment ties', fontsize=8.5, fontweight='bold', va='top')
    fig.text(.06, .929, 'Same three tracks and detections; fixed threshold 0.80', fontsize=6.8, color=MUTED)
    fig.text(.06, .858, '(a) Base cost: two tied optima', fontsize=8.0)
    assignment_panel(fig, [.09, .60, .83, .23], c, c['base_assignment'], np.array(c['base_cost']), show_alternative=True)
    fig.text(.06, .55, 'Shown: the host LAP choice among equal-cost optima.', fontsize=6.7, color=MUTED)
    fig.text(.06, .467, '(b) SRC cost: one unique optimum', fontsize=8.0)
    assignment_panel(fig, [.09, .205, .83, .23], c, c['src_assignment'], np.array(c['src_cost']))
    fig.text(.06, .160, r'$T_2$–$D_3$: 0.013 → 0.989;  $T_3$–$D_2$: 0.031 → 0.992', fontsize=7.1, color=CONFLICT)
    fig.text(.06, .120, 'Conflicting pairs exceed 0.80; the consistent pairs stay.', fontsize=6.8, color=MUTED)
    handles = [Line2D([0], [0], color=GOOD, lw=1.6, label='Same owner'),
               Line2D([0], [0], color=CONFLICT, lw=1.6, ls='--', label='Different owner')]
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.06, .050), frameon=False,
        fontsize=7.0, ncol=2, handlelength=1.8, columnspacing=1.0)
    fig.text(.06, .025, 'Real, closed calibration component; no trajectory rerun.', fontsize=6.7, color=MUTED)
    return fig


def text_bounds(fig):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer(); canvas = fig.bbox
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    outside, overlaps = [], []
    for t in texts:
        b = t.get_window_extent(renderer)
        if b.x0 < canvas.x0-1 or b.y0 < canvas.y0-1 or b.x1 > canvas.x1+1 or b.y1 > canvas.y1+1:
            outside.append(t.get_text())
    for i, t in enumerate(texts):
        b = t.get_window_extent(renderer)
        for other in texts[i+1:]:
            o = other.get_window_extent(renderer)
            if min(b.x1,o.x1)-max(b.x0,o.x0)>1 and min(b.y1,o.y1)-max(b.y0,o.y0)>1:
                overlaps.append([t.get_text(), other.get_text()])
    return {'outside_canvas': outside, 'overlapping_text': overlaps,
        'minimum_visible_font_pt': min(t.get_fontsize() for t in texts), 'text_count':len(texts), 'skill_audit':audit_layout(fig)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--export', action='store_true'); args=parser.parse_args()
    r=json.loads((HERE/'SELECTED_CASE.json').read_text())
    a=json.loads((HERE/'INDEPENDENT_CASE_AUDIT.json').read_text())
    assert a['selected_case_sha256']==hashlib.sha256((HERE/'SELECTED_CASE.json').read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True); checks={}
    for name, maker in [('src_cost_refinement_triptych',triptych),('src_assignment_links',links)]:
        fig=maker(r);fig.savefig(OUT/(name+'_preview.png'),dpi=300,facecolor='white')
        with Image.open(OUT/(name+'_preview.png')) as im:
            ImageOps.grayscale(im).save(OUT/(name+'_grayscale.png'),dpi=(300,300))
        checked=text_bounds(fig); checks[name]=checked
        if args.export:
            assert not checked['outside_canvas'] and not checked['overlapping_text'],checked
            assert checked['minimum_visible_font_pt']>=6.5
            for extension in ['pdf','svg','png']:
                fig.savefig(OUT/(name+'.'+extension),dpi=600,facecolor='white')
        plt.close(fig)
    record={'status':'PASS_VECTOR_EXPORT' if args.export else 'PREVIEW_ONLY','figures':checks,
        'selected_case_sha256':hashlib.sha256((HERE/'SELECTED_CASE.json').read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'output_root':str(OUT)}
    (HERE/'FIGURE_QA.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
