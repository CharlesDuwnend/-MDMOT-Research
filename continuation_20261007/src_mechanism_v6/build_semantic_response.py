#!/usr/bin/env python3
"""Five-class illustrative bars, actual-operator response, and fixed real costs."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageOps

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent / 'src_assignment_visuals'
PALETTE_ROOT = HERE.parent / 'src_fig4_palette'
PAPER = Path('/home/chenhc/src_egia_icme_paper')
OUT = PAPER / 'figures/src_assignment_mechanism_v6'
NAME = 'src_semantic_response_triptych'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def semantic_response(belief, zh, prior):
    # Preserve both saved prior entries, including their float32 rounding.
    return np.clip(1 - belief[0] * zh / prior[0] - belief[1] * (1 - zh) / prior[1], 0, 1)


def historical_semantics(fig, base, case):
    """Specified five-class illustration; not a stored five-class SRC state."""
    classes=['car','van','truck','ped.','other']
    stored=np.array([.70,.08,.12,.05,.05])
    incoming=np.array([.08,.04,.08,.74,.06])
    assert np.isclose(stored.sum(),1,rtol=0,atol=1e-14)
    assert np.isclose(incoming.sum(),1,rtol=0,atol=1e-14)
    ax=fig.add_axes([.076,.345,.180,.36])
    x=np.arange(5)
    bars=[]
    for offset,values,color,hatch in [(-.17,stored,base.HUMAN,None),(.17,incoming,base.VEHICLE,'//')]:
        container=ax.bar(x+offset,values,width=.29,color=color,edgecolor='#705847' if hatch else color,linewidth=.4,hatch=hatch)
        bars.extend(container.patches)
    assert np.array_equal(np.array([bar.get_height() for bar in bars]),np.r_[stored,incoming])
    ax.set_xlim(-.6,4.6);ax.set_ylim(0,1.05)
    ax.set_xticks(x,classes);ax.set_yticks([0,.5,1],['0','0.5','1'])
    ax.tick_params(labelsize=6.7,length=2,pad=2)
    ax.set_ylabel('Semantic probability',fontsize=6.8,labelpad=2)
    for spine in ['top','right']:ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(base.MUTED);ax.spines['left'].set_color(base.MUTED)
    fig.text(.150,.790,'Five-class illustration',fontsize=7.1,ha='center')
    handles=[base.Rectangle((0,0),1,1,facecolor=base.HUMAN,edgecolor=base.HUMAN,label='Track'),
             base.Rectangle((0,0),1,1,facecolor=base.VEHICLE,edgecolor='#705847',hatch='//',label='Detection')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.033,.779),ncol=2,
        frameon=False,fontsize=6.8,handlelength=1.2,columnspacing=.9,borderaxespad=0)
    fig.text(.033,.232,'Different dominant classes.',fontsize=6.6,color=base.MUTED)
    fig.text(.033,.179,'SRC stores human/vehicle beliefs.',fontsize=6.6,color=base.MUTED)
    return {'classes':['car','van','truck','pedestrian','other'],
        'specified_track_probabilities':stored.tolist(),'specified_detection_probabilities':incoming.tolist(),
        'evidence_type':'illustrative five-class evidence, not measured probabilities or an implemented five-class belief head',
        'bars_exactly_match_specified_inputs':True,'probability_normalization':'PASS',
        'source_encoding':'solid track bars; hatched detection bars',
        'used_as_input_to_panels_b_or_c':False,'actual_SRC_groups':['Human','Vehicle']}


def response(fig, base, case):
    """Continuous operator response using the actual prior and track beliefs."""
    prior = np.array(case['prior'])
    prior_h = float(prior[0])
    zh = np.linspace(0, 1, 1001)
    ax = fig.add_axes([.330, .365, .20, .34])
    beliefs = [('H track', np.array([1.,0.]), base.HUMAN, '-'),
               ('V track', np.array([0.,1.]), base.CONFLICT, (0, (4, 1.5))),
               ('Neutral belief: '+r'$b=\pi$', prior.copy(), '#777777', (0, (1, 1.5)))]
    values = []
    for label, belief, color, style in beliefs:
        u = semantic_response(belief, zh, prior)
        values.append(u)
        ax.plot(zh, u, color=color, linewidth=1.3, linestyle=style, label=label)
    weaker=[]
    for index in [0,1]:
        label,belief,color,style=beliefs[index]
        weak_belief=.5*prior+.5*belief
        weak_u=semantic_response(weak_belief,zh,prior)
        assert np.allclose(weak_u,.5*values[index],rtol=0,atol=1e-14)
        ax.fill_between(zh,0,values[index],color=color,alpha=.10,linewidth=0,zorder=0)
        ax.plot(zh,weak_u,color=color,alpha=.5,linewidth=.8,linestyle=style,zorder=1)
        weaker.append({'group':label,'belief':weak_belief.tolist(),'interpolation_strength':.5})
    ax.axvline(prior_h, color='#9CA5AD', linestyle=(0, (2, 2)), linewidth=.65, zorder=0)
    ax.text(prior_h, 1.07, rf'$\pi(H)={prior_h:.3f}$', ha='center', va='bottom', fontsize=6.5)
    real_points = []
    for i, j in case['base_assignment']:
        if np.argmax(case['beliefs'][i]) == np.argmax(case['observation_probabilities'][j]):
            continue
        belief = np.array(case['beliefs'][i]); bh = float(belief[0]); observed = float(case['observation_probabilities'][j][0])
        u = float(semantic_response(belief, observed, prior))
        assert np.isclose(u, case['semantic_penalty'][i][j], rtol=0, atol=1e-12)
        ax.plot(observed, u, 'o', markersize=4.0, markerfacecolor='white',
            markeredgecolor=base.HUMAN if bh >= prior_h else base.CONFLICT, markeredgewidth=1.0, zorder=6)
        real_points.append({'track':i + 1, 'detection':j + 1, 'belief_H':bh, 'observation_H':observed, 'penalty':u})
    ax.set_xlim(-.035, 1.035); ax.set_ylim(-.045, 1.06)
    ax.set_xticks([0, prior_h, 1], ['0', r'$\pi(H)$', '1'])
    ax.set_yticks([0, .5, 1], ['0', '0.5', '1'])
    ax.tick_params(labelsize=6.6, length=2, pad=2)
    ax.set_xlabel(r'Observation $z_j(H)$', fontsize=6.8, labelpad=2)
    ax.set_ylabel(r'Penalty $u_{ij}$', fontsize=6.8, labelpad=2)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color(base.MUTED); ax.spines['left'].set_color(base.MUTED)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles[:2],labels[:2],loc='upper left',bbox_to_anchor=(.306,.237),ncol=2,
        frameon=False,fontsize=6.6,handlelength=1.4,columnspacing=.7,borderaxespad=0,borderpad=0)
    fig.legend(handles[2:],labels[2:],loc='upper left',bbox_to_anchor=(.306,.196),
        frameon=False,fontsize=6.6,handlelength=1.4,borderaxespad=0,borderpad=0)
    fig.text(.310,.131,'Faint: weaker beliefs.',fontsize=6.5,color=base.MUTED)
    vectorized = np.clip(1 - np.array([item[1] for item in beliefs]) @
        (np.array([zh, 1-zh]) / prior[:,None]), 0, 1)
    assert np.allclose(np.array(values), vectorized, atol=1e-14)
    assert np.all(np.diff(values[0]) <= 1e-14) and np.all(np.diff(values[1]) >= -1e-14)
    assert np.allclose(values[2], 0, atol=1e-14)
    for bh in np.linspace(0,1,101):
        assert np.allclose(semantic_response(np.array([bh,1-bh]), prior_h, prior), 0, atol=1e-14)
    all_pairs=np.array([[semantic_response(np.array(b),float(z[0]),prior) for z in case['observation_probabilities']] for b in case['beliefs']])
    replay_error=float(np.max(np.abs(all_pairs-np.array(case['semantic_penalty']))))
    assert replay_error<=1e-12
    return {'prior_H':prior_h,'saved_prior':prior.tolist(),'grid_points':len(zh),'real_case_markers':real_points,
        'maximum_all_nine_pairs_replay_error':replay_error,
        'weaker_belief_curves':weaker,'weaker_belief_halves_penalty':'PASS',
        'curves':'analytic operator response; the observation axis is swept, not sampled dataset observations',
        'independent_vectorized_formula':'PASS','neutral_observation_and_belief':'PASS','graded_monotonic_response':'PASS'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    old_receipt=json.loads((PALETTE_ROOT/'PALETTE_QA.json').read_text())
    protected={path:sha(Path(path)) for path in old_receipt['protected_artifacts_unchanged']}
    old_root=Path(old_receipt['output_root'])
    protected.update({str(old_root/name):sha(old_root/name) for name in old_receipt['artifact_sha256']})
    record=json.loads((CASE_ROOT/'SELECTED_CASE.json').read_text());case=record['case']
    audit=json.loads((CASE_ROOT/'INDEPENDENT_CASE_AUDIT.json').read_text())
    assert audit['selected_case_sha256']==sha(CASE_ROOT/'SELECTED_CASE.json')
    palette_module=module(PALETTE_ROOT/'build_palette_variant.py','src_palette_tools')
    base=module(CASE_ROOT/'plot_case.py','src_base_plot')
    palette=palette_module.palette_from_source()
    base.CMAP=base.colors.LinearSegmentedColormap.from_list('manuscript_figure4',palette)
    base.INK,base.MUTED,base.HUMAN,base.VEHICLE,base.GOOD,base.CONFLICT=('#111111','#58636F','#6C839D','#B88469','#506B87','#A86644')
    base.plt.rcParams.update({'text.color':base.INK,'axes.labelcolor':base.INK,'xtick.color':base.INK,'ytick.color':base.INK})
    fig=base.triptych(record)
    old_axes=list(fig.axes)
    for ax in [old_axes[0],old_axes[1],old_axes[4]]:
        ax.remove()
    for text in list(fig.texts):
        if text.get_position() in [(.420,.795),(.042,.198),(.042,.144)]:
            text.remove()
        elif text.get_text()=='(a) Semantic evidence':
            text.set_text('(a) Historical semantics')
        elif text.get_text()=='(b) Semantic disagreement':
            text.set_text('(b) Disagreement response')
        elif text.get_text().startswith('Real calibration example;'):
            text.set_text('(a) Five-class illustration; (b) analytic response; (c) real calibration case.')
        elif text.get_text()=='SRC resolves ambiguity in a real candidate group':
            text.set_text('SRC: semantic evidence and cost refinement')
    cost_axes=[old_axes[2],old_axes[3]]
    for ax,key in zip(cost_axes,['base_cost','src_cost']):
        values=np.array(case[key])
        for text in ax.texts:
            if re.fullmatch(r'[0-9]+\.[0-9]+',text.get_text()):
                text.set_color(base.INK);text.set_fontsize(7.5);text.set_zorder(6)
        for index,rectangle in enumerate(ax.patches[9:]):
            x,y=rectangle.get_xy();col,row=round(x+.445),round(y+.445)
            rectangle.set_bounds(col-.48,row-.48,.96,.96);rectangle.set_linewidth(2.1 if index%2==0 else 1.1)
        assert [t.get_text() for t in ax.texts if re.fullmatch(r'[0-9]+\.[0-9]+',t.get_text())]==[f'{v:.3f}' for v in values.flat]
    semantic_bars=historical_semantics(fig,base,case)
    mechanism=response(fig,base,case)
    fig.text(.421,.79,r'$u_{ij}=\mathrm{clip}(1-b_i^\top(z_j/\pi),0,1)$',fontsize=7.1,ha='center')
    qa=base.text_bounds(fig)
    OUT.mkdir(parents=True,exist_ok=True)
    preview=OUT/(NAME+'_preview.png')
    fig.savefig(preview,dpi=300,facecolor='white')
    with Image.open(preview) as im:
        ImageOps.grayscale(im).save(OUT/(NAME+'_grayscale.png'),dpi=(300,300))
    with Image.open(preview) as current,Image.open(old_root/'src_cost_refinement_triptych_preview.png') as previous:
        w,h=current.size
        roi=(round(.56*w),round(.07*h),round(.985*w),round(.868*h))
        unchanged=ImageChops.difference(current.crop(roi).convert('RGB'),previous.crop(roi).convert('RGB')).getbbox() is None
    artifacts=[preview,OUT/(NAME+'_grayscale.png')]
    if args.export:
        assert not qa['outside_canvas'] and not qa['overlapping_text'] and not qa['skill_audit'],qa
        assert unchanged,'Cost panel changed unexpectedly.'
        assert qa['minimum_visible_font_pt']>=6.5
        for extension in ['pdf','svg','png']:
            path=OUT/(NAME+'.'+extension);fig.savefig(path,dpi=600,facecolor='white');artifacts.append(path)
    base.plt.close(fig)
    assert all(sha(Path(path))==value for path,value in protected.items())
    receipt={'status':'PASS_EXPORT_PENDING_PDF_VISUAL_REVIEW' if args.export else 'PREVIEW_ONLY',
        'output_root':str(OUT),'script_sha256':sha(Path(__file__)),
        'selected_case_sha256':sha(CASE_ROOT/'SELECTED_CASE.json'),'case_audit_sha256':sha(CASE_ROOT/'INDEPENDENT_CASE_AUDIT.json'),
        'base_plot_script_sha256':sha(CASE_ROOT/'plot_case.py'),'palette_script_sha256':sha(PALETTE_ROOT/'build_palette_variant.py'),
        'palette':palette,'historical_semantics':semantic_bars,'mechanism_response':mechanism,'layout_qa':qa,
        'panel_c_pixel_identical_to_v5_roi':unchanged,'panel_c_roi_fraction':[.56,.07,.985,.868],
        'protected_artifacts_sha256':protected,'dimensions_inches':[7.16,2.95],
        'artifact_sha256':{path.name:sha(path) for path in artifacts}}
    (HERE/'FIGURE_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'panel_c_unchanged':unchanged,'layout':qa,'response':mechanism},indent=2))


if __name__=='__main__':main()
