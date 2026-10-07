#!/usr/bin/env python3
"""Compact SRC mechanism layout with explanations moved to its caption."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
PAPER=Path('/home/chenhc/src_egia_icme_paper')
OUT=PAPER/'figures/src_assignment_mechanism_v8'
NAME='src_semantic_response_triptych'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    previous=json.loads((PARENT/'src_mechanism_v7/FIGURE_QA.json').read_text())
    old_root=Path(previous['output_root'])
    protected={str(old_root/name):sha(old_root/name) for name in previous['artifact_sha256']}
    protected.update({str(PAPER/name):sha(PAPER/name) for name in ['main.pdf','main.tex']})
    case_path=PARENT/'src_assignment_visuals/SELECTED_CASE.json'
    audit_path=PARENT/'src_assignment_visuals/INDEPENDENT_CASE_AUDIT.json'
    record=json.loads(case_path.read_text());case=record['case']
    audit=json.loads(audit_path.read_text())
    assert sha(case_path)==audit['selected_case_sha256']==previous['selected_case_sha256']
    base_path=PARENT/'src_assignment_visuals/plot_case.py'
    operator_path=PARENT/'src_mechanism_v6/build_semantic_response.py'
    palette_path=PARENT/'src_fig4_palette/build_palette_variant.py'
    base=module(base_path,'src_clean_base')
    operator=module(operator_path,'src_clean_operator')
    palette_tools=module(palette_path,'src_clean_palette')
    palette=palette_tools.palette_from_source()
    assert palette==previous['palette']
    base.CMAP=base.colors.LinearSegmentedColormap.from_list('manuscript_figure4',palette)
    base.INK,base.MUTED,base.HUMAN,base.VEHICLE,base.GOOD,base.CONFLICT=('#111111','#58636F','#6C839D','#B88469','#506B87','#A86644')
    base.plt.rcParams.update({'font.size':7.2,'axes.labelsize':7.2,'xtick.labelsize':6.8,'ytick.labelsize':6.8,
        'text.color':base.INK,'axes.labelcolor':base.INK,'xtick.color':base.INK,'ytick.color':base.INK})
    fig=base.plt.figure(figsize=(7.16,2.40))
    bottom=.295;height=.156*7.16/2.40
    for x,label in [(.150,'(a) Historical semantics'),(.430,'(b) Semantic disagreement'),(.785,'(c) Cost refinement')]:
        fig.text(x,.930,label,ha='center',fontsize=8.4)

    # A: the same specified illustrative distributions, with only a source key.
    a=fig.add_axes([.075,bottom,.180,height])
    inputs=previous['historical_semantics']
    track=np.array(inputs['specified_track_probabilities'])
    detection=np.array(inputs['specified_detection_probabilities'])
    bars=[]
    for offset,values,color,label in [(-.17,track,'#0072B2','Track'),(.17,detection,'#E69F00','Detection')]:
        container=a.bar(np.arange(5)+offset,values,width=.29,color=color,edgecolor='none',label=label)
        bars.extend(container.patches)
    assert np.array_equal([patch.get_height() for patch in bars],np.r_[track,detection])
    a.set_xlim(-.6,4.6);a.set_ylim(0,1.05)
    a.set_xticks(np.arange(5),['car','van','truck','ped.','other'])
    a.set_yticks([0,.5,1],['0','0.5','1'])
    a.set_ylabel('Semantic probability',labelpad=3)
    fig.legend(*a.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.075,.855),ncol=2,
        frameon=False,fontsize=7.0,handlelength=1.1,columnspacing=.8,borderaxespad=0,borderpad=0)

    # B: preserve the operator, curves and actual case markers; remove formula text.
    b=fig.add_axes([.330,bottom,.200,height])
    prior=np.array(case['prior']);zh=np.linspace(0,1,1001)
    hard_beliefs=[np.array([1.,0.]),np.array([0.,1.])]
    response_data=[];weak_data=[]
    for belief,color,style,label in [(hard_beliefs[0],base.HUMAN,'-','H track'),
                                    (hard_beliefs[1],base.CONFLICT,(0,(4,1.5)),'V track')]:
        u=operator.semantic_response(belief,zh,prior)
        weak_belief=.5*belief+.5*prior
        weak=operator.semantic_response(weak_belief,zh,prior)
        assert np.allclose(weak,.5*u,rtol=0,atol=1e-14)
        b.plot(zh,u,color=color,linewidth=1.3,linestyle=style,label=label)
        b.fill_between(zh,0,u,color=color,alpha=.10,linewidth=0,zorder=0)
        b.plot(zh,weak,color=color,alpha=.5,linewidth=.8,linestyle=style,zorder=1)
        response_data.append(u);weak_data.append({'belief':weak_belief.tolist(),'half_penalty_check':'PASS'})
    neutral=operator.semantic_response(prior,zh,prior)
    assert np.allclose(neutral,0,rtol=0,atol=1e-14)
    b.plot(zh,neutral,color='#777777',linewidth=1.0,linestyle=(0,(1,1.5)))
    b.axvline(prior[0],color='#9CA5AD',linestyle=(0,(2,2)),linewidth=.65,zorder=0)
    markers=[]
    for i,j in case['base_assignment']:
        if np.argmax(case['beliefs'][i])==np.argmax(case['observation_probabilities'][j]):continue
        belief=np.array(case['beliefs'][i]);observed=float(case['observation_probabilities'][j][0])
        u=float(operator.semantic_response(belief,observed,prior))
        assert np.isclose(u,case['semantic_penalty'][i][j],rtol=0,atol=1e-12)
        b.plot(observed,u,'o',markersize=4,markerfacecolor='white',
            markeredgecolor=base.HUMAN if belief[0]>=prior[0] else base.CONFLICT,markeredgewidth=1,zorder=6)
        markers.append({'track':i+1,'detection':j+1,'observation_H':observed,'penalty':u})
    b.set_xlim(-.035,1.035);b.set_ylim(-.045,1.06)
    b.set_xticks([0,prior[0],1],['0',r'$\pi(H)$','1'])
    b.set_yticks([0,.5,1],['0','0.5','1'])
    b.set_xlabel(r'Observation $z_j(H)$',labelpad=3)
    b.set_ylabel(r'Semantic penalty $u_{ij}$',labelpad=3)
    fig.legend(*b.get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.330,.855),ncol=2,
        frameon=False,fontsize=7.0,handlelength=1.5,columnspacing=.8,borderaxespad=0,borderpad=0)
    for ax in [a,b]:
        ax.tick_params(length=2,pad=2)
        for name in ['top','right']:ax.spines[name].set_visible(False)
        for name in ['bottom','left']:ax.spines[name].set_color(base.MUTED)
    replay=np.array([[operator.semantic_response(np.array(belief),float(z[0]),prior)
        for z in case['observation_probabilities']] for belief in case['beliefs']])
    error=float(np.max(np.abs(replay-np.array(case['semantic_penalty']))));assert error<=1e-12

    # C: all costs, matching boxes and gate marks retain the audited real case.
    matrices=[]
    for x,key,pairs,forbidden in [(.592,'base_cost',case['base_assignment'],False),
                                 (.821,'src_cost',case['src_assignment'],True)]:
        values=np.array(case[key])
        ax=base.matching_matrix(fig,[x,bottom,.156,height],values,pairs,case,forbidden=forbidden)
        matrices.append(ax)
        for text in ax.texts:
            if re.fullmatch(r'[0-9]+\.[0-9]+',text.get_text()):
                text.set_color(base.INK);text.set_fontsize(7.5);text.set_zorder(6)
        for index,rectangle in enumerate(ax.patches[9:]):
            px,py=rectangle.get_xy();col,row=round(px+.445),round(py+.445)
            rectangle.set_bounds(col-.48,row-.48,.96,.96);rectangle.set_linewidth(2.1 if index%2==0 else 1.1)
        assert [text.get_text() for text in ax.texts if re.fullmatch(r'[0-9]+\.[0-9]+',text.get_text())]==[f'{v:.3f}' for v in values.flat]
    matrices[1].set_yticklabels([])
    fig.text(.670,.852,r'Base $C$',ha='center',fontsize=7.6)
    fig.text(.899,.852,r'SRC $\widetilde C$',ha='center',fontsize=7.6)
    fig.text(.784,bottom+height/2,'→',ha='center',va='center',fontsize=12,color=base.MUTED)
    scalar=base.plt.cm.ScalarMappable(norm=base.colors.Normalize(0,1),cmap=base.CMAP)
    cb=fig.colorbar(scalar,cax=fig.add_axes([.592,.236,.385,.024]),orientation='horizontal',ticks=[0,.5,1])
    cb.ax.tick_params(labelsize=6.8,length=1.6,pad=1)
    cb.outline.set_linewidth(.45);cb.set_label('Cost',fontsize=7.2,labelpad=1)
    qa=base.text_bounds(fig)
    texts=[t.get_text() for t in fig.findobj(base.Text) if t.get_visible() and t.get_text()]
    forbidden_text=['Five-class illustration','Different dominant','SRC stores','Two tied','Faint:',
        'SRC: semantic','analytic response','Real calibration','\\mathrm{clip}']
    assert not any(part in text for part in forbidden_text for text in texts)
    assert sum(text.startswith('(') for text in texts)==3
    OUT.mkdir(parents=True,exist_ok=True)
    preview=OUT/(NAME+'_preview.png');fig.savefig(preview,dpi=300,facecolor='white')
    with Image.open(preview) as image:ImageOps.grayscale(image).save(OUT/(NAME+'_grayscale.png'),dpi=(300,300))
    artifacts=[preview,OUT/(NAME+'_grayscale.png')]
    if args.export:
        assert not qa['outside_canvas'] and not qa['overlapping_text'] and not qa['skill_audit'],qa
        assert qa['minimum_visible_font_pt']>=6.5
        for extension in ['pdf','svg','png']:
            path=OUT/(NAME+'.'+extension);fig.savefig(path,dpi=600,facecolor='white');artifacts.append(path)
    base.plt.close(fig)
    assert all(sha(Path(path))==value for path,value in protected.items())
    receipt={'status':'PASS_EXPORT_PENDING_PDF_VISUAL_REVIEW' if args.export else 'PREVIEW_ONLY',
        'script_sha256':sha(Path(__file__)),'source_sha256':{str(path):sha(path) for path in [case_path,audit_path,base_path,operator_path,palette_path]},
        'previous_v7_receipt_sha256':sha(PARENT/'src_mechanism_v7/FIGURE_QA.json'),
        'historical_semantics':inputs,'response':{'prior':prior.tolist(),'grid_points':len(zh),'weak_beliefs':weak_data,
            'neutral_belief':'PASS','real_markers':markers,'nine_pair_replay_max_error':error},
        'real_cost_data_unchanged':True,'legend_explanations_and_formula':'moved to external caption',
        'layout_qa':qa,'visible_text_count':len(texts),'user_requested_text_removals':'PASS',
        'dimensions_inches':[7.16,2.40],'output_root':str(OUT),'protected_artifacts_sha256':protected,
        'artifact_sha256':{path.name:sha(path) for path in artifacts}}
    (HERE/'FIGURE_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'layout':qa,'visible_text_count':len(texts),
        'data_unchanged':True,'text_removals':'PASS','output_root':str(OUT)},indent=2))


if __name__=='__main__':main()
