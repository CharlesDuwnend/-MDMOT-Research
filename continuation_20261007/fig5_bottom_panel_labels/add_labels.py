#!/usr/bin/env python3
"""Add centered group letters below Figure 5 without restoring annotations."""
from pathlib import Path
import hashlib,json,shutil,difflib,types
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
PAPER=Path('/home/chenhc/src_egia_icme_paper')
WORK=PAPER/'output/pdf/fig5_bottom_panel_labels_20261007'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    script=PAPER/'scripts/build_paper_case_figures.py';old=script.read_text()
    start=old.index('def qualitative():');end=old.index('def motivation():',start)
    block=old[start:end];anchor="    c.showPage();c.save();render_pdf(OUT/'qualitative_cases.pdf')\n"
    assert block.count(anchor)==1 and "group_center" not in block
    labels="    for idx,x0 in enumerate(origins):\n        group_center=x0+(3*pw+2*gap)/2\n        text(c,group_center,10,'(a)' if idx==0 else '(b)',8.0,center=True)\n"
    block=block.replace(anchor,labels+anchor)
    block=block.replace("'Left: Native maintains two overlapping tracks on a moving vehicle while Ours keeps one. '","'(a) On a moving vehicle, Native maintains two overlapping tracks while Ours keeps one. '")
    block=block.replace("'Right: the Native identity changes during pedestrian motion whereas Ours preserves one identity. '","'(b) On a pedestrian, the Native identity changes during motion whereas Ours preserves one identity. '")
    updated=old[:start]+block+old[end:]
    tex=(PAPER/'main.tex').read_text()
    old_cap='''\\caption{Qualitative tracking examples on VisDrone2019 test-dev. Left:
Native maintains two overlapping tracks on a moving vehicle while Ours keeps
one. Right: the Native identity changes during pedestrian motion whereas
Ours preserves one identity. Each column uses the same crop for both methods.}'''
    new_cap='''\\caption{Qualitative tracking examples on VisDrone2019 test-dev. (a) On a
moving vehicle, Native maintains two overlapping tracks while Ours keeps one.
(b) On a pedestrian, the Native identity changes during motion whereas Ours
preserves one identity. Each column uses the same crop for both methods.}'''
    assert tex.count(old_cap)==1;new_tex=tex.replace(old_cap,new_cap)
    before=WORK/'before';before.mkdir(parents=True,exist_ok=False)
    assets=[script,PAPER/'main.tex',PAPER/'main.pdf',PAPER/'main.aux',PAPER/'main.log',PAPER/'main.bbl',
            PAPER/'research/figure_case_audit.json',PAPER/'research/figure_case_audit.md']
    assets.extend(p for p in (PAPER/'figures/qualitative').iterdir() if p.is_file() and (p.name.startswith('qualitative_cases') or p.name=='figure_caption.txt'))
    for p in assets:
        dst=before/p.relative_to(PAPER);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    protected=[PAPER/'figures/motivation/motivation.pdf',PAPER/'figures/src_assignment_mechanism_v8/src_semantic_response_triptych.pdf']
    protected_sha={str(p):sha(p) for p in protected}
    old_audit=json.loads((PAPER/'research/figure_case_audit.json').read_text())
    script.write_text(updated);(PAPER/'main.tex').write_text(new_tex)
    module=types.ModuleType('fig5_panel_labels');module.__file__=str(script);exec(compile(updated,str(script),'exec'),module.__dict__)
    module.qualitative();module.report()
    new_audit=json.loads((PAPER/'research/figure_case_audit.json').read_text());assert old_audit['qualitative_cases']==new_audit['qualitative_cases']
    figure=PAPER/'figures/qualitative/qualitative_cases.pdf'
    with Image.open(before/'figures/qualitative/qualitative_cases.png') as a,Image.open(figure.with_suffix('.png')) as b:
        assert a.size==b.size
        upper=round(a.height*890/967)
        assert np.array_equal(np.array(a.convert('RGB'))[:upper],np.array(b.convert('RGB'))[:upper])
    assert all(sha(Path(p))==h for p,h in protected_sha.items())
    patch=''
    for a,b,name in [(old,updated,'scripts/build_paper_case_figures.py'),(tex,new_tex,'main.tex')]:
        patch+=''.join(difflib.unified_diff(a.splitlines(keepends=True),b.splitlines(keepends=True),fromfile=name+'.before',tofile=name))
    (HERE/'paper_changes.patch').write_text(patch)
    qa={'status':'BOTTOM_LABELS_ADDED_PENDING_MANUSCRIPT_REVIEW','work_root':str(WORK),
        'new_script_sha256':sha(script),'new_tex_sha256':sha(PAPER/'main.tex'),
        'backup_sha256':{str(p):sha(before/p.relative_to(PAPER)) for p in assets},'protected_sha256':protected_sha,
        'all_content_above_bottom_label_band_pixel_identical':True,'qualitative_case_data_unchanged':True,
        'label_positions':{'a':{'x_pt':20+(3*((180*72/25.4-20-6-4*3.8-10)/6)+2*3.8)/2,'y_pt':10},
                           'b':{'x_pt':20+3*((180*72/25.4-20-6-4*3.8-10)/6)+2*3.8+10+(3*((180*72/25.4-20-6-4*3.8-10)/6)+2*3.8)/2,'y_pt':10}},
        'label_font_pt':8.0,'figure_sha256':sha(figure)}
    (HERE/'FIG5_PANEL_LABEL_QA.json').write_text(json.dumps(qa,indent=2)+'\n')
    print(json.dumps({'status':qa['status'],'image_content_unchanged':True,'labels':'centered (a) and (b) below left/right groups'},indent=2))


if __name__=='__main__':main()
