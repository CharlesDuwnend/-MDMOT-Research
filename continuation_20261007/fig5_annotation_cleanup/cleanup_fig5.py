#!/usr/bin/env python3
"""Remove only the requested Figure 5 drawing annotations and rebuild it."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import types
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
PAPER=Path('/home/chenhc/src_egia_icme_paper')
WORK=PAPER/'output/pdf/fig5_annotation_cleanup_20261007'
SCRIPT=PAPER/'scripts/build_paper_case_figures.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original=SCRIPT.read_text()
    title="        text(c,x0,221,case['title'],8.2,True)\n"
    footer="    text(c,page_w/2,10,'Colored boxes: tracking outputs   |   Track IDs are local to each run',7.0,center=True)\n"
    assert original.count(title)==original.count(footer)==1
    updated=original.replace(title,'').replace(footer,'')
    updated=updated.replace("'(a) On a moving vehicle, Native maintains two overlapping tracks while Ours keeps one. '",
        "'Left: Native maintains two overlapping tracks on a moving vehicle while Ours keeps one. '")
    updated=updated.replace("'(b) On a pedestrian, the Native identity changes during motion whereas Ours preserves one identity. '",
        "'Right: the Native identity changes during pedestrian motion whereas Ours preserves one identity. '")
    tex=(PAPER/'main.tex').read_text()
    old_caption='''\\caption{Qualitative tracking examples on VisDrone2019 test-dev. (a) On a
moving vehicle, Native maintains two overlapping tracks while Ours keeps one.
(b) On a pedestrian, the Native identity changes during motion whereas Ours
preserves one identity. Each column uses the same crop for both methods.}'''
    new_caption='''\\caption{Qualitative tracking examples on VisDrone2019 test-dev. Left:
Native maintains two overlapping tracks on a moving vehicle while Ours keeps
one. Right: the Native identity changes during pedestrian motion whereas
Ours preserves one identity. Each column uses the same crop for both methods.}'''
    assert tex.count(old_caption)==1
    new_tex=tex.replace(old_caption,new_caption)
    before=WORK/'before';before.mkdir(parents=True,exist_ok=False)
    figures=PAPER/'figures/qualitative'
    backup=[SCRIPT,PAPER/'main.tex',PAPER/'main.pdf',PAPER/'main.aux',PAPER/'main.bbl',PAPER/'main.log',
            PAPER/'research/figure_case_audit.json',PAPER/'research/figure_case_audit.md']
    backup.extend(p for p in figures.iterdir() if p.is_file() and
                  (p.name.startswith('qualitative_cases') or p.name=='figure_caption.txt'))
    for p in backup:
        target=before/p.relative_to(PAPER);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    protected=[PAPER/'figures/motivation/motivation.pdf',PAPER/'figures/motivation/motivation.png',
               PAPER/'figures/src_assignment_mechanism_v8/src_semantic_response_triptych.pdf']
    protected_sha={str(p):sha(p) for p in protected}
    old_audit=json.loads((PAPER/'research/figure_case_audit.json').read_text())
    SCRIPT.write_text(updated);(PAPER/'main.tex').write_text(new_tex)
    module=types.ModuleType('fig5_cleanup_renderer');module.__file__=str(SCRIPT)
    exec(compile(updated,str(SCRIPT),'exec'),module.__dict__)
    module.qualitative();module.report()
    new_audit=json.loads((PAPER/'research/figure_case_audit.json').read_text())
    assert old_audit['qualitative_cases']==new_audit['qualitative_cases']
    with Image.open(before/'figures/qualitative/qualitative_cases.png') as old_image, Image.open(figures/'qualitative_cases.png') as new_image:
        assert old_image.size==new_image.size
        # Only the old title and footer text bands may differ. Frame numbers,
        # methods, all twelve image crops, prediction boxes and IDs stay exact.
        old_pixels=np.array(old_image.convert('RGB'));new_pixels=np.array(new_image.convert('RGB'))
        top=round(old_image.height*65/967);bottom=round(old_image.height*890/967)
        assert np.array_equal(old_pixels[top:bottom],new_pixels[top:bottom]),'Unexpected content change outside annotations.'
        changed=np.any(old_pixels!=new_pixels,axis=2)
        assert changed.sum()>0
        changed_rows=np.flatnonzero(changed.any(axis=1))
    assert all(sha(Path(p))==h for p,h in protected_sha.items())
    patches=[]
    for a,b,name in [(original,updated,'scripts/build_paper_case_figures.py'),(tex,new_tex,'main.tex')]:
        patches.extend(difflib.unified_diff(a.splitlines(keepends=True),b.splitlines(keepends=True),fromfile=name+'.before',tofile=name))
    (HERE/'paper_changes.patch').write_text(''.join(patches))
    receipt={'status':'FIGURE_UPDATED_PENDING_MANUSCRIPT_BUILD_AND_REVIEW','work_root':str(WORK),
        'backup_sha256':{str(p):sha(before/p.relative_to(PAPER)) for p in backup},'protected_sha256':protected_sha,
        'new_script_sha256':sha(SCRIPT),'new_tex_sha256':sha(PAPER/'main.tex'),
        'qualitative_case_data_unchanged':True,'all_twelve_crops_boxes_ids_and_frame_labels_pixel_identical':True,
        'pixel_change_row_ranges':[int(changed_rows.min()),int(changed_rows.max())],
        'unchanged_content_pixel_rows':[top,bottom-1],
        'removed_annotations':['(a) Vehicle: duplicate tracks','(b) Pedestrian: identity change',
                               'Colored boxes: tracking outputs | Track IDs are local to each run'],
        'formal_caption':'left/right group references replace removed a/b titles',
        'new_figure_sha256':sha(figures/'qualitative_cases.pdf')}
    (HERE/'FIG5_CLEANUP_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'unchanged_figure_content':'PASS','case_data':'UNCHANGED',
                      'figure':str(figures/'qualitative_cases.pdf')},indent=2))


if __name__=='__main__':
    main()
