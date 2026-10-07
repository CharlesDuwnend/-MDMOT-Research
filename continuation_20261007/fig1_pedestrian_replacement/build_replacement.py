#!/usr/bin/env python3
"""Preview/install the verified clean-background pedestrian Figure 1 case."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json
import shutil
import types
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/chenhc/src_egia_icme_paper')
WORK = PAPER / 'output/pdf/fig1_pedestrian_replacement_20261007'
SCRIPT = PAPER / 'scripts/build_paper_case_figures.py'
OLD = "dict(key='pedestrian_redundancy',title='(a) Pedestrian',sequence='uav0000297_02761_v',frame=256,target_id=40,native_ids=[167,232],leaf_ids=[118],interval=[248,264]),"
NEW = "dict(key='pedestrian_redundancy',title='(a) Pedestrian',sequence='uav0000073_04464_v',frame=272,target_id=18,native_ids=[110,289],leaf_ids=[92],interval=[272,273]),"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--install', action='store_true')
    args = parser.parse_args()
    original = SCRIPT.read_text()
    assert original.count(OLD) == 1
    updated = original.replace(OLD, NEW)
    protected = [PAPER/'main.tex', PAPER/'main.pdf',
                 PAPER/'figures/src_assignment_mechanism_v8/src_semantic_response_triptych.pdf',
                 PAPER/'figures/qualitative/qualitative_cases.pdf',
                 PAPER/'figures/qualitative/qualitative_cases.png']
    protected_sha = {str(p): sha(p) for p in protected}
    module = types.ModuleType('fig1_clean_preview')
    module.__file__ = str(SCRIPT)
    exec(compile(updated, str(SCRIPT), 'exec'), module.__dict__)
    preview = WORK/'preview'
    module.ROOT = preview
    (preview/'figures/motivation').mkdir(parents=True, exist_ok=True)
    module.AUDIT = WORK/'preview_figure_case_audit.json'
    shutil.copy2(PAPER/'research/figure_case_audit.json', module.AUDIT)
    module.motivation()
    new = preview/'figures/motivation'
    old = PAPER/'figures/motivation'
    with Image.open(old/'motivation.png') as old_image, Image.open(new/'motivation.png') as new_image:
        assert old_image.size == new_image.size
        # The unchanged vehicle group begins at 51.7% of the figure width.
        region = (int(old_image.width*.52), 0, old_image.width, old_image.height)
        a = np.array(old_image.convert('RGB').crop(region))
        b = np.array(new_image.convert('RGB').crop(region))
        assert np.array_equal(a, b), 'Vehicle group must remain pixel-identical.'
    audit = json.loads(module.AUDIT.read_text())
    case = audit['motivation_cases'][0]
    assert case['sequence'] == 'uav0000073_04464_v' and case['frame'] == 272
    assert case['native_ids'] == [110,289] and case['leaf_ids'] == [92]
    assert len(case['continuous_window']) == 2
    assert all(sha(Path(p)) == h for p,h in protected_sha.items())
    receipt = {'status': 'PREVIEW_PENDING_VISUAL_REVIEW', 'new_pedestrian_case': case,
               'input_source_sha256': {str(p): module.sha(p) for p in module.inputs(case['sequence'])},
               'photo_sha256': case['photo_sha256'], 'source_script_before_sha256': sha(SCRIPT),
               'source_script_after_sha256': hashlib.sha256(updated.encode()).hexdigest(),
               'protected_sha256': protected_sha, 'vehicle_group_pixel_identical': True,
               'preview_root': str(new), 'work_root': str(WORK),
               'selected_case_reason': 'Original crop shows an isolated pedestrian on a plaza without the previous banner; the displayed Native duplicate boxes both have target IoU above 0.73 and no overlap with another annotated object.',
               'no_training_inference_or_prediction_rewriting': True,
               'publication_scope': 'Illustrative local output comparison; two adjacent frames verified, without claiming long-duration persistence or module-specific attribution.',
               'preview_artifact_sha256': {p.name: sha(p) for p in sorted(new.iterdir()) if p.is_file()}}
    if args.install:
        before = WORK/'before'
        before.mkdir(exist_ok=False)
        backup = [SCRIPT, PAPER/'research/figure_case_audit.json', PAPER/'research/figure_case_audit.md',
                  PAPER/'main.pdf', PAPER/'main.tex', *[p for p in old.iterdir() if p.is_file() and
                  p.name in ['motivation.pdf','motivation.png','motivation_grayscale.png','figure_caption.txt',
                             'motivation_180mm_96dpi.png','motivation_180mm_96dpi_grayscale.png']]]
        for p in backup:
            target = before/p.relative_to(PAPER)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        SCRIPT.write_text(updated)
        receipt['before_artifact_sha256'] = {str(p): sha(before/p.relative_to(PAPER)) for p in backup}
        for p in new.iterdir():
            if p.is_file():
                shutil.copy2(p, old/p.name)
        audit['motivation_figure']['path'] = str(old/'motivation.pdf')
        module.AUDIT.write_text(json.dumps(audit,indent=2)+'\n')
        shutil.copy2(module.AUDIT, PAPER/'research/figure_case_audit.json')
        actual = types.ModuleType('fig1_clean_installed')
        actual.__file__ = str(SCRIPT)
        exec(compile(updated, str(SCRIPT), 'exec'), actual.__dict__)
        actual.report()
        receipt['status'] = 'INSTALLED_PENDING_MANUSCRIPT_REBUILD'
        receipt['installed_figure_sha256'] = sha(old/'motivation.pdf')
        receipt['installed_figure_path'] = str(old/'motivation.pdf')
        receipt['script_patch_path'] = str(HERE/'figure_case_script.patch')
        (HERE/'figure_case_script.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(keepends=True), updated.splitlines(keepends=True),
            fromfile='scripts/build_paper_case_figures.py.before', tofile='scripts/build_paper_case_figures.py')))
        assert all(sha(Path(p)) == h for p,h in protected_sha.items())
    (HERE/'FIG1_REPLACEMENT_QA.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status': receipt['status'], 'preview': str(new/'motivation.png'),
                      'vehicle_group_pixel_identical': True, 'sequence': case['sequence'],
                      'frame': case['frame'], 'window': case['interval']},indent=2))


if __name__ == '__main__':
    main()
