#!/usr/bin/env python3
"""Recolor panel A only; preserve the five-class illustration and panels B/C."""
import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
V6 = HERE.parent / 'src_mechanism_v6'
OUT = Path('/home/chenhc/src_egia_icme_paper/figures/src_assignment_mechanism_v7')
TRACK_BLUE = '#0072B2'
DETECTION_ORANGE = '#E69F00'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path = V6 / 'build_semantic_response.py'
    spec = importlib.util.spec_from_file_location('src_triptych_v6', path)
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    original = template.historical_semantics
    previous = json.loads((V6 / 'FIGURE_QA.json').read_text())
    previous_root = Path(previous['output_root'])
    protected = {str(previous_root / name): sha(previous_root / name) for name in previous['artifact_sha256']}

    def recolored(fig, base, case):
        old_colors = base.HUMAN, base.VEHICLE
        try:
            base.HUMAN, base.VEHICLE = TRACK_BLUE, DETECTION_ORANGE
            data = original(fig, base, case)
            for rectangle in fig.axes[-1].patches:
                rectangle.set_hatch(None)
                rectangle.set_edgecolor('none')
                rectangle.set_linewidth(0)
            for rectangle in fig.legends[-1].get_patches():
                rectangle.set_hatch(None)
                rectangle.set_edgecolor('none')
                rectangle.set_linewidth(0)
            data['source_encoding'] = 'Solid blue track bars and solid golden-orange detection bars.'
            data['colors'] = {'track': TRACK_BLUE, 'detection': DETECTION_ORANGE}
            return data
        finally:
            base.HUMAN, base.VEHICLE = old_colors

    template.historical_semantics = recolored
    template.HERE = HERE
    template.OUT = OUT
    template.main()
    qa_path = HERE / 'FIGURE_QA.json'
    receipt = json.loads(qa_path.read_text())
    with Image.open(OUT / 'src_semantic_response_triptych_preview.png') as current, Image.open(previous_root / 'src_semantic_response_triptych_preview.png') as old:
        w, h = current.size
        for name, box in {'b': [.270, .07, .550, .90], 'c': [.560, .07, .985, .868]}.items():
            roi = tuple(round(v * (w if index % 2 == 0 else h)) for index, v in enumerate(box))
            unchanged = ImageChops.difference(current.crop(roi).convert('RGB'), old.crop(roi).convert('RGB')).getbbox() is None
            assert unchanged, 'Panel ' + name.upper() + ' changed.'
            receipt['panel_' + name + '_pixel_identical_to_v6'] = True
    assert all(sha(Path(name)) == value for name, value in protected.items())
    receipt['wrapper_script_sha256'] = sha(Path(__file__))
    receipt['original_v6_receipt_sha256'] = sha(V6 / 'FIGURE_QA.json')
    receipt['protected_v6_artifacts_sha256'] = protected
    receipt['panel_a_style'] = {'track': TRACK_BLUE, 'detection': DETECTION_ORANGE,
        'bars': 'solid, no hatching or heavy outlines', 'palette': 'Okabe-Ito blue and orange'}
    qa_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'panels_b_c_unchanged': True, 'colors_a': receipt['panel_a_style'], 'output_root': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
