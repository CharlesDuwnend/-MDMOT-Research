#!/usr/bin/env python3
"""Reuse the audited real SRC case with the manuscript Figure 4 palette."""
import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
CASE_ROOT = HERE.parent / 'src_assignment_visuals'
PAPER = Path('/home/chenhc/src_egia_icme_paper')
OUT = PAPER / 'figures/src_assignment_mechanism_v5_fig4_palette'
REFERENCE_SCRIPT = PAPER / 'scripts/build_semantic_analysis.py'
REFERENCE_EVIDENCE = PAPER / 'figures/semantic_analysis/evidence.json'
NAME = 'src_cost_refinement_triptych'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path):
    spec = importlib.util.spec_from_file_location('audited_src_case_plot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def palette_from_source():
    for statement in ast.parse(REFERENCE_SCRIPT.read_text()).body:
        if isinstance(statement, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == 'PASTEL_COLORS' for t in statement.targets):
                return ast.literal_eval(statement.value)
    raise RuntimeError('The referenced Figure 4 palette is missing.')


def luminance(rgb):
    rgb = np.asarray(rgb)
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return linear @ np.array([.2126, .7152, .0722])


def protected_artifacts():
    paths = [PAPER / 'main.pdf', PAPER / 'main.tex']
    original = PAPER / 'figures/src_assignment_mechanism_v4'
    paths.extend(original / name for name in json.loads((CASE_ROOT / 'FINAL_EXPORT_QA.json').read_text())['artifact_sha256'])
    return {str(path): sha(path) for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    protected = protected_artifacts()
    palette = palette_from_source()
    reference = json.loads(REFERENCE_EVIDENCE.read_text())
    assert palette == reference['design']['palette']
    record = json.loads((CASE_ROOT / 'SELECTED_CASE.json').read_text())
    audit = json.loads((CASE_ROOT / 'INDEPENDENT_CASE_AUDIT.json').read_text())
    assert audit['selected_case_sha256'] == sha(CASE_ROOT / 'SELECTED_CASE.json')
    base = load_module(CASE_ROOT / 'plot_case.py')
    base.CMAP = base.colors.LinearSegmentedColormap.from_list('manuscript_figure4', palette)
    # Numerical scale is identical to Figure 4. Categorical accents are separate.
    base.INK = '#111111'
    base.MUTED = '#58636F'
    base.HUMAN = '#6C839D'
    base.VEHICLE = '#B88469'
    base.GOOD = '#506B87'
    base.CONFLICT = '#A86644'
    base.plt.rcParams.update({'text.color': base.INK, 'axes.labelcolor': base.INK,
        'xtick.color': base.INK, 'ytick.color': base.INK})
    rgb = base.CMAP(np.linspace(0, 1, 256))[:, :3]
    lightness = luminance(rgb)
    assert np.all(np.diff(lightness) < 0), 'Grayscale ordering must be monotonic.'
    contrast = (lightness + .05) / (luminance(base.colors.to_rgb(base.INK)) + .05)
    assert float(contrast.min()) >= 4.5, 'All matrix numbers must remain readable.'
    fig = base.triptych(record)
    # The original renderer uses white digits for the dark end of cividis.
    # This pastel scale needs dark digits throughout, including zero penalties.
    for text in fig.findobj(base.Text):
        if re.fullmatch(r'[0-9]+\.[0-9]+', text.get_text()):
            text.set_color(base.INK)
    expected = [np.array(record['case'][key]) for key in ['semantic_penalty', 'base_cost', 'src_cost']]
    matrix_axes = [ax for ax in fig.axes if np.allclose(ax.get_xlim(), [-.5, 2.5])]
    assert len(matrix_axes) == 3
    for ax, values in zip(matrix_axes, expected):
        labels = [t.get_text() for t in ax.texts if re.fullmatch(r'[0-9]+\.[0-9]+', t.get_text())]
        assert labels == [f'{value:.3f}' for value in values.flat]
        for rectangle, value in zip(ax.patches[:9], values.flat):
            assert np.allclose(rectangle.get_facecolor(), base.CMAP(float(value)))
        # Thin edge boxes leave three-decimal values fully visible on pale cells.
        for text in ax.texts:
            if re.fullmatch(r'[0-9]+\.[0-9]+', text.get_text()):
                text.set_fontsize(7.5)
                text.set_zorder(6)
        for index, rectangle in enumerate(ax.patches[9:]):
            x, y = rectangle.get_xy()
            col, row = round(x + .445), round(y + .445)
            rectangle.set_bounds(col - .48, row - .48, .96, .96)
            rectangle.set_linewidth(2.1 if index % 2 == 0 else 1.1)
    qa = base.text_bounds(fig)
    assert not qa['outside_canvas'] and not qa['overlapping_text'] and not qa['skill_audit'], qa
    assert qa['minimum_visible_font_pt'] >= 6.5
    renderer = fig.canvas.get_renderer()
    for ax in matrix_axes:
        for rectangle in ax.patches[9::2]:
            x, y = rectangle.get_xy()
            col, row = round(x + .48), round(y + .48)
            label = next(t for t in ax.texts if t.get_position() == (col, row))
            bounds = label.get_window_extent(renderer)
            box = rectangle.get_window_extent(renderer)
            margin = renderer.points_to_pixels(rectangle.get_linewidth() / 2)
            assert bounds.x0 > box.x0 + margin and bounds.x1 < box.x1 - margin
            assert bounds.y0 > box.y0 + margin and bounds.y1 < box.y1 - margin
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (NAME + '_preview.png'), dpi=300, facecolor='white')
    with Image.open(OUT / (NAME + '_preview.png')) as preview:
        ImageOps.grayscale(preview).save(OUT / (NAME + '_grayscale.png'), dpi=(300, 300))
    if args.export:
        for extension in ['pdf', 'svg', 'png']:
            fig.savefig(OUT / (NAME + '.' + extension), dpi=600, facecolor='white')
    base.plt.close(fig)
    assert protected_artifacts() == protected
    artifacts = [OUT / (NAME + suffix) for suffix in ['_preview.png', '_grayscale.png']]
    if args.export:
        artifacts.extend(OUT / (NAME + '.' + extension) for extension in ['pdf', 'svg', 'png'])
    receipt = {
        'status': 'PASS_EXPORT_PENDING_PDF_VISUAL_REVIEW' if args.export else 'PASS_PREVIEW',
        'palette': palette,
        'numeric_scale': [0, 1],
        'palette_source': str(REFERENCE_SCRIPT),
        'palette_source_sha256': sha(REFERENCE_SCRIPT),
        'reference_evidence_sha256': sha(REFERENCE_EVIDENCE),
        'reference_main_pdf_sha256': protected[str(PAPER / 'main.pdf')],
        'base_plot_script_sha256': sha(CASE_ROOT / 'plot_case.py'),
        'variant_script_sha256': sha(Path(__file__)),
        'selected_case_sha256': sha(CASE_ROOT / 'SELECTED_CASE.json'),
        'independent_case_audit_sha256': sha(CASE_ROOT / 'INDEPENDENT_CASE_AUDIT.json'),
        'case_and_operator_data': 'Unchanged; all 27 displayed cells checked against the audited real case.',
        'scope': 'One recolored triptych; the original matching-link figure retains its previous colors.',
        'accent_colors': {'human': base.HUMAN, 'vehicle': base.VEHICLE,
            'same_owner': base.GOOD, 'different_owner': base.CONFLICT},
        'grayscale_luminance': {'strictly_decreasing': True, 'low_value': float(lightness[0]), 'high_value': float(lightness[-1])},
        'minimum_matrix_text_contrast_ratio': float(contrast.min()),
        'redundant_encoding': 'Solid same-owner outline, dashed different-owner outline, numerical costs, and forbidden-edge crosses.',
        'dimensions_inches': [7.16, 2.95],
        'png_dpi': 600 if args.export else None,
        'layout_qa': qa,
        'selected_cell_digits_clear_of_outline': True,
        'protected_artifacts_unchanged': protected,
        'artifact_sha256': {path.name: sha(path) for path in artifacts},
        'output_root': str(OUT),
    }
    (HERE / 'PALETTE_QA.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'status': receipt['status'], 'palette': palette,
        'minimum_matrix_text_contrast_ratio': receipt['minimum_matrix_text_contrast_ratio'],
        'all_27_displayed_cells': 'PASS', 'layout': 'PASS', 'output_root': str(OUT)}, indent=2))


if __name__ == '__main__':
    main()
