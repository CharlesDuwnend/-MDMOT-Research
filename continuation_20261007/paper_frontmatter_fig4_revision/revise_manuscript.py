#!/usr/bin/env python3
"""Apply the scoped manuscript edits, preserving data and figure graphics."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
PAPER = Path('/home/chenhc/src_egia_icme_paper')
WORK = PAPER / 'output/pdf/paper_frontmatter_fig4_revision_20261007'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def main():
    before = WORK / 'before'
    before.mkdir(parents=True, exist_ok=False)
    names = ['main.tex', 'main.pdf', 'main.aux', 'main.log', 'main.bbl',
             'scripts/build_intro_bubble.py',
             'figures/intro_bubble/v4single/figure_caption.txt',
             'research/figure_case_audit.json', 'research/figure_case_audit.md']
    for name in names:
        dst = before / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / name, dst)

    tex = (PAPER / 'main.tex').read_text()
    protected = list(PAPER.glob('table*.tex')) + [PAPER / 'references.bib']
    protected += [PAPER / 'figures/motivation/motivation.pdf',
                  PAPER / 'figures/src_assignment_mechanism_v8/src_semantic_response_triptych.pdf',
                  PAPER / 'figures/qualitative/qualitative_cases.pdf',
                  PAPER / 'scripts/build_paper_case_figures.py']
    protected += [p for p in (PAPER / 'figures/intro_bubble/v4single').iterdir()
                  if p.is_file() and p.name != 'figure_caption.txt']
    protected_hashes = {str(p): sha(p) for p in protected}
    # All experimental descriptions and metrics are outside the editing scope.
    experiments = tex[tex.index('\\section{Experiments}'):]
    updated = replace_once(tex, r'''coverage. On VisDrone2019 test-dev, \method{} with U2MOT reaches 58.03\%
MOTA and 71.37\% IDF1, improving the native tracker by 2.10 and 1.44
percentage points while reducing false positives by 21.5\% and identity
switches by 38.6\%. In the vehicle-centric UAVDT setting, MOTA improves by''',
        r'''coverage. On VisDrone2019 test-dev, \method{} improves the native tracker by
2.10 MOTA and 1.44 IDF1 percentage points, reaching 58.03\% and 71.37\%,
respectively, with 21.5\% fewer false positives and 38.6\% fewer identity
switches. In the vehicle-centric UAVDT setting, MOTA improves by''')
    updated = replace_once(updated, r'''\caption{Illustrative redundant outputs on VisDrone2019 test-dev. (a) Native
U2MOT produces two overlapping tracks for a pedestrian, whereas Ours retains
one. (b) Native again produces two tracks for a vehicle, whereas Ours retains
one. Each pair shows the same crop under the two tracking outputs.}''',
        r'''\caption{Illustrative redundant outputs on VisDrone2019 test-dev.
Native produces two overlapping tracks for (a) a pedestrian and (b) a vehicle,
whereas Ours retains one. Each pair shows the same crop under both tracking
outputs.}''')
    updated = replace_once(updated, r'''For context, Fig.~\ref{fig:bubble} places the paired VisDrone result among
the reported methods using the same MOTA--IDF1 axes as Table~\ref{tab:main}.''',
        r'''We assess \method{} through paired comparisons with each native tracker,
including host-specific adaptations to ByteTrack and BoT-SORT
(Table~\ref{tab:transfer}). Fig.~\ref{fig:bubble} places the VisDrone result
among reported methods on the same MOTA--IDF1 axes as Table~\ref{tab:main}.''')
    updated = replace_once(updated, r'''\caption{MOTA, IDF1, and identity switches (IDs) on VisDrone2019 test-dev.
Bubble area is inversely proportional to IDs, so larger bubbles indicate
fewer switches. The displayed literature methods use their reported
configurations, while Ours is evaluated with the U2MOT host.}''',
        r'''\caption{MOTA, IDF1, and identity switches (IDs) on VisDrone2019 test-dev.
Bubble area is inversely proportional to IDs, so larger bubbles indicate
fewer switches. Displayed literature results follow their reported
configurations.}''')
    updated = replace_once(updated, r'''Fig.~\ref{fig:semantic_analysis} connects stored semantic evidence to candidate
cost refinement. Panel (a) illustrates how historical track evidence can
differ from an incoming detection; SRC aggregates this evidence into human
and vehicle groups. In (b), disagreement increases as the observation
contradicts the pre-assignment belief. Weaker beliefs incur smaller
penalties, while a prior-like belief is neutral. In the real calibration
component of (c), refinement raises the conflicting cross-owner costs above
the unchanged host threshold, resolving two tied base assignments into a
unique same-owner assignment. Reliability updates memory after a committed
match; disagreement and cost modulation use the beliefs available before
assignment.''',
        r'''Fig.~\ref{fig:semantic_analysis} illustrates how stored semantic belief shapes
candidate costs. Panel (a) presents schematic five-class evidence, aggregated
into human and vehicle groups in the implementation. Panel (b) holds
pre-assignment beliefs fixed: contradictory observations produce larger
$u_{ij}$, halfway-to-prior beliefs halve the penalty, and prior-like belief is
neutral. Panel (c) uses a closed $3\times3$ component from frozen calibration,
where conflicting costs rise from $0.013$ to $0.989$ and from $0.031$ to
$0.992$. At the unchanged $0.80$ threshold, two tied base optima become a
unique same-owner optimum. Reliability controls the subsequent committed-match
memory update. This example demonstrates cost refinement on a fixed state.''')
    assert updated[updated.index('\\section{Experiments}'):] == experiments

    builder = PAPER / 'scripts/build_intro_bubble.py'
    builder_old = builder.read_text()
    old = """    caption = ('MOTA, IDF1, and identity switches on VisDrone2019 test-dev for 11 displayed methods. '
               'Bubble area is proportional to 1/IDs; larger bubbles indicate fewer switches. '
               'Every displayed method is labeled at its unchanged coordinates. '
               'MOTR, ByteTrack, and native U2MOT are omitted from this compact view; Table I is unchanged. '
               'MM-Tracker has no reported IDs. Other methods retain their reported configurations in Table I.\\n')"""
    new = """    caption = ('MOTA, IDF1, and identity switches (IDs) on VisDrone2019 test-dev. '
               'Bubble area is inversely proportional to IDs, so larger bubbles indicate fewer switches. '
               'Displayed literature results follow their reported configurations.\\n')"""
    builder_new = replace_once(builder_old, old, new)
    caption_path = PAPER / 'figures/intro_bubble/v4single/figure_caption.txt'
    caption_old = caption_path.read_text()
    caption_new = ('MOTA, IDF1, and identity switches (IDs) on VisDrone2019 test-dev. '
                   'Bubble area is inversely proportional to IDs, so larger bubbles indicate fewer switches. '
                   'Displayed literature results follow their reported configurations.\n')

    changes = [('main.tex', tex, updated),
               ('scripts/build_intro_bubble.py', builder_old, builder_new),
               ('figures/intro_bubble/v4single/figure_caption.txt', caption_old, caption_new)]
    patch = ''
    for name, old_text, new_text in changes:
        (PAPER / name).write_text(new_text)
        patch += ''.join(difflib.unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile=name + '.before', tofile=name))
    assert all(sha(Path(p)) == h for p, h in protected_hashes.items())
    (HERE / 'paper_changes.patch').write_text(patch)
    qa = {'status': 'SOURCE_REVISED_PENDING_PDF_REVIEW', 'work_root': str(WORK),
          'backup_sha256': {name: sha(before / name) for name in names},
          'protected_sha256': protected_hashes,
          'experiment_section_unchanged': True,
          'experiment_section_sha256': hashlib.sha256(experiments.encode()).hexdigest(),
          'figure_graphics_and_benchmark_data_unchanged': True,
          'abstract_metrics_preserved': ['58.03% MOTA', '71.37% IDF1', '+2.10 MOTA',
                                         '+1.44 IDF1', '-21.5% FP', '-38.6% IDs', '+0.61 UAVDT MOTA'],
          'new_claims_of_baseline_independence': False,
          'fig4_evidence_boundary': {'a': 'illustrative five classes; actual state has two groups',
                                    'b': 'analytic response using fixed pre-assignment beliefs',
                                    'c': 'selected closed frozen-calibration component on a fixed state',
                                    'base_optimal_assignment_count': 2,
                                    'src_optimal_assignment_count': 1,
                                    'threshold_unchanged': 0.8,
                                    'formal_metrics_or_new_online_inference': False}}
    (HERE / 'MANUSCRIPT_REVISION_QA.json').write_text(json.dumps(qa, indent=2) + '\n')
    print(json.dumps({'status': qa['status'], 'backup_root': str(before),
                      'changed_files': [x[0] for x in changes]}, indent=2))


if __name__ == '__main__':
    main()
