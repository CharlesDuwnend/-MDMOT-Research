#!/usr/bin/env python3
"""Build four independent single-column figures from frozen duplicate audits.

No inference, fitting, threshold selection, or manuscript insertion occurs.
The broad, prediction-class-unrestricted event definition is fixed in the
source audits.  The resolution plots distinguish paired residual events from
all LEAF duplicates and do not equate event absence with target preservation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = Path("/home/chenhc/src_egia_icme_paper")
VIS_AUDIT = PAPER / "research/duplicate_motivation_audit.json"
UAV_AUDIT = HERE / "uavdt_duplicate_audit.json"
SIZE = (3.5, 3.6)  # 88.9 x 91.44 mm; fixed PDF media box, no tight cropping.
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#D55E00"
GRAY = "#8993A1"
INK = "#243746"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verified_data():
    vis = json.loads(VIS_AUDIT.read_text())
    uav = json.loads(UAV_AUDIT.read_text())
    checks = {}
    for arm in vis["inputs"].values():
        checks[arm["manifest"]] = arm["manifest_sha256"]
        for row in arm["files"]:
            for key in ("gt", "pred"):
                checks[row[key]] = row[key + "_sha256"]
    for arm in ("native", "leaf"):
        checks.update(uav[arm]["source_sha256"])
        checks[uav["eval_receipts"][arm]] = uav["eval_receipts"][arm + "_sha256"]
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError("Frozen input SHA-256 mismatch: " + path)

    n = {tuple(k) for k in vis["arms"]["native"]["event_keys"]["class_agnostic"]}
    l = {tuple(k) for k in vis["arms"]["v5"]["event_keys"]["class_agnostic"]}
    assert vis["sequence_count"] == 17 and uav["sequence_count"] == 20
    assert vis["arms"]["native"]["totals"]["gt_frames"] == vis["arms"]["v5"]["totals"]["gt_frames"]
    assert len(n) == vis["arms"]["native"]["totals"]["class_agnostic_duplicate_gt_frames"]
    assert len(l) == vis["arms"]["v5"]["totals"]["class_agnostic_duplicate_gt_frames"]
    classes = ["pedestrian", "car", "van", "truck", "bus"]
    class_ids = [1, 4, 5, 6, 9]
    by_class = Counter(k[2] for k in n)
    class_counts = [vis["arms"]["native"]["by_class"][c]["class_agnostic_duplicate_gt_frames"] for c in classes]
    assert class_counts == [by_class[c] for c in class_ids]
    assert sum(class_counts) == len(n)

    un = {tuple(k) for k in uav["native"]["event_keys"]}
    ul = {tuple(k) for k in uav["leaf"]["event_keys"]}
    assert len(un) == uav["native"]["totals"]["duplicate_target_frames"]
    assert len(ul) == uav["leaf"]["totals"]["duplicate_target_frames"]
    assert uav["native"]["totals"]["gt_target_frames"] == uav["leaf"]["totals"]["gt_target_frames"]
    uc = Counter(k[0] for k in un)
    seqs = uav["native"]["by_sequence"]
    assert set(seqs) == set(uav["leaf"]["by_sequence"])
    assert all(uc[s] == row.get("duplicate_target_frames", 0) for s, row in seqs.items())
    ranked = sorted(uc.items(), key=lambda x: (-x[1], x[0]))
    top = ranked[:5]
    other = len(un) - sum(c for _, c in top)

    outcome = uav["paired"]["strict_leaf_outcome"]
    sets = {k: {tuple(e) for e in v} for k, v in outcome["event_keys"].items()}
    for a, b in itertools.combinations(sets.values(), 2):
        assert not a & b
    assert set().union(*sets.values()) == un
    assert sets["still_duplicate"] == un & ul
    assert sets["one_track"] | sets["no_related_prediction"] == un - ul
    assert {k: len(v) for k, v in sets.items()} == outcome["counts"]
    assert uav["paired"]["residual"] == len(un & ul)
    assert uav["paired"]["resolved"] == len(un - ul)
    counts = outcome["counts"]
    captions = {
        "fig3_visdrone_motivation": (
            "Repeated-output events in Native U2MOT on VisDrone2019 test-dev "
            "(17 sequences). One event is one valid ground-truth target in one "
            "frame with at least two distinct predicted track IDs. Each box has "
            "IoU >= 0.5 with that target and IoU < 0.4 with every other valid "
            "target. Predicted category is unrestricted. Bars group events by "
            "GT category; Ped. denotes pedestrian. The five counts sum to "
            f"{len(n):,} events among {vis['arms']['native']['totals']['gt_frames']:,} "
            "valid target-frames. GT is used only after inference as a locator."
        ),
        "fig6_visdrone_resolution": (
            "Paired repeated-output events under LEAF on VisDrone2019 test-dev, "
            "using the same broad event definition as the motivation figure. "
            "Events are paired by (sequence, frame, GT category, GT ID). Of "
            f"{len(n):,} Native events, {len(n-l):,} are absent from LEAF's "
            f"duplicate-event set and {len(n&l):,} remain at the same keys. "
            "Event absence does not establish that exactly one track remains. "
            f"LEAF has {len(l):,} duplicate events in total, including "
            f"{len(l-n):,} events outside the Native event set. "
            "The bars report paired descriptive counts, not official MOT scores."
        ),
        "fig3_uavdt_motivation": (
            "Repeated-output events in Native U2MOT on UAVDT Protocol-B "
            "test20, with original GT and the frozen evaluator's ignore-region "
            "preprocessing. The five sequences with the most events are shown "
            f"individually; Other 15 combines all remaining sequences ({other:,} "
            f"events), so every one of the {len(un):,} events is represented. "
            "An event requires at least two distinct predicted track IDs on one "
            "valid vehicle GT target in one frame, each with target IoU >= 0.5 "
            "and IoU < 0.4 with every other valid GT target. All counts are "
            f"post-hoc observations among {uav['native']['totals']['gt_target_frames']:,} "
            "valid target-frames; GT does not enter inference."
        ),
        "fig6_uavdt_resolution": (
            "LEAF outcomes at Native repeated-output event keys on UAVDT "
            "Protocol-B original-GT test20. The same IoU and uniqueness rules "
            "are used for both configurations after ignore-region removal. "
            f"Of {len(un):,} Native events, {counts['one_track']:,} have exactly "
            f"one related LEAF track, {counts['no_related_prediction']:,} have "
            f"no related prediction, and {counts['still_duplicate']:,} still "
            "have at least two track IDs. The latter three bars partition the "
            "Native event set; no-related-prediction events are excluded from "
            "single-track resolution. These are descriptive counts, not MOT "
            "scores or isolated attribution to a particular module."
        ),
    }
    figures = [
        dict(stem="fig3_visdrone_motivation", dataset="VisDrone2019 test-dev",
             subtitle="Native repeated-output events", context=f"{len(n):,} events across 17 sequences",
             labels=["Ped.", "Car", "Van", "Truck", "Bus"], values=class_counts,
             colors=[BLUE]*5, hatches=[""]*5, xlabel="Ground-truth category", rotation=0,
             footer="Each event: one target in one frame.\nPed. = pedestrian; prediction class unrestricted."),
        dict(stem="fig6_visdrone_resolution", dataset="VisDrone2019 test-dev",
             subtitle="LEAF outcomes on Native events", context="Paired by sequence, frame and GT identity",
             labels=["Native\nbaseline", "Duplicate\nabsent", "Still\nduplicate"],
             values=[len(n), len(n-l), len(n&l)], colors=[BLUE,TEAL,ORANGE],
             hatches=["", "", "///"], xlabel="", rotation=0,
             footer=f"Absent does not imply one track remains.\nLEAF total: {len(l):,}; {len(l-n):,} events outside Native set.", paired=True),
        dict(stem="fig3_uavdt_motivation", dataset="UAVDT test20",
             subtitle="Native repeated-output events", context=f"Protocol-B original GT | {len(un):,} events",
             labels=[s for s, _ in top]+["Other 15"], values=[c for _,c in top]+[other],
             colors=[BLUE]*5+[GRAY], hatches=[""]*5+["///"],
             xlabel="Sequence (top five + remaining fifteen)", rotation=45,
             footer="All 20 sequences contribute to these counts.\nEach event: one vehicle target in one frame."),
        dict(stem="fig6_uavdt_resolution", dataset="UAVDT test20",
             subtitle="LEAF outcomes on Native events", context="Protocol-B original GT | paired event keys",
             labels=["Native\nbaseline", "Single\ntrack", "No related\nprediction", "Still\nduplicate"],
             values=[len(un), counts["one_track"], counts["no_related_prediction"], counts["still_duplicate"]],
             colors=[BLUE,TEAL,GRAY,ORANGE], hatches=["", "", "...", "///"],
             xlabel="", rotation=0,
             footer="The three LEAF outcomes partition Native events.\nSingle-track resolution excludes uncovered targets.", paired=True),
    ]
    evidence = {
        "source_thread": "01a11542-f887-7631-86c3-689689be96de",
        "source_audits": {str(p):sha256(p) for p in (VIS_AUDIT,UAV_AUDIT)},
        "input_hashes_checked": len(checks), "input_hash_failures": [],
        "inference_or_fitting_run": False, "manuscript_modified": False,
        "visdrone": dict(native=len(n), leaf_total=len(l), paired_absent=len(n-l),
                         paired_residual=len(n&l), leaf_only=len(l-n), classes=dict(zip(classes,class_counts))),
        "uavdt": dict(native=len(un), leaf_total=len(ul), paired_absent=len(un-ul),
                      paired_residual=len(un&ul), leaf_only=len(ul-un), outcomes=counts,
                      top_five=dict(top), other_fifteen=other),
        "limitations": ["Descriptive post-hoc counts; not official MOT performance.",
                        "VisDrone event absence is not audited as single-track resolution.",
                        "UAVDT Other 15 is an aggregate, not a single sequence."],
    }
    return figures, captions, evidence


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.labelsize": 8.5, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": .65, "hatch.linewidth": .45,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def check_text_layout(fig, texts):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    boxes = [(t.get_text(),t.get_window_extent(renderer)) for t in texts if t.get_visible() and t.get_text()]
    for label, b in boxes:
        if b.x0 < bounds.x0 or b.y0 < bounds.y0 or b.x1 > bounds.x1 or b.y1 > bounds.y1:
            raise ValueError("Text outside fixed canvas: " + label)
    for (a,ba),(b,bb) in itertools.combinations(boxes,2):
        if ba.overlaps(bb):
            raise ValueError(f"Overlapping text: {a!r}, {b!r}")
    return {"text_items_checked":len(boxes),"outside_canvas":0,"overlapping_text_pairs":0}


def draw(spec, out):
    fig, ax = plt.subplots(figsize=SIZE, dpi=150)
    fig.subplots_adjust(left=.205, right=.965, bottom=.310, top=.795)
    texts = [
        fig.text(.205,.962,spec["dataset"],fontsize=11.5,weight="bold",va="top"),
        fig.text(.205,.907,spec["subtitle"],fontsize=9.2,va="top"),
        fig.text(.205,.858,spec["context"],fontsize=7.6,color="#637487",va="top"),
        fig.text(.05,.092,spec["footer"],fontsize=7.1,color="#637487",va="top",linespacing=1.55),
    ]
    xs = np.arange(len(spec["labels"]))
    bars = ax.bar(xs,spec["values"],width=.60,color=spec["colors"],
                  edgecolor="white",linewidth=.65,zorder=3)
    for bar,hatch in zip(bars,spec["hatches"]):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_edgecolor("#354A5A")
            bar.set_linewidth(.4)
    texts.extend(ax.bar_label(bars,labels=[f"{v:,}" for v in spec["values"]],
                              padding=4,fontsize=8.6,color=INK))
    ax.set_ylabel("Duplicate target-frames (count)",labelpad=8)
    ax.set_xlabel(spec["xlabel"],labelpad=9,fontsize=7.8)
    ax.set_xticks(xs,spec["labels"])
    ax.tick_params(axis="x",length=0,pad=8)
    if spec["rotation"]:
        plt.setp(ax.get_xticklabels(),rotation=spec["rotation"],ha="right",rotation_mode="anchor")
    ax.tick_params(axis="y",length=3,width=.6,pad=4)
    ax.set_ylim(0,max(spec["values"])*1.20)
    ax.set_xlim(-.62,len(xs)-.38)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f"{v:,.0f}"))
    ax.grid(axis="y",color="#E4E9EE",linewidth=.6,zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    for key in ("left","bottom"):
        ax.spines[key].set_color("#A8B3BE")
    if spec.get("paired"):
        ax.axvline(.5,color="#BAC4CE",linewidth=.7,linestyle=(0,(3,3)),zorder=1)
    texts.extend(ax.get_xticklabels())
    texts.extend(t for t in ax.get_yticklabels() if ax.get_ylim()[0] <= t.get_position()[1] <= ax.get_ylim()[1])
    texts.extend([ax.xaxis.label,ax.yaxis.label])
    qa = check_text_layout(fig,texts)
    stem = out/spec["stem"]
    fig.savefig(stem.with_suffix(".pdf"),metadata={"Title":spec["dataset"]+": "+spec["subtitle"],
                                                 "Creator":"build_four_figures.py","CreationDate":None,"ModDate":None})
    fig.savefig(stem.with_suffix(".svg"),metadata={"Date":None})
    fig.savefig(stem.with_suffix(".png"),dpi=600)
    plt.close(fig)
    return qa


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=PAPER/"figures/motivation/four_figures")
    args=parser.parse_args()
    specs,captions,evidence=verified_data()
    set_style()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    evidence["layout"]={"width_inches":SIZE[0],"height_inches":SIZE[1],"png_dpi":600,
                        "fixed_pdf_media_box":True,"zero_based_linear_y_axes":True,
                        "counts_not_percentages":True,"error_bars":"not applicable to census counts"}
    evidence["text_layout_checks"]={s["stem"]:draw(s,args.out_dir) for s in specs}
    rows=[]
    for s in specs:
        for label,value in zip(s["labels"],s["values"]):
            rows.append([s["stem"],label.replace("\n"," "),value])
    buff=io.StringIO()
    writer=csv.writer(buff,lineterminator="\n")
    writer.writerow(["figure","bar","count"])
    writer.writerows(rows)
    csv_text=buff.getvalue()
    (HERE/"FOUR_FIGURES_DATA.csv").write_text(csv_text)
    (args.out_dir/"figure_data.csv").write_text(csv_text)
    tex=["% Four independent figures. Insert selectively; numbers follow manuscript order."]
    for s in specs:
        caption=captions[s["stem"]].replace(">=",r"$\geq$").replace("<",r"$<$")
        tex.extend([r"\begin{figure}[t]",r"\centering",
                    r"\includegraphics[width=\columnwidth]{motivation/four_figures/"+s["stem"]+".pdf}",
                    r"\caption{"+caption+"}",r"\label{fig:"+s["stem"].replace("_","-")+"}",r"\end{figure}",""])
    (args.out_dir/"FIGURE_CAPTIONS.tex").write_text("\n".join(tex)+"\n")
    (HERE/"FOUR_FIGURES_CAPTIONS.md").write_text("# Four independent duplicate-event figures\n\n"+
        "\n\n".join("## "+s["stem"]+"\n\n"+captions[s["stem"]] for s in specs)+"\n")
    (args.out_dir/"README.md").write_text(
        "# Four independent single-column figures\n\n"
        "Four separate figures are supplied; each has PDF, SVG and 600-DPI PNG versions. "
        "The fixed canvas is 88.9 x 91.44 mm. All bars use absolute counts and linear "
        "axes starting at zero. Blue denotes Native, teal a LEAF outcome, vermillion "
        "residual duplicates, and gray an aggregate or uncovered outcome as labeled.\n\n"
        "| Dataset | Motivation | LEAF outcomes |\n| --- | --- | --- |\n"
        "| VisDrone2019 test-dev | fig3_visdrone_motivation | fig6_visdrone_resolution |\n"
        "| UAVDT Protocol-B test20 | fig3_uavdt_motivation | fig6_uavdt_resolution |\n\n"
        "FIGURE_CAPTIONS.tex contains independent figure snippets; final figure numbers "
        "follow manuscript insertion order. The manuscript has not been changed. "
        "figure_data.csv records every plotted count, and SHA256SUMS covers exports.\n\n"
        "VisDrone's 397 residual events are paired to Native keys; LEAF has 415 duplicates "
        "in total, including 18 additional events. The 2,516 absent duplicate events do "
        "not establish single-track preservation. UAVDT explicitly separates 1,516 "
        "single-track outcomes, 4 without related predictions, and 10 duplicates. "
        "Other 15 is the complete aggregate of the remaining test20 sequences.\n\n"
        "Source and compact evidence: " + str(HERE) + "/build_four_figures.py and "
        "FOUR_FIGURES_{DATA.csv,CAPTIONS.md,RECEIPT.json,QA.json}.\n")
    evidence["output_directory"]=str(args.out_dir)
    evidence["artifacts"]=[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p)}
                           for p in sorted(args.out_dir.iterdir())
                           if p.suffix in (".pdf",".png",".svg",".csv",".tex",".md")]
    evidence["status"]="FOUR_INDEPENDENT_SINGLE_COLUMN_FIGURES_EXPORTED"
    (HERE/"FOUR_FIGURES_RECEIPT.json").write_text(json.dumps(evidence,indent=2)+"\n")
    (args.out_dir/"SHA256SUMS").write_text("".join(a["sha256"]+"  "+Path(a["path"]).name+"\n" for a in evidence["artifacts"]))
    print(json.dumps({"status":evidence["status"],"input_hashes_checked":evidence["input_hashes_checked"],
                      "output_directory":str(args.out_dir),"visdrone":evidence["visdrone"],"uavdt":evidence["uavdt"]},indent=2))


if __name__=="__main__":
    main()
