#!/usr/bin/env python3
"""SRC mechanism explanation and complete paired-sequence statistics, CPU only.

The boundary is analytic. Errors and sequence effects are measured in the
already completed N0-N5 frozen diagnostic, not new inference or training.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.text import Text
from PIL import Image, ImageOps
import numpy as np

HERE = Path(__file__).resolve().parent
EXPERIMENT = Path("/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/native_src_factorial_20261007_v1")
PAPER = Path("/home/chenhc/src_egia_icme_paper")
OUT = PAPER / "figures/src_mechanism_analysis_v2"
SKILL = Path("/home/chenhc/.codex/skills/scipilot-figure-skill/scripts")
BLUE, TEAL, ORANGE = "#0072B2", "#009E73", "#D55E00"
INK, MUTED = "#243746", "#647487"
MAIN_SIZE, SEQ_SIZE = (3.5, 4.8), (3.5, 5.2)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def load_and_verify():
    receipt_path = EXPERIMENT / "artifacts/receipt.json"
    manifest_path = EXPERIMENT / "manifest.json"
    receipt = json.loads(receipt_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert sha(manifest_path) == receipt["manifest_sha256"]
    assert receipt["training_performed"] is False
    assert receipt["testdev_parameter_selection"] is False
    assert len(receipt["completed_arms"]) == 6
    sources = {str(receipt_path): sha(receipt_path), str(manifest_path): sha(manifest_path)}
    hash_contracts = {}
    metrics, counters, per_seq_counters, arm_receipts = {}, {}, {}, {}
    for arm in receipt["completed_arms"]:
        name = arm["name"]
        assert arm["sequence_count"] == 17 and arm["frames"] == 6635
        assert not arm["arm_config"]["use_egia"]
        assert arm["arm_config"] == next(a for a in manifest["arms"] if a["name"] == name)
        result_dir = Path(arm["result_dir"])
        metric_path = result_dir.parent / "tracking_metrics.json"
        hash_contracts[str(metric_path)] = arm["metrics_sha256"]
        hash_contracts[arm["bundle"]] = arm["bundle_sha256"]
        m = json.loads(metric_path.read_text())
        assert m["metrics"]["OVERALL"] == arm["metrics"]
        assert set(m["sequence_names"]) == set(manifest["sequence_frames"])
        assert m["result_files_sha256"] == arm["result_sha256"]
        gtroot = Path(m["groundtruth_folder"])
        for seq, h in m["groundtruth_files_sha256"].items():
            hash_contracts[str(gtroot/(seq+".txt"))] = h
        for seq, h in arm["result_sha256"].items():
            hash_contracts[str(result_dir/(seq+".txt"))] = h
        metrics[name] = m["metrics"]
        per_seq_counters[name] = {}
        count = Counter()
        for seq in manifest["sequence_frames"]:
            p = result_dir/(seq+".txt.runtime.json")
            sources[str(p)] = sha(p)
            r = json.loads(p.read_text())
            assert r["sequence"] == seq
            assert r["frames_seen"] == manifest["sequence_frames"][seq]
            assert r["use_egia"] is False and r["egia_source_probability_calls"] == 0
            assert r["gt_read"] is False and r["birth_class_gate"] == arm["arm_config"]["birth_class_gate"]
            assert r["src_cost_enabled"] == arm["arm_config"]["src_cost_enabled"]
            assert r["src_update_enabled"] == arm["arm_config"]["src_update_enabled"]
            count.update(r["counts"])
            per_seq_counters[name][seq] = r
        counters[name] = dict(count)
        arm_receipts[name] = arm
    for path, expected in hash_contracts.items():
        assert sha(path) == expected, path
    sources.update(hash_contracts)
    for name in metrics:
        for field in ("num_objects", "num_false_positives", "num_misses", "num_switches"):
            assert sum(metrics[name][s][field] for s in manifest["sequence_frames"]) == metrics[name]["OVERALL"][field]
        m = metrics[name]["OVERALL"]
        assert abs(m["mota"]-(1-(m["num_false_positives"]+m["num_misses"]+m["num_switches"])/m["num_objects"])) < 1e-12
    assert arm_receipts["N1"]["result_sha256"] == arm_receipts["N3"]["result_sha256"]
    assert counters["N3"].get("src_cost_evaluation_calls",0) == 0
    assert counters["N2"].get("belief_updates",0) == 0
    assert counters["N4"]["src_shadow_only_seams"] == 0
    assert counters["N4"]["committed_changed_vs_native_seams"] == counters["N4"]["association_changed_vs_native_seams"]
    assert arm_receipts["N2"]["bundle_sha256"] == arm_receipts["N4"]["bundle_sha256"]
    common_gt = [json.loads((Path(a["result_dir"]).parent/"tracking_metrics.json").read_text())["groundtruth_files_sha256"] for a in arm_receipts.values()]
    assert all(g == common_gt[0] for g in common_gt)
    return manifest, metrics, counters, per_seq_counters, arm_receipts, sources


def operator_checks(tau, sources):
    # Load the exact scalar/array operator without importing the GPU tracker.
    path = EXPERIMENT / "belief/models.py"
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ("semantic_penalty","semantic_update")]
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),"exec"),namespace)
    sources[str(path)] = sha(path)
    runtime = EXPERIMENT / "belief/runtime.py"
    sources[str(runtime)] = sha(runtime)
    assert "cost[finite] += (1.-base[finite])*penalty[finite]" in runtime.read_text()
    p = np.array([.4,.6])
    b = np.array([[.1,.9],[.5,.5],[.9,.1]])
    assert np.allclose(namespace["semantic_penalty"](b,np.ones((1,2))),0)
    assert np.allclose(namespace["semantic_update"](b[0],np.array([2.,.3]),0),b[0])
    assert np.allclose(namespace["semantic_update"](b[0],np.ones(2),.8),b[0])
    z = np.array([[.02,.98],[.4,.6],[.98,.02]])
    u = namespace["semantic_penalty"](b,z/p)
    assert np.all((u>=0)&(u<=1))
    c, penalty = np.meshgrid(np.linspace(0,1,501),np.linspace(0,1,501))
    adjusted = c+(1-c)*penalty
    assert np.all(adjusted >= c) and np.all(adjusted <= 1)
    assert not np.any((c>tau)&(adjusted<=tau))
    legal_c = np.linspace(0,tau,501)
    budget = (tau-legal_c)/(1-legal_c)
    assert np.allclose(legal_c+(1-legal_c)*budget,tau)
    assert np.all(np.diff(budget)<0)
    return {"exact_runtime_operator_loaded":True,"prior_like_observation_has_zero_penalty":True,
            "zero_reliability_and_uninformative_evidence_preserve_memory":True,
            "cost_non_decreasing_vs_pre_SRC_base_and_bounded_by_one":True,
            "no_illegal_base_edge_becomes_eligible":True,
            "analytic_budget_matches_cost_boundary":True,
            "budget_decreases_with_base_cost":True,
            "first_stage_threshold_from_frozen_manifest":tau,
            "no_reliability_multiplier_in_candidate_cost":True}


def csv_write(path, rows):
    with path.open("w",newline="") as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,
        "axes.labelsize":8,"xtick.labelsize":7.5,"ytick.labelsize":7.5,
        "text.color":INK,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,
        "axes.linewidth":.6,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "figure.facecolor":"white","savefig.facecolor":"white"})


def clean_axes(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#A8B3BE")
    ax.tick_params(length=3,width=.6)
    ax.grid(color="#E4E9EE",linewidth=.5,axis="y")
    ax.set_axisbelow(True)


def mechanism_figure(tau, summary):
    fig = plt.figure(figsize=MAIN_SIZE,dpi=150)
    fig.text(.065,.975,"SRC: bounded semantic intervention",fontsize=10,weight="bold",va="top")
    fig.text(.065,.938,"Analytic mechanism and controlled VisDrone evidence",fontsize=7.5,color=MUTED,va="top")
    fig.text(.185,.891,"(a) Conflict tolerance of a legal edge",fontsize=8.8,weight="bold",va="top")
    ax = fig.add_axes([.185,.60,.755,.245])
    c = np.linspace(0,tau,301)
    budget = (tau-c)/(1-c)
    ax.fill_between(c,0,budget,color="#E8F4EF")
    ax.fill_between(c,budget,1,color="#FBECE4")
    ax.plot(c,budget,color=TEAL,lw=1.6)
    ax.text(.48,.88,"Becomes ineligible",ha="center",fontsize=7.5,color="#994B28")
    ax.text(.29,.22,"Remains eligible",ha="center",fontsize=7.5,color="#19715A")
    ax.set_xlim(0,tau);ax.set_ylim(0,1)
    ax.set_xticks([0,.2,.4,.6,.8]);ax.set_yticks([0,.5,1])
    ax.set_xlabel(r"Pre-SRC base cost $C$",labelpad=4)
    ax.set_ylabel(r"Semantic disagreement $u$",labelpad=5)
    clean_axes(ax)
    ax.grid(False)
    fig.text(.185,.495,r"$u_{\max}=(\tau-C)/(1-C)$,  $\tau=0.80$",fontsize=8,color=MUTED)
    fig.text(.185,.467,"(b) Fewer errors at the same birth gate",fontsize=8.8,weight="bold",va="top")
    fig.text(.185,.434,"Gate control vs full SRC; EGIA disabled",fontsize=7.5,color=MUTED,va="top")
    ax = fig.add_axes([.185,.162,.755,.245])
    values = [summary["error_reductions"][k] for k in ("FP","FN","IDs")]
    bars = ax.bar(np.arange(3),values,width=.56,color=TEAL,edgecolor="white",lw=.6)
    ax.bar_label(bars,labels=[str(v) for v in values],padding=4,fontsize=9)
    ax.set_xticks([0,1,2],["FP","FN","ID switches"])
    ax.tick_params(axis="x",length=0,pad=6)
    ax.set_ylabel("Errors reduced (count)",labelpad=6)
    ax.set_ylim(0,max(values)*1.22)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4,integer=True))
    clean_axes(ax)
    fig.text(.065,.075,f"Overall: +{summary['delta_mota_pp']:.3f} MOTA / +{summary['delta_idf1_pp']:.3f} IDF1 (pp)",fontsize=7.5)
    fig.text(.065,.031,f"{summary['changed_calls']:,}/{summary['cost_calls']:,} nonempty assignment calls changed ({summary['changed_call_percent']:.2f}%).",fontsize=7.1,color=MUTED)
    return fig


def sequence_figure(rows, summary):
    fig = plt.figure(figsize=SEQ_SIZE,dpi=150)
    fig.text(.055,.975,"Paired SRC effects across sequences",fontsize=10,weight="bold",va="top")
    fig.text(.055,.929,"Same 0.70 birth gate; EGIA off; all 17 sequences",fontsize=7.5,color=MUTED,va="top")
    fig.text(.055,.888,f"IDF1 improves in {summary['idf1_positive_sequences']}/17 sequences",fontsize=8.6,color=TEAL,va="top")
    ax = fig.add_axes([.285,.20,.675,.635])
    y = np.arange(len(rows))
    dm = [r["delta_mota_pp"] for r in rows]
    di = [r["delta_idf1_pp"] for r in rows]
    ax.axvspan(0,2.8,color="#F2F8F5",zorder=0)
    ax.axvline(0,color="#758391",linewidth=.8)
    for ys,vs,color in [(y-.19,dm,BLUE),(y+.19,di,TEAL)]:
        for yi, v in zip(ys,vs): ax.plot([0,v],[yi,yi],color=color,lw=.55,alpha=.65)
    ax.plot(dm,y-.19,linestyle="none",marker="s",markersize=4,color=BLUE,label="MOTA")
    ax.plot(di,y+.19,linestyle="none",marker="o",markersize=4,color=TEAL,label="IDF1")
    ax.set_yticks(y,[r["short_label"] for r in rows])
    ax.set_ylim(len(rows)-.55,-.55)
    ax.set_xlim(-.5,2.8)
    ax.set_xticks([0,1,2])
    ax.set_xlabel("Change vs gate control (pp)",labelpad=7)
    ax.tick_params(axis="y",length=0,pad=5)
    clean_axes(ax)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.125),ncol=2,frameon=False,fontsize=8,
              handlelength=1,handletextpad=.4,columnspacing=1.7)
    fig.text(.055,.041,f"Aggregate: MOTA +{summary['delta_mota_pp']:.3f}, IDF1 +{summary['delta_idf1_pp']:.3f} pp.",fontsize=7.5)
    fig.text(.055,.014,"Short labels retain flight ID / segment; no sequence omitted.",fontsize=7.1,color=MUTED)
    return fig


def export(fig, stem, do_export):
    sys.path.insert(0,str(SKILL))
    from visual_qa import audit_layout
    issues = audit_layout(fig)
    # Skill QA catches clipping and tick collisions; also compare all visible
    # headers, formulas, axis labels, data labels and legend text to each other.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        texts.extend([ax.xaxis.label,ax.yaxis.label])
        for ticks,lim,axis in [(ax.get_xticklabels(),ax.get_xlim(),0),(ax.get_yticklabels(),ax.get_ylim(),1)]:
            texts.extend(t for t in ticks if min(lim)<=t.get_position()[axis]<=max(lim))
        if ax.get_legend(): texts.extend(ax.get_legend().get_texts())
    boxes = [(t.get_text(),t.get_window_extent(renderer)) for t in texts if t.get_text() and t.get_visible()]
    for label,b in boxes:
        if b.x0<fig.bbox.x0 or b.x1>fig.bbox.x1 or b.y0<fig.bbox.y0 or b.y1>fig.bbox.y1:
            issues.append(("FAIL","Text outside fixed canvas: "+label))
    for (a,ba),(b,bb) in itertools.combinations(boxes,2):
        if ba.overlaps(bb): issues.append(("FAIL",f"Text overlap: {a!r} / {b!r}"))
    font_min = min(t.get_fontsize() for t in fig.findobj(Text) if t.get_text())
    assert font_min >= 7
    preview = OUT/(stem+"_preview.png")
    fig.savefig(preview,dpi=300)
    with Image.open(preview) as im:
        ImageOps.grayscale(im).save(OUT/(stem+"_grayscale.png"),dpi=(300,300))
    if do_export:
        assert not issues,issues
        fig.savefig(OUT/(stem+".pdf"),metadata={"CreationDate":None,"ModDate":None})
        fig.savefig(OUT/(stem+".svg"),metadata={"Date":None})
        fig.savefig(OUT/(stem+".png"),dpi=600)
    plt.close(fig)
    return {"skill_layout_checks":issues,"minimum_font_pt":font_min,"all_text_items_checked":len(boxes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export",action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest,metrics,counters,runtime,arms,sources = load_and_verify()
    checks = operator_checks(manifest["fixed"]["match_thresh"],sources)
    n1,n4 = metrics["N1"]["OVERALL"],metrics["N4"]["OVERALL"]
    error_fields = {"FP":"num_false_positives","FN":"num_misses","IDs":"num_switches"}
    reductions = {k:n1[v]-n4[v] for k,v in error_fields.items()}
    assert all(v>0 for v in reductions.values())
    summary = {"comparison":"N4 full SRC minus N1 gate control",
        "birth_class_gate":.7,"egia_enabled":False,"sequences":17,"frames":6635,
        "error_reductions":reductions,"baseline_errors":{k:n1[v] for k,v in error_fields.items()},
        "src_errors":{k:n4[v] for k,v in error_fields.items()},
        "delta_mota_pp":100*(n4["mota"]-n1["mota"]),
        "delta_idf1_pp":100*(n4["idf1"]-n1["idf1"]),
        "changed_calls":counters["N4"]["committed_changed_vs_native_seams"],
        "cost_calls":counters["N4"]["src_cost_evaluation_calls"],
        "all_calls_including_empty":counters["N4"]["association_seams"]}
    summary["changed_call_percent"] = 100*summary["changed_calls"]/summary["cost_calls"]
    assert abs(summary["delta_mota_pp"]-100*sum(reductions.values())/n1["num_objects"]) < 1e-12
    rows = []
    for seq in sorted(manifest["sequence_frames"]):
        a,b = metrics["N1"][seq],metrics["N4"][seq]
        part = seq.split("_")
        rows.append({"sequence":seq,"short_label":part[0][6:]+"/"+part[1],
            "delta_mota_pp":100*(b["mota"]-a["mota"]),
            "delta_idf1_pp":100*(b["idf1"]-a["idf1"]),
            "FP_reduction":a["num_false_positives"]-b["num_false_positives"],
            "FN_reduction":a["num_misses"]-b["num_misses"],
            "IDs_reduction":a["num_switches"]-b["num_switches"]})
    summary["idf1_positive_sequences"] = sum(r["delta_idf1_pp"]>0 for r in rows)
    summary["mota_positive_sequences"] = sum(r["delta_mota_pp"]>0 for r in rows)
    for short,col in [("FP","FP_reduction"),("FN","FN_reduction"),("IDs","IDs_reduction")]:
        assert sum(r[col] for r in rows)==reductions[short]
    csv_write(HERE/"PAIRED_SEQUENCE_DATA.csv",rows)
    csv_write(OUT/"paired_sequence_data.csv",rows)
    error_rows = [{"error":k,"gate_control":n1[v],"full_SRC":n4[v],"reduction":reductions[k]} for k,v in error_fields.items()]
    csv_write(HERE/"ERROR_REDUCTION_DATA.csv",error_rows)
    style()
    layouts = {"src_bounded_intervention":export(mechanism_figure(manifest["fixed"]["match_thresh"],summary),"src_bounded_intervention",args.export),
               "src_paired_sequence_effects":export(sequence_figure(rows,summary),"src_paired_sequence_effects",args.export)}
    # Keep every control and the small/null update effects in the compact audit.
    control_audit = {a:metrics[a]["OVERALL"] for a in metrics}
    update_limit = {
        "N3_update_only_equals_N1_predictions":arms["N3"]["result_sha256"]==arms["N1"]["result_sha256"],
        "N4_minus_N2_delta_mota_pp":100*(metrics["N4"]["OVERALL"]["mota"]-metrics["N2"]["OVERALL"]["mota"]),
        "N4_minus_N2_delta_idf1_pp":100*(metrics["N4"]["OVERALL"]["idf1"]-metrics["N2"]["OVERALL"]["idf1"]),
        "N4_minus_N2_IDs_change":metrics["N4"]["OVERALL"]["num_switches"]-metrics["N2"]["OVERALL"]["num_switches"],
        "interpretation":"No material positive incremental update effect demonstrated in this frozen diagnostic."}
    caption = (
        "SRC as a bounded semantic intervention. (a) Analytic eligibility boundary "
        "for the frozen first-stage threshold tau=0.80. For a legal pre-SRC base "
        "cost C, the adjusted cost C+(1-C)u remains within the threshold exactly "
        "when u <= (tau-C)/(1-C). The plot describes per-edge eligibility, not "
        "the final global assignment, and contains no measured edge samples. "
        "Neutral evidence leaves C unchanged; no base edge above the threshold "
        "can become eligible. (b) Measured error reductions of full SRC (N4) "
        "relative to gate control (N1), using the same 0.70 birth gate and no EGIA "
        "on all 17 VisDrone test-dev sequences (6,635 frames): 139 FP, 185 FN, "
        "and 27 ID switches fewer, giving +0.153 MOTA and +0.150 IDF1 percentage "
        "points. The runtime changed 888 of 14,700 nonempty association calls "
        "relative to the native shadow assignment on the same current state. "
        "These changes are not labeled correct/incorrect matches. N1 uses the "
        "host's native semantic rule; the analytic C is the masked pre-SRC base "
        "cost. Counts are completed frozen diagnostic results, not a new main "
        "benchmark, significance test or isolated responsibility-update benefit.\n")
    second = (
        "Paired full-SRC minus gate-control effects for all 17 VisDrone test-dev "
        "sequences. Both arms use the 0.70 birth gate, and EGIA is disabled. "
        "Squares show MOTA changes and circles show IDF1 changes, in percentage "
        "points. Every sequence is retained, including negative changes. IDF1 "
        "improves in 13/17 sequences and MOTA in 12/17. Short labels retain the "
        "flight ID and segment; the CSV supplies full names. Overall changes "
        "are from aggregate evaluator metrics, not an unweighted average of "
        "the plotted sequence changes. No confidence interval or independence "
        "assumption across sequences is asserted.\n")
    (OUT/"figure_captions.md").write_text("# Figure captions\n\n"+caption+"\n"+second)
    (HERE/"FIGURE_CAPTIONS.md").write_text("# Figure captions\n\n"+caption+"\n"+second)
    tex = []
    for stem, cap in [("src_bounded_intervention",caption),("src_paired_sequence_effects",second)]:
        cap = cap.strip().replace("<=",r"$\leq$")
        tex.extend([r"\begin{figure}[t]",r"\centering",
            r"\includegraphics[width=\columnwidth]{src_mechanism_analysis_v2/"+stem+".pdf}",
            r"\caption{"+cap+"}",r"\label{fig:"+stem.replace("_","-")+"}",r"\end{figure}",""])
    (OUT/"FIGURE_CAPTIONS.tex").write_text("\n".join(tex)+"\n")
    evidence = {"status":"FINAL_EXPORT" if args.export else "PREVIEW_PENDING_VISUAL_REVIEW",
        "method":"current SRC belief/cost implementation; N4 versus same-gate N1",
        "source_experiment":str(EXPERIMENT),"new_training":False,"new_inference":False,
        "testdev_tuning":False,"manuscript_modified":False,
        "source_sha256":sources,"source_files_checked":len(sources),
        "operator_checks":checks,"summary":summary,"all_control_metrics":control_audit,
        "runtime_counters":counters,"update_claim_boundary":update_limit,
        "layouts":layouts,"figure_sizes_inches":{"src_bounded_intervention":MAIN_SIZE,"src_paired_sequence_effects":SEQ_SIZE},
        "limitations":["Analytic boundary is eligibility, not a guarantee of global selection or correctness.",
            "Changed-call counts compare the native shadow solver on SRC's current state; not whole-run pair identity.",
            "Benefit is conditional on this gate/host/frozen bundle; both negative and positive sequence effects are retained.",
            "No specific cross-human/vehicle correction or additional reliability-update benefit is claimed."]}
    files = sorted(p for p in OUT.iterdir() if p.suffix in (".pdf",".svg",".png",".csv",".md",".tex"))
    evidence["output_sha256"] = {str(p):sha(p) for p in files}
    write_json(HERE/"ANALYSIS_RECEIPT.json",evidence)
    (OUT/"SHA256SUMS").write_text("".join(sha(p)+"  "+p.name+"\n" for p in files))
    print(json.dumps({"source_files_checked":len(sources),"summary":summary,"layouts":layouts,"update_claim_boundary":update_limit},indent=2))


if __name__ == "__main__":
    main()
