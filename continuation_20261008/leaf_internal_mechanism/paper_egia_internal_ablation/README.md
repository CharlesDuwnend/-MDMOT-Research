# EGIA internal policy ablation inserted into experiments

User requested a standalone small internal ablation table, without graphs. The existing unused three-row table_policy.tex is now a complete fusion × selective four-row table, included once in a new EGIA Internal Ablation subsection. SRC, evidence-head weights, coherence and host settings are fixed; fusion/selective off still runs EGIA. Both positive conditional MOTA/IDF1 effects and FP/FN/IDs tradeoffs are stated.

All four rows use corrected complete 17-sequence / 6635-frame online runs from leaf_internal_mechanism_20261008_v2_metadata. Scoring JSONs are bound to current experiment receipts, and all 68 actual prediction files were rehashed. Metrics, scope and source paths are recorded in egia_internal_ablation_verified.json.

Authoritative manuscript: /home/chenhc/src_egia_icme_paper/main.pdf. Table V appears on page 8. Two pdflatex passes produced a nine-page PDF with resolved references and no overfull boxes; affected pages 6-9 were rendered and inspected. Old manuscript/source files remain in its output/pdf/egia_internal_ablation_20261008/before archive. Only table_policy.tex and the new subsection in main.tex changed; no figures or experiment/model/baseline artifacts changed. This is a newly authorized manuscript revision after the earlier experiment freeze, so the previous protected-paper hashes remain historical evidence.

Git archives both LaTeX revision patches, compact row provenance, before/after hashes and the review receipt. Full .tex copies remain local because this repository whitelist excludes .tex; its policy is unchanged. PDFs and rendered pages stay local. Ordinary snapshot/push policy applies; no remote backup is claimed when GitHub DNS is unavailable.
