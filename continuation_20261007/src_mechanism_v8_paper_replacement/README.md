# SRC Figure 4 manuscript replacement

User approved insertion of the accepted v8 SRC triptych into the paper. `main_change.patch` records the exact manuscript change: the Figure 4 asset path, caption, and directly associated explanatory paragraph. Existing label `fig:semantic_analysis` is preserved. Other manuscript sections are unchanged.

The caption identifies panel A as illustrative five-class evidence, panel B as an analytic response of Eq. (3), and panel C as a real frozen-calibration component. It distinguishes the two tied base optima from the unique refined optimum and documents the unchanged eligibility threshold. There is no new aggregate tracking result.

Manuscript: `/home/chenhc/src_egia_icme_paper/main.tex`.
Final reviewed PDF: `/home/chenhc/src_egia_icme_paper/main.pdf`.
The figure remains Figure 4 on page 5; the paper remains 9 pages.

Before-state backup, four successful LaTeX/BibTeX step logs, compiled output and page renders are under `/home/chenhc/src_egia_icme_paper/output/pdf/src_mechanism_v8_replacement_20261007/`. The first BibTeX attempt used an absolute output basename and was denied by TeX's openout policy; rerunning from the build directory resolved this without changing the policy. Final cross-references and citations resolve, no overfull boxes occur, all PDF fonts are embedded, and the two existing float-page underfull notices remain unchanged. Page 5, neighboring pages 4 and 6, and a contact sheet of all nine pages were visually reviewed.

Only this README, the exact manuscript patch and compact hash-indexed QA receipt are archived in the MDMT Git repository. Full paper source/PDF, compiled intermediates and backups stay local under the existing repository policy. The original five P63 staged changes are preserved with an isolated Git index for the phase snapshot.
