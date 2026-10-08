# Completed mechanism figure captions

**Conditional operator grids.** (a) Appearance-class masking and semantic
penalty are toggled while SRC memory, its original responsibility feature
definition and EGIA remain fixed. (b) Fusion and selective birth policies are
toggled while source heads, SRC, coherence and other settings remain fixed.
The top-right cells share C11. Cells show pooled MOTA/IDF1; arrows show all
conditional differences in percentage points. Background colors are decorative,
not a cost or performance scale. All nine runs cover the same 17 sequences and
6635 frames with identical ordered detector/ReID inputs. These are conditional
closed-loop effects at the frozen operating point, without significance claims.
C00 is a native-assignment-cost control retaining SRC memory and EGIA.

**Conditional error tradeoffs.** All four conditional contrasts in each
factorial and the paired hierarchy-minus-flat comparison are retained. FP, FN
and ID-switch cells show favorable MOTA contributions
100*(errors_control-errors_treatment)/229506 and errors removed in parentheses.
The three terms sum to ΔMOTA; IDF1 has its separate definition. Blue/ochre denote
positive/negative values on one shared symmetric scale. Zeros and negative
results are retained. The paired heads share the predeclared fit rows, scaler,
event weights and optimizer settings; their parameterization and regularization
geometry differ. Historical metadata-contaminated runs enter neither figure.

The first figure is the compact main-text candidate; the second explains its
error tradeoffs and retains the negative head result. These are author-review
artifacts, exported at IEEE double-column width; the protected manuscript has
not been modified. No actual per-edge cost matrix can be reconstructed from
the current aggregate runtime logs, so these plots do not invent such a case.
