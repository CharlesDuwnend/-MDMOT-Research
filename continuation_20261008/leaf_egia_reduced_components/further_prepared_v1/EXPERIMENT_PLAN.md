# Native + EGIA further internal ablation

All arms: native association, no SRC, no class-confidence gate 0.7, no H2
classifier loaded or fused; detector birth score 0.6. No fitting or tuning.

|Arm|Foreground|Coverage|Source coherence|Selective|Role|
|---|---|---|---|---|---|
|R00 full, completed reference|on|on|on|on|Reused full pure EGIA|
|R03 two heads, completed reference|on|on|off|on|Reused two-head core|
|A04 coverage only|off|on|off|on|New full inference; foreground effect without coherence|
|A05 two heads argmax|on|on|off|off|New full inference; core without abstention|
|A06 full argmax|on|on|on|off|New full inference; coherence without abstention|

R00/R03/A05/A06 form a 2x2 coherence x selective comparison with fixed
foreground and coverage heads. A04 vs R03 isolates foreground deletion when
coherence is already removed. Dropping FG sets f=1 and forbids clutter;
dropping coverage sets c=0 and forbids existing-target actions. These are
functional deletions, not refitted architectural comparisons.

Coherence is log(n * sum_i p_geometry(i) p_appearance(i)). If either cue is
uninformative across owners, it is mathematically zero. Single-cue drops
therefore duplicate the completed no-coherence control. Similarly coverage
deletion cancels E-only coherence authority; this is an action equivalence,
not a new one-classifier tracking result.

17 sequences / 6635 frames / 229506 GT target observations. All ordered
detector/ReID input hashes and native association seams must match the
completed reference. GT is forbidden during inference and used only in the
independent post-hoc scorer. Sequence/flight are the comparison units; no
frame-level significance or parameter-efficiency claims.

First run one 200-frame pure full reference smoke, then the three new smokes.
After all pass, run the three independent full arms on GPUs 0, 1, 3.
Prescribed hypotheses: FG may improve clutter rejection even without cues;
selective may limit harmful uncertain interventions; coherence may interact
with selective. All negative and null outcomes will be retained. No choice
of deployment config or paper numbers is made from this test-dev grid.
