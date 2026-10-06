# P52 novelty screen

- Mayer et al., **Learning Target Candidate Association to Keep Track of What Not to Track**, ICCV 2021, [primary paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Mayer_Learning_Target_Candidate_Association_To_Keep_Track_of_What_Not_ICCV2021_paper.pdf): direct adjacent prior on learned target-candidate association and what-not-to-track behavior. P52 cannot claim the dustbin idea in isolation.
- Lee & Kim, **Systematized Event-Aware Learning for Multi-Object Tracking**, UAI 2022, [primary paper](https://proceedings.mlr.press/v180/lee22a.html): event-aware association losses for missing/difficult tracking events. P52 must compare its positive-removal and cardinality-matched control at mechanism level.
- Lee et al., **Set Transformer**, ICLR 2019, [primary paper](https://proceedings.mlr.press/v97/lee19d/lee19d.pdf): permutation-invariant set interactions. Candidate-set context is established prior art.

Decision before the gate: HOLD broad novelty. The only defensible possible boundary is a narrow MDMT cross-view K<=64 assignment objective with explicit positive-removal versus matched negative-removal causal control. A gain on calibration pairs is feasibility evidence, not a first/novel claim.
