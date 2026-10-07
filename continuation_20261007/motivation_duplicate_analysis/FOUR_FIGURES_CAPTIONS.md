# Four independent duplicate-event figures

## fig3_visdrone_motivation

Repeated-output events in Native U2MOT on VisDrone2019 test-dev (17 sequences). One event is one valid ground-truth target in one frame with at least two distinct predicted track IDs. Each box has IoU >= 0.5 with that target and IoU < 0.4 with every other valid target. Predicted category is unrestricted. Bars group events by GT category; Ped. denotes pedestrian. The five counts sum to 2,913 events among 232,378 valid target-frames. GT is used only after inference as a locator.

## fig6_visdrone_resolution

Paired repeated-output events under LEAF on VisDrone2019 test-dev, using the same broad event definition as the motivation figure. Events are paired by (sequence, frame, GT category, GT ID). Of 2,913 Native events, 2,516 are absent from LEAF's duplicate-event set and 397 remain at the same keys. Event absence does not establish that exactly one track remains. LEAF has 415 duplicate events in total, including 18 events outside the Native event set. The bars report paired descriptive counts, not official MOT scores.

## fig3_uavdt_motivation

Repeated-output events in Native U2MOT on UAVDT Protocol-B test20, with original GT and the frozen evaluator's ignore-region preprocessing. The five sequences with the most events are shown individually; Other 15 combines all remaining sequences (317 events), so every one of the 1,530 events is represented. An event requires at least two distinct predicted track IDs on one valid vehicle GT target in one frame, each with target IoU >= 0.5 and IoU < 0.4 with every other valid GT target. All counts are post-hoc observations among 340,906 valid target-frames; GT does not enter inference.

## fig6_uavdt_resolution

LEAF outcomes at Native repeated-output event keys on UAVDT Protocol-B original-GT test20. The same IoU and uniqueness rules are used for both configurations after ignore-region removal. Of 1,530 Native events, 1,516 have exactly one related LEAF track, 4 have no related prediction, and 10 still have at least two track IDs. The latter three bars partition the Native event set; no-related-prediction events are excluded from single-track resolution. These are descriptive counts, not MOT scores or isolated attribution to a particular module.
