# Saved diagnostic schema

`rows/{pair}-{view}.npz` is sorted by stream frame and raw XML ID. Pair and view are bound by the filename; the primary key is `(pair,view,frame,raw_id,class)`. Every supported outside=0 XML box in the complete stream range occurs once. Class is 0 pedestrian, 1 bicycle, 2 car. Arrays are equal length and load with `numpy.load(..., allow_pickle=False)`.

| Fields | Meaning |
|---|---|
| frame, raw_id, class | XML frame + 1 and view-local annotation identity/class; never a tracker or verified global ID |
| bbox_x1/y1/x2/y2, minimum_side_px | Raw GT xyxy geometry and minimum side in source pixels; float32 |
| occluded | XML occluded attribute, 0/1 |
| minside_bin | 0: <16; 1: 16 to <32; 2: >=32 |
| current_detected | Boolean class-aware one-to-one frozen detector match at IoU >=0.5 |
| matched_detector_index, matched_label_row | Index in frozen frame's detections and full frozen label NPZ respectively; -1 when missing |
| matched_iou | Frozen match IoU, 0 when missing |
| first_annotation | First annotated observation of this local identity/class in the stream |
| stream_start_left_censored | first_annotation at stream frame 1; not proven physical birth |
| segment_start, reappearance | First observation of a contiguous annotation segment; reappearance excludes first_annotation |
| segment_start_frame, annotation_age | Segment start and current prefix annotation count |
| prior_detector_frame, prior_detector_age | Latest strictly prior matched detection in this contiguous segment and t-minus-that-frame; both -1 when absent |
| seen_prior_1/4/8 | Any strictly prior same-segment detection in [t-H,t-1], fixed H=1/4/8 |
| first_detector_in_segment | CURRENT matched and no earlier detection in this annotation segment |
| first_detector_local_lifetime | CURRENT matched and no earlier detection in this view-local annotation lifetime; gaps do not erase lifetime status |
| prefix_miss_run_age | Number of consecutive missing observations through CURRENT; 0 if CURRENT detected; reset on annotation gaps |
| miss_category | 0 current detected; 1 first annotation missing; 2 reappearance missing; 3 continuous segment never previously detected; 4 continuous prior detection older than 8; 5 continuous prior detection within 8 |

All GT and GT-derived fields are OFFLINE ORACLE diagnostics, not model inference features. Only the history predicates and prefix ages obey the causal prefix contract; the existence/geometry of the current GT box is itself Oracle information.

`miss_runs/{pair}-{view}.npz` contains full retrospective missing intervals. `start_row` and `end_row` reference the corresponding GT row table. `length=end_frame-start_frame+1`; `preceded_by_current_detection` refers to the immediately preceding annotated frame. `end_reason` is 1 for immediate next-frame detection, 2 for annotation gap followed by a later annotation, and 3 for no subsequent annotation in the available stream. `prior_8_at_start` copies the causal prior flag at the run start. Full run lengths/endpoints use later annotations and are summary-only, never history inputs.

`cross_view/{pair}.npz` includes same-frame, same raw-ID and normalized-label coannotated observations, excluding cross-label conflicts. `view1_row/view2_row` reference the two GT tables. `current_v1/current_v2`, `both_missing`, `occluded_v1/v2`, and `minside_bin_v1/v2` describe the current pair. `prior_H_either/both` is OR/AND of the independently gap-reset strictly-past view histories. Each view may have observed the identity at a different prior frame. These are conventional labels; physical identity and synchronized evidence transfer are not established.

`cases.csv` gives the first two missing rows per sequence × miss category × occlusion × size cell, in deterministic stream order. It is an index into all-row evidence, not a visual quality selection.
