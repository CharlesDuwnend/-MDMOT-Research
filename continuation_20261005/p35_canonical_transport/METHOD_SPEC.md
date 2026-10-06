# P35: object-excluded canonical point transport (candidate)

## Claim and mechanism

MIA represents an object by its bounding-box center in inter-view homography estimation and association. Under different viewing angles and partial occlusion, two box centers may refer to different physical locations. The candidate learns a bounded, crop-conditioned point inside each predicted box. It fits camera transport on other objects, and trains the held-out object to agree across views. A candidate must not explain its own geometric correspondence.

Input -> operator -> objective -> output:
`frozen predicted crop feature + box shape -> shared anchor head (bounded xy offset) -> bidirectional object-excluded DLT reprojection loss -> anchor coordinates used by unchanged MIA association code`.

There are no identity labels, labels derived from evaluation GT, or relation ground-truth fractions in inference input. The protocol still retains the exact MIA first-frame GT initialization, which is declared rather than called GT-free.

## Fixed protocol

* Existing complete-pair split: fit15 [23,25,28,29,39,44,45,51,53,63,66,69,70,74,78]; calibration5 [27,32,42,64,65]; dev5 [30,50,54,58,76]. Existing detector exposure and previous diagnostics prevent claims of unseen scenes.
* Training inputs: frozen P4 DINO CLS descriptors of predicted boxes, keyed by frame/view/detector. Labels: separate aligned PID file, used only to form training correspondences.
* Operator: shared MLP 389 -> 128 -> 2, zero-initialized last layer; bounded offset <=0.45 box width/height. Camera index and current tracker IDs do not enter the head.
* Training: 800 fixed updates, seed42, Adam 0.0003. Four disjoint object folds per frame; held-out targets cannot enter DLT support. Symmetric reprojection plus a small bounded-offset regularizer.
* Representation gate: final weights only, same samples for center/fixed point/learned point. Fit/calibration reprojection is a supervision-based diagnostic, not tracking performance. Calibration macro median error must improve >=5%, on >=3/5 pairs, before full host dev.
* Host gate: center and learned point use the same MIA runtime including any documented crash-only compatibility repair. Detector/weights/thresholds/GT initialization frozen. Full output MDA, IDF1, MOTA; no official test.
* Failure gate: audit feature/label separation, data-key alignment, fold exclusion, solver conditioning, nonfinite gradients, zero-init equivalence, and host attachment. Repair implementation errors once before making a STOP decision.

## Basic novelty screen

This is a provisional mechanism candidate, not a novelty claim. Comparison dimensions are input, operator, learning target, inference attachment, and supervision:

| Prior work | Mechanism overlap | Relevant difference / action |
|---|---|---|
| MIA-Net (TMM2023) | object points and inter-view geometry | Baseline uses box centers; this head learns correspondence points via object-excluded fitting. Direct same-task comparison required. |
| STCA | image matching for drone view transport | Does not establish that an object-conditioned box-interior point is learned with target exclusion. No claim for homography or image matching. |
| UAVST-HM | position/appearance/distribution cue fusion | Candidate modifies the spatial measurement itself, not matching weights. |
| SAME-MDMT | motion correction + position-IoU fusion | Basic abstract does not show the present learning objective. Full-chain uncertainty remains. |
| ObjectMatch CVPR2023 | learned canonical object correspondences | Tier2 high reviewer risk. Its RGB-D object registration and canonical 3D correspondences differ from a 2D box-interior, crossfit transport point. Need narrow claim. |
| Sim2Real Object-Centric Keypoints AAAI2022 | cross-view object keypoint consistency | Tier2 overlap; sim2real dense keypoint/descriptor formulation does not settle the current MDMT mechanism. |
| Multi-camera ground/feet point tracking (2006/2009/2011) | physical corresponding point under homography | Points and ground-plane matching are established. Never claim them as new. |
| HVC-Net ECCV2022 | homography/visibility/confidence learning | Planar single-object image tracking, not the current multi-object excluded target estimator; still related. |

Primary sources: MIA project https://github.com/VisDrone/Multi-Drone-Multi-Object-Detection-and-Tracking ; STCA local primary PDF snapshot `/home/chenhc/mdmot_research_20261002/sources/direct_stca_pdf.txt`; UAVST-HM https://www.mdpi.com/2504-446X/8/12/704 ; SAME https://ieeexplore.ieee.org/document/11486840/ ; ObjectMatch https://openaccess.thecvf.com/content/CVPR2023/html/Gumeli_ObjectMatch_Robust_Registration_Using_Canonical_Object_Correspondences_CVPR_2023_paper.html ; Sim2Real https://arxiv.org/abs/2202.00448 ; HVC https://arxiv.org/abs/2209.08924 .

Decision: `KEEP_FOR_TRAIN_CALIBRATION_SHORT_GATE; NOVELTY_PROVISIONAL`. No direct same-task full-chain collision was established by this basic screen. Adjacent overlaps narrow the candidate to its target-excluded learning target and baseline attachment. A basic screen is not a comprehensive novelty clearance.
