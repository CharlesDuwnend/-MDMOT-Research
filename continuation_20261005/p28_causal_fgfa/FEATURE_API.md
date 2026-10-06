# Native P26 FPN cache API

`{role}/FEATURE_INDEX.json` is a JSON object. `entries` lists one record per selected unique image; `by_key` maps frame key to the integer position in `entries`. `groups` retains the fixed plan, ordered `current,t-1,t-4,t-8`; its `current_metadata_key` identifies the metadata to use for the group's current-frame output. Each entry includes the original frame/role/pair/view/image path and SHA, archive path/SHA/bytes, FPN shapes, detection counts and JSON-safe metadata.

`{role}/features` is a workspace symlink into `/raid/datasets/chc_data/claude_try_MDMOT_p28_causal_fgfa_20261005/{role}/features`.

| NPZ field | Stored representation |
|---|---|
| f0..f4 | Unquantized float32 `[1,256,H,W]`; FPN output levels at strides 8,16,32,64,128 |
| det0..det2 | Float32 `[N,5]` arrays in pedestrian/bicycle/car order; native post-NMS original-image xyxy and score |

The NPZ is uncompressed `numpy.savez`; no leading batch dimension is removed and no half conversion is performed. Spatial dimensions use ceil division of the padded input at each FPN stride. For the first FIT image, the padded input is 768×1344 and the last level is 6×11. Do not assume floor division by 128.

The metadata object preserves `img_shape` (resized unpadded), `pad_shape` (tensor domain), `ori_shape`, `scale_factor` (four values), `img_norm_cfg`, filename and flip fields, and `batch_input_shape`. NumPy arrays/scalars and tuples are converted to JSON lists/scalars. Convert lists back to NumPy arrays only where the consuming API explicitly requires arrays; preserve the four distinct scale factors.

Each native detection array is produced by `detector.bbox_head.simple_test` on the exact FPN tensors being exported, followed by `bbox2result`. Original detector settings remain score 0.05, NMS IoU 0.6 and maximum 100 detections per image. There is no additional cache threshold or truncation. This fresh baseline is independent of the historical stage1 score>=0.1 streams. Detector-only native equivalence does not claim MIA tracking-output equivalence.

Native score semantics: the installed `BaseDenseHead._get_bboxes_single` applies `cfg.score_thr` before `_bbox_post_process` multiplies `score_factors`. Consequently a returned detection can have score slightly below 0.05. The FIT cache minimum is 0.04997823387384415, which is retained exactly. Do not apply an extra 0.05 postfilter. The first readback attempt incorrectly asserted a final-score floor of 0.05; its code/log are preserved under `fit/verification_attempt_001`. Only the checker was corrected to the native [0,1] score contract; export tensors, detections and the original export receipt were unchanged.

`SMOKE.json` compares the first image against a separate `detector.simple_test` call with exact array equality and rtol=atol=0; every class passed with maximum difference zero. The first archive is also checked by an exact NPZ round trip. `verify_fpn.py --role ROLE` subsequently audits all archive hashes, float32 finite values, shapes, counts, fixed group lags and metadata references without opening images, GT or a model. Its result is separate `{role}/FEATURE_VERIFY.json`.

`FEATURE_RECEIPT.json` binds the source/config/checkpoint/plans, imported runtime modules, all image hashes and all archive hashes. P26 source/config/checkpoint and plan hashes are checked before and after the export. The model is in eval mode, all parameters have `requires_grad=False`, and extraction/head inference runs under `no_grad`. Runtime metadata records TF32 settings rather than changing them silently.

The FIT launcher is `bash run_fpn.sh fit`, executed in persistent tmux `mdmt_p28_fpn`; it writes `fit/fpn.log` and `fit/fpn.exit`. It refuses existing log/exit targets, while the exporter refuses existing feature/index/receipt targets. A later authorized calibration export must be an explicit separate invocation. The FIT run does not open calibration images or any GT/labels.

Read-only independent code review by `/root/p24_protocol_review` found no material bug in equivalence, role isolation, freezing, metadata or provenance checks. It reviewed code only and did not read the dataset or change files.
