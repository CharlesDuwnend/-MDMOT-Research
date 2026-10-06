# P52/P26 compatibility replay

The replay used only MDMT training pair `23` and its existing P26 fit replay.
P26 detector inference, ByteTrack state, first-frame XML initialization,
homography updates, geometry gates, owner continuity, and collision checks were
left in the pinned wrapper. The runtime hook changed candidate selection only
after the existing geometry gate and resolved native stage-3 features by frame,
box IoU, and view.

The result is an engineering compatibility diagnostic. It is not official
validation/test evidence, a new MOT score, or authorization to train longer.
