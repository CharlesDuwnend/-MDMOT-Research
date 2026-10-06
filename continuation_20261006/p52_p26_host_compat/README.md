# P52 to P26 host compatibility

This stage is a development-only compatibility replay on MDMT fit pair 23.
The protected P26 detector, ByteTrack, first-frame XML initialization,
homography update, and ID collision checks remain in the P26 wrapper.  The
adapter only changes the candidate choice after the existing projected-center
geometry gate by scoring the same-frame native 256-D feature rows with the
P52 checkpoint.  Missing or invalid feature rows use the original geometry
choice and are counted explicitly.

No official validation or test data are read.  The P52 checkpoint is treated
as an adapter control after its collision audit; this stage does not establish
paper novelty or a formal MOT result.
