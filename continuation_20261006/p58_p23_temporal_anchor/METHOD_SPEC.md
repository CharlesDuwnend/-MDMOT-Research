# P58: P23 head-only causal temporal anchor signal

P57 showed that dense patch correspondence is a negative signal on the P23 identity
head. P58 tests a different causal representation: for each current P23 head-only
embedding, average the current observation with up to four strictly past
same-view observations from the same local track within an eight-frame age limit,
then L2-normalize. The numeric local-track ID is only a history pointer; it is not
an input feature or identity label.

Input -> operator -> output:
`P23 frozen head features + strict past local-track history -> normalized temporal
anchor -> same P4 cross-view candidate ranking`.

This is a train-free signal gate on the five P23 calibration pairs. It does not
change MIA, use geometry, read future frames, train, or read official val/test.
A learned residual adapter is authorized only if the fixed anchor clears the
pre-registered gate in `PREREG.json`.
