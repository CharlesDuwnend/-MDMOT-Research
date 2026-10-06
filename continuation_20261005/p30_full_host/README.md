# P30 full continuous cache and MIA host integration

This isolated phase uses calibration pairs 27/32/42/64/65 only. It exports
every image's native FPN features and every frame with available lags
1/4/8, then evaluates the P29 fixed flow-centred residual adapter inside the
official MIA host. It retains the first-frame XML box/ID/confirmed-track/Kalman
initialization contract. No official val/test is read.
