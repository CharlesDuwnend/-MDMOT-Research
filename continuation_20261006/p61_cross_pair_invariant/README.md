# P61 CPIA

P61 is a train-only test of whether the P23 identity residual can be made more
cross-pair generalizable by aligning update directions from two distinct fit-pair
environments. It is designed in response to P59's cross-pair generalization failure.
The candidate is adjacent to established domain-generalization and meta-learning
ReID, so the repository makes no novelty claim before evidence.

The first CPU attempt exposed an indexing defect in the diagnostic summary after
known-candidate filtering. It was repaired before the accepted contract; no
training or calibration number came from that failed attempt. The repaired CPU
contract covers all 15 fit-pair environments and 1,200 schedule groups.

The accepted A100 run completed 1,200 updates on physical GPU3. Calibration fell
from `0.4791246` to `0.4608756` (`-0.0182490`), with 0/5 pair wins. The final
three-round audit passed data split/receipt closure, zero-init and unknown-label
contracts, finite/unit outputs, and independent scorer replay. CPIA is stopped as
a valid negative result; no P26 attachment or official evaluation is authorized.
