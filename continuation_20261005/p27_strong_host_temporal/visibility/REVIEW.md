# Independent read-only review

Reviewed by collaborating agent `/root/p24_protocol_review` on 2026-10-05 using only `audit_visibility.py` and `PROTOCOL.md`; the review did not read dataset contents or modify code.

No core strict-past, gap-reset, birth, or one-to-one error was found. State is updated after the current row's history is emitted; the latest prior detection is sufficient for an any-in-window predicate. Annotation gaps clear segment detection history while local-lifetime detection state remains separate. First annotation, reappearance, segment first detection, and lifetime first detection are distinct. Duplicate detector row and matched GT keys are rejected, and every matched label must be consumed by an annotated GT observation.

Review boundary: the main audit's diagnostic PASS alone is insufficient. Completion requires the separate saved-row recomputation and final hash receipt. `verify_visibility.py` provides this independent grouped-trajectory implementation; `VERIFY.json` and `RECEIPT.json` record the result.

The reviewer confirmed that the old stream score threshold 0.1 / P26 reported 0.05 distinction, local raw-ID scope, and cross-view annotation-convention-only appendix are explicit.
